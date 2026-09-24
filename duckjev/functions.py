"""Arrow UDF bodies and their registration on a DuckDB connection.

DuckDB calls each UDF once per vector (about 2,048 rows), so each call is a
batch: build the (state, questions) pairs, hand them to ``JevClient.judge``
(which dedupes, consults the cache and fans out the misses), then marshal the
answers into the declared Arrow type in the vector's original order.

NULL handling is ``'special'``: DuckDB's default rejects a UDF that returns
NULL for a non-NULL input, and we want an empty or whitespace-only state to
come back NULL without a request. The UDF itself maps any NULL argument to a
NULL result, so the observable behavior is still null-in, null-out.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import duckdb
import pyarrow as pa
from duckdb import sqltypes as T

from . import marshal
from .client import JevClient

PROB_MAP = duckdb.map_type(T.VARCHAR, T.DOUBLE)
CHOICE_SQL_TYPE = duckdb.struct_type(
    {"choice": T.VARCHAR, "confidence": T.DOUBLE, "probabilities": PROB_MAP}
)
SCORE_SQL_TYPE = duckdb.struct_type(
    {
        "score": T.DOUBLE,
        "confidence": T.DOUBLE,
        "probabilities": PROB_MAP,
        "legend": duckdb.map_type(T.VARCHAR, T.VARCHAR),
    }
)

#: UDFs registered by register(); the typed noul arities are exposed through
#: an overloaded ``jev_noul`` macro because DuckDB does not overload Python UDFs.
UDF_NAMES = ("jev", "jev_noul2", "jev_noul3", "jev_choice", "jev_score")


def _column(arr: pa.Array | pa.ChunkedArray, n: int) -> list[Any]:
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    values = arr.to_pylist()
    if len(values) == 1 and n != 1:  # defensive: broadcast a constant
        values = values * n
    return values


def _columns(args: Sequence[pa.Array | pa.ChunkedArray]) -> list[list[Any]]:
    n = max(len(a) for a in args)
    return [_column(a, n) for a in args]


def _judge_rows(
    client: JevClient,
    states: list[Any],
    build: Callable[..., marshal.Questions],
    spec_cols: list[list[Any]],
) -> list[dict[str, Any] | None]:
    """Judge every usable row; returns the ``answers`` map per row or None."""
    n = len(states)
    specs: dict[tuple[Any, ...], marshal.Questions] = {}
    rows: list[int] = []
    batch: list[tuple[Any, marshal.Questions]] = []
    for i in range(n):
        state = states[i]
        spec_args = tuple(col[i] for col in spec_cols)
        if state is None or not str(state).strip() or any(a is None for a in spec_args):
            continue
        if spec_args not in specs:
            specs[spec_args] = build(*spec_args)
        rows.append(i)
        batch.append((state, specs[spec_args]))
    out: list[dict[str, Any] | None] = [None] * n
    if not batch:  # all NULL/empty: never touch the network (bind-time folding, no key)
        return out
    for i, answers in zip(rows, client.judge(batch), strict=True):
        out[i] = answers
    return out


def _one(answers: list[dict[str, Any] | None]) -> list[dict[str, Any] | None]:
    return [None if a is None else a[marshal.QID] for a in answers]


def make_udfs(client: JevClient) -> dict[str, Callable[..., pa.Array]]:
    def jev(state: pa.Array, questions_json: pa.Array) -> pa.Array:
        states, qs = _columns([state, questions_json])
        return marshal.json_array(_judge_rows(client, states, marshal.parse_questions, [qs]))

    def jev_noul2(state: pa.Array, instructions: pa.Array) -> pa.Array:
        states, ins = _columns([state, instructions])

        def build(i: str) -> marshal.Questions:
            return {marshal.QID: marshal.noul_question(i)}

        return marshal.noul_array(_one(_judge_rows(client, states, build, [ins])))

    def jev_noul3(state: pa.Array, instructions: pa.Array, criteria: pa.Array) -> pa.Array:
        states, ins, crit = _columns([state, instructions, criteria])

        def build(i: str, c: str) -> marshal.Questions:
            return {marshal.QID: marshal.noul_question(i, c)}

        return marshal.noul_array(_one(_judge_rows(client, states, build, [ins, crit])))

    def jev_choice(state: pa.Array, instructions: pa.Array, criteria: pa.Array) -> pa.Array:
        states, ins, crit = _columns([state, instructions, criteria])

        def build(i: str, c: str) -> marshal.Questions:
            return {marshal.QID: marshal.choice_question(i, c)}

        return marshal.choice_array(_one(_judge_rows(client, states, build, [ins, crit])))

    def jev_score(state: pa.Array, instructions: pa.Array, levels: pa.Array) -> pa.Array:
        states, ins, lev = _columns([state, instructions, levels])

        def build(i: str, lv: str) -> marshal.Questions:
            return {marshal.QID: marshal.score_question(i, lv)}

        return marshal.score_array(_one(_judge_rows(client, states, build, [ins, lev])))

    return {
        "jev": jev,
        "jev_noul2": jev_noul2,
        "jev_noul3": jev_noul3,
        "jev_choice": jev_choice,
        "jev_score": jev_score,
    }


_SIGNATURES: dict[str, tuple[list[duckdb.sqltype], duckdb.sqltype]] = {
    "jev": ([T.VARCHAR, T.VARCHAR], T.VARCHAR),
    "jev_noul2": ([T.VARCHAR, T.VARCHAR], T.DOUBLE),
    "jev_noul3": ([T.VARCHAR, T.VARCHAR, T.VARCHAR], T.DOUBLE),
    "jev_choice": ([T.VARCHAR, T.VARCHAR, T.VARCHAR], CHOICE_SQL_TYPE),
    "jev_score": ([T.VARCHAR, T.VARCHAR, T.VARCHAR], SCORE_SQL_TYPE),
}


def register_udfs(con: duckdb.DuckDBPyConnection, client: JevClient) -> None:
    for name, fn in make_udfs(client).items():
        try:
            con.remove_function(name)
        except (duckdb.InvalidInputException, duckdb.CatalogException):
            pass
        params, ret = _SIGNATURES[name]
        con.create_function(
            name, fn, params, ret, type="arrow", side_effects=False, null_handling="special"
        )
