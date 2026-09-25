from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest
from fake import FakeTransport

import duckjev

CRITERIA = json.dumps({"card": "about a card", "transfer": "about a transfer", "other": None})
LEVELS = json.dumps(["calm", "annoyed", "angry"])
ROWS = [
    ("my card is lost card",),
    ("where is my transfer",),
    ("my card is lost card",),
    ("hello there",),
    (None,),
    ("",),
]


@pytest.fixture
def fake() -> FakeTransport:
    return FakeTransport()


@pytest.fixture
def con(fake: FakeTransport) -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    duckjev.register(c, cache=False, api_key="test-key", transport=fake)
    c.execute("CREATE TABLE t(text VARCHAR)")
    c.executemany("INSERT INTO t VALUES (?)", ROWS)
    c.execute(
        "CREATE TABLE p AS SELECT p::DOUBLE AS p FROM (VALUES (0.9), (0.5), (0.2), (1.0)) v(p)"
    )
    return c


def test_register_installs_functions_and_macros(con: duckdb.DuckDBPyConnection) -> None:
    names = {
        r[0]
        for r in con.execute(
            "SELECT DISTINCT function_name FROM duckdb_functions() "
            "WHERE function_name LIKE 'jev%' OR function_name LIKE 'sem_%' "
            "OR function_name LIKE 'expected_count%'"
        ).fetchall()
    }
    assert {
        "jev",
        "jev_noul",
        "jev_choice",
        "jev_score",
        "sem_where",
        "expected_count",
        "expected_count_var",
        "expected_count_stderr",
        "jev_argmax",
        "jev_p",
        "jev_runner_up",
        "jev_extract",
        "jev_field",
        "jev_money_spans",
        "jev_date_spans",
        "jev_line_windows",
    } <= names


def test_jev_noul_both_arities(con: duckdb.DuckDBPyConnection, fake: FakeTransport) -> None:
    rows = con.execute(
        "SELECT jev_noul(text, 'about a card?'), "
        """jev_noul(text, 'about a card?', '{"true": "yes card", "false": "no"}') FROM t"""
    ).fetchall()
    assert rows == [(0.8, 0.8), (0.1, 0.1), (0.8, 0.8), (0.1, 0.1), (None, None), (None, None)]
    with_criteria = [r for r in fake.requests if "criteria" in r["questions"]["q"]]
    assert with_criteria and with_criteria[0]["questions"]["q"]["criteria"]["true"] == "yes card"
    # 3 unique non-empty states per function, NULL and '' never sent
    assert len(fake.requests) == 6


def test_jev_choice_struct(con: duckdb.DuckDBPyConnection) -> None:
    rel = con.execute(f"SELECT jev_choice(text, 'topic?', '{CRITERIA}') AS c FROM t")
    assert "STRUCT(choice VARCHAR, confidence DOUBLE, probabilities MAP(VARCHAR, DOUBLE))" in str(
        rel.description[0][1]
    )
    rows = [r[0] for r in rel.fetchall()]
    assert rows[0]["choice"] == "card"
    assert rows[0]["probabilities"]["card"] == pytest.approx(0.7)
    assert rows[1]["choice"] == "transfer"
    assert rows[4] is None and rows[5] is None


def test_jev_score_struct(con: duckdb.DuckDBPyConnection) -> None:
    row = con.execute(
        f"SELECT s.score, s.legend['2'], s.probabilities['2'], s.confidence "
        f"FROM (SELECT jev_score(text, 'how upset?', '{LEVELS}') s FROM t LIMIT 1)"
    ).fetchone()
    assert row == (2.0, "angry", 1.0, 1.0)


def test_fused_jev(con: duckdb.DuckDBPyConnection, fake: FakeTransport) -> None:
    questions = json.dumps(
        {
            "card": {"type": "noul", "instructions": "card?"},
            "topic": {"type": "choice", "instructions": "topic?", "criteria": json.loads(CRITERIA)},
        }
    )
    rows = con.execute(f"SELECT jev(text, '{questions}') FROM t").fetchall()
    first = json.loads(rows[0][0])
    assert first["card"]["noul"] == 0.8 and first["topic"]["choice"] == "card"
    assert rows[4][0] is None
    assert len(fake.requests) == 3  # both questions in one request per unique state
    assert all(set(r["questions"]) == {"card", "topic"} for r in fake.requests)
    val = con.execute(
        f"SELECT (jev(text, '{questions}')::JSON ->> '$.topic.choice') FROM t LIMIT 1"
    ).fetchone()
    assert val == ("card",)


def test_choice_helpers_and_runner_up(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        f"""SELECT jev_argmax(c), jev_p(c, 'card'), jev_p(c, 'missing'), jev_runner_up(c)
            FROM (SELECT jev_choice(text, 'topic?', '{CRITERIA}') c FROM t WHERE text <> '')"""
    ).fetchall()
    # card 0.7, transfer 0.15, other 0.15 -> ties broken by ORDER BY; runner-up is never argmax
    assert rows[0][0] == "card"
    assert rows[0][1] == pytest.approx(0.7)
    assert rows[0][2] == 0.0
    assert rows[0][3] in {"transfer", "other"}
    assert rows[1][0] == "transfer" and rows[1][3] in {"card", "other"}


def test_runner_up_distinct_probabilities() -> None:
    fake = FakeTransport(choice=lambda s, opts: {"a": 0.5, "b": 0.1, "c": 0.4})
    c = duckdb.connect()
    duckjev.register(c, cache=False, api_key="k", transport=fake)
    row = c.execute(
        """SELECT jev_argmax(x), jev_runner_up(x)
           FROM (SELECT jev_choice('s', 'q', '["a","b","c"]') x)"""
    ).fetchone()
    assert row == ("a", "c")


def test_soft_group_by_sums(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        f"""SELECT e.key AS intent, SUM(e.value) AS expected_rows
            FROM t, UNNEST(map_entries(jev_choice(t.text, 'topic?', '{CRITERIA}').probabilities))
                    AS u(e)
            GROUP BY intent ORDER BY intent"""
    ).fetchall()
    got = dict(rows)
    # 4 judged rows: card, transfer, card, other(first option fallback -> card)
    # card: 0.7 + 0.15 + 0.7 + 0.7 ; transfer: 0.15 + 0.7 + 0.15 + 0.15 ; other: 0.15 * 4
    assert got["card"] == pytest.approx(2.25)
    assert got["transfer"] == pytest.approx(1.15)
    assert got["other"] == pytest.approx(0.6)
    assert sum(got.values()) == pytest.approx(4.0)


def test_calibrated_aggregation_macros(con: duckdb.DuckDBPyConnection) -> None:
    ec, var, se = con.execute(
        "SELECT expected_count(p), expected_count_var(p), expected_count_stderr(p) FROM p"
    ).fetchone()
    assert ec == pytest.approx(2.6)
    assert var == pytest.approx(0.09 + 0.25 + 0.16 + 0.0)
    assert se == pytest.approx(0.5**0.5 * 1.0)


def test_sem_where(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        "SELECT text FROM t WHERE sem_where(text, 'about a card?', 0.5) ORDER BY text"
    ).fetchall()
    assert rows == [("my card is lost card",), ("my card is lost card",)]


def test_register_without_key_then_first_call_raises() -> None:
    fake = FakeTransport()
    c = duckdb.connect()
    duckjev.register(c, cache=False, transport=fake)  # no key anywhere: register still works
    assert c.execute(
        "SELECT expected_count(x::DOUBLE) FROM (VALUES (0.5), (0.25)) v(x)"
    ).fetchone() == (0.75,)
    # all-NULL / empty input never needs the key
    assert c.execute("SELECT jev_noul(NULL::VARCHAR, 'q'), jev_noul('', 'q')").fetchone() == (
        None,
        None,
    )
    with pytest.raises(duckdb.Error, match="TYPESAFE_API_KEY is not set"):
        c.execute("SELECT jev_noul('my card', 'card?')").fetchall()
    assert fake.attempts == 0


def test_bad_criteria_surfaces_as_sql_error(con: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(duckdb.Error, match="not valid JSON"):
        con.execute("SELECT jev_choice(text, 'q', 'nope') FROM t").fetchall()


def test_cache_rerun_is_free_and_inspectable(tmp_path: Path) -> None:
    fake = FakeTransport()
    c = duckdb.connect()
    duckjev.register(c, api_key="k", transport=fake, cache_path=tmp_path / "cache.duckdb")
    c.execute("CREATE TABLE t AS SELECT 'row ' || (i % 7) AS text FROM range(5000) r(i)")
    q = f"SELECT jev_choice(text, 'topic?', '{CRITERIA}').choice FROM t"
    first = c.execute(q).fetchall()
    assert len(fake.requests) == 7  # deduped within and across vectors via the cache
    duckjev.usage(reset=True)
    assert c.execute(q).fetchall() == first
    u = duckjev.usage()
    assert u["requests"] == 0 and u["est_usd"] == 0 and u["cache_hits"] > 0
    duckjev.flush(c)
    duckjev.cache_table(c, "cache_rows")
    n, model = c.execute("SELECT count(*), any_value(model) FROM cache_rows").fetchone()
    assert (n, model) == (7, "jev-1.13.0")
    assert c.execute("SELECT answers->>'$.q.type' FROM cache_rows LIMIT 1").fetchone() == (
        "choice",
    )


def test_multiple_vectors_keep_order(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        "CREATE TABLE big AS SELECT i, CASE WHEN i % 3 = 0 THEN 'card ' || i ELSE 'x ' || i END "
        "AS text FROM range(4500) r(i)"
    )
    rows = con.execute("SELECT i, jev_noul(text, 'card?') FROM big ORDER BY i").fetchall()
    assert len(rows) == 4500
    assert all((p == 0.8) == (i % 3 == 0) for i, p in rows)
