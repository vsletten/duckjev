"""Question builders and conversion of Jev answers to Arrow arrays.

Everything here is pure: no network, no DuckDB connection. The Arrow types
declared here must match the DuckDB return types declared in ``functions.py``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pyarrow as pa

Answer = dict[str, Any]
Question = dict[str, Any]
Questions = dict[str, Question]

#: The question id used by the one-question typed functions.
QID = "q"

PROB_MAP = pa.map_(pa.string(), pa.float64())
LEGEND_MAP = pa.map_(pa.string(), pa.string())

CHOICE_TYPE = pa.struct(
    [
        pa.field("choice", pa.string()),
        pa.field("confidence", pa.float64()),
        pa.field("probabilities", PROB_MAP),
    ]
)
SCORE_TYPE = pa.struct(
    [
        pa.field("score", pa.float64()),
        pa.field("confidence", pa.float64()),
        pa.field("probabilities", PROB_MAP),
        pa.field("legend", LEGEND_MAP),
    ]
)


class JevQuestionError(ValueError):
    """A question spec passed from SQL is malformed."""


# --------------------------------------------------------------------------- questions


def _parse_json(text: str, what: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise JevQuestionError(f"{what} is not valid JSON: {exc}") from exc


def noul_question(instructions: str, criteria_json: str | None = None) -> Question:
    """Build a Noul question; ``criteria_json`` is an optional ``{"true":..,"false":..}``."""
    q: Question = {"type": "noul", "instructions": instructions}
    if criteria_json is not None:
        criteria = _parse_json(criteria_json, "noul criteria")
        if not isinstance(criteria, dict):
            raise JevQuestionError('noul criteria must be a JSON object {"true": .., "false": ..}')
        q["criteria"] = criteria
    return q


def choice_question(instructions: str, criteria_json: str) -> Question:
    """Build a Choice question from a JSON object (option -> description) or array of options."""
    criteria = _parse_json(criteria_json, "choice criteria")
    if isinstance(criteria, list):
        criteria = {str(k): None for k in criteria}
    if not isinstance(criteria, dict) or not criteria:
        raise JevQuestionError("choice criteria must be a non-empty JSON object or array")
    if len(criteria) > 255:
        raise JevQuestionError(f"choice supports at most 255 options, got {len(criteria)}")
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score_question(instructions: str, levels_json: str) -> Question:
    """Build a Score question from a JSON array of 2..10 ordered level descriptions."""
    levels = _parse_json(levels_json, "score levels")
    if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
        raise JevQuestionError("score levels must be a JSON array of 2 to 10 levels")
    return {"type": "score", "instructions": instructions, "criteria": levels}


def parse_questions(questions_json: str) -> Questions:
    """Parse the fused ``jev()`` questions map."""
    questions = _parse_json(questions_json, "questions")
    if not isinstance(questions, dict) or not questions:
        raise JevQuestionError("questions must be a non-empty JSON object of id -> question")
    for qid, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in ("noul", "choice", "score"):
            raise JevQuestionError(f"question {qid!r} needs type noul, choice or score")
    return questions


# --------------------------------------------------------------------------- answers


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _map_array(
    maps: Sequence[dict[str, Any] | None], value_type: pa.DataType, map_type: pa.DataType
) -> pa.MapArray:
    """Build a MAP array; a None entry becomes an empty map (the parent struct carries nulls)."""
    offsets = [0]
    keys: list[str] = []
    items: list[Any] = []
    for m in maps:
        if m:
            for k, v in m.items():
                keys.append(str(k))
                items.append(v)
        offsets.append(len(keys))
    return pa.MapArray.from_arrays(
        pa.array(offsets, pa.int32()),
        pa.array(keys, pa.string()),
        pa.array(items, value_type),
        type=map_type,
    )


def _mask(answers: Sequence[Answer | None]) -> pa.Array:
    return pa.array([a is None for a in answers], pa.bool_())


def noul_array(answers: Sequence[Answer | None]) -> pa.Array:
    return pa.array([None if a is None else float(a["noul"]) for a in answers], pa.float64())


def choice_array(answers: Sequence[Answer | None]) -> pa.StructArray:
    choice = pa.array([None if a is None else a["choice"] for a in answers], pa.string())
    conf = pa.array(
        [None if a is None else float(a.get("confidence", 0.0)) for a in answers], pa.float64()
    )
    probs = _map_array(
        [
            None if a is None else {k: float(v) for k, v in a["probabilities"].items()}
            for a in answers
        ],
        pa.float64(),
        PROB_MAP,
    )
    return pa.StructArray.from_arrays(
        [choice, conf, probs], fields=list(CHOICE_TYPE), mask=_mask(answers)
    )


def score_array(answers: Sequence[Answer | None]) -> pa.StructArray:
    score = pa.array([None if a is None else float(a["score"]) for a in answers], pa.float64())
    conf = pa.array(
        [None if a is None else float(a.get("confidence", 0.0)) for a in answers], pa.float64()
    )
    probs = _map_array(
        [
            None if a is None else {k: float(v) for k, v in a["probabilities"].items()}
            for a in answers
        ],
        pa.float64(),
        PROB_MAP,
    )
    legend = _map_array(
        [
            None if a is None else {k: _as_text(v) for k, v in a.get("legend", {}).items()}
            for a in answers
        ],
        pa.string(),
        LEGEND_MAP,
    )
    return pa.StructArray.from_arrays(
        [score, conf, probs, legend], fields=list(SCORE_TYPE), mask=_mask(answers)
    )


def json_array(answers: Sequence[dict[str, Answer] | None]) -> pa.Array:
    """The fused ``jev()`` result: the API's ``answers`` map as a JSON string, verbatim."""
    return pa.array(
        [None if a is None else json.dumps(a, separators=(",", ":")) for a in answers], pa.string()
    )
