"""Question builders and conversion of Jev answers to Arrow arrays.

Everything here is pure: no network, no DuckDB connection. The Arrow types
declared here must match the DuckDB return types declared in ``functions.py``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
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
#: One extracted field: the selected candidate (NULL when ``none`` wins or nothing was
#: asked), the probability of what was returned, the ``none`` mass, Jev's confidence,
#: how many candidates were offered, and the distribution over the candidates.
EXTRACT_FIELD_TYPE = pa.struct(
    [
        pa.field("value", pa.string()),
        pa.field("p", pa.float64()),
        pa.field("p_none", pa.float64()),
        pa.field("confidence", pa.float64()),
        pa.field("n_candidates", pa.int32()),
        pa.field("probabilities", PROB_MAP),
    ]
)
EXTRACT_TYPE = pa.map_(pa.string(), EXTRACT_FIELD_TYPE)

#: Choice caps a question at 255 options; one is reserved for ``none``.
MAX_OPTIONS = 255
NONE_KEY = "none"
NONE_KEY_FALLBACK = "none of these"
NONE_DESCRIPTION = "None of these candidates is the requested value."


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


@dataclass(frozen=True)
class ExtractField:
    """One field of a ``jev_extract`` spec after parsing: its candidates and ``none`` key."""

    name: str
    candidates: tuple[str, ...]
    none_key: str | None


def _candidate_options(name: str, raw: Any) -> dict[str, Any]:
    """Candidates as ordered ``option -> description``: stripped, deduped, empties dropped."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        pairs = list(raw.items())
    elif isinstance(raw, list):
        pairs = [(c, None) for c in raw]
    else:
        raise JevQuestionError(f"field {name!r}: candidates must be a JSON array or object")
    options: dict[str, Any] = {}
    for cand, desc in pairs:
        if cand is None or isinstance(cand, (dict, list)):
            continue
        text = (cand if isinstance(cand, str) else json.dumps(cand)).strip()
        if text and text not in options:
            options[text] = desc
    return options


def extract_questions(spec_json: str) -> tuple[Questions, list[ExtractField]]:
    """Parse a ``jev_extract`` spec into one Choice question per field that has candidates.

    The spec is ``{field: {"instructions": .., "candidates": [..] | {cand: description},
    "none": description | false}}``. Each question's options are the candidate strings
    themselves plus a ``none`` option (unless ``"none": false``), so the answer is always a
    verbatim copy of a candidate. Fields with no candidates are kept but not asked.
    """
    spec = _parse_json(spec_json, "extract spec")
    if not isinstance(spec, dict) or not spec:
        raise JevQuestionError("extract spec must be a non-empty JSON object of field -> spec")
    questions: Questions = {}
    fields: list[ExtractField] = []
    for name, f in spec.items():
        if not isinstance(f, dict) or not f.get("instructions"):
            raise JevQuestionError(f"field {name!r} needs an object with instructions")
        options = _candidate_options(name, f.get("candidates"))
        none = f.get("none", NONE_DESCRIPTION)
        none_key = None
        if none is not False:
            none_key = NONE_KEY_FALLBACK if NONE_KEY in options else NONE_KEY
            if none_key in options:
                raise JevQuestionError(f"field {name!r}: candidates collide with the none option")
        limit = MAX_OPTIONS - (none_key is not None)
        if len(options) > limit:
            raise JevQuestionError(
                f"field {name!r} has {len(options)} distinct candidates, more than the "
                f"{limit} a Choice can take{' beside none' if none_key else ''}. "
                f"Narrow the list in SQL, e.g. candidates[1:{limit}]"
            )
        fields.append(ExtractField(name, tuple(options), none_key))
        if not options:
            continue
        criteria = dict(options)
        if none_key is not None:
            criteria[none_key] = NONE_DESCRIPTION if none is None or none is True else none
        questions[name] = {
            "type": "choice",
            "instructions": f["instructions"],
            "criteria": criteria,
        }
    return questions, fields


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


def _extract_field(field: ExtractField, answer: Answer | None) -> dict[str, Any]:
    if answer is None:  # no candidates: the field was not asked
        return {
            "value": None,
            "p": None,
            "p_none": None,
            "confidence": None,
            "n_candidates": len(field.candidates),
            "probabilities": [],
        }
    probs = {str(k): float(v) for k, v in answer["probabilities"].items()}
    choice = answer["choice"]
    return {
        "value": None if choice == field.none_key else choice,
        "p": probs.get(choice, 0.0),
        "p_none": 0.0 if field.none_key is None else probs.get(field.none_key, 0.0),
        "confidence": float(answer.get("confidence", 0.0)),
        "n_candidates": len(field.candidates),
        "probabilities": [(c, probs.get(c, 0.0)) for c in field.candidates],
    }


def extract_array(
    fields: Sequence[list[ExtractField] | None], answers: Sequence[dict[str, Answer] | None]
) -> pa.MapArray:
    """``MAP(field -> EXTRACT_FIELD_TYPE)`` per row; a row with no parsed spec is NULL."""
    rows: list[list[tuple[str, dict[str, Any]]] | None] = []
    for row_fields, row_answers in zip(fields, answers, strict=True):
        if row_fields is None:
            rows.append(None)
            continue
        got = row_answers or {}
        rows.append([(f.name, _extract_field(f, got.get(f.name))) for f in row_fields])
    return pa.array(rows, type=EXTRACT_TYPE)
