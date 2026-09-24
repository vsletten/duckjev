from __future__ import annotations

import json

import pyarrow as pa
import pytest

from duckjev import marshal


def test_noul_array_with_null() -> None:
    arr = marshal.noul_array([{"type": "noul", "noul": 0.25}, None])
    assert arr.type == pa.float64()
    assert arr.to_pylist() == [0.25, None]


def test_choice_array_type_map_and_nulls() -> None:
    a = {
        "type": "choice",
        "choice": "x",
        "probabilities": {"x": 0.9, "y": 0.1},
        "confidence": 0.8,
    }
    arr = marshal.choice_array([a, None, a])
    assert arr.type == marshal.CHOICE_TYPE
    assert arr.null_count == 1
    rows = arr.to_pylist()
    assert rows[1] is None
    assert rows[0]["choice"] == "x"
    assert rows[0]["confidence"] == 0.8
    assert rows[0]["probabilities"] == [("x", 0.9), ("y", 0.1)]
    probs = arr.field("probabilities")
    assert isinstance(probs, pa.MapArray)
    assert probs.offsets.type == pa.int32()
    assert probs.offsets.to_pylist() == [0, 2, 2, 4]  # the NULL row is an empty map


def test_score_array_legend_mapping() -> None:
    a = {
        "type": "score",
        "score": 1.43,
        "legend": {"0": "low", "1": "mid", "2": {"what": "high"}},
        "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43},
        "confidence": 0.35,
    }
    arr = marshal.score_array([None, a])
    assert arr.type == marshal.SCORE_TYPE
    row = arr.to_pylist()[1]
    assert row["score"] == 1.43
    assert dict(row["legend"]) == {"0": "low", "1": "mid", "2": '{"what": "high"}'}
    assert dict(row["probabilities"])["1"] == 0.57
    assert arr.to_pylist()[0] is None


def test_json_array_verbatim() -> None:
    answers = {"a": {"type": "noul", "noul": 0.5}}
    arr = marshal.json_array([answers, None])
    assert json.loads(arr[0].as_py()) == answers
    assert arr[1].as_py() is None


def test_empty_batch_arrays() -> None:
    assert len(marshal.choice_array([])) == 0
    assert len(marshal.score_array([])) == 0


def test_question_builders() -> None:
    assert marshal.noul_question("q?") == {"type": "noul", "instructions": "q?"}
    assert marshal.noul_question("q?", '{"true":"t","false":"f"}')["criteria"]["true"] == "t"
    c = marshal.choice_question("which?", '{"a":"A","b":null}')
    assert c["criteria"] == {"a": "A", "b": None}
    assert marshal.choice_question("which?", '["a","b"]')["criteria"] == {"a": None, "b": None}
    s = marshal.score_question("how?", '["lo","hi"]')
    assert s == {"type": "score", "instructions": "how?", "criteria": ["lo", "hi"]}


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (marshal.choice_question, ("q", "not json")),
        (marshal.choice_question, ("q", "{}")),
        (marshal.choice_question, ("q", json.dumps({str(i): None for i in range(256)}))),
        (marshal.score_question, ("q", '["only one"]')),
        (marshal.score_question, ("q", json.dumps([str(i) for i in range(11)]))),
        (marshal.noul_question, ("q", '"just a string"')),
        (marshal.parse_questions, ('{"a": {"type": "essay"}}',)),
    ],
)
def test_bad_specs_raise(fn, args) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(marshal.JevQuestionError):
        fn(*args)
