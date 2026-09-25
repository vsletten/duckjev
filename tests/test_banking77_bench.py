"""Offline checks of bench/banking77.py: round configs, question building, flattening, and the
whole pipeline (run, rescore, report) against the fake transport on a tiny split."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import duckdb
import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_bench():
    path = ROOT / "bench" / "banking77.py"
    spec = importlib.util.spec_from_file_location("banking77_bench", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["banking77_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load_bench()


def test_divisions_partition_the_intents() -> None:
    intents = list(bench.intent_criteria("v1"))
    listed = [i for d in bench.divisions() for i in d["intents"]]
    assert sorted(listed) == sorted(intents)
    assert len(listed) == len(set(listed)) == 77
    assert all(d["label"] and d["what"] for d in bench.divisions())


def test_every_gloss_set_covers_the_intents_in_v1_order() -> None:
    v1 = list(bench.intent_criteria("v1"))
    for name in bench.CRITERIA_FILES:
        assert list(bench.intent_criteria(name)) == v1, name
    v2 = bench.intent_criteria("v2")
    structured = {k for k, v in v2.items() if isinstance(v, dict)}
    assert structured
    assert all({"what", "not_for", "examples"} <= set(v2[k]) for k in structured)
    assert all(isinstance(v, str) and v for v in bench.intent_criteria("short").values())


def test_flat_questions() -> None:
    q = bench.questions_for(bench.ROUNDS["R0"])
    assert list(q) == ["intent"]  # R0 asks the top-up question separately
    assert q["intent"]["type"] == "choice" and len(q["intent"]["criteria"]) == 77
    rev = bench.questions_for(bench.Round("x", "v1", "flat", "reverse", "fused"))
    assert list(rev["intent"]["criteria"]) == list(reversed(q["intent"]["criteria"]))
    assert rev["topup"] == {"type": "noul", "instructions": bench.TOPUP_Q}


def test_two_level_questions() -> None:
    q = bench.questions_for(bench.ROUNDS["R2"])
    divs = bench.divisions()
    assert list(q) == ["division", *[f"intent_{d['key']}" for d in divs]]
    assert list(q["division"]["criteria"]) == [d["key"] for d in divs]
    for d in divs:
        leaf = q[f"intent_{d['key']}"]
        assert list(leaf["criteria"]) == d["intents"]
        assert d["label"] in leaf["instructions"]
    rev = bench.questions_for(bench.Round("x", "v2", "two_level", "reverse", "none"))
    assert list(rev["division"]["criteria"]) == [d["key"] for d in reversed(divs)]
    first = divs[0]
    assert list(rev[f"intent_{first['key']}"]["criteria"]) == list(reversed(first["intents"]))


def test_flatten_flat() -> None:
    answers = {
        "intent": {
            "type": "choice",
            "choice": "a1",
            "confidence": 0.7,
            "probabilities": {"a1": 0.8, "b1": 0.2},
        },
        "topup": {"type": "noul", "noul": 0.3},
    }
    rec = bench.flatten(answers, bench.ROUNDS["R0"], {"a1": "A", "b1": "B"})
    assert rec["choice"] == rec["greedy_choice"] == rec["answer"] == "a1"
    assert rec["top_p"] == 0.8 and rec["confidence"] == 0.7 and rec["division"] == "A"
    assert rec["deferred"] is False and rec["topup_p"] == 0.3


def test_flatten_two_level_product_argmax_and_deferral() -> None:
    div_of = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    answers = {
        "division": {
            "type": "choice",
            "choice": "A",
            "confidence": 0.3,
            "probabilities": {"A": 0.55, "B": 0.45},
        },
        "intent_A": {
            "type": "choice",
            "choice": "a1",
            "confidence": 0.1,
            "probabilities": {"a1": 0.5, "a2": 0.5},
        },
        "intent_B": {
            "type": "choice",
            "choice": "b1",
            "confidence": 0.9,
            "probabilities": {"b1": 0.9, "b2": 0.1},
        },
    }
    rec = bench.flatten(answers, bench.ROUNDS["R2"], div_of)
    assert rec["choice"] == "b1"  # 0.45 × 0.9 beats 0.55 × 0.5
    assert rec["greedy_choice"] == "a1"
    assert rec["top_p"] == pytest.approx(0.405)
    assert rec["confidence"] == 0.9  # the intent Choice that produced the answer
    assert sum(rec["probabilities"].values()) == pytest.approx(1.0)
    assert rec["deferred"] is True and rec["answer"] == "A" and rec["division_p"] == 0.55
    assert rec["topup_p"] is None
    answers["division"]["confidence"] = 0.95
    rec = bench.flatten(answers, bench.ROUNDS["R2"], div_of)
    assert rec["deferred"] is False and rec["answer"] == "b1"


@pytest.fixture
def tiny_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A 24-row dev split (three messages for the first intent of each division) in a tmp DATA."""
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    for d in bench.divisions():
        intent = d["intents"][0]
        for j in range(3):
            text = f"message {j} about {intent.replace('_', ' ')}"
            rows.append({"text": text, "label_text": intent})
    con = duckdb.connect()
    con.register("r", pa.Table.from_pylist(rows))
    con.execute("COPY r TO ? (FORMAT parquet)", [str(data / "banking77_dev.parquet")])
    monkeypatch.setattr(bench, "DATA", data)
    monkeypatch.setattr(bench, "RUNS_FILE", tmp_path / "runs.json")
    monkeypatch.setattr(bench, "RESULTS", tmp_path / "banking77.md")
    return data


def test_dry_run_writes_suffixed_files_and_records_nothing(tiny_data: Path) -> None:
    assert bench.main(["run", "R2", "--split", "dev", "--dry-run"]) == 0
    assert (tiny_data / "summary_dev_R2_dry.json").exists()
    assert (tiny_data / "scored_dev_R2_dry.parquet").exists()
    assert (tiny_data / "cache_dev_R2_dry.duckdb").exists()
    assert not bench.RUNS_FILE.exists()
    s = json.loads((tiny_data / "summary_dev_R2_dry.json").read_text())
    assert s["rows"] == 24 and s["questions"] == 9 and 0 <= s["accuracy"] <= 1
    assert s["deferral"] is not None and s["rerun_usage"]["requests"] == 0
    assert s["rerun_identical_rows"] == 24

    assert bench.main(["run", "R0", "--split", "dev", "--dry-run", "--limit", "10"]) == 0
    s0 = json.loads((tiny_data / "summary_dev_R0_n10_dry.json").read_text())
    assert s0["rows"] == 10 and s0["topup"] is not None and "projected_full_usd" in s0
    assert s0["topup_usage"]["requests"] == 10


def test_full_run_needs_a_preflight(tiny_data: Path) -> None:
    assert bench.main(["run", "R1", "--split", "dev"]) == 2
    assert not (tiny_data / "scored_dev_R1.parquet").exists()


def test_rescore_and_report_from_the_run_log(tiny_data: Path) -> None:
    bench.main(["run", "R0", "--split", "dev", "--dry-run"])
    bench.main(["run", "R2", "--split", "dev", "--dry-run"])
    runs = {}
    for rnd in ("R0", "R2"):
        s = json.loads((tiny_data / f"summary_dev_{rnd}_dry.json").read_text())
        live_cache = tiny_data / f"cache_dev_{rnd}.duckdb"
        shutil.copy(tiny_data / f"cache_dev_{rnd}_dry.duckdb", live_cache)
        runs[f"dev/{rnd}"] = s
        runs[f"test/{rnd}"] = {**s, "split": "test"}  # no cache for test, so rescore skips it
    legacy = {k: v for k, v in runs["test/R0"].items() if not isinstance(v, (list, dict))}
    runs["pr1/R0"] = {**legacy, "legacy": True, "split": "PR #1 test", "accuracy": 0.5}
    bench.RUNS_FILE.write_text(json.dumps(runs))
    before = {k: r["accuracy"] for k, r in runs.items()}

    assert bench.main(["rescore"]) == 0
    after = json.loads(bench.RUNS_FILE.read_text())
    assert {k: r["accuracy"] for k, r in after.items()} == before
    assert (tiny_data / "scored_dev_R2.parquet").exists()

    assert bench.main(["report"]) == 0
    text = bench.RESULTS.read_text()
    best = bench.chosen_round(after)
    assert f"## Held-out test split, round {best}" in text
    assert "## Tuning rounds on the dev split" in text
    assert "dev/R0" in text and "dev/R2" in text
    assert "| PR #1 test/R0 | **0.500** |" in text  # the imported run leads the held-out table
    assert "### Two-level rounds: deferral to the division" in text


def test_chosen_round_rule() -> None:
    runs = {
        "dev/R0": {"round": "R0", "accuracy": 0.80, "input_tokens_per_request": 2000},
        "dev/R1": {"round": "R1", "accuracy": 0.85, "input_tokens_per_request": 5000},
        "dev/R2": {"round": "R2", "accuracy": 0.85, "input_tokens_per_request": 4000},
        "test/R1": {"round": "R1", "accuracy": 0.99, "input_tokens_per_request": 1},
    }
    assert bench.chosen_round(runs) == "R2"  # ties go to fewer tokens; test runs never count
