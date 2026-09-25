"""Offline checks of bench/entity_matching.py: corpus and round configs, the metrics helpers,
and the whole pipeline (coverage, run, demo, rescore, report) on a tiny corpus against the
fake transport."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_bench():
    path = ROOT / "bench" / "entity_matching.py"
    spec = importlib.util.spec_from_file_location("entity_matching_bench", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["entity_matching_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load_bench()


def test_corpora_and_rounds_are_well_formed() -> None:
    for key, c in bench.CORPORA.items():
        assert c.fields and c.block and c.plain and c.colleague, key
        assert set(c.criteria) == {"true", "false"}
        assert 2 <= len(c.topk_levels) <= 10
    for name, r in bench.ROUNDS.items():
        assert r.wording in ("plain", "colleague") and r.order in ("ab", "ba"), name
        assert (r.layout, r.order) in bench.STATE_SQL


def test_judge_params_follow_the_round() -> None:
    sql, params = bench.judge_params("abt", bench.ROUNDS["R0"])
    assert params == {"q": bench.CORPORA["abt"].plain}
    assert "jev_noul(jev_pair(a_rec, b_rec), $q)" in sql and "{table}" in sql
    sql, params = bench.judge_params("dblp", bench.ROUNDS["R3"])
    assert params["q"] == bench.CORPORA["dblp"].colleague and "criteria" not in params
    assert "rec_text(b_rec) || chr(10) || 'B: ' || rec_text(a_rec)" in sql
    sql, params = bench.judge_params("abt", bench.ROUNDS["R4"])
    assert json.loads(params["criteria"]) == bench.CORPORA["abt"].criteria
    assert "$criteria)" in sql


def test_auroc_and_prf() -> None:
    assert bench._auroc([(0.9, 1), (0.8, 0), (0.7, 1), (0.2, 0)]) == pytest.approx(0.75)
    assert bench._auroc([(0.5, 1), (0.5, 0)]) == pytest.approx(0.5)  # a tie counts half
    assert bench._auroc([(0.9, 1), (0.1, 1)]) is None
    assert bench._prf(0, 0, 5) == {
        "predicted": 0,
        "true_positives": 0,
        "precision": None,
        "recall": 0.0,
        "f1": 0.0,
    }
    assert bench._prf(4, 3, 6)["f1"] == pytest.approx(0.6)


def test_two_sides_handles_both_layouts() -> None:
    assert bench._two_sides("A: x; y\nB: z") == ("x; y", "z")
    a, b = bench._two_sides(json.dumps({"a": {"name": "n"}, "b": "s"}), width=6)
    assert (a, b) == ('{"name', '"s"')


def test_chosen_round_rule() -> None:
    runs = {
        "abt/dev/R0": {"round": "R0", "at_threshold": {"f1": 0.9}, "input_tokens_per_request": 400},
        "abt/dev/R1": {
            "round": "R1",
            "at_threshold": {"f1": 0.95},
            "input_tokens_per_request": 500,
        },
        "abt/dev/R2": {
            "round": "R2",
            "at_threshold": {"f1": 0.95},
            "input_tokens_per_request": 450,
        },
        "dblp/dev/R0": {"round": "R0", "at_threshold": {"f1": 0.99}, "input_tokens_per_request": 1},
        "abt/test/R0": {"round": "R0", "at_threshold": {"f1": 0.99}, "input_tokens_per_request": 1},
    }
    assert bench.chosen_round(runs, "abt") == "R2"  # ties go to fewer tokens; other keys ignored
    assert bench.chosen_round(runs, "dblp") == "R0"


@pytest.fixture
def tiny_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A four-by-four Abt-Buy shaped corpus with labeled pairs, under a tmp DATA directory."""
    data = tmp_path / "data"
    data.mkdir()
    a = [
        (0, "sony turntable x1", "belt drive turntable", "99.0"),
        (1, "sony walkman w2", "portable player", ""),
        (2, "bose speaker s3", "bookshelf speaker", "199.0"),
        (3, "canon camera c4", "digital camera body", "899.0"),
    ]
    b = [
        (0, "sony turntable x1 belt drive", "", "95.0"),
        (1, "sony portable player", "", ""),
        (2, "bose speaker s3 system", "two bookshelf speakers", ""),
        (3, "canon lens 50mm", "", "120.0"),
    ]
    for name, rows in (("tableA", a), ("tableB", b)):
        lines = ["id,name,description,price"] + [f'{i},"{n}","{d}",{p}' for i, n, d, p in rows]
        (data / f"abt_{name}.csv").write_text("\n".join(lines) + "\n")
    pairs = {
        "train": [(0, 1, 0), (2, 3, 0)],
        "valid": [(0, 0, 1), (0, 1, 0), (1, 1, 0), (2, 2, 1), (1, 0, 0), (3, 3, 0)],
        "test": [(0, 0, 1), (1, 1, 0), (2, 2, 1), (3, 3, 0), (2, 3, 0)],
    }
    for split, rows in pairs.items():
        lines = ["ltable_id,rtable_id,label"] + [f"{lid},{rid},{y}" for lid, rid, y in rows]
        (data / f"abt_{split}.csv").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(bench, "DATA", data)
    monkeypatch.setattr(bench, "RUNS_FILE", tmp_path / "runs.json")
    monkeypatch.setattr(bench, "RESULTS", tmp_path / "entity_matching.md")
    monkeypatch.setattr(bench, "CORPORA", {"abt": bench.CORPORA["abt"]})
    return data


def test_coverage_counts_blocked_gold(tiny_data: Path) -> None:
    assert bench.main(["coverage", "--record"]) == 0
    row = json.loads(bench.RUNS_FILE.read_text())["coverage"]["rows"][0]
    # first-word blocks: sony (2 x 2), bose (1 x 1), canon (1 x 1) = 6 pairs; gold (0,0) (2,2) kept
    assert (row["cross_pairs"], row["blocked_pairs"]) == (16, 6)
    assert (row["gold_pairs"], row["gold_pairs_blocked"]) == (2, 2)


def test_dry_run_writes_suffixed_files_and_records_nothing(tiny_data: Path) -> None:
    assert bench.main(["run", "R2", "--corpus", "abt", "--split", "dev", "--dry-run"]) == 0
    s = json.loads((tiny_data / "summary_abt_dev_R2_dry.json").read_text())
    assert (tiny_data / "judged_abt_dev_R2_dry.parquet").exists()
    assert (tiny_data / "cache_abt_dev_R2_dry.duckdb").exists()
    assert not bench.RUNS_FILE.exists()
    assert s["pairs"] == 6 and s["positives"] == 2 and s["rerun_identical_rows"] == 6
    assert s["at_threshold"]["predicted"] + 0 >= 0 and len(s["sweep"]) == 19
    assert len(s["false_negatives"]) == 2 and len(s["false_positives"]) == 4
    assert (
        bench.main(["run", "R0", "--corpus", "abt", "--split", "dev", "--dry-run", "--limit", "3"])
        == 0
    )
    s0 = json.loads((tiny_data / "summary_abt_dev_R0_n3_dry.json").read_text())
    assert s0["pairs"] == 3 and "projected_full_usd" in s0


def test_full_run_needs_a_preflight(tiny_data: Path) -> None:
    assert bench.main(["run", "R1", "--corpus", "abt", "--split", "dev"]) == 2
    assert not (tiny_data / "judged_abt_dev_R1.parquet").exists()


def test_demo_rescore_and_report(tiny_data: Path) -> None:
    for rnd in ("R0", "R2"):
        assert bench.main(["run", rnd, "--corpus", "abt", "--split", "dev", "--dry-run"]) == 0
    assert bench.main(["demo", "--corpus", "abt", "--dry-run", "--n", "2"]) == 0
    demo = json.loads((tiny_data / "summary_abt_demo_dry.json").read_text())
    assert demo["join"]["blocked_pairs"] >= 1 and len(demo["topk"]) == 2
    assert demo["dedup"]["rows_in"] >= 2

    runs: dict = {}
    for rnd in ("R0", "R2"):
        s = json.loads((tiny_data / f"summary_abt_dev_{rnd}_dry.json").read_text())
        shutil.copy(
            tiny_data / f"cache_abt_dev_{rnd}_dry.duckdb", tiny_data / f"cache_abt_dev_{rnd}.duckdb"
        )
        runs[f"abt/dev/{rnd}"] = s
        runs[f"abt/test/{rnd}"] = {**s, "split": "test"}  # no cache for test, so rescore skips it
    runs["demo/abt"] = demo
    bench.RUNS_FILE.write_text(json.dumps(runs))
    assert bench.main(["coverage", "--record"]) == 0
    before = {k: r["at_threshold"]["f1"] for k, r in runs.items() if "at_threshold" in r}

    assert bench.main(["rescore"]) == 0
    after = json.loads(bench.RUNS_FILE.read_text())
    assert {k: r["at_threshold"]["f1"] for k, r in after.items() if "at_threshold" in r} == before
    assert (tiny_data / "judged_abt_dev_R2.parquet").exists()

    assert bench.main(["report"]) == 0
    text = bench.RESULTS.read_text()
    best = bench.chosen_round(after, "abt")
    assert "### Held-out test split, Abt-Buy" in text and f"round {best}" in text
    assert "## Blocking coverage (no Jev)" in text
    assert "## The macros live: Abt-Buy, 2 sampled left rows" in text
