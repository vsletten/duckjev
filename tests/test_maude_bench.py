"""Offline checks of bench/maude.py: round and corpus configs, the question builder for every
round, the multi-label metric rule, the split and allocation rules, the chosen-round rule,
the budget and held-out gates, and the whole pipeline (prepare from a fixture, run, demo,
rescore, report) on a dozen synthetic reports against the keyword fake transport."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_bench():
    path = ROOT / "bench" / "maude.py"
    spec = importlib.util.spec_from_file_location("maude_bench", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["maude_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load_bench()
IDS = bench.load_json(bench.IDS_FILE)


def test_rounds_and_codes_are_well_formed() -> None:
    assert list(bench.ROUNDS) == [f"R{i}" for i in range(7)]
    for name, r in bench.ROUNDS.items():
        assert r.criteria in ("none", "what", "full"), name
        assert r.state in bench.STATE_COLUMN and r.order in ("given", "reversed"), name
        assert r.vocabulary in ("code", "global"), name
    assert [n for n, r in bench.ROUNDS.items() if r.selectable] == ["R0", "R1", "R2", "R3"]
    assert set(bench.CODES) == set(IDS["codes"]) == {"QBJ", "FTR", "LWS"}
    assert 2 <= len(bench.SEVERITY_LEVELS) <= 10 and set(bench.HARM_CRITERIA) == set(bench.HARMS)
    assert len(bench.JUDGE_SQLS) == 2 * len(bench.STATE_COLUMN) * 2


def test_committed_option_sets_follow_the_rule() -> None:
    for code, c in IDS["codes"].items():
        counts = [n for _, n in c["options"]]
        assert counts == sorted(counts, reverse=True), code
        assert len(counts) <= bench.OPTIONS_PER_CODE and min(counts) >= bench.MIN_TERM_REPORTS
        splits = c["splits"]
        assert [len(splits[s]) for s in ("dev", "test", "gloss")] == [500, 1000, 200], code
        keys = [k for s in splits.values() for k in s]
        assert len(keys) == len(set(keys)), f"{code} splits overlap"
    assert len(IDS["global_options"]) <= bench.MAX_OPTIONS - 1


def test_every_option_has_a_definition_or_a_gloss() -> None:
    vocab, crit = bench.terms_by_name(), bench.load_criteria()
    assert all(v in vocab for v in crit["aliases"].values())
    for t, _ in IDS["global_options"]:
        assert bench.vocab_name(t, crit) in vocab or t in crit["glosses"], t
    for t, entry in crit["terms"].items():
        assert set(entry) <= {"also_not_for", "example"}, t
        assert all(n in crit["short"] for n in entry.get("also_not_for", [])), t


def test_questions_for_every_round() -> None:
    q = {r: bench.questions_for(bench.ROUNDS[r], "QBJ") for r in bench.ROUNDS}
    own = [t for t, _ in IDS["codes"]["QBJ"]["options"]] + [bench.CATCH_ALL]
    for r in ("R0", "R1", "R2", "R3", "R6"):
        assert list(q[r]["problem"]["criteria"]) == own, r
        assert list(q[r]) == ["problem", "harm", "severity"]
        assert q[r]["severity"]["criteria"] == bench.SEVERITY_LEVELS
    # R0 bare, R1 the official definition only, R2 on with not_for and one example
    assert set(q["R0"]["problem"]["criteria"].values()) == {None}
    assert q["R1"]["problem"]["criteria"]["Low Readings"] == {
        "what": bench.terms_by_name()["Low Readings"]["definition"]
    }
    assert q["R1"]["problem"]["criteria"][bench.CATCH_ALL] == {
        "what": bench.load_criteria()["glosses"][bench.CATCH_ALL]
    }
    low = q["R2"]["problem"]["criteria"]["Low Readings"]
    assert set(low) == {"what", "not_for", "examples"} and len(low["examples"]) == 1
    assert "(High Readings)" in low["not_for"]
    assert q["R0"]["harm"]["criteria"] == bench.HARM_CRITERIA
    assert set(q["R2"]["harm"]["criteria"]["Malfunction"]) == {"what", "not_for", "examples"}
    # R3 changes the state, not the questions; R4 reverses both Choices, not the Score
    assert q["R3"] == q["R2"] == q["R6"]
    assert list(q["R4"]["problem"]["criteria"]) == own[::-1]
    assert list(q["R4"]["harm"]["criteria"]) == list(bench.HARMS)[::-1]
    assert q["R4"]["problem"]["criteria"]["Low Readings"] == low
    # R5: every term seen, the code's own options exactly as in R3, the rest defined only
    r5 = q["R5"]["problem"]["criteria"]
    assert len(r5) == len(IDS["global_options"]) + 1 and list(r5)[-1] == bench.CATCH_ALL
    assert all(r5[t] == q["R3"]["problem"]["criteria"][t] for t in own)
    assert r5["Over-Sensing"] == {"what": bench.terms_by_name()["Oversensing"]["definition"]}


def test_neighbours_follow_the_hierarchy() -> None:
    vocab, crit = bench.terms_by_name(), bench.load_criteria()
    # siblings under a level-2 parent, and the parent
    low = bench.neighbours("Low Readings", vocab, crit)
    assert {"High Readings", "Incorrect, Inadequate or Imprecise Result or Readings"} <= set(low)
    # a level-2 term under a top-level category: parent and children only, no siblings
    sensing = bench.neighbours("Device Sensing Problem", vocab, crit)
    assert "Battery Problem" not in sensing and "Over-Sensing" in sensing  # label, via alias
    top = bench.neighbours("Protective Measures Problem", vocab, crit)  # a top-level term
    assert "Device Alarm System" in top and "Material Integrity Problem" not in top
    assert (
        bench.neighbours("Adverse Event Without Identified Device or Use Problem", vocab, crit)
        == []
    )


def test_object_state_keeps_the_description_first_and_drops_missing_fields() -> None:
    s = json.loads(bench.object_state("PUMP FAILED (B)(6)", "BRAND", None, "EVAL"))
    assert list(s) == [
        "description of event or problem",
        "brand name",
        "additional manufacturer narrative",
    ]
    assert s["description of event or problem"] == "PUMP FAILED (B)(6)"


def _answers(problem: dict[str, float], harm: str = "Malfunction") -> dict:
    top = max(problem, key=problem.__getitem__)
    return {
        "problem": {"choice": top, "probabilities": problem, "confidence": problem[top]},
        "harm": {"choice": harm, "probabilities": {harm: 1.0}, "confidence": 1.0},
        "severity": {"score": 0.2, "probabilities": {"0": 0.8, "1": 0.2}, "confidence": 0.8},
    }


def test_multi_label_metric_rule() -> None:
    opts = {"A", "B", "C", bench.CATCH_ALL}
    one = bench.flatten(_answers({"A": 0.6, "B": 0.3, "C": 0.1}), ["A"], opts, opts)
    assert (one["top1_in_set"], one["strict_correct"], one["covered"]) == (True, True, True)
    assert one["set_mass"] == pytest.approx(0.6)
    two = bench.flatten(_answers({"A": 0.2, "B": 0.5, "C": 0.3}), ["A", "B"], opts, opts)
    assert two["top1_in_set"] and two["strict_correct"] is None  # strict needs one filed term
    assert two["set_mass"] == pytest.approx(0.7)
    miss = bench.flatten(_answers({"A": 0.2, "B": 0.5, "C": 0.3}), ["C"], opts, opts)
    assert not miss["top1_in_set"] and miss["strict_correct"] is False
    out = bench.flatten(_answers({"A": 0.9, "B": 0.1}), ["Z"], opts, opts)
    assert not out["covered"] and not out["top1_in_set"]  # left out of accuracy, counts coverage


def test_allocate_and_draw_splits() -> None:
    assert bench.allocate(
        {"Death": 150, "Injury": 8000, "Malfunction": 12000, "Other": 2}, 500, 50
    ) == {
        "Death": 50,
        "Injury": 180,
        "Malfunction": 270,
        "Other": 0,
    }
    # the floor does not apply where fewer than the floor are available
    assert (
        bench.allocate({"Death": 3, "Injury": 30000, "Malfunction": 420, "Other": 20}, 500, 50)[
            "Death"
        ]
        == 0
    )
    assert sum(bench.allocate({"Injury": 3, "Malfunction": 4}, 10, 2).values()) == 7
    # floors that do not all fit go to the rarest categories first
    assert bench.allocate({"Death": 1, "Injury": 1, "Malfunction": 2}, 2, 1) == {
        "Death": 1,
        "Injury": 1,
        "Malfunction": 0,
    }
    pool = [
        {"mdr_report_key": str(i), "event_type": "Injury" if i % 3 else "Death"} for i in range(30)
    ]
    old = bench.SPLIT_SIZES, bench.FLOOR
    bench.SPLIT_SIZES, bench.FLOOR = {"dev": 6, "test": 6, "gloss": 3}, 3
    try:
        s = bench.draw_splits(pool, "QBJ")
        assert s == bench.draw_splits(list(reversed(pool)), "QBJ")  # order-free, by hash
    finally:
        bench.SPLIT_SIZES, bench.FLOOR = old
    keys = s["dev"] + s["test"] + s["gloss"]
    assert len(keys) == len(set(keys)) == 15
    death = {r["mdr_report_key"] for r in pool if r["event_type"] == "Death"}
    assert len(death & set(s["dev"])) == 3 and len(death & set(s["test"])) == 3


def test_chosen_round_rule() -> None:
    def r(name: str, top1: float, tokens: int) -> dict:
        return {"round": name, "pooled": {"top1_in_set": top1}, "input_tokens_per_request": tokens}

    runs = {
        "dev/R0": r("R0", 0.5, 1000),
        "dev/R2": r("R2", 0.8, 5000),
        "dev/R3": r("R3", 0.8, 4000),
        "dev/R4": r("R4", 0.9, 1),  # a check, never chosen
        "test/R1": r("R1", 0.99, 1),
    }
    assert bench.chosen_round(runs) == "R3"


# --------------------------------------------------------------------------- tiny pipeline


def _report(key: int, code: str, event: str, problems: list[str], text: str, day: str) -> dict:
    return {
        "mdr_report_key": str(key),
        "date_received": day,
        "date_of_event": "20250705",
        "event_type": event,
        "product_problems": problems,
        "mdr_text": [
            {"text_type_code": bench.DESCRIPTION, "text": text},
            {"text_type_code": bench.ADDITIONAL, "text": "EVALUATION COMPLETED."},
        ],
        "device": [
            {
                "device_report_product_code": code,
                "brand_name": f"{code} BRAND",
                "generic_name": "GENERIC",
                "manufacturer_d_name": "MAKER",
            }
        ],
        "remedial_action": ["Recall"] if key % 2 else [],
        "report_source_code": "Manufacturer report",
    }


PAD = " THE DEVICE WAS RETURNED FOR EVALUATION AND THE INVESTIGATION IS ONGOING AT THIS TIME."
FIXTURE_ROWS = {
    "QBJ": [
        ("Malfunction", ["Low Readings"], "THE SENSOR GAVE LOW READINGS AGAINST A METER."),
        ("Malfunction", ["Wireless Communication Problem"], "A WIRELESS COMMUNICATION PROBLEM."),
        ("Injury", ["Low Readings"], "LOW READINGS; THE PATIENT WAS TAKEN TO HOSPITAL."),
        ("Malfunction", ["Protective Measures Problem"], "NO ALERT; PROTECTIVE MEASURES PROBLEM."),
    ],
    "FTR": [
        ("Injury", ["Material Rupture"], "MATERIAL RUPTURE FOUND AT SURGERY; EXPLANTED."),
        ("Injury", ["Device Appears to Trigger Rejection"], "CAPSULAR CONTRACTURE, INJURY."),
        ("Malfunction", ["Break"], "THE SHELL SHOWED A BREAK WHEN INTRODUCED."),
        ("Injury", ["Material Rupture", "Gel Leak"], "GEL LEAK AND MATERIAL RUPTURE; SURGERY."),
    ],
    "LWS": [
        ("Death", ["Over-Sensing"], "OVER-SENSING WAS SEEN AND THE PATIENT DIED."),
        ("Injury", ["High impedance"], "HIGH IMPEDANCE; THE LEAD WAS REPLACED IN SURGERY."),
        ("Malfunction", ["Fracture"], "A LEAD FRACTURE WAS SUSPECTED; LEAD REMAINS IN USE."),
        ("Malfunction", ["Over-Sensing", "Fracture"], "OVER-SENSING FROM A FRACTURE."),
    ],
}


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A dozen synthetic reports, four per code, prepared into a tmp DATA directory."""
    reports: dict[str, list[dict]] = {}
    key = 1000
    for code, rows in FIXTURE_ROWS.items():
        reports[code] = []
        for i, (event, problems, text) in enumerate(rows):
            key += 1
            day = "20250708" if i % 2 else "20250722"
            reports[code].append(_report(key, code, event, problems, text + PAD, day))
    recalls = [
        {
            "product_code": "QBJ",
            "product_res_number": "Z-1-2025",
            "product_description": "CGM APP",
            "reason_for_recall": "THE APP GAVE LOW READINGS AGAINST A METER.",
            "root_cause_description": "Software design",
            "event_date_initiated": "2025-07-01",
        }
    ]
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"reports": reports, "recalls": recalls}))
    data = tmp_path / "data"
    monkeypatch.setattr(bench, "DATA", data)
    monkeypatch.setattr(bench, "IDS_FILE", tmp_path / "maude_ids.json")
    monkeypatch.setattr(bench, "RUNS_FILE", tmp_path / "maude_runs.json")
    monkeypatch.setattr(bench, "RESULTS", tmp_path / "maude.md")
    monkeypatch.setattr(bench, "SPLIT_SIZES", {"dev": 2, "test": 1, "gloss": 1})
    monkeypatch.setattr(bench, "FLOOR", 1)
    monkeypatch.setattr(bench, "MIN_TERM_REPORTS", 1)
    assert bench.main(["prepare", "--from-fixture", str(fixture)]) == 0
    return data


def test_prepare_from_fixture(tiny: Path) -> None:
    ids = json.loads(bench.IDS_FILE.read_text())
    for code in bench.CODES:
        c = ids["codes"][code]
        assert c["pool"] == 4 and [len(c["splits"][s]) for s in ("dev", "test", "gloss")] == [
            2,
            1,
            1,
        ]
        assert (tiny / f"maude_{code}_dev.parquet").exists()
    assert ["Material Rupture", 2] in ids["codes"]["FTR"]["options"]
    assert (tiny / "maude_recalls.parquet").exists()


def test_prepare_refuses_a_missing_committed_report(tiny: Path) -> None:
    ids = json.loads(bench.IDS_FILE.read_text())
    missing = ids["codes"]["QBJ"]["splits"]["test"][0]
    fixture = tiny.parent / "fixture.json"
    data = json.loads(fixture.read_text())
    data["reports"]["QBJ"] = [r for r in data["reports"]["QBJ"] if r["mdr_report_key"] != missing]
    fixture.write_text(json.dumps(data))
    split = tiny / "maude_QBJ_test.parquet"
    before = split.read_bytes()

    with pytest.raises(SystemExit, match="committed split keys are missing"):
        bench.main(["prepare", "--from-fixture", str(fixture)])
    assert split.read_bytes() == before


def test_dry_run_keyword_answers_and_gates(tiny: Path) -> None:
    assert bench.main(["run", "R1", "--split", "dev"]) == 2  # no pre-flight
    assert bench.main(["run", "R0", "--split", "dev", "--dry-run"]) == 0
    s = json.loads((tiny / "summary_dev_R0_dry.json").read_text())
    assert not bench.RUNS_FILE.exists()  # a dry run records nothing
    assert (
        s["reports"] == 6 and s["rerun_identical_rows"] == 6 and s["rerun_usage"]["requests"] == 0
    )
    # every fixture description names one of its filed terms, so the keyword fake is right
    assert s["pooled"]["top1_in_set"] == 1.0 and s["pooled"]["coverage"] == 1.0
    assert bench.main(["run", "R0", "--split", "dev", "--dry-run"]) == 0  # a dry pre-flight...
    assert not bench._preflighted("R0")  # ...does not open the gate
    assert bench.main(["run", "R0", "--split", "dev", "--limit", "3", "--dry-run"]) == 0
    assert json.loads((tiny / "summary_dev_R0_n3_dry.json").read_text())["reports"] == 3
    # the held-out split: only after the reading, and only R0 and the chosen round
    (tiny / "summary_dev_R0_n40.json").write_text("{}")  # stands for a live pre-flight
    assert bench.main(["run", "R0", "--split", "test"]) == 2
    bench.RUNS_FILE.write_text(json.dumps({"dev/R0": {**s, "dry_run": False}, "reading": {}}))
    assert bench.main(["run", "R2", "--split", "test"]) == 2


def test_metrics_scope_binds_untrusted_code(tiny: Path) -> None:
    assert bench.main(["run", "R0", "--split", "dev", "--dry-run"]) == 0
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE scored AS SELECT * FROM read_parquet(?)",
        [str(tiny / "scored_dev_R0_dry.parquet")],
    )
    original_rows = con.execute("SELECT count(*) FROM scored").fetchone()[0]
    result = bench.metrics_for(con, "QBJ'; DROP TABLE scored; --")
    assert result["rows"] == 0
    assert con.execute("SELECT count(*) FROM scored").fetchone()[0] == original_rows


def test_metric_queries_are_literal_and_scoped_by_parameter() -> None:
    names = [
        "SUMMARY_SQL",
        "HARM_SQL",
        "HARM_MATRIX_SQL",
        "RELIABILITY_SQL",
        "DEFERRAL_SQL",
        "SEVERITY_SQL",
        "CONFUSIONS_SQL",
        "ERRORS_SQL",
        "CONFUSION_EXAMPLES_SQL",
        "COUNTS_SQL",
    ]
    for name in names:
        sql = getattr(bench, name)
        assert "{" not in sql and "$code IS NULL OR product_code = $code" in sql, name


def test_reading_must_say_something(tiny: Path) -> None:
    empty = tiny / "reading.md"
    empty.write_text("  \n\n")
    assert bench.main(["reading", "--file", str(empty)]) == 2
    assert "reading" not in bench.load_runs()


def test_counts_keep_filed_terms_outside_the_options(tiny: Path) -> None:
    import duckdb

    con = duckdb.connect()
    con.execute(
        "CREATE TABLE scored AS SELECT * FROM (VALUES ('QBJ', ['A'], 'A', MAP {'A': 0.9}), "
        "('QBJ', ['Z'], 'A', MAP {'A': 0.8})) t(product_code, filed, problem, problem_probs)"
    )
    rows = con.execute(bench.COUNTS_SQL, {"code": None, "k": 10}).fetchall()
    assert [r[:3] for r in rows] == [("A", 1, 2), ("Z", 1, 0)]
    assert rows[1][3:] == (0.0, 0.0)


def test_budget_hard_stop(tiny: Path) -> None:
    bench.RUNS_FILE.write_text(json.dumps({"spend": [{"usd": 2.9}]}))
    assert bench.budget_tokens(0.5, dry_run=False) == int(0.1 / bench.USD_PER_INPUT_TOKEN)
    assert bench.budget_tokens(0.5, dry_run=True) == int(0.5 / bench.USD_PER_INPUT_TOKEN)
    bench.record_spend("x", {"est_usd": 0.1, "requests": 1, "input_tokens": 1})
    with pytest.raises(SystemExit):
        bench.budget_tokens(0.5, dry_run=False)


def test_demo_rescore_and_report(tiny: Path, tmp_path: Path) -> None:
    for rnd in ("R0", "R3", "R6"):
        assert bench.main(["run", rnd, "--split", "dev", "--dry-run"]) == 0
    assert bench.main(["demo", "--code", "QBJ", "--n", "2", "--round", "R3", "--dry-run"]) == 0
    demo = json.loads((tiny / "summary_demo_QBJ_dry.json").read_text())
    assert demo["sample_rows"] == 1 and demo["join"]["recalls"] == 1 and demo["join"]["pairs"] == 1
    assert demo["trend_rows"] >= 1 and len(demo["topk"]) == 1
    assert demo["dedup"]["rows_in"] >= 1

    runs: dict = {}
    for rnd in ("R0", "R3", "R6"):
        s = json.loads((tiny / f"summary_dev_{rnd}_dry.json").read_text())
        shutil.copy(tiny / f"cache_dev_{rnd}_dry.duckdb", tiny / f"cache_dev_{rnd}.duckdb")
        runs[f"dev/{rnd}"] = s
    best = bench.chosen_round(runs)
    for rnd in ("R0", best):
        runs[f"test/{rnd}"] = {**runs[f"dev/{rnd}"], "split": "test"}  # no cache: rescore skips
    runs["reading"] = {
        "written": "2026-01-01 00:00 UTC",
        "text": "Written before the held-out run.",
    }
    runs["demo/QBJ"] = demo
    runs["spend"] = [{"what": "x", "usd": 0.01, "requests": 1, "input_tokens": 1, "timestamp": "t"}]
    bench.RUNS_FILE.write_text(json.dumps(runs))
    before = {k: r["pooled"]["top1_in_set"] for k, r in runs.items() if "pooled" in r}

    assert bench.main(["rescore"]) == 0
    after = json.loads(bench.RUNS_FILE.read_text())
    assert {k: r["pooled"]["top1_in_set"] for k, r in after.items() if "pooled" in r} == before
    assert (tiny / "scored_dev_R3.parquet").exists()

    assert bench.main(["report"]) == 0
    text = bench.RESULTS.read_text()
    assert best == "R0"  # every round ties at 1.0 on the fixture; ties go to fewer tokens
    assert "## Held-out test split" in text and f"| {best} top-1 in set |" in text
    assert "## Reading the numbers" in text and "Written before the held-out run." in text
    assert "## The demo queries live: QBJ, 1 sampled test reports" in text
    assert "Total live spend for the benchmark: **$0.010**" in text
