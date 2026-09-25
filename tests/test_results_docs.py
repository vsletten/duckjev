"""The hand-written SROIE headline in the README must match the committed run log."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = json.loads((ROOT / "docs" / "results" / "sroie_runs.json").read_text())
README = (ROOT / "README.md").read_text()


def _pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def _reported_round() -> str:
    """The rule in bench/sroie.py: best all-fields exact on train among selectable rounds."""
    train = [
        r for k, r in RUNS.items() if k.startswith("train/") and r["round_config"]["selectable"]
    ]
    return max(train, key=lambda r: r["fields"]["all"]["exact"])["round"]


def test_readme_sroie_table_matches_run_log() -> None:
    run = RUNS[f"test/{_reported_round()}"]
    section = README.split("## SROIE receipts numbers")[1].split("\n## ")[0]
    for field, label in [
        ("company", "company"),
        ("date", "date"),
        ("address", "address"),
        ("total", "total"),
        ("all", "all fields"),
    ]:
        m = run["fields"][field]
        row = next(
            line
            for line in section.splitlines()
            if line.replace("*", "").startswith(f"| {label} |")
        )
        cells = [c.strip().strip("*") for c in row.split("|")[2:5]]
        assert cells == [_pct(m["coverage"]), _pct(m["exact"]), _pct(m["selection"])], field
    four = re.search(r"All four fields were exact on ([\d.]+%) of receipts", section)
    assert four and four.group(1) == _pct(run["record_all_fields_exact"])
    base = re.search(r"against ([\d.]+%) for the untuned\s+baseline", section)
    assert base and base.group(1) == _pct(RUNS["test/R0"]["record_all_fields_exact"])


BANKING = json.loads((ROOT / "docs" / "results" / "banking77_runs.json").read_text())


def _banking_chosen_round() -> str:
    """The rule in bench/banking77.py: best dev accuracy, ties to fewer tokens per request."""
    dev = [
        (r["accuracy"], -r["input_tokens_per_request"], r["round"])
        for k, r in BANKING.items()
        if k.startswith("dev/") and r["round_config"]["selectable"]
    ]
    return max(dev)[2]


def _row(section: str, label: str) -> list[str]:
    line = next(ln for ln in section.splitlines() if ln.startswith(f"| {label} |"))
    return [c.strip() for c in line.split("|")[2:4]]


def test_readme_banking77_table_matches_run_log() -> None:
    best = _banking_chosen_round()
    section = README.split("## Banking77 numbers")[1].split("\n## ")[0]
    header = section.splitlines()[section.splitlines().index("|---|---|---|") - 1]
    assert header.split("|")[2].strip().startswith("R0:")
    assert header.split("|")[3].strip().startswith(f"{best}:")
    for i, run in enumerate([BANKING["test/R0"], BANKING[f"test/{best}"]]):
        u = run["usage"]
        expected = {
            "accuracy (argmax = gold)": f"**{run['accuracy']:.3f}** ± {run['accuracy_se']:.3f}",
            "ECE, 10 bins, over `confidence` / top-1 p": (
                f"{run['ece_confidence']:.3f} / {run['ece_top_p']:.3f}"
            ),
            "Σ over intents of abs(hard − true)": str(int(run["sum_abs_hard_minus_true"])),
            "Σ over intents of abs(expected − true)": f"{run['sum_abs_expected_minus_true']:.1f}",
            "intents with abs(expected − true) ≤ 2·SE": (
                f"{run['intents_within_2se']} / {run['intents']}"
            ),
            "input tokens per request": f"{run['input_tokens_per_request']:,.0f}",
            "cost": f"**${run['usd_per_1k_rows']:.3f} per 1,000 rows**",
            "throughput": f"{run['rows_per_second']:.0f} rows/s ({u['rate_limited']:,} × 429",
            "cache re-run": (
                f"{run['rerun_seconds']:.2f} s, {run['rerun_usage']['requests']} requests, $0"
            ),
        }
        for label, want in expected.items():
            cell = _row(section, label)[i]
            assert cell.startswith(want), (label, cell, want)
