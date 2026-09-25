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
