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


EM = json.loads((ROOT / "docs" / "results" / "entity_matching_runs.json").read_text())
EM_NAMES = {"abt": "Abt-Buy", "dblp": "DBLP-ACM"}


def _em_chosen_round(corpus: str) -> str:
    """The rule in bench/entity_matching.py: best dev F1 at 0.5, ties to fewer tokens."""
    dev = [
        (r["at_threshold"]["f1"], -r["input_tokens_per_request"], r["round"])
        for k, r in EM.items()
        if k.startswith(f"{corpus}/dev/") and r["round_config"]["selectable"]
    ]
    return max(dev)[2]


def test_readme_entity_matching_table_matches_run_log() -> None:
    section = README.split("## Entity-matching numbers")[1].split("\n## ")[0]
    for corpus, name in EM_NAMES.items():
        best = _em_chosen_round(corpus)
        assert f"R{best[1:]} F1 at 0.5" in section  # the header names the chosen round
        base, run = EM[f"{corpus}/test/R0"], EM[f"{corpus}/test/{best}"]
        a = run["at_threshold"]
        line = next(ln for ln in section.splitlines() if ln.startswith(f"| {name} |"))
        cells = [c.strip() for c in line.split("|")[2:9]]
        assert cells == [
            f"{base['at_threshold']['f1']:.3f}",
            f"**{a['f1']:.3f}**",
            f"{a['precision']:.3f} / {a['recall']:.3f}",
            f"{run['auroc']:.3f}",
            f"{run['ece']:.3f}",
            f"{run['expected_count']:.1f} ± {run['expected_stderr']:.1f} vs {run['positives']}",
            f"${run['usd_per_1k_pairs']:.4f}",
        ], corpus


MAUDE = json.loads((ROOT / "docs" / "results" / "maude_runs.json").read_text())
STREAM_2026 = 2_503_728  # bench/maude.py: MAUDE reports received 2026-01-01 to 2026-09-26


def _maude_chosen_round() -> str:
    """The rule in bench/maude.py: best pooled dev top-1 in set, ties to fewer tokens."""
    dev = [
        (r["pooled"]["top1_in_set"], -r["input_tokens_per_request"], r["round"])
        for k, r in MAUDE.items()
        if k.startswith("dev/") and r["round_config"]["selectable"]
    ]
    return max(dev)[2]


def test_readme_maude_table_matches_run_log() -> None:
    best = _maude_chosen_round()
    section = README.split("## MAUDE numbers")[1].split("\n## ")[0]
    header = section.splitlines()[section.splitlines().index("|---|---|---|") - 1]
    assert header.split("|")[2].strip().startswith("R0:")
    assert header.split("|")[3].strip().startswith(f"{best}:")
    for i, run in enumerate([MAUDE["test/R0"], MAUDE[f"test/{best}"]]):
        m = run["pooled"]
        stream = run["input_tokens_per_report"] * STREAM_2026 * 42e-9
        expected = {
            "problem: top-1 in set": (
                f"{m['top1_in_set']:.3f} ± {m['top1_in_set_se']:.3f}",
                f"**{m['top1_in_set']:.3f}** ± {m['top1_in_set_se']:.3f}",
            ),
            "harm: accuracy (macro average over event types)": (
                f"{m['harm_accuracy']:.3f} ({m['harm_macro']:.3f})",
                f"**{m['harm_accuracy']:.3f}** ({m['harm_macro']:.3f})",
            ),
            "ECE over `confidence`, problem / harm": (
                f"{m['problem_ece_confidence']:.3f} / {m['harm_ece_confidence']:.3f}",
            )
            * 2,
            "input tokens per request": (f"{run['input_tokens_per_request']:,.0f}",) * 2,
            "cost": (
                f"${run['usd_per_1k_reports']:.3f} per 1,000 reports",
                f"**${run['usd_per_1k_reports']:.3f} per 1,000 reports**",
            ),
            "the 2026 stream to date, 2.5M reports": (f"${stream:,.0f}",) * 2,
        }
        for label, want in expected.items():
            assert _row(section, label)[i] == want[i], (label, want[i])
