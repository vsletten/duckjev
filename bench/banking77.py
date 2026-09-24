"""Live Banking77 benchmark for duckjev (workstation only; needs TYPESAFE_API_KEY).

    uv run python bench/banking77.py sample          # 200-row check, projects full cost
    uv run python bench/banking77.py full            # full test split, writes docs/results
    uv run python bench/banking77.py full --dry-run  # same pipeline against a fake transport

``full`` refuses to run unless a ``sample`` run projected the full split at or
under ``--max-usd`` (default $0.50), and it also sets that amount as the
client's ``max_input_tokens`` budget. Each phase uses a fresh cache file so the
timed run pays for every row; the cache re-run then reuses that file.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import httpx

import duckjev
from duckjev.client import USD_PER_INPUT_TOKEN

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bench" / "data"
TEST_URL = "https://huggingface.co/datasets/mteb/banking77/resolve/main/test.jsonl"
TEST_FILE = DATA / "banking77_test.jsonl"
CRITERIA_FILE = ROOT / "bench" / "banking77_criteria.json"
SAMPLE_FILE = DATA / "sample_result.json"
RESULTS = ROOT / "docs" / "results" / "banking77.md"

INSTR = "Which banking-support intent does this customer message express?"
TOPUP_Q = "Is the customer asking about topping up (adding money to) their account?"
TOPUP_INTENTS = (
    "automatic_top_up",
    "pending_top_up",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "verify_top_up",
)
FULL_ROWS = 3080

JUDGE_SQL = """CREATE OR REPLACE TABLE {table} AS
SELECT text, label_text, jev_choice(text, $instr, $criteria) AS j
FROM {source}"""

ACCURACY_SQL = """SELECT count(*) AS n, avg((j.choice = label_text)::INT) AS accuracy
FROM judged"""

RELIABILITY_SQL = """WITH b AS (
  SELECT least(floor({conf} * 10), 9)::INT AS bin,
         {conf} AS conf,
         (j.choice = label_text)::INT AS correct
  FROM judged)
SELECT bin, count(*) AS n, avg(conf) AS mean_conf, avg(correct) AS accuracy
FROM b GROUP BY bin ORDER BY bin"""

ECE_SQL = """WITH b AS (
  SELECT least(floor({conf} * 10), 9)::INT AS bin, {conf} AS conf,
         (j.choice = label_text)::INT AS correct
  FROM judged),
bins AS (SELECT count(*) AS n, avg(conf) AS c, avg(correct) AS a FROM b GROUP BY bin)
SELECT sum(n * abs(a - c)) / sum(n) AS ece FROM bins"""

SOFT_VS_HARD_SQL = """WITH truth AS (
  SELECT label_text AS intent, count(*) AS true_count FROM judged GROUP BY ALL),
hard AS (
  SELECT j.choice AS intent, count(*) AS hard_count FROM judged GROUP BY ALL),
soft AS (
  SELECT e.key AS intent,
         SUM(e.value) AS expected_count,
         sqrt(SUM(e.value * (1 - e.value))) AS stderr
  FROM judged, UNNEST(map_entries(j.probabilities)) AS u(e)
  GROUP BY ALL)
SELECT t.intent, t.true_count,
       coalesce(h.hard_count, 0) AS hard_count,
       s.expected_count, s.stderr,
       abs(coalesce(h.hard_count, 0) - t.true_count) AS hard_abs_err,
       abs(s.expected_count - t.true_count) AS soft_abs_err,
       abs(s.expected_count - t.true_count) <= 2 * s.stderr AS within_2se
FROM truth t LEFT JOIN hard h USING (intent) LEFT JOIN soft s USING (intent)
ORDER BY t.intent"""

DEMO_SOFT_GROUP_BY_SQL = """SELECT e.key AS intent, SUM(e.value) AS expected_rows
FROM banking77_test t,
     UNNEST(map_entries(jev_choice(t.text, $instr, $criteria).probabilities)) AS u(e)
GROUP BY intent ORDER BY expected_rows DESC
LIMIT 5"""

SEM_WHERE_SQL = """SELECT
  count(*) FILTER (WHERE sem_where(text, $q, 0.5))          AS sem_where_count,
  expected_count(jev_noul(text, $q))                        AS expected,
  expected_count_stderr(jev_noul(text, $q))                 AS stderr,
  count(*) FILTER (WHERE list_contains($intents, label_text)) AS true_count,
  count(*) FILTER (WHERE sem_where(text, $q, 0.5)
                     AND list_contains($intents, label_text)) AS true_positives
FROM banking77_test"""


# --------------------------------------------------------------------------- setup


def ensure_data() -> Path:
    if not TEST_FILE.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(TEST_URL, follow_redirects=True, timeout=60)
        resp.raise_for_status()
        TEST_FILE.write_bytes(resp.content)
    return TEST_FILE


def fake_transport() -> httpx.MockTransport:
    """Dry-run stand-in: a peaked random distribution that is right ~80% of the time."""
    rng = random.Random(7)
    labels: dict[str, str] = {}
    con = duckdb.connect()
    for text, label in con.execute(
        "SELECT text, label_text FROM read_json(?)", [str(ensure_data())]
    ).fetchall():
        labels[text] = label

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        answers: dict[str, Any] = {}
        for qid, q in body["questions"].items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": rng.random()}
                continue
            opts = list(q["criteria"])
            truth = labels.get(body["state"], opts[0])
            top = truth if rng.random() < 0.8 else rng.choice(opts)
            probs = {o: 0.2 / (len(opts) - 1) for o in opts}
            probs[top] = 0.8
            answers[qid] = {
                "type": "choice",
                "choice": top,
                "probabilities": probs,
                "confidence": 0.6,
            }
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "answers": answers,
                "usage": {"input_tokens": 900, "output_tokens": 30},
            },
        )

    return httpx.MockTransport(handle)


def connect(cache_file: Path, args: argparse.Namespace) -> tuple[duckdb.DuckDBPyConnection, str]:
    if cache_file.exists():
        cache_file.unlink()
    con = duckdb.connect()
    budget = int(args.max_usd / USD_PER_INPUT_TOKEN)
    duckjev.register(
        con,
        cache_path=cache_file,
        concurrency=args.concurrency,
        max_input_tokens=budget,
        transport=fake_transport() if args.dry_run else None,
        api_key="dry-run" if args.dry_run else None,
    )
    con.execute(
        "CREATE TABLE banking77_test AS "
        "SELECT text, label, label_text FROM read_json(?) ORDER BY label, text",
        [str(ensure_data())],
    )
    criteria = CRITERIA_FILE.read_text(encoding="utf-8")
    json.loads(criteria)  # fail early on a broken gloss file
    return con, criteria


def timed(con: duckdb.DuckDBPyConnection, sql: str, params: dict[str, Any]) -> float:
    t0 = time.perf_counter()
    con.execute(sql, params)
    return time.perf_counter() - t0


# --------------------------------------------------------------------------- phases


def run_sample(args: argparse.Namespace) -> int:
    con, criteria = connect(DATA / "sample_cache.duckdb", args)
    con.execute(
        f"CREATE TABLE sample AS SELECT * FROM banking77_test "
        f"USING SAMPLE {args.n} ROWS (reservoir, 42)"
    )
    duckjev.usage(reset=True)
    secs = timed(
        con,
        JUDGE_SQL.format(table="judged", source="sample"),
        {"instr": INSTR, "criteria": criteria},
    )
    use = duckjev.usage()
    n, acc = con.execute(ACCURACY_SQL).fetchone()
    projected = use["est_usd"] / n * FULL_ROWS
    result = {
        "n": n,
        "seconds": secs,
        "accuracy": acc,
        "usage": use,
        "tokens_per_request": use["input_tokens"] / max(use["requests"], 1),
        "projected_full_usd": projected,
        "dry_run": args.dry_run,
        "ok": projected <= args.max_usd,
    }
    SAMPLE_FILE.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        print(f"HARD STOP: projected ${projected:.4f} for the full split exceeds ${args.max_usd}")
        return 2
    return 0


def run_full(args: argparse.Namespace) -> int:
    if not SAMPLE_FILE.exists():
        print("run the sample phase first")
        return 2
    sample = json.loads(SAMPLE_FILE.read_text())
    if not sample["ok"] or sample["dry_run"] != args.dry_run:
        print("the last sample run did not clear the cost gate for this mode; refusing")
        return 2

    con, criteria = connect(
        DATA / ("dry_cache.duckdb" if args.dry_run else "full_cache.duckdb"), args
    )
    params = {"instr": INSTR, "criteria": criteria}
    judge_sql = JUDGE_SQL.format(table="judged", source="banking77_test")

    duckjev.usage(reset=True)
    secs = timed(con, judge_sql, params)
    use = duckjev.usage(reset=True)

    rerun_sql = JUDGE_SQL.format(table="judged_rerun", source="banking77_test")
    rerun_secs = timed(con, rerun_sql, params)
    rerun_use = duckjev.usage(reset=True)
    identical = con.execute(
        "SELECT count(*) FROM judged a JOIN judged_rerun b USING (text) "
        "WHERE a.j IS NOT DISTINCT FROM b.j"
    ).fetchone()[0]

    n, acc = con.execute(ACCURACY_SQL).fetchone()
    conf_expr = "j.confidence"
    top_expr = "j.probabilities[j.choice]"
    reliability = con.execute(RELIABILITY_SQL.format(conf=conf_expr)).fetchall()
    reliability_top = con.execute(RELIABILITY_SQL.format(conf=top_expr)).fetchall()
    ece = con.execute(ECE_SQL.format(conf=conf_expr)).fetchone()[0]
    ece_top = con.execute(ECE_SQL.format(conf=top_expr)).fetchone()[0]
    per_intent = con.execute(SOFT_VS_HARD_SQL).fetchall()

    t0 = time.perf_counter()
    demo = con.execute(DEMO_SOFT_GROUP_BY_SQL, params).fetchall()
    demo_secs = time.perf_counter() - t0
    demo_use = duckjev.usage(reset=True)

    t0 = time.perf_counter()
    sem = con.execute(SEM_WHERE_SQL, {"q": TOPUP_Q, "intents": list(TOPUP_INTENTS)}).fetchone()
    sem_secs = time.perf_counter() - t0
    sem_use = duckjev.usage(reset=True)
    duckjev.flush(con)

    report = {
        "n": n,
        "seconds": secs,
        "usage": use,
        "rerun_seconds": rerun_secs,
        "rerun_usage": rerun_use,
        "rerun_identical_rows": identical,
        "accuracy": acc,
        "ece_confidence": ece,
        "ece_top_prob": ece_top,
        "reliability": reliability,
        "reliability_top": reliability_top,
        "per_intent": per_intent,
        "demo": demo,
        "demo_seconds": demo_secs,
        "demo_usage": demo_use,
        "sem_where": sem,
        "sem_where_seconds": sem_secs,
        "sem_where_usage": sem_use,
        "sample": sample,
        "concurrency": args.concurrency,
        "dry_run": args.dry_run,
    }
    out = DATA / "banking77_dryrun.md" if args.dry_run else RESULTS
    out.write_text(render(report, judge_sql))
    headline = headline_numbers(report)
    (out.with_suffix(".json")).write_text(json.dumps(headline, indent=2) + "\n")
    print(json.dumps(headline, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


# --------------------------------------------------------------------------- report


def headline_numbers(r: dict[str, Any]) -> dict[str, Any]:
    use = r["usage"]
    rows = r["per_intent"]
    hard_total = sum(row[5] for row in rows)
    soft_total = sum(row[6] for row in rows)
    within = sum(1 for row in rows if row[7])
    return {
        "rows": r["n"],
        "wall_seconds": round(r["seconds"], 2),
        "rows_per_second": round(r["n"] / r["seconds"], 2),
        "requests_per_second": round(use["requests"] / r["seconds"], 2),
        "input_tokens_per_second": round(use["input_tokens"] / r["seconds"], 1),
        "requests": use["requests"],
        "input_tokens": use["input_tokens"],
        "input_tokens_per_request": round(use["input_tokens"] / max(use["requests"], 1), 1),
        "est_usd": round(use["est_usd"], 5),
        "usd_per_1k_rows": round(use["est_usd"] / r["n"] * 1000, 5),
        "rate_limited_429": use["rate_limited"],
        "overloaded_529": use["overloaded"],
        "retries": use["retries"],
        "accuracy": round(r["accuracy"], 4),
        "ece_confidence": round(r["ece_confidence"], 4),
        "ece_top_prob": round(r["ece_top_prob"], 4),
        "sum_abs_hard_minus_true": int(hard_total),
        "sum_abs_expected_minus_true": round(soft_total, 1),
        "intents_within_2se": within,
        "intents": len(rows),
        "frac_within_2se": round(within / len(rows), 3),
        "cache_rerun_seconds": round(r["rerun_seconds"], 3),
        "cache_rerun_usd": r["rerun_usage"]["est_usd"],
        "cache_rerun_requests": r["rerun_usage"]["requests"],
    }


def _table(header: list[str], rows: list[list[Any]]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 1000 else f"{v:,.1f}"
        return str(v)

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _sql(s: str) -> str:
    return f"```sql\n{s}\n```"


def render(r: dict[str, Any], judge_sql: str) -> str:
    h = headline_numbers(r)
    use = r["usage"]
    sem = r["sem_where"]
    s = r["sample"]
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    per_intent = [list(row) for row in r["per_intent"]]
    sem_precision = sem[4] / sem[0] if sem[0] else math.nan
    sem_recall = sem[4] / sem[3] if sem[3] else math.nan
    sem_within = abs(sem[1] - sem[3]) <= 2 * sem[2]
    mode = "DRY RUN against a fake transport (numbers are synthetic)" if r["dry_run"] else "live"

    parts = [
        "# Banking77 benchmark",
        "",
        f"Generated by `bench/banking77.py full` on {stamp} ({mode}). Model `jev-1.13.0`, "
        f"duckjev {duckjev.__version__}, DuckDB {duckdb.__version__}, Python "
        f"{platform.python_version()}, client concurrency {r['concurrency']}.",
        "",
        "Corpus: the Banking77 test split, 3,080 customer-support messages with 77 gold "
        "intents, from `mteb/banking77` `test.jsonl` (the auto-converted parquet in that repo "
        "drops 4 rows, so the benchmark reads the JSONL). Each row is one Jev Choice question "
        "over all 77 intents; option descriptions are the one-line glosses in "
        "`bench/banking77_criteria.json`, written from the label names and train-split "
        "examples only.",
        "",
        "## Headline",
        "",
        _table(
            ["metric", "value"],
            [
                ["rows", h["rows"]],
                ["wall time (s)", h["wall_seconds"]],
                ["rows / s", h["rows_per_second"]],
                ["requests / s", h["requests_per_second"]],
                ["input tokens / s", h["input_tokens_per_second"]],
                ["requests", h["requests"]],
                ["input tokens (billed)", h["input_tokens"]],
                ["input tokens / request", h["input_tokens_per_request"]],
                ["estimated cost (USD)", f"${h['est_usd']:.4f}"],
                ["USD per 1,000 rows", f"${h['usd_per_1k_rows']:.4f}"],
                ["429 responses (retried)", h["rate_limited_429"]],
                ["529 responses (retried)", h["overloaded_529"]],
                ["accuracy (argmax = gold)", h["accuracy"]],
                ["ECE, 10 bins over `j.confidence`", h["ece_confidence"]],
                ["ECE, 10 bins over top-1 probability", h["ece_top_prob"]],
                ["Σ over intents of abs(hard − true)", h["sum_abs_hard_minus_true"]],
                ["Σ over intents of abs(expected − true)", h["sum_abs_expected_minus_true"]],
                [
                    "intents with abs(expected − true) ≤ 2·SE",
                    f"{h['intents_within_2se']} / {h['intents']} ({h['frac_within_2se']:.1%})",
                ],
                ["cache re-run wall time (s)", h["cache_rerun_seconds"]],
                ["cache re-run cost (USD)", f"${h['cache_rerun_usd']:.4f}"],
            ],
        ),
        "",
        "Cost is `duckjev.usage()['est_usd']`: billed input tokens × $42 per billion "
        "(output tokens are free).",
        "",
        "## Pre-flight sample",
        "",
        f"A {s['n']}-row reservoir sample ran first on its own fresh cache: "
        f"{s['usage']['input_tokens']:,} input tokens, ${s['usage']['est_usd']:.4f}, "
        f"{s['tokens_per_request']:.0f} tokens per request, accuracy {s['accuracy']:.3f}, "
        f"projecting ${s['projected_full_usd']:.4f} for the full split against a $0.50 hard stop.",
        "",
        "## The judged table",
        "",
        "Timed from `CREATE TABLE` to completion, on a fresh cache file, with "
        "`$instr` and `$criteria` bound as prepared-statement parameters:",
        "",
        _sql(judge_sql),
        "",
        f"`$instr` = `{INSTR}`; `$criteria` = the contents of `bench/banking77_criteria.json`.",
        "",
        "## Throughput and cost",
        "",
        _table(
            ["rows", "seconds", "rows/s", "requests", "requests/s", "input tokens", "tokens/s"],
            [
                [
                    r["n"],
                    r["seconds"],
                    h["rows_per_second"],
                    use["requests"],
                    h["requests_per_second"],
                    use["input_tokens"],
                    h["input_tokens_per_second"],
                ]
            ],
        ),
        "",
        _table(
            ["est_usd", "USD / 1k rows", "429s", "529s", "retries"],
            [
                [
                    f"${use['est_usd']:.4f}",
                    f"${h['usd_per_1k_rows']:.4f}",
                    use["rate_limited"],
                    use["overloaded"],
                    use["retries"],
                ]
            ],
        ),
        "",
        "## Accuracy",
        "",
        _sql(ACCURACY_SQL),
        "",
        f"Accuracy: **{r['accuracy']:.4f}** over {r['n']} rows.",
        "",
        "## Calibration",
        "",
        "Expected calibration error over 10 equal-width bins, as the handoff specifies, over "
        "`j.confidence`. Jev's Choice `confidence` is derived from how concentrated the whole "
        "distribution is, not the probability of the chosen option, so the table after it "
        "repeats the analysis over the top-1 probability `j.probabilities[j.choice]`, which "
        "is the quantity classical calibration measures.",
        "",
        _sql(ECE_SQL.format(conf="j.confidence")),
        "",
        f"ECE over `j.confidence`: **{r['ece_confidence']:.4f}**. "
        f"ECE over the top-1 probability: **{r['ece_top_prob']:.4f}**.",
        "",
        "### Reliability over `j.confidence`",
        "",
        _sql(RELIABILITY_SQL.format(conf="j.confidence")),
        "",
        _table(["bin", "n", "mean confidence", "accuracy"], [list(x) for x in r["reliability"]]),
        "",
        "### Reliability over the top-1 probability",
        "",
        _table(["bin", "n", "mean top-1 p", "accuracy"], [list(x) for x in r["reliability_top"]]),
        "",
        "## Soft vs hard group-by",
        "",
        "Per intent: the gold count, the hard count (rows whose argmax is the intent), the "
        "expected count Σp (each row's probability mass on the intent, summed), and its "
        "standard error √Σp(1−p) under the tuple-independent Bernoulli model.",
        "",
        _sql(SOFT_VS_HARD_SQL),
        "",
        f"Totals across {h['intents']} intents: Σ abs(hard − true) = "
        f"**{h['sum_abs_hard_minus_true']}**, Σ abs(expected − true) = "
        f"**{h['sum_abs_expected_minus_true']}**; abs(expected − true) ≤ 2·SE for "
        f"**{h['intents_within_2se']} / {h['intents']}** intents.",
        "",
        _table(
            [
                "intent",
                "true",
                "hard",
                "expected",
                "stderr",
                "abs(hard−true)",
                "abs(exp−true)",
                "within 2 SE",
            ],
            per_intent,
        ),
        "",
        "### The same soft group-by, straight off the source table",
        "",
        "The demo form needs no judged table; with a warm cache it costs nothing "
        f"({r['demo_usage']['requests']} requests, {r['demo_seconds']:.2f} s). Top five:",
        "",
        _sql(DEMO_SOFT_GROUP_BY_SQL),
        "",
        _table(["intent", "expected_rows"], [list(x) for x in r["demo"]]),
        "",
        "## Cache re-run",
        "",
        "The identical `CREATE TABLE` run again on the same connection and cache file:",
        "",
        _table(
            ["wall seconds", "requests", "cache hits", "est_usd", "rows identical to first run"],
            [
                [
                    f"{r['rerun_seconds']:.3f}",
                    r["rerun_usage"]["requests"],
                    r["rerun_usage"]["cache_hits"],
                    f"${r['rerun_usage']['est_usd']:.4f}",
                    f"{r['rerun_identical_rows']} / {r['n']}",
                ]
            ],
        ),
        "",
        "## `sem_where` and calibrated counting",
        "",
        f"Noul question `{TOPUP_Q}` over all rows. Ground truth is the {len(TOPUP_INTENTS)} "
        "intents whose label is about top-ups: " + ", ".join(f"`{i}`" for i in TOPUP_INTENTS) + ".",
        "",
        _sql(SEM_WHERE_SQL),
        "",
        _table(
            [
                "sem_where ≥ 0.5",
                "expected_count",
                "stderr",
                "true count",
                "precision",
                "recall",
                "within 2 SE",
            ],
            [[sem[0], sem[1], sem[2], sem[3], sem_precision, sem_recall, sem_within]],
        ),
        "",
        f"Cost of this query: {r['sem_where_usage']['requests']} requests, "
        f"{r['sem_where_usage']['input_tokens']:,} input tokens, "
        f"${r['sem_where_usage']['est_usd']:.4f}, {r['sem_where_seconds']:.1f} s.",
        "",
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("phase", choices=["sample", "full"])
    p.add_argument("--n", type=int, default=200, help="sample size (sample phase)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-usd", type=float, default=0.50, help="hard stop for the full split")
    p.add_argument("--dry-run", action="store_true", help="use a fake transport, no network")
    args = p.parse_args(argv)
    return run_sample(args) if args.phase == "sample" else run_full(args)


if __name__ == "__main__":
    sys.exit(main())
