"""Live SROIE receipts benchmark for ``jev_extract`` (workstation only; needs TYPESAFE_API_KEY).

    uv run python bench/sroie.py prepare                        # text columns, once, no key
    uv run python bench/sroie.py coverage                       # candidate coverage, offline
    uv run python bench/sroie.py run R0 --split train --limit 40   # pre-flight sample
    uv run python bench/sroie.py run R0 --split train           # one tuning round on dev
    uv run python bench/sroie.py run R2 --split test            # the held-out split
    uv run python bench/sroie.py rescore                        # rebuild from answer caches, free
    uv run python bench/sroie.py report                         # docs/results/sroie.md
    uv run python bench/sroie.py run R0 --split train --dry-run # fake transport, no key

Select, don't generate: SQL builds candidate spans per receipt (money and date regexes,
runs of header lines), ``jev_extract`` sends one request per receipt with one Choice per
field over that receipt's own candidates, and the answer is a verbatim copy of a candidate.
Candidate coverage (is the gold value among the candidates at all) is a property of the
builders and is measured without Jev; selection is measured given coverage.

The train split (626 receipts) is the dev set for tuning rounds; the test split (347) is
held out. Every full run appends its metrics to ``docs/results/sroie_runs.json``, and
``report`` renders ``docs/results/sroie.md`` from that file alone.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import httpx
import pyarrow as pa

import duckjev
from duckjev.client import USD_PER_INPUT_TOKEN

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bench" / "data"
RUNS_FILE = ROOT / "docs" / "results" / "sroie_runs.json"
RESULTS = ROOT / "docs" / "results" / "sroie.md"
SOURCE = "rth/sroie-2019-v2"
SOURCE_URL = (
    "https://huggingface.co/datasets/rth/sroie-2019-v2/resolve/"
    "refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"
)
FIELDS = ("company", "date", "address", "total")
TUNING_SPLIT, HELDOUT_SPLIT = "train", "test"

# --------------------------------------------------------------------------- rounds

PLAIN = {
    "company": "What is the company name on this receipt?",
    "date": "What is the date of this receipt?",
    "address": "What is the address on this receipt?",
    "total": "What is the total amount on this receipt?",
}
COLLEAGUE = {
    "company": (
        "Which candidate is the name of the business that issued this receipt, exactly as "
        "printed? When the receipt prints a registered company name (usually ending in SDN "
        "BHD, S/B, BHD or BERHAD), that registered name is the answer, even if a shop brand "
        "or outlet name is printed more prominently. Leave out registration numbers in "
        "brackets. Not a person's name, the cashier, the customer, or the mall."
    ),
    "date": (
        "Which candidate is the date of this purchase, the day the receipt was issued? Not a "
        "date printed in a promotion, a membership expiry or a return-policy deadline."
    ),
    "address": (
        "Which candidate is the full postal address of the business that issued this "
        "receipt, as printed near the top: all of its lines in order, from the building or "
        "street line through the postcode, city and state? Leave out the business name, "
        "registration or GST numbers, and telephone or fax lines."
    ),
    "total": (
        "Which amount is the final total of this receipt: what the customer had to pay after "
        "tax, discounts and rounding? Not the subtotal before tax, the tax or GST amount, the "
        "cash handed over, or the change given back."
    ),
}


# R4 hypothesis, from R2's misses: the gold address ends with the branch, outlet or mall
# line printed under the postcode and state, and R1's wording told Jev to stop early.
COLLEAGUE_R4 = {
    **COLLEAGUE,
    "address": (
        "Which candidate is the full address block of the business that issued this "
        "receipt, as printed near the top: every line of it in order, from the building or "
        "street line through the postcode, city and state, including a branch, outlet or mall "
        "name printed as the last line of that block? Leave out the business name, "
        "registration or GST numbers, and telephone or fax lines."
    ),
}


# R5: on train, 36% of gold company names are the shop name, and in 39 of those the shop
# name is printed above a registered name; R1's "the registered name wins" picks the latter.
COLLEAGUE_R5 = {
    **COLLEAGUE,
    "company": (
        "Which candidate is the name of the business that issued this receipt, exactly as "
        "printed where the receipt names its issuer at the top? If a shop or brand name is "
        "printed above a registered company name (one ending in SDN BHD, S/B, BHD or "
        "BERHAD), the shop or brand name is the answer; if the registered name comes first "
        "or is the only name, it is the answer. Leave out registration numbers in brackets. "
        "Not a person's name, the cashier, the customer, or the mall."
    ),
}


@dataclass(frozen=True)
class Round:
    note: str
    state: str  # column holding the state text: raw_text or layout_text
    order: str  # forward, or reverse (every candidate list reversed)
    wording: dict[str, str]
    selectable: bool = True  # False for robustness checks that are not a candidate config


ROUNDS: dict[str, Round] = {
    "R0": Round(
        "baseline: plain one-line questions; lines in annotation order",
        "raw_text",
        "forward",
        PLAIN,
    ),
    "R1": Round(
        "colleague-style questions that say what to exclude", "raw_text", "forward", COLLEAGUE
    ),
    "R2": Round(
        "R1 plus the receipt rebuilt into visual rows from the boxes",
        "layout_text",
        "forward",
        COLLEAGUE,
    ),
    "R3": Round(
        "R2 with every candidate list reversed, as an option-order check; it lifted "
        "address selection, so order became a lever: reversed puts the longest runs of "
        "lines first",
        "layout_text",
        "reverse",
        COLLEAGUE,
    ),
    "R4": Round(
        "R3 with the address question asking for a trailing branch or outlet line as part "
        "of the block, the hypothesis from R2's too-short misses; the gold mostly leaves it "
        "out, so this hurt",
        "layout_text",
        "reverse",
        COLLEAGUE_R4,
    ),
    "R5": Round(
        "R3 with the company question preferring whichever issuer name is printed "
        "first, shop or registered",
        "layout_text",
        "reverse",
        COLLEAGUE_R5,
    ),
}

# --------------------------------------------------------------------------- SQL

# Candidate builders: fixed across rounds, so coverage is a property of the builders only.
# company: runs of 1..4 of the first 10 lines, the same with a trailing "(...)" removed, and
#          legal-name spans ending in SDN BHD / S/B / BHD / BERHAD from the first 12 lines
# address: runs of 1..6 of the first 20 lines
CANDIDATES_SQL = r"""CREATE OR REPLACE TABLE cands AS
SELECT id, gold, raw_text, layout_text,
       jev_money_spans(raw_text) AS total_c,
       jev_date_spans(raw_text)  AS date_c,
       flatten([
         jev_line_windows(lines[1:10], 4),
         list_transform(jev_line_windows(lines[1:10], 4),
                        lambda x: trim(regexp_replace(x, '\s*\([^)]*\)\s*$', ''))),
         list_transform(
           regexp_extract_all(array_to_string(lines[1:12], chr(10)),
                              '[A-Z][A-Z0-9''&.() -]*?\b(?:SDN\.? ?BHD|S/B|BHD|BERHAD)\.?'),
           lambda x: trim(x))
       ]) AS company_c,
       jev_line_windows(lines[1:20], 6) AS address_c
FROM receipts"""

# Fixed SQL text: $layout picks the rebuilt visual rows over the raw lines, and $reverse
# reverses every candidate list. Nothing is formatted into the query.
EXTRACT_SQL = """CREATE OR REPLACE TABLE extracted AS
SELECT id, jev_extract(CASE WHEN $layout THEN layout_text ELSE raw_text END, json_object(
         'company', jev_field($company_q, ordered(company_c, $reverse)),
         'date',    jev_field($date_q,    ordered(date_c,    $reverse)),
         'address', jev_field($address_q, ordered(address_c, $reverse)),
         'total',   jev_field($total_q,   ordered(total_c,   $reverse)))) AS x
FROM cands"""
EXTRACT_RERUN_SQL = EXTRACT_SQL.replace("TABLE extracted AS", "TABLE extracted_rerun AS", 1)

LOAD_SQL = """CREATE OR REPLACE TABLE receipts AS
SELECT * FROM (SELECT * FROM read_parquet($path) ORDER BY hash(id || 'sroie-sample') LIMIT $n)
ORDER BY id"""

# ordered(): candidate order is a round input. primary match: case- and whitespace-
# insensitive; loose: letters and digits only. Amounts compare as digits and the decimal
# point under both (gold writes "$8.20" for "8.20").
BENCH_MACROS = [
    "CREATE OR REPLACE MACRO ordered(l, rev) AS CASE WHEN rev THEN list_reverse(l) ELSE l END",
    r"""CREATE OR REPLACE MACRO norm_primary(f, s) AS CASE WHEN f = 'total'
          THEN regexp_replace(coalesce(s, ''), '[^0-9.]+', '', 'g')
          ELSE upper(regexp_replace(coalesce(s, ''), '\s+', '', 'g')) END""",
    r"""CREATE OR REPLACE MACRO norm_loose(f, s) AS CASE WHEN f = 'total'
          THEN regexp_replace(coalesce(s, ''), '[^0-9.]+', '', 'g')
          ELSE upper(regexp_replace(coalesce(s, ''), '[^A-Za-z0-9]+', '', 'g')) END""",
]

SCORED_SQL = """CREATE OR REPLACE TABLE scored AS
WITH per_field AS (
  SELECT c.id, u.f.field, u.f.gold, u.f.cands, e.x[u.f.field] AS r
  FROM cands c JOIN extracted e USING (id),
       UNNEST([{'field': 'company', 'gold': c.gold.company, 'cands': c.company_c},
               {'field': 'date',    'gold': c.gold.date,    'cands': c.date_c},
               {'field': 'address', 'gold': c.gold.address, 'cands': c.address_c},
               {'field': 'total',   'gold': c.gold.total,   'cands': c.total_c}]) AS u(f)
  WHERE coalesce(u.f.gold, '') <> '')
SELECT id, field, gold, r.value AS value, r.p AS p, r.p_none AS p_none,
       r.n_candidates AS n_candidates,
       list_contains(list_transform(cands, lambda x: norm_primary(field, x)),
                     norm_primary(field, gold)) AS covered,
       list_contains(list_transform(cands, lambda x: norm_loose(field, x)),
                     norm_loose(field, gold)) AS covered_loose,
       coalesce(norm_primary(field, r.value) = norm_primary(field, gold), false) AS exact,
       coalesce(norm_loose(field, r.value) = norm_loose(field, gold), false) AS exact_loose,
       coalesce(1 - levenshtein(norm_loose(field, r.value), norm_loose(field, gold))
                    / greatest(length(norm_loose(field, gold)), 1) >= 0.9, false) AS near
FROM per_field"""

FIELD_METRICS_SQL = """SELECT coalesce(field, 'all') AS field, count(*) AS n,
  avg(covered::INT) AS coverage, avg(covered_loose::INT) AS coverage_loose,
  avg(exact::INT) AS exact, avg(exact_loose::INT) AS exact_loose,
  count(*) FILTER (covered) AS n_covered,
  avg(exact::INT) FILTER (covered) AS selection,
  avg(exact_loose::INT) FILTER (covered_loose) AS selection_loose,
  avg((value IS NULL)::INT) AS none_rate,
  avg((value IS NULL)::INT) FILTER (covered) AS none_when_covered,
  count(*) FILTER (NOT covered) AS n_uncovered,
  count(*) FILTER (NOT covered AND value IS NULL) AS uncovered_none,
  count(*) FILTER (NOT covered AND value IS NOT NULL AND near) AS uncovered_near,
  count(*) FILTER (NOT covered AND value IS NOT NULL AND NOT near) AS uncovered_other,
  avg(n_candidates) AS mean_candidates, max(n_candidates) AS max_candidates
FROM scored GROUP BY ROLLUP (field) ORDER BY field NULLS LAST"""

RECORD_SQL = """SELECT count(*) AS receipts, avg(all_exact::INT) AS all_fields_exact
FROM (SELECT id, bool_and(exact) AS all_exact FROM scored GROUP BY id)"""

RELIABILITY_SQL = """WITH b AS (
  SELECT least(floor(p * 10), 9)::INT AS bin, p, exact::INT AS correct
  FROM scored WHERE value IS NOT NULL AND (covered OR NOT $covered_only))
SELECT bin, count(*) AS n, avg(p) AS mean_p, avg(correct) AS accuracy
FROM b GROUP BY bin ORDER BY bin"""

THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.99)
THRESHOLD_SQL = """SELECT $t AS threshold,
  avg((value IS NOT NULL AND p >= $t)::INT) AS answered,
  avg(exact::INT) FILTER (value IS NOT NULL AND p >= $t) AS precision
FROM scored"""

# --------------------------------------------------------------------------- data


def data_file(split: str) -> Path:
    return DATA / f"sroie_{split}.parquet"


def run_tag(split: str, rnd: str, limit: int | None = None, dry_run: bool = False) -> str:
    """Names a run's cache and per-row files; only full live runs get the bare tag."""
    return f"{split}_{rnd}" + (f"_n{limit}" if limit else "") + ("_dry" if dry_run else "")


def extract_params(rnd: Round) -> dict[str, Any]:
    return {
        "layout": rnd.state == "layout_text",
        "reverse": rnd.order == "reverse",
        **{f"{f}_q": rnd.wording[f] for f in FIELDS},
    }


def layout_text(lines: list[str], boxes: list[list[int]]) -> str:
    """Rebuild visual rows: sort boxes top to bottom, merge boxes whose vertical centers lie
    within half a line height of the row, and order each row left to right."""
    items = sorted(zip(lines, boxes, strict=True), key=lambda t: ((t[1][1] + t[1][3]) / 2, t[1][0]))
    rows: list[dict[str, Any]] = []
    for text, (x0, y0, _x1, y1) in items:
        yc, h = (y0 + y1) / 2, max(y1 - y0, 1)
        if rows and abs(yc - rows[-1]["yc"]) <= 0.5 * min(h, rows[-1]["h"]):
            rows[-1]["items"].append((x0, text))
        else:
            rows.append({"yc": yc, "h": h, "items": [(x0, text)]})
    return "\n".join("  ".join(t for _, t in sorted(r["items"])) for r in rows)


def prepare() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    for split in (TUNING_SPLIT, HELDOUT_SPLIT):
        rows = con.execute(
            "SELECT replace(image.path, '.jpg', '') AS id, objects.entities AS gold, "
            "objects.text AS lines, objects.bbox AS bbox FROM read_parquet(?) ORDER BY id",
            [SOURCE_URL.format(split=split)],
        ).fetchall()
        records = []
        for rid, gold, lines, bbox in rows:
            boxes = [[min(b[0]), min(b[1]), max(b[0]), max(b[1])] for b in bbox]
            records.append(
                {
                    "id": rid,
                    "gold": gold,
                    "lines": lines,
                    "raw_text": "\n".join(lines),
                    "layout_text": layout_text(lines, boxes),
                }
            )
        con.from_arrow(pa.Table.from_pylist(records)).write_parquet(str(data_file(split)))
        print(f"{split}: {len(records)} receipts -> {data_file(split).relative_to(ROOT)}")
    return 0


def load(con: duckdb.DuckDBPyConnection, split: str, limit: int | None) -> int:
    if not data_file(split).exists():
        raise SystemExit("run `bench/sroie.py prepare` first")
    con.execute(LOAD_SQL, {"path": str(data_file(split)), "n": limit or 1_000_000})
    con.execute(CANDIDATES_SQL)
    for m in BENCH_MACROS:
        con.execute(m)
    return con.execute("SELECT count(*) FROM receipts").fetchone()[0]


# --------------------------------------------------------------------------- coverage


EMPTY_EXTRACTED_SQL = """CREATE OR REPLACE TABLE extracted AS
SELECT id, MAP([]::VARCHAR[], []::STRUCT(value VARCHAR, p DOUBLE, p_none DOUBLE,
  confidence DOUBLE, n_candidates INTEGER, probabilities MAP(VARCHAR, DOUBLE))[]) AS x
FROM cands"""

COVERAGE_SQL = """SELECT field, count(*), avg(covered::INT), avg(covered_loose::INT),
  avg(len(list_distinct(cands))), max(len(list_distinct(cands)))
FROM (SELECT s.*, CASE s.field WHEN 'company' THEN c.company_c WHEN 'date' THEN c.date_c
                  WHEN 'address' THEN c.address_c ELSE c.total_c END AS cands
      FROM scored s JOIN cands c USING (id))
GROUP BY field ORDER BY field"""


def coverage(args: argparse.Namespace) -> int:
    con = duckdb.connect()
    duckjev.register(con, cache=False)  # macros only; nothing here calls Jev
    for split in (TUNING_SPLIT, HELDOUT_SPLIT):
        load(con, split, None)
        con.execute(EMPTY_EXTRACTED_SQL)  # no answers: scoring then measures coverage only
        con.execute(SCORED_SQL)
        print(f"== {split}")
        for f, n, cov, cov_l, mean_c, max_c in con.execute(COVERAGE_SQL).fetchall():
            print(
                f"  {f:8s} n={n:4d} coverage {cov:.3f}  loose {cov_l:.3f}  "
                f"candidates mean {mean_c:.1f} max {max_c}"
            )
        if args.show_misses and split == args.split:
            for field, gold in con.execute(
                "SELECT field, gold FROM scored WHERE NOT covered ORDER BY field, id"
            ).fetchall():
                print(f"  miss {field:8s} {gold}")
    return 0


# --------------------------------------------------------------------------- live run


def fake_transport(split: str) -> httpx.MockTransport:
    """Dry-run stand-in: picks the gold candidate 80% of the time when it is offered."""
    rng = random.Random(7)
    con = duckdb.connect()
    gold: dict[str, dict[str, str]] = {}
    for g, raw, lay in con.execute(
        "SELECT gold, raw_text, layout_text FROM read_parquet(?)", [str(data_file(split))]
    ).fetchall():
        gold[raw] = gold[lay] = g

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        g = gold.get(body["state"], {})
        answers = {}
        for qid, q in body["questions"].items():
            opts = list(q["criteria"])
            want = next(
                (o for o in opts if o.replace(" ", "") == (g.get(qid) or "").replace(" ", "")),
                "none",
            )
            top = want if rng.random() < 0.8 else rng.choice(opts)
            probs = {o: 0.2 / max(len(opts) - 1, 1) for o in opts}
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
                "usage": {"input_tokens": 1500, "output_tokens": 40},
            },
        )

    return httpx.MockTransport(handle)


def _reliability(
    con: duckdb.DuckDBPyConnection, covered_only: bool
) -> tuple[list[Any], float | None]:
    rows = [
        list(r) for r in con.execute(RELIABILITY_SQL, {"covered_only": covered_only}).fetchall()
    ]
    answered = sum(r[1] for r in rows)
    ece = sum(r[1] * abs(r[3] - r[2]) for r in rows) / answered if answered else None
    return rows, ece


def metrics(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Every reported metric, from the ``scored`` table alone."""
    cols = [d[0] for d in con.execute(FIELD_METRICS_SQL).description]
    fields = {
        r[0]: dict(zip(cols, r, strict=True)) for r in con.execute(FIELD_METRICS_SQL).fetchall()
    }
    _, record_exact = con.execute(RECORD_SQL).fetchone()
    reliability, ece = _reliability(con, covered_only=False)
    reliability_covered, ece_covered = _reliability(con, covered_only=True)
    return {
        "fields": fields,
        "record_all_fields_exact": record_exact,
        "reliability": reliability,
        "ece": ece,
        "reliability_covered": reliability_covered,
        "ece_covered": ece_covered,
        "thresholds": [list(con.execute(THRESHOLD_SQL, {"t": t}).fetchone()) for t in THRESHOLDS],
    }


def _refuse(request: httpx.Request) -> httpx.Response:
    raise RuntimeError("rescore is served from the run's own cache; this request missed it")


def rescore(args: argparse.Namespace) -> int:
    """Rebuild every recorded run's rows and metrics from that run's Jev answer cache.

    The cache is the durable record of what Jev answered. The transport refuses every
    request, so a rescore can neither spend nor read anything but the run's own answers;
    timing and usage in the run log are kept from the live run.
    """
    runs = json.loads(RUNS_FILE.read_text())
    for key, summary in runs.items():
        tag = run_tag(summary["split"], summary["round"])
        cache = DATA / f"cache_{tag}.duckdb"
        if not cache.exists():
            print(f"skip {key}: {cache.relative_to(ROOT)} is missing")
            continue
        con = duckdb.connect()
        duckjev.register(
            con, cache_path=cache, api_key="cache-only", transport=httpx.MockTransport(_refuse)
        )
        load(con, summary["split"], None)
        duckjev.usage(reset=True)
        con.execute(EXTRACT_SQL, extract_params(ROUNDS[summary["round"]]))
        assert duckjev.usage()["requests"] == 0
        con.execute(SCORED_SQL)
        con.table("scored").write_parquet(str(DATA / f"scored_{tag}.parquet"))
        before = (summary["fields"]["all"]["exact"], summary["record_all_fields_exact"])
        summary.update(metrics(con))
        after = (summary["fields"]["all"]["exact"], summary["record_all_fields_exact"])
        print(
            f"rescored {key}: all-fields exact {before[0]:.4f} -> {after[0]:.4f}, "
            f"all four {before[1]:.4f} -> {after[1]:.4f}"
        )
    RUNS_FILE.write_text(json.dumps(runs, indent=1, default=float) + "\n")
    return 0


def run(args: argparse.Namespace) -> int:
    rnd = ROUNDS[args.round]
    DATA.mkdir(parents=True, exist_ok=True)
    tag = run_tag(args.split, args.round, args.limit, args.dry_run)
    cache_file = DATA / f"cache_{tag}.duckdb"
    if cache_file.exists():
        cache_file.unlink()  # a fresh cache per run, so the timed run pays for every receipt
    con = duckdb.connect()
    duckjev.register(
        con,
        cache_path=cache_file,
        concurrency=args.concurrency,
        max_input_tokens=int(args.max_usd / USD_PER_INPUT_TOKEN),
        transport=fake_transport(args.split) if args.dry_run else None,
        api_key="dry-run" if args.dry_run else None,
    )
    n = load(con, args.split, args.limit)
    params = extract_params(rnd)

    duckjev.usage(reset=True)
    t0 = time.perf_counter()
    con.execute(EXTRACT_SQL, params)
    secs = time.perf_counter() - t0
    use = duckjev.usage(reset=True)

    t0 = time.perf_counter()
    con.execute(EXTRACT_RERUN_SQL, params)
    rerun_secs = time.perf_counter() - t0
    rerun = duckjev.usage(reset=True)
    duckjev.flush(con)

    con.execute(SCORED_SQL)
    con.table("scored").write_parquet(str(DATA / f"scored_{tag}.parquet"))

    summary = {
        "round": args.round,
        "split": args.split,
        "limit": args.limit,
        "dry_run": args.dry_run,
        "round_config": asdict(rnd),
        "receipts": n,
        "model": duckjev.client_for(con).model,
        "duckjev": duckjev.__version__,
        "duckdb": duckdb.__version__,
        "python": platform.python_version(),
        "concurrency": args.concurrency,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "seconds": secs,
        "usage": use,
        "receipts_per_second": n / secs,
        "input_tokens_per_request": use["input_tokens"] / max(use["requests"], 1),
        "usd_per_1k_receipts": use["est_usd"] / n * 1000,
        "rerun_seconds": rerun_secs,
        "rerun_usage": rerun,
        **metrics(con),
    }
    print(json.dumps(_brief(summary), indent=2))
    if args.limit is None and not args.dry_run:
        runs = json.loads(RUNS_FILE.read_text()) if RUNS_FILE.exists() else {}
        runs[f"{args.split}/{args.round}"] = summary
        RUNS_FILE.write_text(json.dumps(runs, indent=1, default=float) + "\n")
        print(f"recorded {args.split}/{args.round} in {RUNS_FILE.relative_to(ROOT)}")
    else:
        (DATA / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1, default=float))
    return 0


def _brief(s: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": f"{s['split']}/{s['round']}" + (f" (n={s['limit']})" if s["limit"] else ""),
        "receipts": s["receipts"],
        "seconds": round(s["seconds"], 1),
        "requests": s["usage"]["requests"],
        "tokens_per_request": round(s["input_tokens_per_request"]),
        "est_usd": round(s["usage"]["est_usd"], 4),
        "usd_per_1k_receipts": round(s["usd_per_1k_receipts"], 4),
        "429s": s["usage"]["rate_limited"],
        "record_all_fields_exact": round(s["record_all_fields_exact"], 3),
        "ece": None if s["ece"] is None else round(s["ece"], 3),
        "ece_covered": None if s["ece_covered"] is None else round(s["ece_covered"], 3),
        "fields": {
            f: {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in m.items()
                if k in ("n", "coverage", "exact", "selection", "exact_loose", "none_when_covered")
            }
            for f, m in s["fields"].items()
        },
        "rerun": {
            "seconds": round(s["rerun_seconds"], 2),
            "requests": s["rerun_usage"]["requests"],
        },
    }


# --------------------------------------------------------------------------- report


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:.1f}%"


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def chosen_round(runs: dict[str, Any]) -> str:
    """The selectable round with the best all-fields exact match on the tuning split."""
    tuned = [
        (r["fields"]["all"]["exact"], r["round"])
        for k, r in runs.items()
        if k.startswith(f"{TUNING_SPLIT}/") and ROUNDS[r["round"]].selectable
    ]
    return max(tuned)[1]


def report(args: argparse.Namespace) -> int:
    runs = json.loads(RUNS_FILE.read_text())
    best = chosen_round(runs)
    dev = [runs[k] for k in sorted(runs) if k.startswith(f"{TUNING_SPLIT}/")]
    held = [runs[k] for k in sorted(runs) if k.startswith(f"{HELDOUT_SPLIT}/")]
    final = runs.get(f"{HELDOUT_SPLIT}/{best}")
    base = runs.get(f"{HELDOUT_SPLIT}/R0")
    parts = [
        "# SROIE receipts: `jev_extract` benchmark",
        "",
        "Generated by `bench/sroie.py report` from `docs/results/sroie_runs.json`; do not edit "
        "by hand. Every number below is from a live run against `jev-1.13.0`.",
        "",
        "Corpus: ICDAR 2019 SROIE (Scanned Receipts OCR and Information Extraction; Huang "
        "et al., ICDAR 2019, arXiv:2103.10213), task 3 "
        f"key fields `company`, `date`, `address`, `total`, from `{SOURCE}` on Hugging Face: "
        "the task 1 line transcriptions with their boxes, and the gold key fields. The train "
        "split (626 receipts) is the dev set for the tuning rounds; the test split (347 "
        "receipts) is held out and was run only with the baseline and the round chosen on "
        "dev. The chosen round is the selectable round with the best all-fields exact match "
        f"on train: **{best}**.",
        "",
        "How it works: SQL builds the candidates for each receipt, `jev_extract` sends one "
        "request per receipt with one Choice per field over that receipt's candidates plus "
        "`none`, and each answer is a verbatim copy of a candidate. So there are two separate "
        "numbers per field. **Coverage** is how often the gold value is among the candidates "
        "at all; it depends on the candidate builders, and Jev plays no part in it. "
        "**Selection** is how often Jev picks the gold value when it is there.",
        "",
        "Matching: *exact* compares case- and whitespace-insensitively; *loose* compares "
        "letters and digits only. Amounts compare as digits and the decimal point under both, "
        "because the gold writes `$8.20` where the receipt prints `8.20`. The SROIE gold "
        "values were typed by annotators. Where the gold corrects a misprint (`SDN BHD` for "
        "the printed `SDN BND`) or has its own typo (`TED HENG` for the printed `TEO HENG`), no "
        "candidate matches, and the row counts against coverage.",
        "",
    ]
    if final:
        parts += [
            f"## Held-out test split, round {best}",
            "",
            _field_table(final),
            "",
            f"All four fields exact on the same receipt: "
            f"**{_pct(final['record_all_fields_exact'])}** of {final['receipts']} receipts. "
            f"Throughput {final['receipts_per_second']:.1f} receipts/s "
            f"({final['usage']['requests']} requests in {final['seconds']:.1f} s at "
            f"concurrency {final['concurrency']}, {final['usage']['rate_limited']} × 429); "
            f"{final['input_tokens_per_request']:,.0f} input tokens per request; "
            f"**${final['usd_per_1k_receipts']:.4f} per 1,000 receipts** "
            f"(${final['usage']['est_usd']:.4f} for the split). A cache re-run took "
            f"{final['rerun_seconds']:.2f} s with {final['rerun_usage']['requests']} requests.",
            "",
        ]
        if base and best != "R0":
            parts += [
                "### Held-out runs: the baseline and the chosen round",
                "",
                "Exact match, with selection given coverage in brackets.",
                "",
                _rounds_table(held),
                "",
            ]
    parts += [
        "## Tuning rounds on the dev split (train)",
        "",
        "Candidates are identical in every round, so coverage does not move; only selection "
        "does. Cells are exact match with selection given coverage in brackets.",
        "",
        _rounds_table(dev),
        "",
        *[f"- **{r['round']}**: {r['round_config']['note']}." for r in dev],
        "",
    ]
    if final:
        parts += [
            f"## Calibration on the held-out split, round {best}",
            "",
            "`p` of the returned candidate against exact match, 10 equal-width bins. Two "
            "views: over the fields whose gold value was among the candidates, which is the "
            "calibration of Jev's selection, and over every returned field, where an "
            "uncovered row is wrong whatever Jev picks.",
            "",
            f"Covered fields: ECE **{final['ece_covered']:.3f}**.",
            "",
            _reliability_table(final["reliability_covered"]),
            "",
            f"Every returned field: ECE **{final['ece']:.3f}**.",
            "",
            _reliability_table(final["reliability"]),
            "",
            "Acting only on answers at or above a threshold on `p` (answered is the share of "
            "all gold fields, precision is exact match among them):",
            "",
            _table(
                ["p ≥", "answered", "precision"],
                [[t, _pct(a), _pct(pr)] for t, a, pr in final["thresholds"]],
            ),
            "",
            "## Where coverage is lost, held-out split",
            "",
            "For gold values no candidate matches, what `jev_extract` returned instead. *Near* "
            "means a candidate within 10% edit distance of the gold, letters and digits only. "
            "That is almost always the printed text the gold was typed from.",
            "",
            _table(
                ["field", "uncovered", "returned none", "returned a near match", "other"],
                [
                    [
                        f,
                        m["n_uncovered"],
                        m["uncovered_none"],
                        m["uncovered_near"],
                        m["uncovered_other"],
                    ]
                    for f, m in final["fields"].items()
                ],
            ),
            "",
        ]
    parts += [
        "## The SQL",
        "",
        "Candidates (the `jev_*_spans` and `jev_line_windows` builders are macros installed by "
        "`duckjev.register`):",
        "",
        f"```sql\n{CANDIDATES_SQL}\n```",
        "",
        "Extraction. `$layout` picks the rebuilt visual rows over the raw lines, and "
        "`$reverse` reverses every candidate list through the bench macro "
        "`ordered(l, rev)`:",
        "",
        f"```sql\n{EXTRACT_SQL}\n```",
        "",
        "Question wording, plain (R0):",
        "",
        *[f"- `{f}`: {PLAIN[f]}" for f in FIELDS],
        "",
        "Colleague-style (R1 onward):",
        "",
        *[f"- `{f}`: {COLLEAGUE[f]}" for f in FIELDS],
        "",
        f"R4 changes only `address`: {COLLEAGUE_R4['address']}",
        "",
        f"R5 changes only `company` (from R3): {COLLEAGUE_R5['company']}",
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python bench/sroie.py prepare",
        "uv run python bench/sroie.py coverage",
        *[
            f"uv run python bench/sroie.py run {r['round']} --split {r['split']}"
            for r in dev + held
        ],
        "uv run python bench/sroie.py report",
        "```",
        "",
    ]
    RESULTS.write_text("\n".join(parts))
    print(f"wrote {RESULTS.relative_to(ROOT)} (chosen round {best})")
    return 0


def _reliability_table(rows: list[list[Any]]) -> str:
    return _table(
        ["bin", "n", "mean p", "exact"],
        [
            [f"{b / 10:.1f}–{(b + 1) / 10:.1f}", n, f"{mp:.3f}", f"{acc:.3f}"]
            for b, n, mp, acc in rows
        ],
    )


def _field_table(s: dict[str, Any]) -> str:
    rows = []
    for f in (*FIELDS, "all"):
        m = s["fields"][f]
        rows.append(
            [
                f,
                m["n"],
                _pct(m["coverage"]),
                f"**{_pct(m['exact'])}**",
                _pct(m["selection"]),
                _pct(m["coverage_loose"]),
                _pct(m["exact_loose"]),
                _pct(m["selection_loose"]),
                _pct(m["none_when_covered"]),
                f"{m['mean_candidates']:.1f}",
            ]
        )
    return _table(
        [
            "field",
            "n",
            "coverage",
            "exact",
            "selection given coverage",
            "coverage (loose)",
            "exact (loose)",
            "selection (loose)",
            "none when covered",
            "candidates",
        ],
        rows,
    )


def _rounds_table(runs: list[dict[str, Any]]) -> str:
    def cell(r: dict[str, Any], f: str) -> str:
        m = r["fields"][f]
        return f"{_pct(m['exact'])} ({_pct(m['selection'])})"

    return _table(
        ["run", *FIELDS, "all", "all four", "tokens / req", "$ / 1k"],
        [
            [
                f"{r['split']}/{r['round']}",
                *[cell(r, f) for f in (*FIELDS, "all")],
                _pct(r["record_all_fields_exact"]),
                f"{r['input_tokens_per_request']:,.0f}",
                f"${r['usd_per_1k_receipts']:.4f}",
            ]
            for r in runs
        ],
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    c = sub.add_parser("coverage")
    c.add_argument("--show-misses", action="store_true")
    c.add_argument("--split", default=TUNING_SPLIT, choices=[TUNING_SPLIT, HELDOUT_SPLIT])
    r = sub.add_parser("run")
    r.add_argument("round", choices=list(ROUNDS))
    r.add_argument("--split", default=TUNING_SPLIT, choices=[TUNING_SPLIT, HELDOUT_SPLIT])
    r.add_argument("--limit", type=int, default=None, help="reservoir sample size")
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--max-usd", type=float, default=0.50, help="hard budget for this run")
    r.add_argument("--dry-run", action="store_true", help="fake transport, no network")
    sub.add_parser("rescore", help="rebuild recorded runs from their answer caches; no requests")
    sub.add_parser("report")
    args = p.parse_args(argv)
    commands = {
        "prepare": lambda a: prepare(),
        "coverage": coverage,
        "run": run,
        "rescore": rescore,
        "report": report,
    }
    return commands[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
