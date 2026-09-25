"""Live Banking77 benchmark for duckjev, tuned in rounds (workstation only; needs TYPESAFE_API_KEY).

    uv run python bench/banking77.py prepare                        # downloads + dev sample, no key
    uv run python bench/banking77.py run R0 --split dev --limit 40  # pre-flight, not recorded
    uv run python bench/banking77.py run R0 --split dev             # one tuning round on dev
    uv run python bench/banking77.py run R2 --split test            # the held-out split
    uv run python bench/banking77.py confusions R0 --split dev      # top confusions, offline
    uv run python bench/banking77.py rescore                        # rebuild from caches, free
    uv run python bench/banking77.py report                         # docs/results/banking77.md
    uv run python bench/banking77.py run R0 --split dev --dry-run   # fake transport, no key

Every round sends one fused ``jev()`` request per message. A flat round asks one Choice over
all 77 intents; a two-level round asks a division Choice and one intent Choice per division,
speculatively, in the same request, and the code consumes the branch the division picks
(the product of the two probabilities is the intent distribution). A round with a fused
top-up Noul answers the ``sem_where`` question in that same request.

The dev split is a stratified sample of the train split, 20 messages per intent (1,540
rows); the test split (3,080 rows) is held out and runs only with the baseline and the
round chosen on dev. Every full live run appends its metrics to
``docs/results/banking77_runs.json``, and ``report`` renders ``docs/results/banking77.md``
from that file alone. Pre-flight samples and dry runs write their own suffixed files under
``bench/data/`` and record nothing.
"""

from __future__ import annotations

import argparse
import json
import math
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
BENCH = ROOT / "bench"
RUNS_FILE = ROOT / "docs" / "results" / "banking77_runs.json"
RESULTS = ROOT / "docs" / "results" / "banking77.md"
SOURCE_URL = "https://huggingface.co/datasets/mteb/banking77/resolve/main/{split}.jsonl"
SPLITS = ("dev", "test")
DEV_PER_INTENT = 20
CRITERIA_FILES = {
    "v1": BENCH / "banking77_criteria.json",
    "v2": BENCH / "banking77_criteria_v2.json",
    "short": BENCH / "banking77_criteria_short.json",
}
DIVISIONS_FILE = BENCH / "banking77_divisions.json"

INSTR = "Which banking-support intent does this customer message express?"
DIVISION_INSTR = "Which area of banking support is this customer message about?"
LEAF_INSTR = "If this customer message is about {label}, which of these intents does it express?"
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
DEFER_BELOW = 0.9  # two-level rounds report the division when its confidence is below this

# --------------------------------------------------------------------------- rounds


@dataclass(frozen=True)
class Round:
    note: str
    criteria: str  # gloss set for the 77 intents: v1, v2 (structured confusables), short
    structure: str  # flat (one 77-option Choice) or two_level (division + per-division Choices)
    order: str  # forward, or reverse (every option list reversed)
    topup: str  # separate (its own jev_noul query), fused (in the same request), none
    selectable: bool = True  # False for checks that are not a candidate config


ROUNDS: dict[str, Round] = {
    "R0": Round(
        "baseline: the PR #1 config, one 77-option Choice with one-line glosses; the top-up "
        "Noul as a separate query",
        "v1",
        "flat",
        "forward",
        "separate",
    ),
    "R1": Round(
        "R0 with the intents in the top dev confusions rewritten as structured criteria "
        "(what / not_for / examples from train)",
        "v2",
        "flat",
        "forward",
        "none",
    ),
    "R2": Round(
        "R1 as two levels: a division Choice and one intent Choice per division, asked "
        "speculatively in one request; the intent distribution is the product",
        "v2",
        "two_level",
        "forward",
        "none",
    ),
    "R3": Round(
        "R1 with every option list reversed: the option-order check on the best round",
        "v2",
        "flat",
        "reverse",
        "none",
    ),
    "R4": Round(
        "R0 with the top-up Noul fused into the intent request instead of a separate query: "
        "the fusion cost lever, and a check that the intent answers do not move",
        "v1",
        "flat",
        "forward",
        "fused",
    ),
    "R5": Round(
        "R0 with short glosses, a few words per intent: the gloss-length cost lever",
        "short",
        "flat",
        "forward",
        "none",
    ),
}

# --------------------------------------------------------------------------- SQL

LOAD_SQL = {
    "dev": "CREATE OR REPLACE TABLE rows AS SELECT text, label_text FROM read_parquet($path)",
    "test": "CREATE OR REPLACE TABLE rows AS SELECT text, label_text FROM read_json($path)",
}
SAMPLE_SQL = """CREATE OR REPLACE TABLE rows AS
SELECT * FROM (SELECT * FROM rows ORDER BY md5(text || 'banking77-sample') LIMIT $n)"""
ORDER_SQL = "CREATE OR REPLACE TABLE rows AS SELECT * FROM rows ORDER BY label_text, text"

# One fused request per message; $questions is the JSON questions map of the round.
JUDGE_SQL = """CREATE OR REPLACE TABLE judged AS
SELECT text, label_text, jev(text, $questions) AS a FROM rows"""
JUDGE_RERUN_SQL = JUDGE_SQL.replace("TABLE judged AS", "TABLE judged_rerun AS", 1)
IDENTICAL_SQL = """SELECT count(*) FROM judged j JOIN judged_rerun r USING (text)
WHERE j.a IS NOT DISTINCT FROM r.a"""

# The separate top-up query of the baseline: the typed function, one Noul per message.
TOPUP_SQL = """CREATE OR REPLACE TABLE topup AS
SELECT text, jev_noul(text, $q) AS p FROM rows"""

ACCURACY_SQL = """SELECT count(*) AS n,
  avg((choice = label_text)::INT) AS accuracy,
  avg((greedy_choice = label_text)::INT) AS greedy_accuracy,
  avg((answer = label_text)::INT) AS strict_accuracy,
  avg((division = division_of(label_text))::INT) AS division_accuracy
FROM scored"""

DEFERRAL_SQL = """SELECT avg(deferred::INT) AS deferred,
  avg((choice = label_text)::INT) FILTER (NOT deferred) AS precision_when_answered,
  avg((division = division_of(label_text))::INT) FILTER (deferred)
    AS division_accuracy_when_deferred,
  avg((choice = label_text)::INT) FILTER (deferred) AS leaf_accuracy_when_deferred
FROM scored"""

RELIABILITY_SQL = """WITH b AS (
  SELECT least(floor({conf} * 10), 9)::INT AS bin, {conf} AS conf,
         (choice = label_text)::INT AS correct
  FROM scored)
SELECT bin, count(*) AS n, avg(conf) AS mean_conf, avg(correct) AS accuracy
FROM b GROUP BY bin ORDER BY bin"""

SOFT_VS_HARD_SQL = """WITH truth AS (
  SELECT label_text AS intent, count(*) AS true_count FROM scored GROUP BY ALL),
hard AS (
  SELECT choice AS intent, count(*) AS hard_count FROM scored GROUP BY ALL),
soft AS (
  SELECT e.key AS intent,
         SUM(e.value) AS expected_count,
         sqrt(SUM(e.value * (1 - e.value))) AS stderr
  FROM scored, UNNEST(map_entries(probabilities)) AS u(e)
  GROUP BY ALL)
SELECT t.intent, t.true_count,
       coalesce(h.hard_count, 0) AS hard_count,
       s.expected_count, s.stderr,
       abs(coalesce(h.hard_count, 0) - t.true_count) AS hard_abs_err,
       abs(s.expected_count - t.true_count) AS soft_abs_err,
       abs(s.expected_count - t.true_count) <= 2 * s.stderr AS within_2se
FROM truth t LEFT JOIN hard h USING (intent) LEFT JOIN soft s USING (intent)
ORDER BY t.intent"""

# Division readout of any round's intent distribution: the mass on the argmax's division. A
# flat round can defer to a division from this alone, with no second question.
DIVISION_MASS_SQL = """WITH mass AS (
  SELECT s.text, m.division, sum(e.value) AS p
  FROM scored s, UNNEST(map_entries(s.probabilities)) AS u(e)
  JOIN division_map m ON m.intent = e.key
  GROUP BY ALL),
top AS (SELECT text, arg_max(division, p) AS division, max(p) AS p FROM mass GROUP BY text)
SELECT avg((t.division = division_of(s.label_text))::INT) AS division_accuracy,
  avg((t.p < $thr)::INT) AS deferred,
  avg((s.choice = s.label_text)::INT) FILTER (t.p >= $thr) AS precision_when_answered,
  avg((t.division = division_of(s.label_text))::INT) FILTER (t.p < $thr)
    AS division_accuracy_when_deferred,
  avg((s.choice = s.label_text)::INT) FILTER (t.p < $thr) AS leaf_accuracy_when_deferred
FROM scored s JOIN top t USING (text)"""

CONFUSIONS_SQL = """SELECT label_text AS gold, choice AS predicted, count(*) AS n
FROM scored WHERE choice <> label_text GROUP BY ALL ORDER BY n DESC, gold, predicted LIMIT $k"""

# sem_where and calibrated counting over the top-up probability, whichever request answered it
TOPUP_METRICS_SQL = """SELECT
  count(*) FILTER (WHERE topup_p >= 0.5)                                     AS sem_where_count,
  expected_count(topup_p)                                                    AS expected,
  expected_count_stderr(topup_p)                                             AS stderr,
  count(*) FILTER (WHERE list_contains($intents, label_text))                AS true_count,
  count(*) FILTER (WHERE topup_p >= 0.5 AND list_contains($intents, label_text))
    AS true_positives
FROM scored"""

# --------------------------------------------------------------------------- questions


def _rel(path: Path) -> str:
    """A path for messages: relative to the repo when inside it, else as is."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def divisions() -> list[dict[str, Any]]:
    return load_json(DIVISIONS_FILE)["divisions"]


def division_of() -> dict[str, str]:
    return {i: d["key"] for d in divisions() for i in d["intents"]}


def intent_criteria(name: str) -> dict[str, Any]:
    """The 77 option descriptions of a gloss set. ``v2`` overlays structured entries on v1."""
    base = load_json(CRITERIA_FILES["v1"])
    if name == "v1":
        return base
    overlay = {k: v for k, v in load_json(CRITERIA_FILES[name]).items() if not k.startswith("_")}
    unknown = set(overlay) - set(base)
    if unknown:
        raise ValueError(f"{name} glosses name intents that do not exist: {sorted(unknown)}")
    if name == "short":
        missing = set(base) - set(overlay)
        if missing:
            raise ValueError(f"short glosses are missing intents: {sorted(missing)}")
        return {k: overlay[k] for k in base}
    return {k: overlay.get(k, v) for k, v in base.items()}


def ordered(items: list[Any], rnd: Round) -> list[Any]:
    return list(reversed(items)) if rnd.order == "reverse" else list(items)


def questions_for(rnd: Round) -> dict[str, Any]:
    """The fused questions map of a round: what one ``jev(text, $questions)`` call asks."""
    crit = intent_criteria(rnd.criteria)
    q: dict[str, Any] = {}
    if rnd.structure == "flat":
        keys = ordered(list(crit), rnd)
        q["intent"] = {
            "type": "choice",
            "instructions": INSTR,
            "criteria": {k: crit[k] for k in keys},
        }
    else:
        divs = ordered(divisions(), rnd)
        q["division"] = {
            "type": "choice",
            "instructions": DIVISION_INSTR,
            "criteria": {d["key"]: d["what"] for d in divs},
        }
        for d in divs:
            q[f"intent_{d['key']}"] = {
                "type": "choice",
                "instructions": LEAF_INSTR.format(label=d["label"]),
                "criteria": {k: crit[k] for k in ordered(d["intents"], rnd)},
            }
    if rnd.topup == "fused":
        q["topup"] = {"type": "noul", "instructions": TOPUP_Q}
    return q


def flatten(answers: dict[str, Any], rnd: Round, div_of: dict[str, str]) -> dict[str, Any]:
    """One uniform record per message from a round's answers map.

    Flat: the Choice as returned. Two-level: the intent distribution is p(division) ×
    p(intent | division) over every division's speculative Choice; ``choice`` is its argmax,
    ``greedy_choice`` follows the division's own pick, and a row whose division confidence
    is under ``DEFER_BELOW`` reports the division as its ``answer``.
    """
    if rnd.structure == "flat":
        a = answers["intent"]
        probs = dict(a["probabilities"])
        choice = a["choice"]
        rec = {
            "choice": choice,
            "greedy_choice": choice,
            "confidence": a["confidence"],
            "division": div_of[choice],
            "division_p": None,
            "division_confidence": None,
            "deferred": False,
            "answer": choice,
        }
    else:
        d = answers["division"]
        probs = {}
        for key, p_div in d["probabilities"].items():
            for intent, p in answers[f"intent_{key}"]["probabilities"].items():
                probs[intent] = p_div * p
        choice = max(probs, key=probs.__getitem__)
        division = d["choice"]
        deferred = d["confidence"] < DEFER_BELOW
        rec = {
            "choice": choice,
            "greedy_choice": answers[f"intent_{division}"]["choice"],
            "confidence": answers[f"intent_{div_of[choice]}"]["confidence"],
            "division": division,
            "division_p": d["probabilities"][division],
            "division_confidence": d["confidence"],
            "deferred": deferred,
            "answer": division if deferred else choice,
        }
    rec["top_p"] = probs[choice]
    rec["probabilities"] = probs
    rec["topup_p"] = answers["topup"]["noul"] if "topup" in answers else None
    return rec


SCORED_SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("label_text", pa.string()),
        ("choice", pa.string()),
        ("greedy_choice", pa.string()),
        ("confidence", pa.float64()),
        ("top_p", pa.float64()),
        ("division", pa.string()),
        ("division_p", pa.float64()),
        ("division_confidence", pa.float64()),
        ("deferred", pa.bool_()),
        ("answer", pa.string()),
        ("topup_p", pa.float64()),
        ("probabilities", pa.map_(pa.string(), pa.float64())),
    ]
)


def score(con: duckdb.DuckDBPyConnection, rnd: Round) -> None:
    """Build ``scored`` from ``judged`` (and ``topup`` when the round asked it separately)."""
    div_of = division_of()
    records = []
    for text, label, a in con.execute("SELECT text, label_text, a FROM judged").fetchall():
        rec = flatten(json.loads(a), rnd, div_of)
        rec["probabilities"] = list(rec["probabilities"].items())
        records.append({"text": text, "label_text": label, **rec})
    table = pa.Table.from_pylist(records, schema=SCORED_SCHEMA)
    con.register("_scored", table)
    if rnd.topup == "separate":
        con.execute(
            "CREATE OR REPLACE TABLE scored AS SELECT s.* EXCLUDE (topup_p), t.p AS topup_p "
            "FROM _scored s JOIN topup t USING (text)"
        )
    else:
        con.execute("CREATE OR REPLACE TABLE scored AS SELECT * FROM _scored")
    con.unregister("_scored")


# --------------------------------------------------------------------------- data


def data_file(split: str) -> Path:
    return DATA / ("banking77_dev.parquet" if split == "dev" else f"banking77_{split}.jsonl")


def run_tag(split: str, rnd: str, limit: int | None = None, dry_run: bool = False) -> str:
    """Names a run's cache and per-row files; only full live runs get the bare tag."""
    return f"{split}_{rnd}" + (f"_n{limit}" if limit else "") + ("_dry" if dry_run else "")


def prepare() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        path = DATA / f"banking77_{split}.jsonl"
        if not path.exists():
            resp = httpx.get(SOURCE_URL.format(split=split), follow_redirects=True, timeout=60)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        print(f"{split}: {sum(1 for _ in path.open())} rows -> {_rel(path)}")
    con = duckdb.connect()
    con.execute(
        "COPY (SELECT text, label_text FROM read_json($train) "
        "QUALIFY row_number() OVER (PARTITION BY label_text "
        "ORDER BY md5(text || 'banking77-dev')) <= $k ORDER BY label_text, text) "
        "TO $out (FORMAT parquet)",
        {
            "train": str(DATA / "banking77_train.jsonl"),
            "k": DEV_PER_INTENT,
            "out": str(data_file("dev")),
        },
    )
    n, k = con.execute(
        "SELECT count(*), count(DISTINCT label_text) FROM read_parquet(?)",
        [str(data_file("dev"))],
    ).fetchone()
    print(f"dev: {n} rows, {k} intents x {DEV_PER_INTENT} -> {_rel(data_file('dev'))}")
    return 0


def load(con: duckdb.DuckDBPyConnection, split: str, limit: int | None) -> int:
    if not data_file(split).exists():
        raise SystemExit("run `bench/banking77.py prepare` first")
    con.execute(LOAD_SQL[split], {"path": str(data_file(split))})
    if limit:
        con.execute(SAMPLE_SQL, {"n": limit})
    con.execute(ORDER_SQL)
    div_of = division_of()
    con.register(
        "_division_map", pa.table({"intent": list(div_of), "division": list(div_of.values())})
    )
    con.execute("CREATE OR REPLACE TABLE division_map AS SELECT * FROM _division_map")
    con.unregister("_division_map")
    con.execute(
        "CREATE OR REPLACE MACRO division_of(i) AS "
        "(SELECT division FROM division_map WHERE intent = i)"
    )
    return con.execute("SELECT count(*) FROM rows").fetchone()[0]


def split_size(split: str) -> int:
    con = duckdb.connect()
    load(con, split, None)
    return con.execute("SELECT count(*) FROM rows").fetchone()[0]


# --------------------------------------------------------------------------- live run


def fake_transport(split: str) -> httpx.MockTransport:
    """Dry-run stand-in: picks the gold option 80% of the time when it is offered."""
    rng = random.Random(7)
    con = duckdb.connect()
    load(con, split, None)
    labels = dict(con.execute("SELECT text, label_text FROM rows").fetchall())
    div_of = division_of()

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        gold = labels.get(body["state"])
        answers: dict[str, Any] = {}
        for qid, q in body["questions"].items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": rng.random()}
                continue
            opts = list(q["criteria"])
            want = div_of.get(gold) if qid == "division" else gold
            top = want if want in opts and rng.random() < 0.8 else rng.choice(opts)
            probs = {o: 0.2 / max(len(opts) - 1, 1) for o in opts}
            probs[top] = 0.8 if len(opts) > 1 else 1.0
            answers[qid] = {
                "type": "choice",
                "choice": top,
                "probabilities": probs,
                "confidence": rng.choice([0.6, 0.95]),
            }
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "answers": answers,
                "usage": {"input_tokens": 700 + 20 * len(body["questions"]), "output_tokens": 40},
            },
        )

    return httpx.MockTransport(handle)


def _row_dict(
    con: duckdb.DuckDBPyConnection, sql: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    cur = con.execute(sql, params or {})
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, cur.fetchone(), strict=True))


def _ece(rows: list[Any]) -> float | None:
    n = sum(r[1] for r in rows)
    return sum(r[1] * abs(r[3] - r[2]) for r in rows) / n if n else None


def metrics(con: duckdb.DuckDBPyConnection, rnd: Round) -> dict[str, Any]:
    """Every reported metric, from the ``scored`` table alone."""
    n, acc, greedy, strict, div_acc = con.execute(ACCURACY_SQL).fetchone()
    reliability = [
        list(r) for r in con.execute(RELIABILITY_SQL.format(conf="confidence")).fetchall()
    ]
    reliability_top = [
        list(r) for r in con.execute(RELIABILITY_SQL.format(conf="top_p")).fetchall()
    ]
    per_intent = [list(r) for r in con.execute(SOFT_VS_HARD_SQL).fetchall()]
    out: dict[str, Any] = {
        "rows": n,
        "accuracy": acc,
        "accuracy_se": math.sqrt(acc * (1 - acc) / n) if n else None,
        "greedy_accuracy": greedy,
        "strict_accuracy": strict,
        "division_accuracy": div_acc,
        "ece_confidence": _ece(reliability),
        "ece_top_p": _ece(reliability_top),
        "reliability": reliability,
        "reliability_top": reliability_top,
        "per_intent": per_intent,
        "sum_abs_hard_minus_true": sum(r[5] for r in per_intent),
        "sum_abs_expected_minus_true": sum(r[6] for r in per_intent),
        "intents_within_2se": sum(1 for r in per_intent if r[7]),
        "intents": len(per_intent),
        "confusions": [list(r) for r in con.execute(CONFUSIONS_SQL, {"k": 15}).fetchall()],
        "division_mass": _row_dict(con, DIVISION_MASS_SQL, {"thr": DEFER_BELOW}),
        "deferral": None,
        "topup": None,
    }
    if rnd.structure == "two_level":
        out["deferral"] = _row_dict(con, DEFERRAL_SQL)
    if rnd.topup != "none":
        s = con.execute(TOPUP_METRICS_SQL, {"intents": list(TOPUP_INTENTS)}).fetchone()
        out["topup"] = {
            "sem_where_count": s[0],
            "expected": s[1],
            "stderr": s[2],
            "true_count": s[3],
            "true_positives": s[4],
            "precision": s[4] / s[0] if s[0] else None,
            "recall": s[4] / s[3] if s[3] else None,
            "within_2se": abs(s[1] - s[3]) <= 2 * s[2],
        }
    return out


def _refuse(request: httpx.Request) -> httpx.Response:
    raise RuntimeError("rescore is served from the run's own cache; this request missed it")


def _preflighted(rnd: str) -> bool:
    runs = json.loads(RUNS_FILE.read_text()) if RUNS_FILE.exists() else {}
    return f"dev/{rnd}" in runs or any(DATA.glob(f"summary_dev_{rnd}_n*.json"))


def run(args: argparse.Namespace) -> int:
    rnd = ROUNDS[args.round]
    if not (args.limit or args.dry_run or _preflighted(args.round)):
        print(f"run `bench/banking77.py run {args.round} --split dev --limit 40` first")
        return 2
    DATA.mkdir(parents=True, exist_ok=True)
    tag = run_tag(args.split, args.round, args.limit, args.dry_run)
    cache_file = DATA / f"cache_{tag}.duckdb"
    if cache_file.exists():
        cache_file.unlink()  # a fresh cache per run, so the timed run pays for every row
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
    params = {"questions": json.dumps(questions_for(rnd))}

    duckjev.usage(reset=True)
    t0 = time.perf_counter()
    con.execute(JUDGE_SQL, params)
    secs = time.perf_counter() - t0
    use = duckjev.usage(reset=True)

    t0 = time.perf_counter()
    con.execute(JUDGE_RERUN_SQL, params)
    rerun_secs = time.perf_counter() - t0
    rerun = duckjev.usage(reset=True)
    identical = con.execute(IDENTICAL_SQL).fetchone()[0]

    topup_secs, topup_use = None, None
    if rnd.topup == "separate":
        t0 = time.perf_counter()
        con.execute(TOPUP_SQL, {"q": TOPUP_Q})
        topup_secs = time.perf_counter() - t0
        topup_use = duckjev.usage(reset=True)
    duckjev.flush(con)

    score(con, rnd)
    con.table("scored").write_parquet(str(DATA / f"scored_{tag}.parquet"))

    summary = {
        "round": args.round,
        "split": args.split,
        "limit": args.limit,
        "dry_run": args.dry_run,
        "round_config": asdict(rnd),
        "questions": len(questions_for(rnd)),
        "model": duckjev.client_for(con).model,
        "duckjev": duckjev.__version__,
        "duckdb": duckdb.__version__,
        "python": platform.python_version(),
        "concurrency": args.concurrency,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "seconds": secs,
        "usage": use,
        "rows_per_second": n / secs,
        "input_tokens_per_request": use["input_tokens"] / max(use["requests"], 1),
        "usd_per_1k_rows": use["est_usd"] / n * 1000,
        "rerun_seconds": rerun_secs,
        "rerun_usage": rerun,
        "rerun_identical_rows": identical,
        "topup_seconds": topup_secs,
        "topup_usage": topup_use,
        **metrics(con, rnd),
    }
    if args.limit:
        summary["projected_full_usd"] = use["est_usd"] / n * split_size(args.split)
    print(json.dumps(_brief(summary), indent=2))
    if args.limit is None and not args.dry_run:
        runs = json.loads(RUNS_FILE.read_text()) if RUNS_FILE.exists() else {}
        runs[f"{args.split}/{args.round}"] = summary
        RUNS_FILE.write_text(json.dumps(runs, indent=1, default=float) + "\n")
        print(f"recorded {args.split}/{args.round} in {_rel(RUNS_FILE)}")
    else:
        (DATA / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1, default=float))
    return 0


def _round3(d: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def _brief(s: dict[str, Any]) -> dict[str, Any]:
    r = {
        "run": f"{s['split']}/{s['round']}" + (f" (n={s['limit']})" if s["limit"] else ""),
        "rows": s["rows"],
        "seconds": round(s["seconds"], 1),
        "requests": s["usage"]["requests"],
        "tokens_per_request": round(s["input_tokens_per_request"]),
        "est_usd": round(s["usage"]["est_usd"], 4),
        "usd_per_1k_rows": round(s["usd_per_1k_rows"], 4),
        "429s": s["usage"]["rate_limited"],
        "accuracy": round(s["accuracy"], 4),
        "accuracy_se": round(s["accuracy_se"], 4),
        "strict_accuracy": round(s["strict_accuracy"], 4),
        "division_accuracy": round(s["division_accuracy"], 4),
        "ece_confidence": round(s["ece_confidence"], 4),
        "ece_top_p": round(s["ece_top_p"], 4),
        "sum_abs_hard": s["sum_abs_hard_minus_true"],
        "sum_abs_expected": round(s["sum_abs_expected_minus_true"], 1),
        "within_2se": f"{s['intents_within_2se']} / {s['intents']}",
        "rerun": {
            "seconds": round(s["rerun_seconds"], 2),
            "requests": s["rerun_usage"]["requests"],
        },
        "top_confusions": s["confusions"][:5],
    }
    r["division_mass"] = _round3(s["division_mass"])
    if s["deferral"]:
        r["deferral"] = _round3(s["deferral"])
    if s["topup"]:
        r["topup"] = _round3(s["topup"])
    if s["topup_usage"]:
        u = s["topup_usage"]
        r["topup_separate"] = {
            "requests": u["requests"],
            "tokens_per_request": round(u["input_tokens"] / max(u["requests"], 1)),
            "est_usd": round(u["est_usd"], 4),
        }
    if "projected_full_usd" in s:
        r["projected_full_usd"] = round(s["projected_full_usd"], 4)
    return r


def rescore(args: argparse.Namespace) -> int:
    """Rebuild every recorded run's rows and metrics from that run's Jev answer cache.

    The cache is the durable record of what Jev answered. The transport refuses every
    request, so a rescore can neither spend nor read anything but the run's own answers;
    timing and usage in the run log are kept from the live run.
    """
    runs = json.loads(RUNS_FILE.read_text())
    for key, summary in runs.items():
        if summary.get("legacy"):
            continue  # imported from an earlier headline file; no cache to rebuild from
        tag = run_tag(summary["split"], summary["round"])
        cache = DATA / f"cache_{tag}.duckdb"
        if not cache.exists():
            print(f"skip {key}: {_rel(cache)} is missing")
            continue
        rnd = ROUNDS[summary["round"]]
        con = duckdb.connect()
        duckjev.register(
            con, cache_path=cache, api_key="cache-only", transport=httpx.MockTransport(_refuse)
        )
        load(con, summary["split"], None)
        duckjev.usage(reset=True)
        con.execute(JUDGE_SQL, {"questions": json.dumps(questions_for(rnd))})
        if rnd.topup == "separate":
            con.execute(TOPUP_SQL, {"q": TOPUP_Q})
        assert duckjev.usage()["requests"] == 0
        score(con, rnd)
        con.table("scored").write_parquet(str(DATA / f"scored_{tag}.parquet"))
        before = summary["accuracy"]
        summary.update(metrics(con, rnd))
        print(f"rescored {key}: accuracy {before:.4f} -> {summary['accuracy']:.4f}")
    RUNS_FILE.write_text(json.dumps(runs, indent=1, default=float) + "\n")
    return 0


def confusions(args: argparse.Namespace) -> int:
    """Top confusions of a run from its per-row file, with the messages behind the top pairs."""
    path = DATA / f"scored_{run_tag(args.split, args.round, args.limit, args.dry_run)}.parquet"
    con = duckdb.connect()
    con.execute("CREATE TABLE scored AS SELECT * FROM read_parquet(?)", [str(path)])
    rows = con.execute(CONFUSIONS_SQL, {"k": args.top}).fetchall()
    total = con.execute("SELECT count(*) FILTER (choice <> label_text) FROM scored").fetchone()[0]
    print(f"{_rel(path)}: {total} errors; top {args.top} pairs")
    for gold, pred, n in rows:
        print(f"  {n:3d}  {gold}  ->  {pred}")
    if args.show:
        for gold, pred, _ in rows[: args.show]:
            print(f"\n== {gold} -> {pred}")
            for text, p in con.execute(
                "SELECT text, top_p FROM scored WHERE label_text = ? AND choice = ? "
                "ORDER BY top_p DESC",
                [gold, pred],
            ).fetchall():
                print(f"  {p:.2f}  {text}")
    return 0


# --------------------------------------------------------------------------- report


def chosen_round(runs: dict[str, Any]) -> str:
    """The selectable round with the best dev accuracy; ties go to fewer tokens per request."""
    dev = [
        (r["accuracy"], -r["input_tokens_per_request"], r["round"])
        for k, r in runs.items()
        if k.startswith("dev/") and ROUNDS[r["round"]].selectable
    ]
    return max(dev)[2]


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:,.1f}"
    return str(v)


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _sql(s: str) -> str:
    return f"```sql\n{s}\n```"


ROUNDS_HEADER = [
    "run",
    "accuracy",
    "± SE",
    "ECE conf",
    "ECE top-1",
    "Σ\\|hard−true\\|",
    "Σ\\|exp−true\\|",
    "within 2 SE",
    "tokens / req",
    "$ / 1k rows",
    "rows / s",
]


def _round_row(r: dict[str, Any]) -> list[Any]:
    return [
        f"{r['split']}/{r['round']}",
        f"**{r['accuracy']:.3f}**",
        f"{r['accuracy_se']:.3f}",
        r["ece_confidence"],
        r["ece_top_p"],
        int(r["sum_abs_hard_minus_true"]),
        f"{r['sum_abs_expected_minus_true']:.1f}",
        f"{r['intents_within_2se']} / {r['intents']}",
        f"{r['input_tokens_per_request']:,.0f}",
        f"${r['usd_per_1k_rows']:.4f}",
        f"{r['rows_per_second']:.0f}",
    ]


def _rounds_table(runs: list[dict[str, Any]]) -> str:
    return _table(ROUNDS_HEADER, [_round_row(r) for r in runs])


def _reliability_table(rows: list[list[Any]], label: str) -> str:
    return _table(
        ["bin", "n", f"mean {label}", "accuracy"],
        [[f"{b / 10:.1f}–{(b + 1) / 10:.1f}", n, mc, acc] for b, n, mc, acc in rows],
    )


def _unfused_sibling(r: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The run on the same split with the same intent questions and no fused top-up."""
    keys = ("criteria", "structure", "order")
    for other in runs:
        same = all(other["round_config"][k] == r["round_config"][k] for k in keys)
        if same and other["split"] == r["split"] and other["round_config"]["topup"] != "fused":
            return other
    return None


def _topup_table(runs: list[dict[str, Any]]) -> str:
    rows = []
    for r in runs:
        t = r["topup"]
        if t is None:
            continue
        if r["round_config"]["topup"] == "separate":
            u = r["topup_usage"]
            how = "separate `jev_noul` query"
            cost = (
                f"{u['input_tokens'] / max(u['requests'], 1):,.0f} tokens / req, "
                f"${u['est_usd'] / r['rows'] * 1000:.4f} / 1k rows on top"
            )
        else:
            how = "fused into the intent request"
            sib = _unfused_sibling(r, runs)
            cost = "one more question in the same request"
            if sib:
                extra = r["input_tokens_per_request"] - sib["input_tokens_per_request"]
                cost = (
                    f"{extra:,.0f} tokens / req more than {sib['split']}/{sib['round']}, "
                    f"${extra * USD_PER_INPUT_TOKEN * 1000:.4f} / 1k rows on top"
                )
        rows.append(
            [
                f"{r['split']}/{r['round']}",
                how,
                t["sem_where_count"],
                t["expected"],
                t["stderr"],
                t["true_count"],
                t["precision"],
                t["recall"],
                t["within_2se"],
                cost,
            ]
        )
    return _table(
        [
            "run",
            "how asked",
            "sem_where ≥ 0.5",
            "expected",
            "stderr",
            "true",
            "precision",
            "recall",
            "within 2 SE",
            "cost of the top-up question",
        ],
        rows,
    )


def _confusion_pair_table(a: dict[str, Any], b: dict[str, Any], k: int = 10) -> str:
    rows_a, rows_b = a["confusions"][:k], b["confusions"][:k]
    rows = []
    for i in range(max(len(rows_a), len(rows_b))):
        ra = rows_a[i] if i < len(rows_a) else ["", "", ""]
        rb = rows_b[i] if i < len(rows_b) else ["", "", ""]
        rows.append(
            [
                f"{ra[0]} → {ra[1]}" if ra[0] else "",
                ra[2],
                f"{rb[0]} → {rb[1]}" if rb[0] else "",
                rb[2],
            ]
        )
    return _table(
        [f"{a['round']}: gold → predicted", "n", f"{b['round']}: gold → predicted", "n"], rows
    )


def _division_mass_table(runs: list[dict[str, Any]]) -> str:
    return _table(
        [
            "run",
            "division accuracy (mass)",
            f"deferred (mass < {DEFER_BELOW})",
            "intent precision when answered",
            "division right when deferred",
            "intent argmax right when deferred",
        ],
        [
            [
                f"{r['split']}/{r['round']}",
                r["division_mass"]["division_accuracy"],
                f"{r['division_mass']['deferred']:.1%}",
                r["division_mass"]["precision_when_answered"],
                r["division_mass"]["division_accuracy_when_deferred"],
                r["division_mass"]["leaf_accuracy_when_deferred"],
            ]
            for r in runs
        ],
    )


def _deferral_table(runs: list[dict[str, Any]]) -> str:
    return _table(
        [
            "run",
            "division accuracy",
            "greedy leaf accuracy",
            "product-argmax accuracy",
            f"deferred (division confidence < {DEFER_BELOW})",
            "intent precision when answered",
            "division right when deferred",
            "strict accuracy",
        ],
        [
            [
                f"{r['split']}/{r['round']}",
                r["division_accuracy"],
                r["greedy_accuracy"],
                r["accuracy"],
                f"{r['deferral']['deferred']:.1%}",
                r["deferral"]["precision_when_answered"],
                r["deferral"]["division_accuracy_when_deferred"],
                r["strict_accuracy"],
            ]
            for r in runs
        ],
    )


def _headline_table(final: dict[str, Any]) -> str:
    u = final["usage"]
    return _table(
        ["metric", "value"],
        [
            ["rows", final["rows"]],
            [
                "throughput",
                f"{final['rows_per_second']:.1f} rows/s ({final['rows']} rows in "
                f"{final['seconds']:.1f} s; {u['rate_limited']} × 429, {u['overloaded']} × 529)",
            ],
            [
                "cost",
                f"${final['usd_per_1k_rows']:.4f} per 1,000 rows (${u['est_usd']:.4f} total; "
                f"{final['input_tokens_per_request']:,.0f} input tokens per request, "
                f"{final['questions']} question{'s' if final['questions'] > 1 else ''})",
            ],
            [
                "accuracy (argmax = gold)",
                f"{final['accuracy']:.4f} ± {final['accuracy_se']:.4f}",
            ],
            ["division accuracy", f"{final['division_accuracy']:.4f}"],
            [
                "ECE, 10 bins",
                f"{final['ece_confidence']:.4f} over `confidence`, "
                f"{final['ece_top_p']:.4f} over top-1 probability",
            ],
            ["Σ over intents of abs(hard − true)", int(final["sum_abs_hard_minus_true"])],
            [
                "Σ over intents of abs(expected − true)",
                f"{final['sum_abs_expected_minus_true']:.1f}",
            ],
            [
                "intents with abs(expected − true) ≤ 2·SE",
                f"{final['intents_within_2se']} / {final['intents']}",
            ],
            [
                "cache re-run",
                f"{final['rerun_seconds']:.2f} s, {final['rerun_usage']['requests']} requests, "
                f"${final['rerun_usage']['est_usd']:.4f}; "
                f"{final['rerun_identical_rows']} / {final['rows']} rows identical",
            ],
        ],
    )


def report(args: argparse.Namespace) -> int:
    runs = json.loads(RUNS_FILE.read_text())
    best = chosen_round(runs)
    dev = [runs[k] for k in sorted(runs) if k.startswith("dev/")]
    held = [runs[k] for k in sorted(runs) if k.startswith("test/")]
    legacy = [r for r in runs.values() if r.get("legacy")]  # the PR #1 run, imported
    final = runs.get(f"test/{best}")
    base = runs.get("test/R0")
    n_dev = dev[0]["rows"] if dev else 0
    parts = [
        "# Banking77 benchmark, tuned in rounds",
        "",
        "Generated by `bench/banking77.py report` from `docs/results/banking77_runs.json`; do not "
        "edit by hand. Every number below is from a live run against `jev-1.13.0` at client "
        f"concurrency {dev[0]['concurrency'] if dev else 16}. The `PR #1 test/R0` row is the "
        "earlier held-out run, imported into the run log from its headline file; its reading "
        "is in `docs/NEXT.md` §2.1.",
        "",
        "Corpus: Banking77 (Casanueva et al., 2020), 77 customer-support intents, from "
        "`mteb/banking77` on Hugging Face. The dev split is a stratified sample of the train "
        f"split, {DEV_PER_INTENT} messages per intent ({n_dev} rows), drawn by a fixed rule in "
        "`prepare`. The test split (3,080 rows) is held out and was run only with the baseline "
        "and the round chosen on dev. Glosses, examples and the division map come from the label "
        "names and the train split only. The chosen round is the selectable round with the best "
        f"dev accuracy, ties to fewer tokens per request: **{best}**.",
        "",
        "How it works: every round is one fused `jev()` request per message. A flat round asks "
        "one Choice over all 77 intents. A two-level round asks a division Choice and one intent "
        "Choice per division in the same request, speculatively; the intent distribution is "
        "p(division) × p(intent | division) over every branch, `choice` is its argmax, and a row "
        f"whose division confidence is under {DEFER_BELOW} reports the division instead of an "
        "intent. *Accuracy* is argmax = gold over all rows, the same quantity in every round; "
        "*strict* accuracy counts a deferred row as wrong. ECE is over 10 equal-width bins, "
        "once over Jev's `confidence` (for a two-level row, the confidence of the intent Choice "
        "that produced the answer) and once over the top-1 probability.",
        "",
    ]
    if final:
        parts += [f"## Held-out test split, round {best}", "", _headline_table(final), ""]
        if final["deferral"]:
            d = final["deferral"]
            parts += [
                f"Deferral: {d['deferred']:.1%} of rows had division confidence under "
                f"{DEFER_BELOW} and reported the division. Intent precision on the answered rows "
                f"{d['precision_when_answered']:.4f}; on the deferred rows the reported division "
                f"was right {_fmt(d['division_accuracy_when_deferred'])} of the time and the "
                f"leaf argmax would have been right {_fmt(d['leaf_accuracy_when_deferred'])}. "
                f"Strict accuracy (deferred = wrong) {final['strict_accuracy']:.4f}.",
                "",
            ]
        parts += [
            "### Held-out runs: the baseline and the chosen round",
            "",
            _rounds_table(legacy + held),
            "",
        ]
        if base and best != "R0":
            parts += [
                "Top confusions on the held-out split, baseline against the chosen round "
                "(gold → predicted, count):",
                "",
                _confusion_pair_table(base, final),
                "",
            ]
    parts += [
        "## Tuning rounds on the dev split",
        "",
        f"{n_dev} rows, {DEV_PER_INTENT} per intent. One input changes per round.",
        "",
        _rounds_table(dev),
        "",
        *[f"- **{r['round']}**: {r['round_config']['note']}." for r in dev],
        "",
    ]
    two_level = [r for r in dev + held if r["deferral"]]
    if two_level:
        parts += [
            "### Two-level rounds: deferral to the division",
            "",
            _deferral_table(two_level),
            "",
        ]
    parts += [
        "### Deferring to a division from the intent distribution alone",
        "",
        "For every round, the division of a row is the one holding the most probability mass "
        "(summed over its intents) in that row's intent distribution, and a row defers when that "
        f"mass is under {DEFER_BELOW}. A flat round gets this readout from its one Choice, with "
        "no second question.",
        "",
        _division_mass_table(dev + held),
        "",
    ]
    if "dev/R0" in runs and best != "R0":
        parts += [
            "### Top confusions on dev, baseline against the chosen round",
            "",
            _confusion_pair_table(runs["dev/R0"], runs[f"dev/{best}"]),
            "",
        ]
    if final:
        parts += [
            f"## Calibration on the held-out split, round {best}",
            "",
            f"ECE over `confidence`: **{final['ece_confidence']:.4f}**.",
            "",
            _reliability_table(final["reliability"], "confidence"),
            "",
            f"ECE over the top-1 probability: **{final['ece_top_p']:.4f}**.",
            "",
            _reliability_table(final["reliability_top"], "top-1 p"),
            "",
            f"## Soft vs hard group-by on the held-out split, round {best}",
            "",
            "Per intent: the gold count, the hard count (rows whose argmax is the intent), the "
            "expected count Σp (each row's probability mass on the intent, summed), and its "
            "standard error √Σp(1−p) under the tuple-independent Bernoulli model.",
            "",
            _sql(SOFT_VS_HARD_SQL),
            "",
            f"Totals across {final['intents']} intents: Σ abs(hard − true) = "
            f"**{int(final['sum_abs_hard_minus_true'])}**, Σ abs(expected − true) = "
            f"**{final['sum_abs_expected_minus_true']:.1f}**; abs(expected − true) ≤ 2·SE for "
            f"**{final['intents_within_2se']} / {final['intents']}** intents.",
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
                final["per_intent"],
            ),
            "",
        ]
    parts += [
        "## `sem_where` and calibrated counting: the top-up question",
        "",
        f"Noul question `{TOPUP_Q}`. Ground truth is the {len(TOPUP_INTENTS)} intents whose label "
        "is about top-ups: " + ", ".join(f"`{i}`" for i in TOPUP_INTENTS) + ". The baseline asks "
        "it as its own query; a fused round adds it to the intent request, where it costs only "
        "its own tokens.",
        "",
        _topup_table(dev + held),
        "",
        "## The SQL and the questions",
        "",
        "One fused request per message, `$questions` bound as a prepared-statement parameter:",
        "",
        _sql(JUDGE_SQL),
        "",
        "The baseline's separate top-up query:",
        "",
        _sql(TOPUP_SQL),
        "",
        f"Flat rounds ask `{INSTR}` with the 77 glosses of the round's gloss set as options. "
        f"Two-level rounds ask `{DIVISION_INSTR}` over the divisions in "
        "`bench/banking77_divisions.json`, and for each division `"
        + LEAF_INSTR.format(label="<the division's label>")
        + "` over that division's intents. Gloss sets: `v1` is `bench/banking77_criteria.json`; "
        "`v2` overlays `bench/banking77_criteria_v2.json` (structured what / not_for / examples "
        "for the confusable intents) on v1; `short` is `bench/banking77_criteria_short.json`.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python bench/banking77.py prepare",
        *[
            f"uv run python bench/banking77.py run {r['round']} --split {r['split']}"
            for r in dev + held
        ],
        "uv run python bench/banking77.py report",
        "```",
        "",
    ]
    RESULTS.write_text("\n".join(parts))
    print(f"wrote {_rel(RESULTS)} (chosen round {best})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("round", choices=list(ROUNDS))
    r.add_argument("--split", default="dev", choices=SPLITS)
    r.add_argument("--limit", type=int, default=None, help="pre-flight sample size")
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--max-usd", type=float, default=0.30, help="hard budget for this run")
    r.add_argument("--dry-run", action="store_true", help="fake transport, no network")
    c = sub.add_parser("confusions")
    c.add_argument("round", choices=list(ROUNDS))
    c.add_argument("--split", default="dev", choices=SPLITS)
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--top", type=int, default=15)
    c.add_argument("--show", type=int, default=0, help="print the messages behind the top N pairs")
    sub.add_parser("rescore", help="rebuild recorded runs from their answer caches; no requests")
    sub.add_parser("report")
    args = p.parse_args(argv)
    commands = {
        "prepare": lambda a: prepare(),
        "run": run,
        "confusions": confusions,
        "rescore": rescore,
        "report": report,
    }
    return commands[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
