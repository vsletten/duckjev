"""Live entity-matching benchmark for jev_match / sem_join (workstation only; needs the key).

    uv run python bench/entity_matching.py prepare                            # downloads, no key
    uv run python bench/entity_matching.py coverage                           # blocking, offline
    uv run python bench/entity_matching.py run R0 --corpus abt --split dev --limit 40  # pre-flight
    uv run python bench/entity_matching.py run R0 --corpus abt --split dev    # one round on dev
    uv run python bench/entity_matching.py run R1 --corpus abt --split test   # the held-out split
    uv run python bench/entity_matching.py demo --corpus abt                  # the macros, live
    uv run python bench/entity_matching.py rescore                            # from caches, free
    uv run python bench/entity_matching.py report                             # docs/results/*.md
    uv run python bench/entity_matching.py run R0 --corpus abt --split dev --dry-run

Two corpora from the DeepMatcher / Magellan collection: Abt-Buy (textual product listings
from two retailers) and DBLP-ACM (structured bibliographic records). Each ships two record
tables and labeled candidate pairs already blocked by its authors, split train / valid /
test. The valid split is the dev set for the tuning rounds; the test split is held out and
runs only with the baseline and the round chosen on dev. Every pair is one ``jev_noul`` over
the pair state, the primitive ``jev_match``, ``sem_join`` and ``sem_dedup`` are built on;
the round decides the question wording, the pair-state layout and which record comes first.
``demo`` runs the table macros themselves, live, on a sample of Abt-Buy.

Every full live run appends its metrics to ``docs/results/entity_matching_runs.json``, and
``report`` renders ``docs/results/entity_matching.md`` from that file alone. Pre-flight
samples and dry runs write their own suffixed files under ``bench/data/`` and record nothing.
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

import duckjev
from duckjev.client import USD_PER_INPUT_TOKEN

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bench" / "data"
RUNS_FILE = ROOT / "docs" / "results" / "entity_matching_runs.json"
RESULTS = ROOT / "docs" / "results" / "entity_matching.md"
SOURCE_URL = "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/{path}/exp_data/{file}"
FILES = ("tableA.csv", "tableB.csv", "train.csv", "valid.csv", "test.csv")
SPLITS = ("dev", "test")
SPLIT_FILE = {"dev": "valid", "test": "test"}
THRESHOLD = 0.5
DEMO_ROWS = 100
DEMO_RIGHT_PER_BLOCK = 5

# --------------------------------------------------------------------------- corpora


@dataclass(frozen=True)
class Corpus:
    name: str
    path: str  # under the DeepMatcher data root
    entity: str
    fields: tuple[str, ...]
    block: str  # SQL over the record columns: the blocking key of sem_join / sem_dedup
    plain: str
    colleague: str
    criteria: dict[str, str]  # the colleague guidance as Noul true / false criteria
    topk_instructions: str
    topk_levels: tuple[str, ...]


CORPORA: dict[str, Corpus] = {
    "abt": Corpus(
        name="Abt-Buy",
        path="Textual/Abt-Buy",
        entity="product",
        fields=("name", "description", "price"),
        block="lower(split_part(name, ' ', 1))",
        plain="Do these two product listings describe the same product?",
        colleague=(
            "Do these two listings, from two different retailers, describe the same product: "
            "the same model, not merely the same brand or product line? Different wording, a "
            "missing description or price, and a different price still mean the same product "
            "when the model number or model name matches. A different model number, a different "
            "size, colour or capacity variant, or an accessory, part or bundle for the product "
            "is a different product."
        ),
        criteria={
            "true": (
                "The same product: the model number or model name matches, allowing different "
                "wording, a missing description or price, and a different price"
            ),
            "false": (
                "Different products: a different model number, a size, colour or capacity "
                "variant, an accessory, part or bundle, or merely the same brand or product line"
            ),
        },
        topk_instructions="How premium is this product?",
        topk_levels=("budget", "mid-range", "premium"),
    ),
    "dblp": Corpus(
        name="DBLP-ACM",
        path="Structured/DBLP-ACM",
        entity="publication",
        fields=("title", "authors", "venue", "year"),
        block="year",
        plain="Do these two records describe the same publication?",
        colleague=(
            "Do these two bibliographic records describe the same publication: the same paper "
            "with the same title and the same authors? Differences in capitalisation, "
            "punctuation, abbreviation, author name formatting or author order, and a venue "
            "written differently (a conference against its proceedings, a journal against its "
            "abbreviation) still mean the same publication. A different paper with a similar "
            "title, a different author list, or a different year is a different publication."
        ),
        criteria={
            "true": (
                "The same publication: the same title and authors, allowing differences in "
                "capitalisation, punctuation, abbreviation, author formatting or order, and in "
                "how the venue is written"
            ),
            "false": (
                "Different publications: a different paper with a similar title, a different "
                "author list, or a different year"
            ),
        },
        topk_instructions="How broad is the audience of this paper?",
        topk_levels=("narrow specialists", "a subfield", "the whole field"),
    ),
}


@dataclass(frozen=True)
class Round:
    note: str
    wording: str  # plain | colleague
    layout: str  # json (jev_pair) | text (labeled lines)
    order: str  # ab | ba: which record is first in the state
    criteria: bool = False  # the guidance as Noul true / false criteria instead of instructions
    selectable: bool = True


ROUNDS: dict[str, Round] = {
    "R0": Round("baseline: the plain question over the JSON pair state", "plain", "json", "ab"),
    "R1": Round(
        "colleague-style question that says what counts as the same and what does not",
        "colleague",
        "json",
        "ab",
    ),
    "R2": Round(
        "R1 with the pair as labeled text lines instead of a JSON object",
        "colleague",
        "text",
        "ab",
    ),
    "R3": Round(
        "R2 with the records swapped, B first: the order check on the best round",
        "colleague",
        "text",
        "ba",
    ),
    "R4": Round(
        "R1 with the guidance moved from the question into the Noul's true / false criteria",
        "colleague",
        "json",
        "ab",
        criteria=True,
    ),
}

# --------------------------------------------------------------------------- SQL


def rec_sql(c: Corpus) -> str:
    return "struct_pack(" + ", ".join(f"{f} := {f}" for f in c.fields) + ")"


def rec_text_sql(c: Corpus) -> str:
    """Bench macro body: 'field: value; ...' with null fields left out."""
    parts = ", ".join(f"'{f}: ' || rec.{f}" for f in c.fields)
    return f"CREATE OR REPLACE MACRO rec_text(rec) AS concat_ws('; ', {parts})"


LOAD_RECORDS_SQL = """CREATE OR REPLACE TABLE {t} AS
SELECT id, {rec} AS rec, {block} AS blk
FROM read_csv($path, header = true, all_varchar = true)"""

LOAD_PAIRS_SQL = """CREATE OR REPLACE TABLE pairs AS
SELECT p.ltable_id::VARCHAR AS lid, p.rtable_id::VARCHAR AS rid, p.label::INT AS label,
       a.rec AS a_rec, b.rec AS b_rec
FROM read_csv($path, header = true) p
JOIN recs_a a ON a.id = p.ltable_id::VARCHAR
JOIN recs_b b ON b.id = p.rtable_id::VARCHAR
ORDER BY lid, rid"""

SAMPLE_SQL = """CREATE OR REPLACE TABLE pairs AS
SELECT * FROM (SELECT * FROM pairs ORDER BY md5(lid || '-' || rid || 'em-sample') LIMIT $n)
ORDER BY lid, rid"""

STATE_SQL = {
    ("json", "ab"): "jev_pair(a_rec, b_rec)",
    ("json", "ba"): "jev_pair(b_rec, a_rec)",
    ("text", "ab"): "'A: ' || rec_text(a_rec) || chr(10) || 'B: ' || rec_text(b_rec)",
    ("text", "ba"): "'A: ' || rec_text(b_rec) || chr(10) || 'B: ' || rec_text(a_rec)",
}

# One Noul per pair. {state} is one of STATE_SQL; {noul} is jev_noul(state, $q) or, when
# the round moves the guidance into criteria, jev_noul(state, $q, $criteria).
JUDGE_SQL = """CREATE OR REPLACE TABLE {table} AS
SELECT lid, rid, label, {state} AS state, {noul} AS p FROM pairs"""
IDENTICAL_SQL = """SELECT count(*) FROM judged j JOIN judged_rerun r USING (lid, rid)
WHERE j.p IS NOT DISTINCT FROM r.p"""

COUNTS_SQL = """SELECT count(*) AS n, sum(label) AS positives,
  count(*) FILTER (p >= $t) AS predicted,
  sum(label) FILTER (p >= $t) AS true_positives,
  expected_count(p) AS expected, expected_count_stderr(p) AS stderr
FROM judged"""

RELIABILITY_SQL = """WITH b AS (SELECT least(floor(p * 10), 9)::INT AS bin, p, label FROM judged)
SELECT bin, count(*) AS n, avg(p) AS mean_p, avg(label) AS positive_rate
FROM b GROUP BY bin ORDER BY bin"""

EXAMPLES_SQL = """SELECT lid, rid, label, p, state FROM judged
WHERE label = $label ORDER BY p {dir}, lid, rid LIMIT $k"""

# Every SQL variant is built here, at import, from the corpus and round constants; the
# execute() calls below take only these constants and bound parameters.
LOAD_RECORDS_SQLS: dict[tuple[str, str], str] = {
    (key, t): LOAD_RECORDS_SQL.format(t=t, rec=rec_sql(c), block=c.block)
    for key, c in CORPORA.items()
    for t in ("recs_a", "recs_b")
}
REC_TEXT_SQLS: dict[str, str] = {key: rec_text_sql(c) for key, c in CORPORA.items()}
JUDGE_SQLS: dict[tuple[str, str, bool, str], str] = {
    (layout, order, crit, table): JUDGE_SQL.format(
        table=table,
        state=state,
        noul=f"jev_noul({state}, $q, $criteria)" if crit else f"jev_noul({state}, $q)",
    )
    for (layout, order), state in STATE_SQL.items()
    for crit in (False, True)
    for table in ("judged", "judged_rerun")
}
EXAMPLES_SQLS: dict[str, str] = {d: EXAMPLES_SQL.format(dir=d) for d in ("ASC", "DESC")}

COVERAGE_SQL = """WITH gold AS (
  SELECT DISTINCT ltable_id::VARCHAR AS lid, rtable_id::VARCHAR AS rid
  FROM read_csv($pairs, header = true, union_by_name = true) WHERE label = 1),
blocked AS (
  SELECT a.id AS lid, b.id AS rid FROM recs_a a JOIN recs_b b USING (blk))
SELECT (SELECT count(*) FROM recs_a) AS left_rows, (SELECT count(*) FROM recs_b) AS right_rows,
       (SELECT count(*) FROM blocked) AS blocked_pairs,
       (SELECT count(*) FROM gold) AS gold_pairs,
       (SELECT count(*) FROM gold JOIN blocked USING (lid, rid)) AS gold_pairs_blocked"""

# --------------------------------------------------------------------------- data


def data_file(corpus: str, file: str) -> Path:
    return DATA / f"{corpus}_{file}"


def run_tag(
    corpus: str, split: str, rnd: str, limit: int | None = None, dry_run: bool = False
) -> str:
    """Names a run's cache and per-row files; only full live runs get the bare tag."""
    return f"{corpus}_{split}_{rnd}" + (f"_n{limit}" if limit else "") + ("_dry" if dry_run else "")


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def prepare() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    for key, c in CORPORA.items():
        for file in FILES:
            path = data_file(key, file)
            if not path.exists():
                url = SOURCE_URL.format(path=c.path, file=file)
                resp = httpx.get(url, follow_redirects=True, timeout=60)
                resp.raise_for_status()
                path.write_bytes(resp.content)
            print(f"{key}/{file}: {sum(1 for _ in path.open()) - 1} rows -> {_rel(path)}")
    return 0


def load_records(con: duckdb.DuckDBPyConnection, corpus: str) -> None:
    for t, file in (("recs_a", "tableA.csv"), ("recs_b", "tableB.csv")):
        if not data_file(corpus, file).exists():
            raise SystemExit("run `bench/entity_matching.py prepare` first")
        con.execute(LOAD_RECORDS_SQLS[(corpus, t)], {"path": str(data_file(corpus, file))})
    con.execute(REC_TEXT_SQLS[corpus])


def load(con: duckdb.DuckDBPyConnection, corpus: str, split: str, limit: int | None) -> int:
    load_records(con, corpus)
    con.execute(LOAD_PAIRS_SQL, {"path": str(data_file(corpus, f"{SPLIT_FILE[split]}.csv"))})
    if limit:
        con.execute(SAMPLE_SQL, {"n": limit})
    return con.execute("SELECT count(*) FROM pairs").fetchone()[0]


def split_size(corpus: str, split: str) -> int:
    con = duckdb.connect()
    return load(con, corpus, split, None)


def judge_sql(rnd: Round, table: str) -> str:
    """The judged SQL of a round, one of the constants built at import."""
    return JUDGE_SQLS[(rnd.layout, rnd.order, rnd.criteria, table)]


def judge_params(corpus: str, rnd: Round) -> dict[str, Any]:
    """The bound parameters of a round's judged SQL: the question, and criteria if asked."""
    c = CORPORA[corpus]
    params: dict[str, Any] = {"q": c.colleague if rnd.wording == "colleague" else c.plain}
    if rnd.criteria:
        params["criteria"] = json.dumps(c.criteria)
    return params


# --------------------------------------------------------------------------- coverage


def coverage(args: argparse.Namespace) -> int:
    """How much of the labeled gold a key-equality block keeps, and how many pairs it makes."""
    rows = []
    for key in CORPORA:
        con = duckdb.connect()
        load_records(con, key)
        r = con.execute(
            COVERAGE_SQL,
            {"pairs": [str(data_file(key, f)) for f in ("train.csv", "valid.csv", "test.csv")]},
        ).fetchone()
        left, right, blocked, gold, gold_blocked = r
        rows.append(
            {
                "corpus": key,
                "block": CORPORA[key].block,
                "left_rows": left,
                "right_rows": right,
                "cross_pairs": left * right,
                "blocked_pairs": blocked,
                "gold_pairs": gold,
                "gold_pairs_blocked": gold_blocked,
                "gold_recall": gold_blocked / gold if gold else None,
            }
        )
        print(
            f"{key}: block {CORPORA[key].block!r} keeps {blocked:,} of {left * right:,} pairs "
            f"({blocked / (left * right):.2%}) and {gold_blocked} of {gold} gold matches "
            f"({gold_blocked / gold:.1%})"
        )
    if args.record:
        runs = json.loads(RUNS_FILE.read_text()) if RUNS_FILE.exists() else {}
        runs["coverage"] = {"legacy": True, "rows": rows}
        RUNS_FILE.write_text(json.dumps(runs, indent=1, default=float) + "\n")
        print(f"recorded coverage in {_rel(RUNS_FILE)}")
    return 0


# --------------------------------------------------------------------------- live run


def fake_transport() -> httpx.MockTransport:
    """Dry-run stand-in: a random probability per request, deterministic per process."""
    rng = random.Random(7)

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": round(rng.random(), 3)}
            else:
                n = len(q["criteria"])
                probs = [rng.random() for _ in range(n)]
                s = sum(probs)
                probs = [x / s for x in probs]
                answers[qid] = {
                    "type": "score",
                    "score": sum(i * p for i, p in enumerate(probs)),
                    "legend": {str(i): lv for i, lv in enumerate(q["criteria"])},
                    "probabilities": {str(i): p for i, p in enumerate(probs)},
                    "confidence": max(probs),
                }
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "answers": answers,
                "usage": {"input_tokens": 200 + len(body["state"]) // 4, "output_tokens": 10},
            },
        )

    return httpx.MockTransport(handle)


def _prf(predicted: int, true_positives: int, positives: int) -> dict[str, float | None]:
    precision = true_positives / predicted if predicted else None
    recall = true_positives / positives if positives else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
    )
    return {
        "predicted": predicted,
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _auroc(scores: list[tuple[float, int]]) -> float | None:
    """Rank-based AUROC with ties averaged."""
    pos = sum(1 for _, y in scores if y == 1)
    neg = len(scores) - pos
    if not pos or not neg:
        return None
    ranked = sorted(scores, key=lambda t: t[0])
    ranks: list[float] = [0.0] * len(ranked)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j + 2) / 2  # 1-based average rank of the tie
        i = j + 1
    rank_sum = sum(r for r, (_, y) in zip(ranks, ranked, strict=True) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def _two_sides(state: str, width: int = 110) -> tuple[str, str]:
    """Both records of a pair state, each cut to ``width`` characters, for the examples."""
    if state.startswith("A: ") and "\nB: " in state:
        a, b = state[3:].split("\nB: ", 1)
    else:
        obj = json.loads(state)
        a, b = json.dumps(obj["a"]), json.dumps(obj["b"])
    return a[:width], b[:width]


def metrics(con: duckdb.DuckDBPyConnection, threshold: float = THRESHOLD) -> dict[str, Any]:
    """Every reported metric, from the ``judged`` table alone."""
    n, positives, predicted, tp, expected, stderr = con.execute(
        COUNTS_SQL, {"t": threshold}
    ).fetchone()
    scores = con.execute("SELECT p, label FROM judged").fetchall()
    sweep = []
    for t in [i / 20 for i in range(1, 20)]:
        pred = sum(1 for p, _ in scores if p >= t)
        tpos = sum(1 for p, y in scores if p >= t and y == 1)
        sweep.append({"threshold": t, **_prf(pred, tpos, positives)})
    best = max(sweep, key=lambda r: (r["f1"], -abs(r["threshold"] - 0.5)))
    reliability = [list(r) for r in con.execute(RELIABILITY_SQL).fetchall()]
    ece = sum(r[1] * abs(r[3] - r[2]) for r in reliability) / n if n else None

    def examples(label: int, direction: str) -> list[list[Any]]:
        rows = con.execute(EXAMPLES_SQLS[direction], {"label": label, "k": 5}).fetchall()
        return [[lid, rid, y, p, *_two_sides(s)] for lid, rid, y, p, s in rows]

    return {
        "pairs": n,
        "positives": positives,
        "threshold": threshold,
        "at_threshold": _prf(predicted, tp, positives),
        "best_dev_threshold": best,
        "sweep": sweep,
        "auroc": _auroc([(p, y) for p, y in scores]),
        "expected_count": expected,
        "expected_stderr": stderr,
        "expected_within_2se": abs(expected - positives) <= 2 * stderr,
        "ece": ece,
        "reliability": reliability,
        "false_negatives": examples(1, "ASC"),
        "false_positives": examples(0, "DESC"),
    }


def _refuse(request: httpx.Request) -> httpx.Response:
    raise RuntimeError("rescore is served from the run's own cache; this request missed it")


def _preflighted(corpus: str, rnd: str) -> bool:
    runs = json.loads(RUNS_FILE.read_text()) if RUNS_FILE.exists() else {}
    return f"{corpus}/dev/{rnd}" in runs or any(DATA.glob(f"summary_{corpus}_dev_{rnd}_n*.json"))


def _connect(cache_file: Path, args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    if cache_file.exists():
        cache_file.unlink()  # a fresh cache per run, so the timed run pays for every pair
    con = duckdb.connect()
    duckjev.register(
        con,
        cache_path=cache_file,
        concurrency=args.concurrency,
        max_input_tokens=int(args.max_usd / USD_PER_INPUT_TOKEN),
        transport=fake_transport() if args.dry_run else None,
        api_key="dry-run" if args.dry_run else None,
    )
    return con


def run(args: argparse.Namespace) -> int:
    rnd = ROUNDS[args.round]
    if not (args.limit or args.dry_run or _preflighted(args.corpus, args.round)):
        print(
            f"run `bench/entity_matching.py run {args.round} --corpus {args.corpus} "
            "--split dev --limit 40` first"
        )
        return 2
    DATA.mkdir(parents=True, exist_ok=True)
    tag = run_tag(args.corpus, args.split, args.round, args.limit, args.dry_run)
    con = _connect(DATA / f"cache_{tag}.duckdb", args)
    n = load(con, args.corpus, args.split, args.limit)
    params = judge_params(args.corpus, rnd)

    duckjev.usage(reset=True)
    t0 = time.perf_counter()
    con.execute(judge_sql(rnd, "judged"), params)
    secs = time.perf_counter() - t0
    use = duckjev.usage(reset=True)

    t0 = time.perf_counter()
    con.execute(judge_sql(rnd, "judged_rerun"), params)
    rerun_secs = time.perf_counter() - t0
    rerun = duckjev.usage(reset=True)
    identical = con.execute(IDENTICAL_SQL).fetchone()[0]
    duckjev.flush(con)
    con.table("judged").write_parquet(str(DATA / f"judged_{tag}.parquet"))

    summary = {
        "corpus": args.corpus,
        "round": args.round,
        "split": args.split,
        "limit": args.limit,
        "dry_run": args.dry_run,
        "round_config": asdict(rnd),
        "question": params["q"],
        "model": duckjev.client_for(con).model,
        "duckjev": duckjev.__version__,
        "duckdb": duckdb.__version__,
        "python": platform.python_version(),
        "concurrency": args.concurrency,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "seconds": secs,
        "usage": use,
        "pairs_per_second": n / secs,
        "input_tokens_per_request": use["input_tokens"] / max(use["requests"], 1),
        "usd_per_1k_pairs": use["est_usd"] / n * 1000,
        "rerun_seconds": rerun_secs,
        "rerun_usage": rerun,
        "rerun_identical_rows": identical,
        **metrics(con),
    }
    if args.limit:
        summary["projected_full_usd"] = use["est_usd"] / n * split_size(args.corpus, args.split)
    print(json.dumps(_brief(summary), indent=2))
    if args.limit is None and not args.dry_run:
        runs = json.loads(RUNS_FILE.read_text()) if RUNS_FILE.exists() else {}
        runs[f"{args.corpus}/{args.split}/{args.round}"] = summary
        RUNS_FILE.write_text(json.dumps(runs, indent=1, default=float) + "\n")
        print(f"recorded {args.corpus}/{args.split}/{args.round} in {_rel(RUNS_FILE)}")
    else:
        (DATA / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1, default=float))
    return 0


def _brief(s: dict[str, Any]) -> dict[str, Any]:
    a, b = s["at_threshold"], s["best_dev_threshold"]
    r = {
        "run": f"{s['corpus']}/{s['split']}/{s['round']}"
        + (f" (n={s['limit']})" if s["limit"] else ""),
        "pairs": s["pairs"],
        "positives": s["positives"],
        "seconds": round(s["seconds"], 1),
        "tokens_per_request": round(s["input_tokens_per_request"]),
        "est_usd": round(s["usage"]["est_usd"], 4),
        "usd_per_1k_pairs": round(s["usd_per_1k_pairs"], 4),
        "429s": s["usage"]["rate_limited"],
        "at_0.5": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items()},
        "best_threshold": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()},
        "auroc": None if s["auroc"] is None else round(s["auroc"], 4),
        "ece": None if s["ece"] is None else round(s["ece"], 4),
        "expected": f"{s['expected_count']:.1f} ± {s['expected_stderr']:.1f} "
        f"vs {s['positives']} ({'within' if s['expected_within_2se'] else 'outside'} 2 SE)",
        "rerun": {
            "seconds": round(s["rerun_seconds"], 2),
            "requests": s["rerun_usage"]["requests"],
        },
    }
    if "projected_full_usd" in s:
        r["projected_full_usd"] = round(s["projected_full_usd"], 4)
    return r


def rescore(args: argparse.Namespace) -> int:
    """Rebuild every recorded run's rows and metrics from that run's Jev answer cache.

    The transport refuses every request, so a rescore can neither spend nor read anything
    but the run's own answers; timing and usage in the run log are kept from the live run.
    """
    runs = json.loads(RUNS_FILE.read_text())
    for key, summary in runs.items():
        if summary.get("legacy") or key.startswith("demo/"):
            continue
        tag = run_tag(summary["corpus"], summary["split"], summary["round"])
        cache = DATA / f"cache_{tag}.duckdb"
        if not cache.exists():
            print(f"skip {key}: {_rel(cache)} is missing")
            continue
        con = duckdb.connect()
        duckjev.register(
            con, cache_path=cache, api_key="cache-only", transport=httpx.MockTransport(_refuse)
        )
        load(con, summary["corpus"], summary["split"], None)
        rnd = ROUNDS[summary["round"]]
        duckjev.usage(reset=True)
        con.execute(judge_sql(rnd, "judged"), judge_params(summary["corpus"], rnd))
        assert duckjev.usage()["requests"] == 0
        con.table("judged").write_parquet(str(DATA / f"judged_{tag}.parquet"))
        before = summary["at_threshold"]["f1"]
        summary.update(metrics(con))
        print(f"rescored {key}: F1 at 0.5 {before:.4f} -> {summary['at_threshold']['f1']:.4f}")
    RUNS_FILE.write_text(json.dumps(runs, indent=1, default=float) + "\n")
    return 0


# --------------------------------------------------------------------------- demo

DEMO_SAMPLE_SQL = """CREATE OR REPLACE TABLE a_s AS
SELECT id, rec, blk, rec_text(rec) AS txt FROM recs_a
ORDER BY md5(id || 'em-demo') LIMIT $n"""
# The dedup demo judges every pair inside a block, right-right pairs included, so it takes
# the sampled left rows plus at most DEMO_RIGHT_PER_BLOCK right rows of each of their blocks.
DEMO_UNION_SQL = """CREATE OR REPLACE TABLE u AS
SELECT 'A' || id AS id, rec, blk FROM a_s
UNION ALL
SELECT 'B' || id AS id, rec, blk FROM recs_b WHERE blk IN (SELECT blk FROM a_s)
QUALIFY row_number() OVER (PARTITION BY blk ORDER BY md5(id || 'em-demo')) <= $k"""
DEMO_JOIN_SQL = """CREATE OR REPLACE TABLE joined AS
SELECT left_row.id AS lid, right_row.id AS rid, p
FROM sem_join('a_s', 'recs_b', 'blk', 'rec', 'rec', $q, 0.0)"""
DEMO_JOIN_METRICS_SQL = """WITH gold AS (
  SELECT DISTINCT ltable_id::VARCHAR AS lid, rtable_id::VARCHAR AS rid, label
  FROM read_csv($pairs, header = true, union_by_name = true))
SELECT count(*) AS blocked_pairs,
  count(*) FILTER (p >= 0.5) AS matched,
  expected_count(p) AS expected, expected_count_stderr(p) AS stderr,
  count(*) FILTER (p >= 0.5 AND g.label = 1) AS matched_gold_positive,
  count(*) FILTER (p >= 0.5 AND g.label = 0) AS matched_gold_negative,
  count(*) FILTER (p >= 0.5 AND g.label IS NULL) AS matched_unlabeled,
  count(*) FILTER (g.label = 1) AS gold_positive_in_blocks,
  count(*) FILTER (g.label = 1 AND p >= 0.5) AS gold_positive_matched
FROM joined j LEFT JOIN gold g USING (lid, rid)"""
DEMO_DEDUP_SQL = """SELECT (SELECT count(*) FROM u) AS rows_in,
  (SELECT count(*) FROM sem_dedup('u', 'id', 'blk', 'rec', $q, 0.5)) AS survivors,
  (SELECT count(*) FROM sem_dups('u', 'id', 'blk', 'rec', $q, 0.5)) AS duplicate_pairs"""
DEMO_TOPK_SQL = """SELECT rec.name, score, confidence
FROM sem_topk('a_s', 'txt', $instr, $levels, 5)"""


def demo(args: argparse.Namespace) -> int:
    """The table macros themselves, live, on a sample of one corpus; recorded as demo/<corpus>."""
    c = CORPORA[args.corpus]
    tag = f"{args.corpus}_demo" + ("_dry" if args.dry_run else "")
    con = _connect(DATA / f"cache_{tag}.duckdb", args)
    load_records(con, args.corpus)
    con.execute(DEMO_SAMPLE_SQL, {"n": args.n})
    con.execute(DEMO_UNION_SQL, {"k": DEMO_RIGHT_PER_BLOCK})
    q = c.colleague
    pairs = [str(data_file(args.corpus, f)) for f in ("train.csv", "valid.csv", "test.csv")]
    out: dict[str, Any] = {
        "corpus": args.corpus,
        "sample_rows": args.n,
        "block": c.block,
        "question": q,
        "dry_run": args.dry_run,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }

    duckjev.usage(reset=True)
    t0 = time.perf_counter()
    con.execute(DEMO_JOIN_SQL, {"q": q})
    out["join_seconds"] = time.perf_counter() - t0
    out["join_usage"] = duckjev.usage(reset=True)
    cols = [d[0] for d in con.execute(DEMO_JOIN_METRICS_SQL, {"pairs": pairs}).description]
    out["join"] = dict(
        zip(cols, con.execute(DEMO_JOIN_METRICS_SQL, {"pairs": pairs}).fetchone(), strict=True)
    )

    t0 = time.perf_counter()
    cols = [d[0] for d in con.execute(DEMO_DEDUP_SQL, {"q": q}).description]
    out["dedup"] = dict(zip(cols, con.execute(DEMO_DEDUP_SQL, {"q": q}).fetchone(), strict=True))
    out["dedup_seconds"] = time.perf_counter() - t0
    out["dedup_usage"] = duckjev.usage(reset=True)

    t0 = time.perf_counter()
    out["topk"] = [
        list(r)
        for r in con.execute(
            DEMO_TOPK_SQL, {"instr": c.topk_instructions, "levels": json.dumps(c.topk_levels)}
        ).fetchall()
    ]
    out["topk_seconds"] = time.perf_counter() - t0
    out["topk_usage"] = duckjev.usage(reset=True)
    duckjev.flush(con)
    print(json.dumps(out, indent=2, default=float))
    if not args.dry_run:
        runs = json.loads(RUNS_FILE.read_text()) if RUNS_FILE.exists() else {}
        runs[f"demo/{args.corpus}"] = out
        RUNS_FILE.write_text(json.dumps(runs, indent=1, default=float) + "\n")
        print(f"recorded demo/{args.corpus} in {_rel(RUNS_FILE)}")
    else:
        (DATA / f"summary_{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    return 0


# --------------------------------------------------------------------------- report


def chosen_round(runs: dict[str, Any], corpus: str) -> str:
    """The selectable round with the best dev F1 at 0.5; ties go to fewer tokens per request."""
    dev = [
        (r["at_threshold"]["f1"], -r["input_tokens_per_request"], r["round"])
        for k, r in runs.items()
        if k.startswith(f"{corpus}/dev/") and ROUNDS[r["round"]].selectable
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


def _rounds_table(runs: list[dict[str, Any]]) -> str:
    def row(r: dict[str, Any]) -> list[Any]:
        a, b = r["at_threshold"], r["best_dev_threshold"]
        return [
            f"{r['split']}/{r['round']}",
            f"**{a['f1']:.3f}**",
            a["precision"],
            a["recall"],
            f"{b['f1']:.3f} at {b['threshold']:.2f}",
            r["auroc"],
            r["ece"],
            f"{r['expected_count']:.1f} ± {r['expected_stderr']:.1f} vs {r['positives']}",
            f"{r['input_tokens_per_request']:,.0f}",
            f"${r['usd_per_1k_pairs']:.4f}",
            f"{r['pairs_per_second']:.0f}",
        ]

    return _table(
        [
            "run",
            "F1 at 0.5",
            "precision",
            "recall",
            "best F1 on this split",
            "AUROC",
            "ECE",
            "expected matches vs true",
            "tokens / req",
            "$ / 1k pairs",
            "pairs / s",
        ],
        [row(r) for r in runs],
    )


def _reliability_table(rows: list[list[Any]]) -> str:
    return _table(
        ["bin", "n", "mean p", "positive rate"],
        [[f"{b / 10:.1f}–{(b + 1) / 10:.1f}", n, mp, pr] for b, n, mp, pr in rows],
    )


def _examples_table(rows: list[list[Any]]) -> str:
    def cell(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ")

    return _table(
        ["p", "gold", "record A (cut)", "record B (cut)"],
        [[f"{p:.2f}", y, cell(a), cell(b)] for _, _, y, p, a, b in rows],
    )


def _corpus_section(runs: dict[str, Any], key: str) -> list[str]:
    c = CORPORA[key]
    dev = [runs[k] for k in sorted(runs) if k.startswith(f"{key}/dev/")]
    held = [runs[k] for k in sorted(runs) if k.startswith(f"{key}/test/")]
    if not dev:
        return []
    best = chosen_round(runs, key)
    final = runs.get(f"{key}/test/{best}")
    parts = [
        f"## {c.name}",
        "",
        f"Records are `{', '.join(c.fields)}`; the chosen round is **{best}**. "
        f"{dev[0]['pairs']:,} labeled dev pairs ({dev[0]['positives']} matches)"
        + (
            f"; {final['pairs']:,} held-out test pairs ({final['positives']} matches)."
            if final
            else "."
        ),
        "",
    ]
    if final:
        parts += [
            f"### Held-out test split, {c.name}",
            "",
            _rounds_table(held),
            "",
            f"Round {best} on the held-out split: F1 **{final['at_threshold']['f1']:.3f}** at "
            f"the default threshold 0.5 (precision {final['at_threshold']['precision']:.3f}, "
            f"recall {final['at_threshold']['recall']:.3f}); AUROC {final['auroc']:.3f}; ECE "
            f"{final['ece']:.3f}. The calibrated join size Σp over all labeled pairs is "
            f"{final['expected_count']:.1f} ± {final['expected_stderr']:.1f} against "
            f"{final['positives']} true matches, "
            f"{'within' if final['expected_within_2se'] else 'outside'} 2 SE. "
            f"{final['pairs_per_second']:.0f} pairs/s at concurrency {final['concurrency']} "
            f"({final['usage']['rate_limited']} × 429), "
            f"{final['input_tokens_per_request']:,.0f} tokens per request, "
            f"**${final['usd_per_1k_pairs']:.4f} per 1,000 pairs**; cache re-run "
            f"{final['rerun_seconds']:.2f} s, {final['rerun_usage']['requests']} requests, "
            f"{final['rerun_identical_rows']} / {final['pairs']} rows identical.",
            "",
        ]
        dev_best = runs[f"{key}/dev/{best}"]["best_dev_threshold"]
        t = dev_best["threshold"]
        at_dev_t = next(r for r in final["sweep"] if abs(r["threshold"] - t) < 1e-9)
        parts += [
            f"Dev picked threshold {t:.2f} for {best} (F1 {dev_best['f1']:.3f} on dev); at that "
            f"threshold the held-out split gives precision {_fmt(at_dev_t['precision'])}, recall "
            f"{_fmt(at_dev_t['recall'])}, F1 {at_dev_t['f1']:.3f}.",
            "",
        ]
    parts += [
        f"### Tuning rounds on the dev split, {c.name}",
        "",
        _rounds_table(dev),
        "",
        *[f"- **{r['round']}**: {r['round_config']['note']}." for r in dev],
        "",
    ]
    if final:
        parts += [
            f"### Calibration on the held-out split, {c.name}, round {best}",
            "",
            "`p` against the share of gold matches in each of 10 equal-width bins.",
            "",
            _reliability_table(final["reliability"]),
            "",
            f"### What round {best} still gets wrong on the held-out split, {c.name}",
            "",
            "The five gold matches with the lowest p:",
            "",
            _examples_table(final["false_negatives"]),
            "",
            "The five gold non-matches with the highest p:",
            "",
            _examples_table(final["false_positives"]),
            "",
        ]
    return parts


def _demo_section(runs: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    for key in CORPORA:
        d = runs.get(f"demo/{key}")
        if not d:
            continue
        c = CORPORA[key]
        j, dd = d["join"], d["dedup"]
        parts += [
            f"## The macros live: {c.name}, {d['sample_rows']} sampled left rows",
            "",
            f"`sem_join('a_s', 'recs_b', 'blk', 'rec', 'rec', $q, 0.0)` with the block key "
            f"`{c.block}` and the colleague-style question, then `count(*) FILTER (p >= 0.5)` "
            "and `expected_count(p)` over the result:",
            "",
            _table(
                ["metric", "value"],
                [
                    ["blocked pairs judged", j["blocked_pairs"]],
                    ["matched at p ≥ 0.5", j["matched"]],
                    [
                        "of which labeled match / labeled non-match / unlabeled",
                        f"{j['matched_gold_positive']} / {j['matched_gold_negative']} / "
                        f"{j['matched_unlabeled']}",
                    ],
                    [
                        "labeled matches inside the blocks, recalled at 0.5",
                        f"{j['gold_positive_matched']} / {j['gold_positive_in_blocks']}",
                    ],
                    [
                        "expected matches Σp ± SE",
                        f"{j['expected']:.1f} ± {j['stderr']:.1f}",
                    ],
                    [
                        "cost",
                        f"{d['join_usage']['requests']} requests, "
                        f"${d['join_usage']['est_usd']:.4f}, {d['join_seconds']:.1f} s",
                    ],
                ],
            ),
            "",
            f"`sem_dedup('u', 'id', 'blk', 'rec', $q, 0.5)` over the sampled left rows plus at "
            f"most {DEMO_RIGHT_PER_BLOCK} right rows of each of their blocks ({dd['rows_in']} "
            f"rows; every pair inside a block is judged, right-right pairs included): "
            f"{dd['survivors']} survivors, {dd['duplicate_pairs']} duplicate pairs found; "
            f"{d['dedup_usage']['requests']} requests, ${d['dedup_usage']['est_usd']:.4f} (the "
            "cache served the pairs the join had already judged).",
            "",
            f"`sem_topk('a_s', 'txt', '{c.topk_instructions}', {json.dumps(list(c.topk_levels))}, "
            f"5)`; {d['topk_usage']['requests']} requests, ${d['topk_usage']['est_usd']:.4f}:",
            "",
            _table(["name", "score", "confidence"], d["topk"]),
            "",
        ]
    return parts


def report(args: argparse.Namespace) -> int:
    runs = json.loads(RUNS_FILE.read_text())
    cov = runs.get("coverage", {}).get("rows", [])
    parts = [
        "# Entity matching: `sem_join`, `sem_dedup`, `sem_topk`",
        "",
        "Generated by `bench/entity_matching.py report` from "
        "`docs/results/entity_matching_runs.json`; do not edit by hand. Every number below is "
        "from a live run against `jev-1.13.0`.",
        "",
        "Corpora: two sets from the DeepMatcher / Magellan entity-matching collection "
        "(Mudgal et al., SIGMOD 2018): Abt-Buy, textual product listings from two retailers "
        "(1,081 × 1,092 records), and DBLP-ACM, structured bibliographic records (2,616 × 2,294). "
        "Each ships labeled candidate pairs, already blocked by its authors and split "
        "train / valid / test. The valid split is the dev set for the tuning rounds; the test "
        "split is held out and was run only with the baseline and the round chosen on dev. The "
        "chosen round is the selectable round with the best dev F1 at the default threshold "
        "0.5, ties to fewer tokens per request.",
        "",
        "How it works: every labeled pair is one `jev_noul` over the pair state, the primitive "
        "`jev_match(a, b, q)` that `sem_join` and `sem_dedup` call for each blocked pair. A "
        "round decides the question wording, the pair-state layout (the JSON object "
        "`jev_pair` builds, or labeled text lines) and which record comes first. *F1 at 0.5* "
        "is precision and recall at the default `sem_join` threshold; *best F1 on this split* "
        "is the best threshold in steps of 0.05, reported for the held-out split only as what "
        "the dev-chosen threshold achieves. *Expected matches* is Σp over the labeled pairs "
        "with its Bernoulli standard error, the calibrated size of the join.",
        "",
    ]
    if cov:
        parts += [
            "## Blocking coverage (no Jev)",
            "",
            "What a key-equality block keeps, before any pair is judged: the candidate pairs it "
            "makes, and how many of the labeled gold matches fall inside them.",
            "",
            _table(
                ["corpus", "block key", "cross product", "blocked pairs", "gold matches kept"],
                [
                    [
                        r["corpus"],
                        f"`{r['block']}`",
                        f"{r['cross_pairs']:,}",
                        f"{r['blocked_pairs']:,} ({r['blocked_pairs'] / r['cross_pairs']:.2%})",
                        f"{r['gold_pairs_blocked']} / {r['gold_pairs']} ({r['gold_recall']:.1%})",
                    ]
                    for r in cov
                ],
            ),
            "",
        ]
    for key in CORPORA:
        parts += _corpus_section(runs, key)
    parts += _demo_section(runs)
    parts += [
        "## The SQL and the questions",
        "",
        "One Noul per labeled pair; `{state}` is one of the pair-state layouts and "
        "`$q` the round's question:",
        "",
        _sql(JUDGE_SQL.format(table="judged", state="{state}", noul="jev_noul({state}, $q)")),
        "",
        "Pair-state layouts (`jev_pair` builds the JSON object; `rec_text` is a bench macro "
        "that writes `field: value; ...` and drops empty fields):",
        "",
        *[f"- `{k[0]}`, order `{k[1]}`: `{v}`" for k, v in STATE_SQL.items()],
        "",
        *[
            line
            for key, c in CORPORA.items()
            for line in (
                f"{c.name}, plain: `{c.plain}`",
                "",
                f"{c.name}, colleague-style: `{c.colleague}`",
                "",
            )
        ],
        "## Reproduce",
        "",
        "```bash",
        "uv run python bench/entity_matching.py prepare",
        "uv run python bench/entity_matching.py coverage --record",
        *[
            f"uv run python bench/entity_matching.py run {r['round']} --corpus {r['corpus']} "
            f"--split {r['split']}"
            for k, r in sorted(runs.items())
            if not r.get("legacy") and not k.startswith("demo/")
        ],
        *[
            f"uv run python bench/entity_matching.py demo --corpus {k.split('/')[1]}"
            for k in sorted(runs)
            if k.startswith("demo/")
        ],
        "uv run python bench/entity_matching.py report",
        "```",
        "",
    ]
    RESULTS.write_text("\n".join(parts))
    print(f"wrote {_rel(RESULTS)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    c = sub.add_parser("coverage")
    c.add_argument("--record", action="store_true", help="write the numbers into the run log")
    r = sub.add_parser("run")
    r.add_argument("round", choices=list(ROUNDS))
    r.add_argument("--corpus", default="abt", choices=list(CORPORA))
    r.add_argument("--split", default="dev", choices=SPLITS)
    r.add_argument("--limit", type=int, default=None, help="pre-flight sample size")
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--max-usd", type=float, default=0.10, help="hard budget for this run")
    r.add_argument("--dry-run", action="store_true", help="fake transport, no network")
    d = sub.add_parser("demo")
    d.add_argument("--corpus", default="abt", choices=list(CORPORA))
    d.add_argument("--n", type=int, default=DEMO_ROWS, help="sampled left rows")
    d.add_argument("--concurrency", type=int, default=16)
    d.add_argument("--max-usd", type=float, default=0.25)
    d.add_argument("--dry-run", action="store_true")
    sub.add_parser("rescore", help="rebuild recorded runs from their answer caches; no requests")
    sub.add_parser("report")
    args = p.parse_args(argv)
    commands = {
        "prepare": lambda a: prepare(),
        "coverage": coverage,
        "run": run,
        "demo": demo,
        "rescore": rescore,
        "report": report,
    }
    return commands[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
