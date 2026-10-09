"""A keyless duckjev demo on 150 FDA adverse event reports for continuous glucose monitors.

``python -m duckjev.demo`` runs four SQL statements over ``reports.csv.gz`` and answers every
Jev question from ``answers.jsonl.gz``, the answers recorded when the demo was built
(``bench/demo_data.py``). It needs no API key and sends no request. ``--live`` asks Jev
again with ``$TYPESAFE_API_KEY``, capped at ``LIVE_BUDGET_TOKENS`` (about one cent).

The reports are public MAUDE records from openFDA (US government work, public domain),
received on 2026-01-08 and 2026-01-22, narratives cut to 1,200 characters.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from importlib.resources import as_file, files
from typing import Any

import duckdb
import httpx

import duckjev
from duckjev.cache import AnswerCache
from duckjev.client import JevError, Usage

MODEL = "jev-1.13.0"  # the model the shipped answers came from; part of every cache key
LIVE_BUDGET_TOKENS = 250_000  # $0.0105 at $42 per billion input tokens

PROBLEM_Q = "What kind of problem with the glucose monitor does this report describe?"
PROBLEMS = {
    "inaccurate readings": "Glucose values that did not match a fingerstick or symptoms: "
    "too high, too low, or erratic",
    "sensor failure": "The sensor stopped working, would not start, showed an error or ended "
    "its session early",
    "adhesive or insertion": "The sensor fell off, did not stick, or the applicator did not "
    "insert it",
    "app, receiver or alerts": "The phone app, receiver, connection, or an alert or alarm "
    "did not work",
    "skin reaction or wound": "Irritation, rash, bleeding, infection, or a wire left in the skin",
}
MISSED_ALERT_Q = "Does the report say an alert or alarm failed to sound, or that one was missed?"
CARE_Q = (
    "Did the patient get care from a medical professional (an emergency room, a hospital, "
    "paramedics or a doctor) because of this event?"
)

PARAMS = {
    "problem_q": PROBLEM_Q,
    "problems": json.dumps(PROBLEMS),
    "missed_alert_q": MISSED_ALERT_Q,
    "care_q": CARE_Q,
}

STEPS: list[tuple[str, str]] = [
    (
        "Judge each report once, into a table. Three questions, three typed columns: a Choice "
        "with its probability distribution and two Nouls (the probability of yes). Judging "
        "into a table, then querying it, is the pattern: every later query is free.",
        """CREATE TABLE judged AS
SELECT mdr_report_key, manufacturer, event_type, narrative,
       jev_choice(narrative, $problem_q, $problems) AS problem,
       jev_noul(narrative, $missed_alert_q)       AS missed_alert,
       jev_noul(narrative, $care_q)               AS got_care
FROM reports""",
    ),
    (
        "What the reports are about. expected_rows sums each option's probability over the "
        "reports (the soft group-by); argmax_rows counts each report once under its top "
        "choice. Where they differ, the reports were genuinely unsure between options.",
        """SELECT e.key AS kind,
       round(sum(e.value), 1)                         AS expected_rows,
       count(*) FILTER (WHERE problem.choice = e.key) AS argmax_rows
FROM judged, UNNEST(map_entries(problem.probabilities)) AS u(e)
GROUP BY kind ORDER BY expected_rows DESC""",
    ),
    (
        "How many reports describe a missed or silent alert? A filter counts the reports "
        "over 0.5; expected_count adds up the probabilities, and its standard error says "
        "how sure that total is.",
        """SELECT count(*)                                   AS reports,
       count(*) FILTER (WHERE missed_alert >= 0.5) AS filtered,
       round(expected_count(missed_alert), 1)        AS expected,
       round(expected_count_stderr(missed_alert), 1) AS stderr
FROM judged""",
    ),
    (
        "Check the answers against the label the manufacturer filed. Reports filed as "
        "Injury should mostly say the patient got care; malfunctions mostly should not.",
        """SELECT event_type AS filed_as, count(*) AS reports,
       round(avg(got_care), 2)              AS mean_p_care,
       round(expected_count(got_care), 1)   AS expected_with_care
FROM judged GROUP BY event_type ORDER BY reports DESC""",
    ),
]


class OfflineMiss(JevError):
    """A question the shipped answers do not cover; only ``--live`` can answer it."""


def _refuse(request: httpx.Request) -> httpx.Response:
    raise OfflineMiss(
        "this question is not in the shipped answers (did a question change?); "
        "rerun with --live and TYPESAFE_API_KEY set"
    )


def data_file(name: str) -> Any:
    return files("duckjev.demo").joinpath(name)


def load_answers(cache: AnswerCache) -> int:
    with as_file(data_file("answers.jsonl.gz")) as path, gzip.open(path, "rt") as f:
        rows = [json.loads(line) for line in f]
    cache.put_many({r["key"]: (r["model"], r["answers"], r["usage"]) for r in rows})
    return len(rows)


def connect(live: bool) -> duckdb.DuckDBPyConnection:
    """A connection with ``reports`` loaded, an in-memory cache and its own usage counters.

    Offline, the cache holds the shipped answers and the transport refuses every request.
    Live, the cache starts empty, so every question goes to Jev under the budget.
    """
    con = duckdb.connect()
    with as_file(data_file("reports.csv.gz")) as path:
        con.execute("CREATE TABLE reports AS SELECT * FROM read_csv(?)", [str(path)])
    if live:
        client = duckjev.register(
            con, model=MODEL, cache=False, max_input_tokens=LIVE_BUDGET_TOKENS
        )
    else:
        client = duckjev.register(
            con,
            model=MODEL,
            cache=False,
            api_key="offline",  # never sent: the transport refuses every request
            transport=httpx.MockTransport(_refuse),
        )
    client.usage = Usage()
    client.cache = AnswerCache(None)
    if not live:
        load_answers(client.cache)
    return con


def run(con: duckdb.DuckDBPyConnection, out: Any = None) -> None:
    out = out or sys.stdout
    for i, (what, sql) in enumerate(STEPS, 1):
        print(f"\n-- {i}. {what}\n{sql};", file=out)
        rel = con.execute(sql, {k: v for k, v in PARAMS.items() if f"${k}" in sql})
        if rel.description:
            cols = [d[0] for d in rel.description]
            rows = rel.fetchall()
            print(_table(cols, rows), file=out)


def _table(cols: list[str], rows: list[tuple[Any, ...]]) -> str:
    cells = [cols, *[["" if v is None else str(v) for v in r] for r in rows]]
    widths = [max(len(r[i]) for r in cells) for i in range(len(cols))]
    line = lambda r: "  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True))  # noqa: E731
    return "\n".join([line(cells[0]), line(["-" * w for w in widths]), *map(line, cells[1:])])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m duckjev.demo", description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true", help="ask Jev again (needs TYPESAFE_API_KEY)")
    args = ap.parse_args(argv)
    con = connect(args.live)
    client = duckjev.client_for(con)
    if not args.live:
        print(
            f"Replaying {len(client.cache)} recorded Jev answers (model {MODEL}): "
            "no key, no requests."
        )
    try:
        run(con)
    except (JevError, duckdb.Error) as exc:
        print(f"\ndemo stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()
        con.close()
        u = client.usage.snapshot()
        print(
            f"\nusage: {u['requests']} requests, {u['input_tokens']:,} input tokens, "
            f"{u['cache_hits']} cache hits, ${u['est_usd']:.4f}"
        )
    return 0
