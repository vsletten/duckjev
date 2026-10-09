"""Build the keyless demo's data: ``duckjev/demo/reports.csv.gz`` and ``answers.jsonl.gz``.

    uv run python bench/demo_data.py pull     # openFDA, free, no key
    uv run python bench/demo_data.py record   # asks Jev live: TYPESAFE_API_KEY, about one cent

``pull`` takes the first 999 continuous glucose monitor (QBJ) reports openFDA returns for
each of 2026-01-08 and 2026-01-22 with an event description of at least 100 characters,
orders them by a hash of the report key and keeps 150: up to 25 filed as Injury first (the
pull had 10), then the rest. Narratives are cut to 1,200 characters.

``record`` runs the demo's statements live on an empty cache under the demo's budget,
writes every answer it received, then replays the demo offline and fails if that sends a
single request.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path
from urllib.parse import quote

import httpx

import duckjev
from duckjev import demo

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "duckjev" / "demo"
REPORTS = DEMO / "reports.csv.gz"
ANSWERS = DEMO / "answers.jsonl.gz"

EVENT_URL = "https://api.fda.gov/device/event.json"
DAYS = ("20260108", "20260122")
DESCRIPTION = "Description of Event or Problem"
ROWS = 150
INJURY_FLOOR = 25
MAX_CHARS = 1200
COLUMNS = ("mdr_report_key", "date_received", "manufacturer", "brand_name", "event_type")


def _description(r: dict) -> str:
    seen: list[str] = []
    for t in r.get("mdr_text") or []:
        text = (t.get("text") or "").strip()
        if t.get("text_type_code") == DESCRIPTION and text and text not in seen:
            seen.append(text)
    return "\n".join(seen)


def _flatten(r: dict) -> dict:
    dev = next(
        (d for d in r.get("device") or [] if d.get("device_report_product_code") == "QBJ"),
        (r.get("device") or [{}])[0],
    )
    narrative = _description(r)
    if len(narrative) > MAX_CHARS:
        narrative = narrative[:MAX_CHARS].rsplit(" ", 1)[0] + " ..."
    d = r.get("date_received") or ""
    return {
        "mdr_report_key": str(r["mdr_report_key"]),
        "date_received": f"{d[:4]}-{d[4:6]}-{d[6:]}",
        "manufacturer": dev.get("manufacturer_d_name") or "",
        "brand_name": dev.get("brand_name") or "",
        "event_type": r.get("event_type") or "",
        "narrative": narrative,
    }


def pull(_: argparse.Namespace) -> int:
    rows: list[dict] = []
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for day in DAYS:
            search = (
                f"device.device_report_product_code:QBJ AND date_received:{day} AND "
                f'mdr_text.text_type_code.exact:"{DESCRIPTION}"'
            )
            url = f"{EVENT_URL}?search={quote(search, safe='')}&limit=999"
            resp = client.get(url)
            resp.raise_for_status()
            got = resp.json()["results"]
            print(f"{day}: {len(got)} reports (of {resp.json()['meta']['results']['total']})")
            rows += [_flatten(r) for r in got]
    eligible = {
        r["mdr_report_key"]: r
        for r in rows
        if len(r["narrative"]) >= 100 and r["event_type"] in ("Injury", "Malfunction", "Death")
    }
    ranked = sorted(
        eligible.values(), key=lambda r: hashlib.sha256(r["mdr_report_key"].encode()).hexdigest()
    )
    injuries = [r for r in ranked if r["event_type"] == "Injury"][:INJURY_FLOOR]
    chosen = injuries + [r for r in ranked if r not in injuries][: ROWS - len(injuries)]
    chosen.sort(key=lambda r: int(r["mdr_report_key"]))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=[*COLUMNS, "narrative"], lineterminator="\n")
    w.writeheader()
    w.writerows(chosen)
    REPORTS.write_bytes(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0))
    kinds = {k: sum(r["event_type"] == k for r in chosen) for k in ("Injury", "Malfunction")}
    print(
        f"wrote {len(chosen)} of {len(eligible)} eligible to {REPORTS.relative_to(ROOT)}: {kinds}"
    )
    return 0


def record(_: argparse.Namespace) -> int:
    if not duckjev.client.os.environ.get(duckjev.client.API_KEY_ENV):
        print("record needs TYPESAFE_API_KEY", file=sys.stderr)
        return 2
    duckjev.usage(reset=True)
    con = demo.connect(live=True)
    client = duckjev.client_for(con)
    demo.run(con)
    use = duckjev.usage()
    cache = client.cache.to_arrow().sort_by("key").to_pylist()
    client.close()
    con.close()
    lines = [
        json.dumps(
            {
                "key": e["key"],
                "model": e["model"],
                "answers": json.loads(e["answers"]),
                "usage": {"input_tokens": e["input_tokens"], "output_tokens": e["output_tokens"]},
            },
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for e in cache
    ]
    ANSWERS.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("utf-8"), mtime=0))
    print(
        f"\nlive: {use['requests']} requests, {use['input_tokens']:,} input tokens, "
        f"${use['est_usd']:.4f}; wrote {len(lines)} answers to {ANSWERS.relative_to(ROOT)}"
    )
    duckjev.usage(reset=True)
    out = io.StringIO()
    con = demo.connect(live=False)
    demo.run(con, out)
    duckjev.client_for(con).close()
    con.close()
    replay = duckjev.usage()
    if replay["requests"] or replay["cache_misses"]:
        print(f"offline replay was not complete: {replay}", file=sys.stderr)
        return 1
    print(f"offline replay: 0 requests, {replay['cache_hits']} cache hits")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pull").set_defaults(fn=pull)
    sub.add_parser("record").set_defaults(fn=record)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
