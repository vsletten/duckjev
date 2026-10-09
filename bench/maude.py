"""MAUDE demo: coded complaint surveillance on FDA device adverse events (workstation only).

    uv run python bench/maude.py prepare                                # openFDA pull, no Jev key
    uv run python bench/maude.py run R0 --split dev --limit 40          # pre-flight, not recorded
    uv run python bench/maude.py run R0 --split dev                     # one round on dev
    uv run python bench/maude.py confusions R0 --split dev --show 3     # top confusions, offline
    uv run python bench/maude.py reading --file reading.md              # hand-written, before test
    uv run python bench/maude.py run R2 --split test                    # the held-out split
    uv run python bench/maude.py demo --code QBJ                        # the demo queries, live
    uv run python bench/maude.py rescore                                # from caches, free
    uv run python bench/maude.py report                                 # docs/results/maude.md
    uv run python bench/maude.py run R0 --split dev --dry-run           # fake transport, no key

The corpus is three FDA product codes pulled from openFDA ``device/event`` over a closed
window: continuous glucose monitors (QBJ, two fixed days a month), silicone gel-filled breast
implants (FTR) and implantable cardioverter defibrillators (LWS). Each report is one fused
``jev()`` request: a Choice over the device-problem terms filed for that product code
(candidates in code, plus a catch-all), a Choice over FDA's event types, and a Score for
severity. The problem code and the harm category are measured against what the
manufacturers filed; the round decides the option descriptions, the state layout, the
option order, the vocabulary and whether the three questions share one request.

Per code, dev is 500 reports and test 1,000, drawn by a fixed hash rule stratified by event
type; a disjoint gloss slice of 200 is the only source of examples for the criteria. The
test split runs only with R0 and the round chosen on dev. Every full live run appends its
metrics to ``docs/results/maude_runs.json`` and every live call its spend; ``report``
renders ``docs/results/maude.md`` from that file alone. Pre-flights and dry runs write their
own suffixed files under ``bench/data/`` and record nothing but spend.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import duckdb
import httpx
import pyarrow as pa

import duckjev
from duckjev.client import USD_PER_INPUT_TOKEN

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
DATA = BENCH / "data"
RUNS_FILE = ROOT / "docs" / "results" / "maude_runs.json"
RESULTS = ROOT / "docs" / "results" / "maude.md"
IDS_FILE = BENCH / "maude_ids.json"
TERMS_FILE = BENCH / "maude_terms.json"
CRITERIA_FILE = BENCH / "maude_criteria_v2.json"

EVENT_URL = "https://api.fda.gov/device/event.json"
RECALL_URL = "https://api.fda.gov/device/recall.json"
ANNEX_URL = "https://www.fda.gov/media/192166/download?attachment"
OPENFDA_KEY_ENV = "OPENFDA_API_KEY"
PAGE = 999  # 1,000 needs a key; 999 does not
WINDOW = ("20250701", "20260630")  # date_received, inclusive, closed
SAMPLED_DAYS = (8, 22)  # QBJ is pulled on these two days of every month in the window
DESCRIPTION = "Description of Event or Problem"
ADDITIONAL = "Additional Manufacturer Narrative"
MIN_DESCRIPTION = 100
HARMS = ("Death", "Injury", "Malfunction", "Other")
SPLITS = ("dev", "test")
SPLIT_SIZES = {"dev": 500, "test": 1000, "gloss": 200}  # per product code
FLOOR = 50  # per event type in dev and test, where that many remain
OPTIONS_PER_CODE = 40
MIN_TERM_REPORTS = 5
MAX_OPTIONS = 255  # Jev's Choice cap, catch-all included
CATCH_ALL = "some other problem"
LEGACY = (
    "Adverse Event Without Identified Device or Use Problem",
    "Appropriate Device Problem Term/Code Not Available",
)
TOTAL_BUDGET_USD = 3.0  # PR #8's token guard for the R rounds and the demo; ledger "spend"
# Issue #10's frozen re-evaluation (rounds F0 and F3): its own authorized budget and ledger.
FROZEN_BUDGET_USD = 1.50
FROZEN = ("F0", "F3")
FROZEN_FILE = BENCH / "maude_frozen.json"
STREAM_2026 = 2_503_728  # MAUDE reports received 2026-01-01 to 2026-09-26 (MAUDE.md §2)
DEMO_ROWS = 200
DEMO_OTHERS_PER_BLOCK = 5


@dataclass(frozen=True)
class Code:
    name: str
    sampled: bool  # True: only SAMPLED_DAYS of each month; False: the whole window


CODES: dict[str, Code] = {
    "QBJ": Code("Continuous glucose monitor", True),
    "FTR": Code("Silicone gel-filled breast implant", False),
    "LWS": Code("Implantable cardioverter defibrillator (non-CRT)", False),
}

# --------------------------------------------------------------------------- helpers


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(o: Any) -> Any:
    """Dates as ISO strings; Decimal and numpy scalars as floats."""
    return o.isoformat() if isinstance(o, date) else float(o)


def _write_json(path: Path, obj: Any, indent: int | None = 1) -> None:
    text = json.dumps(obj, indent=indent, ensure_ascii=False, default=_json_default)
    path.write_text(text + "\n")


def _hash(key: str, salt: str) -> str:
    return hashlib.md5(f"{key}|{salt}".encode()).hexdigest()


def _date(s: str | None) -> date | None:
    if not s or not re.fullmatch(r"\d{8}", s):
        return None
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:]))
    except ValueError:
        return None


# --------------------------------------------------------------------------- prepare: vocabulary

ANNEX_SQL = """SELECT * FROM read_xlsx($path, sheet = 'A', header = true, range = 'A8:K2000',
  all_varchar = true)"""


def build_terms(xlsx: Path) -> list[dict[str, Any]]:
    """Sheet A of the FDA annexes: every device-problem term with its level, parent, codes
    and definition, in sheet order."""
    con = duckdb.connect()
    cur = con.execute(ANNEX_SQL, {"path": str(xlsx)})
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
    by_code: dict[str, str] = {}
    terms = []
    for r in rows:
        level = next((i for i in (1, 2, 3) if r[f"Level {i} Term"]), None)
        if level is None or not r["IMDRF Code"]:
            continue
        term = r[f"Level {level} Term"].strip()
        code = r["IMDRF Code"].strip()
        by_code[code] = term
        chain = (r["CodeHierarchy"] or code).split("|")
        terms.append(
            {
                "term": term,
                "level": level,
                "parent_code": chain[-2] if len(chain) > 1 else None,
                "imdrf_code": code,
                "fda_code": r["FDA Code"],
                "definition": (r["Definition"] or "").strip() or None,
                "status": r["Status"],
            }
        )
    for t in terms:
        t["parent"] = by_code.get(t.pop("parent_code") or "")
    return terms


def terms_by_name() -> dict[str, dict[str, Any]]:
    return {t["term"]: t for t in load_json(TERMS_FILE)["terms"]}


# --------------------------------------------------------------------------- prepare: openFDA


def _openfda_key() -> str | None:
    return os.environ.get(OPENFDA_KEY_ENV) or None


def _with_key(url: str) -> str:
    key = _openfda_key()
    if not key or "api_key=" in url:
        return url
    return url + ("&" if "?" in url else "?") + "api_key=" + quote(key, safe="")


def _get(client: httpx.Client, url: str) -> httpx.Response | None:
    """GET with backoff; None for openFDA's 404, which means no records."""
    for attempt in range(6):
        resp = client.get(_with_key(url))
        if resp.status_code == 404:
            return None
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < 5:
            time.sleep(2**attempt)
            continue
        resp.raise_for_status()
        return resp
    return None


def sampled_days() -> list[str]:
    start, end = _date(WINDOW[0]), _date(WINDOW[1])
    assert start and end
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out += [f"{y:04d}{m:02d}{d:02d}" for d in SAMPLED_DAYS]
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return [d for d in out if WINDOW[0] <= d <= WINDOW[1]]


def event_search(code: str) -> str:
    if CODES[code].sampled:
        dates = "(" + " ".join(f"date_received:{d}" for d in sampled_days()) + ")"
    else:
        dates = f"date_received:[{WINDOW[0]} TO {WINDOW[1]}]"
    return (
        f"device.device_report_product_code:{code} AND {dates} AND "
        f'mdr_text.text_type_code.exact:"{DESCRIPTION}"'
    )


def event_url(code: str, limit: int = PAGE) -> str:
    return (
        f"{EVENT_URL}?search={quote(event_search(code), safe='')}&limit={limit}"
        "&sort=date_received:asc"
    )


def raw_dir(code: str) -> Path:
    return DATA / "maude_raw" / code


def pull_events(client: httpx.Client, code: str) -> int:
    """Every report of the code's search, 999 a page, following the search_after cursor."""
    out = raw_dir(code)
    done = out / "done.json"
    if done.exists():
        return load_json(done)["records"]
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("page_*.json.gz"):
        old.unlink()
    url: str | None = event_url(code)
    pages = records = 0
    total = None
    while url:
        resp = _get(client, url)
        if resp is None:
            break
        body = resp.json()
        total = total or body["meta"]["results"]["total"]
        results = body.get("results") or []
        if not results:
            break
        with gzip.open(out / f"page_{pages:03d}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(results, f)
        pages += 1
        records += len(results)
        print(f"  {code}: page {pages}, {records:,} of {total:,}", flush=True)
        url = resp.links.get("next", {}).get("url")
    meta = {"records": records, "pages": pages, "total": total, "search": event_search(code)}
    meta["pulled_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    _write_json(done, meta)
    return records


def pull_recalls(client: httpx.Client, code: str) -> list[dict[str, Any]]:
    url = f"{RECALL_URL}?search={quote(f'product_code:{code}', safe='')}&limit={PAGE}"
    resp = _get(client, url)
    return [] if resp is None else resp.json().get("results") or []


def raw_reports(code: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for page in sorted(raw_dir(code).glob("page_*.json.gz")):
        with gzip.open(page, "rt", encoding="utf-8") as f:
            out += json.load(f)
    return out


# --------------------------------------------------------------------------- prepare: flatten


def _texts(report: dict[str, Any], kind: str) -> str | None:
    """Every distinct text of one kind, as published, in the order openFDA lists them."""
    seen: list[str] = []
    for t in report.get("mdr_text") or []:
        text = t.get("text") or ""
        if t.get("text_type_code") == kind and text.strip() and text not in seen:
            seen.append(text)
    return "\n".join(seen) or None


def flatten_report(r: dict[str, Any], code: str) -> dict[str, Any]:
    devices = r.get("device") or [{}]
    dev = next((d for d in devices if d.get("device_report_product_code") == code), devices[0])
    return {
        "mdr_report_key": str(r["mdr_report_key"]),
        "product_code": code,
        "report_number": r.get("report_number") or None,
        "date_received": _date(r.get("date_received")),
        "date_of_event": _date(r.get("date_of_event")),
        "event_type": r.get("event_type") or None,
        "product_problems": list(dict.fromkeys(p for p in r.get("product_problems") or [] if p)),
        "narrative": _texts(r, DESCRIPTION),
        "mfr_narrative": _texts(r, ADDITIONAL),
        "brand_name": dev.get("brand_name") or None,
        "generic_name": dev.get("generic_name") or None,
        "manufacturer": dev.get("manufacturer_d_name") or None,
        "report_source_code": r.get("report_source_code") or None,
        "type_of_report": list(r.get("type_of_report") or []),
        "source_type": list(r.get("source_type") or []),
        "remedial_action": list(r.get("remedial_action") or []),
    }


def eligible(rec: dict[str, Any]) -> bool:
    n = rec["narrative"]
    return bool(n) and len(n) >= MIN_DESCRIPTION and rec["event_type"] in HARMS


def flatten_recall(r: dict[str, Any]) -> dict[str, Any]:
    lines = [
        ("Product", r.get("product_description")),
        ("Reason for recall", r.get("reason_for_recall")),
        ("Root cause", r.get("root_cause_description")),
    ]
    return {
        "product_res_number": r.get("product_res_number"),
        "product_code": r.get("product_code"),
        "recalling_firm": r.get("recalling_firm"),
        "event_date_initiated": r.get("event_date_initiated"),
        "recall_text": "\n".join(f"{k}: {v.strip()}" for k, v in lines if v and v.strip()),
    }


REPORT_SCHEMA = pa.schema(
    [
        ("mdr_report_key", pa.string()),
        ("product_code", pa.string()),
        ("report_number", pa.string()),
        ("date_received", pa.date32()),
        ("date_of_event", pa.date32()),
        ("event_type", pa.string()),
        ("product_problems", pa.list_(pa.string())),
        ("narrative", pa.string()),
        ("mfr_narrative", pa.string()),
        ("brand_name", pa.string()),
        ("generic_name", pa.string()),
        ("manufacturer", pa.string()),
        ("report_source_code", pa.string()),
        ("type_of_report", pa.list_(pa.string())),
        ("source_type", pa.list_(pa.string())),
        ("remedial_action", pa.list_(pa.string())),
    ]
)


def pool_file(code: str) -> Path:
    return DATA / f"maude_pool_{code}.parquet"


def slice_file(code: str, split: str) -> Path:
    return DATA / f"maude_{code}_{split}.parquet"


def recalls_file() -> Path:
    return DATA / "maude_recalls.parquet"


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows, schema=REPORT_SCHEMA), path)


# --------------------------------------------------------------------------- prepare: sample


def option_counts(pool: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for rec in pool:
        for t in rec["product_problems"]:
            counts[t] = counts.get(t, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def option_set(pool: list[dict[str, Any]]) -> list[list[Any]]:
    """The code's candidates: its most frequent filed terms with at least MIN_TERM_REPORTS."""
    top = [(t, n) for t, n in option_counts(pool) if n >= MIN_TERM_REPORTS]
    return [[t, n] for t, n in top[:OPTIONS_PER_CODE]]


def allocate(avail: dict[str, int], n: int, floor: int) -> dict[str, int]:
    """Quotas summing to min(n, total): proportional to what is available, with each category
    raised to ``floor`` where at least that many are available (rarest first, while the floors
    fit in ``n``)."""
    n = min(n, sum(avail.values()))
    fixed: dict[str, int] = {}
    while True:
        free = {c: a for c, a in avail.items() if c not in fixed and a}
        left = n - sum(fixed.values())
        tot = sum(free.values())
        share = {c: left * a / tot for c, a in free.items()} if tot else {}
        low = sorted(
            (a, c) for c, a in free.items() if a >= floor and share[c] < floor and floor <= left
        )
        if not low:
            break
        for _, c in low:
            if sum(fixed.values()) + floor <= n:
                fixed[c] = floor
        if sum(fixed.values()) + floor > n:
            free = {c: a for c, a in avail.items() if c not in fixed and a}
            left = n - sum(fixed.values())
            tot = sum(free.values())
            share = {c: left * a / tot for c, a in free.items()} if tot else {}
            break
    quota = {c: int(s) for c, s in share.items()}
    rest = left - sum(quota.values())
    for c in sorted(share, key=lambda c: (-(share[c] - int(share[c])), c))[:rest]:
        quota[c] += 1
    return {c: fixed.get(c, quota.get(c, 0)) for c in avail}


def draw_splits(pool: list[dict[str, Any]], code: str) -> dict[str, list[str]]:
    """dev, test and gloss, disjoint, each stratified by event type in hash order."""
    by_type: dict[str, list[str]] = {h: [] for h in HARMS}
    for rec in sorted(pool, key=lambda r: _hash(r["mdr_report_key"], f"maude-split-{code}")):
        by_type[rec["event_type"]].append(rec["mdr_report_key"])
    out: dict[str, list[str]] = {}
    for split, n in SPLIT_SIZES.items():
        avail = {h: len(keys) for h, keys in by_type.items()}
        quota = allocate(avail, n, FLOOR if split in SPLITS else 0)
        taken: list[str] = []
        for h in HARMS:
            taken += by_type[h][: quota[h]]
            by_type[h] = by_type[h][quota[h] :]
        out[split] = sorted(taken)
    return out


def prepare(args: argparse.Namespace) -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    fixture = load_json(Path(args.from_fixture)) if args.from_fixture else None
    if args.dry_run:
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            for code in CODES:
                resp = _get(client, event_url(code, limit=1))
                total = 0 if resp is None else resp.json()["meta"]["results"]["total"]
                print(f"{code}: {total:,} reports, {math.ceil(total / PAGE)} pages to pull")
        return 0
    if fixture is None:
        xlsx = DATA / "fda_annexes.xlsx"
        if not xlsx.exists():
            resp = httpx.get(
                ANNEX_URL, follow_redirects=True, timeout=120, headers={"User-Agent": "duckjev"}
            )
            resp.raise_for_status()
            xlsx.write_bytes(resp.content)
        terms = build_terms(xlsx)
        _write_json(
            TERMS_FILE,
            {
                "_": "FDA device-problem vocabulary (IMDRF Annex A as FDA publishes it), sheet A "
                f"of the annexes workbook at {ANNEX_URL}; written by bench/maude.py prepare.",
                "terms": terms,
            },
        )
        print(f"terms: {len(terms)} -> {_rel(TERMS_FILE)}")

    pools: dict[str, list[dict[str, Any]]] = {}
    recalls: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    with httpx.Client(timeout=300, follow_redirects=True) as client:
        for code in CODES:
            if fixture is not None:
                raw = fixture["reports"].get(code, [])
                recalls += [r for r in fixture.get("recalls", []) if r.get("product_code") == code]
            else:
                pull_events(client, code)
                raw = raw_reports(code)
                recalls += pull_recalls(client, code)
                meta[code] = load_json(raw_dir(code) / "done.json")
            seen: set[str] = set()
            flat = []
            for r in raw:
                rec = flatten_report(r, code)
                if rec["mdr_report_key"] not in seen:
                    seen.add(rec["mdr_report_key"])
                    flat.append(rec)
            pools[code] = [rec for rec in flat if eligible(rec)]
            _write_rows(pools[code], pool_file(code))
            print(f"{code}: {len(flat):,} pulled, {len(pools[code]):,} eligible")

    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist([flatten_recall(r) for r in recalls]), recalls_file())
    print(f"recalls: {len(recalls)} -> {_rel(recalls_file())}")

    if IDS_FILE.exists() and not args.resample:
        ids = load_json(IDS_FILE)
    else:
        pooled = [rec for code in CODES for rec in pools[code]]
        ids = {
            "_": "Sampled report keys per product code and split, and each code's option set "
            "(term, reports in the pool), written by bench/maude.py prepare. The splits are "
            "drawn from the pool of eligible reports by a fixed rule (md5 of the key, per event "
            "type, in order) and do not move if openFDA re-serves the window.",
            "window": list(WINDOW),
            "sampled_days": list(SAMPLED_DAYS),
            "min_description_chars": MIN_DESCRIPTION,
            "sizes": SPLIT_SIZES,
            "floor": FLOOR,
            "pull": meta,
            "codes": {
                code: {
                    "pool": len(pools[code]),
                    "pool_by_event_type": {
                        h: sum(1 for r in pools[code] if r["event_type"] == h) for h in HARMS
                    },
                    "options": option_set(pools[code]),
                    "splits": draw_splits(pools[code], code),
                }
                for code in CODES
            },
            "global_options": [[t, n] for t, n in option_counts(pooled)][: MAX_OPTIONS - 1],
        }
        _write_json(IDS_FILE, ids, indent=None)
        print(f"splits and option sets -> {_rel(IDS_FILE)}")

    # A refreshed openFDA pull must not turn a committed split into a smaller,
    # apparently valid run. Fail before writing any of the split parquet files.
    for code in CODES:
        present = {r["mdr_report_key"] for r in pools[code]}
        missing = {
            key
            for keys in ids["codes"][code]["splits"].values()
            for key in keys
            if key not in present
        }
        if missing:
            raise SystemExit(
                f"{code}: {len(missing)} committed split keys are missing from the pull; "
                "refusing to rewrite the split files"
            )

    for code in CODES:
        by_key = {r["mdr_report_key"]: r for r in pools[code]}
        for split, keys in ids["codes"][code]["splits"].items():
            rows = [by_key[k] for k in keys]
            _write_rows(rows, slice_file(code, split))
            by_type = {h: sum(1 for r in rows if r["event_type"] == h) for h in HARMS}
            print(f"  {code}/{split}: {len(rows)} reports {by_type}")
    return 0


# --------------------------------------------------------------------------- freeze (issue #10)

CONTENT_FIELDS = (
    "narrative",
    "mfr_narrative",
    "brand_name",
    "generic_name",
    "manufacturer",
    "event_type",
    "product_problems",
)


def content_hash(rec: dict[str, Any]) -> str:
    """sha256 of what a run reads from a report: the state fields and the filed labels."""
    obj = {f: rec.get(f) for f in CONTENT_FIELDS}
    obj["product_problems"] = list(obj["product_problems"] or [])
    return hashlib.sha256(
        json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def freeze(args: argparse.Namespace) -> int:
    """Freeze the content of every split report and the train-only candidate lists.

    Writes ``bench/maude_frozen.json`` (each split key's content hash) and adds
    ``options_train`` to every code in ``bench/maude_ids.json``: the code's option set drawn
    from its eligible pool without the test reports, so no test label reaches a question.
    Refuses to change either once written, unless ``--refreeze``.
    """
    import pyarrow.parquet as pq

    ids = load_ids()
    if FROZEN_FILE.exists() and not args.refreeze:
        print(f"{_rel(FROZEN_FILE)} exists; frozen content does not move (--refreeze to redo)")
        return 2
    frozen: dict[str, Any] = {
        "_": "Content hash of every split report (sha256 of the compact JSON of "
        + ", ".join(CONTENT_FIELDS)
        + ") as of the pull below, written by bench/maude.py freeze (issue #10). Frozen runs "
        "refuse a report whose content differs.",
        "pull": {c: ids.get("pull", {}).get(c) for c in CODES},
        "frozen_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "pool": {},
        "codes": {},
    }
    for code in CODES:
        pool = pq.read_table(pool_file(code)).to_pylist()
        test = set(ids["codes"][code]["splits"]["test"])
        train = [r for r in pool if r["mdr_report_key"] not in test]
        before = ids["codes"][code].get("options_train")
        after = option_set(train)
        if before is not None and before != after and not args.refreeze:
            raise SystemExit(f"{code}: options_train would change; --refreeze to redo")
        ids["codes"][code]["options_train"] = after
        ids["codes"][code]["options_train_pool"] = len(train)
        by_key = {r["mdr_report_key"]: r for r in pool}
        frozen["pool"][code] = len(pool)  # eligible at this pull; PR #8's count is in ids
        frozen["codes"][code] = {
            split: {k: content_hash(by_key[k]) for k in keys}
            for split, keys in ids["codes"][code]["splits"].items()
        }
        print(f"{code}: {len(after)} train-only options from {len(train):,} reports")
    ids["options_train_rule"] = (
        "options_train: the code's most frequent filed terms (at least MIN_TERM_REPORTS, at "
        "most OPTIONS_PER_CODE) counted over its eligible pool without the test split, in "
        "descending frequency; written by bench/maude.py freeze before any held-out call of "
        "the frozen rounds (issue #10). `options` is PR #8's list, counted with the test split."
    )
    _write_json(IDS_FILE, ids, indent=None)
    _write_json(FROZEN_FILE, frozen, indent=None)
    print(f"content hashes -> {_rel(FROZEN_FILE)}; options_train -> {_rel(IDS_FILE)}")
    return 0


def verify_frozen(con: duckdb.DuckDBPyConnection) -> None:
    """Refuse a frozen run if any loaded report's content differs from its frozen hash."""
    frozen = load_json(FROZEN_FILE)["codes"]
    want = {k: h for code in frozen.values() for split in code.values() for k, h in split.items()}
    cols = ", ".join(CONTENT_FIELDS)
    rows = con.execute(f"SELECT mdr_report_key, {cols} FROM reports").fetchall()
    bad = [
        r[0]
        for r in rows
        if want.get(r[0]) != content_hash(dict(zip(CONTENT_FIELDS, r[1:], strict=True)))
    ]
    if bad:
        raise SystemExit(
            f"{len(bad)} reports differ from their frozen content (first: {bad[0]}); "
            "restore the frozen pull before a frozen run"
        )


# --------------------------------------------------------------------------- rounds


@dataclass(frozen=True)
class Round:
    note: str
    criteria: str = (
        "none"  # none: bare terms | what: official definitions | full: + not_for, example
    )
    state: str = "text"  # text: the description | object: description, device, mfr narrative
    order: str = "given"  # given: descending frequency, catch-all last | reversed
    vocabulary: str = "code"  # code: the product code's option set | global: every term seen
    fused: bool = True  # one request with three questions, or three requests
    selectable: bool = True  # False for checks that are not a candidate config
    # pool: PR #8's options, from every eligible report including the test split's labels;
    # train: from the pool without the test reports, frozen before any held-out call (#10)
    candidates: str = "pool"


ROUNDS: dict[str, Round] = {
    "R0": Round(
        "baseline: the product code's option set as bare term strings, the description as "
        "the state, three questions fused"
    ),
    "R1": Round(
        "R0 with the official FDA definition as structured criteria (`what`) on every option",
        criteria="what",
    ),
    "R2": Round(
        "R1 plus `not_for` and an example where available: a problem option's `not_for` "
        "names its confusable neighbours in the hierarchy, an event type's names its "
        "neighbouring type; the examples are written from the gloss slice",
        criteria="full",
    ),
    "R3": Round(
        "R2 (the best of R0 to R2) with the state as an object: description, brand and "
        "generic name, and the manufacturer's additional narrative when present",
        criteria="full",
        state="object",
    ),
    "R4": Round(
        "R3 (the best round) with both Choices' option order reversed: the order check",
        criteria="full",
        state="object",
        order="reversed",
        selectable=False,
    ),
    "R5": Round(
        "R3 (the best round) over the full vocabulary, every term seen on the corpus, instead "
        "of the product code's set; the terms outside the code's set carry their official "
        "definition: the candidates-in-code ablation",
        criteria="full",
        state="object",
        vocabulary="global",
        selectable=False,
    ),
    "R6": Round(
        "R3 (the best round) as three separate requests instead of one fused: the fusion "
        "cost check",
        criteria="full",
        state="object",
        fused=False,
        selectable=False,
    ),
    "F0": Round(
        "R0 with candidates from the pool without the test reports, on frozen report content: "
        "the baseline of the frozen re-evaluation (issue #10)",
        selectable=False,
        candidates="train",
    ),
    "F3": Round(
        "R3 with candidates from the pool without the test reports, on frozen report content: "
        "the chosen round of the frozen re-evaluation, fixed in advance (issue #10)",
        criteria="full",
        state="object",
        selectable=False,
        candidates="train",
    ),
}

# --------------------------------------------------------------------------- questions

PROBLEM_INSTR = (
    "Which device problem does this medical device adverse event report describe? Choose the "
    "FDA device problem term the manufacturer would code it with."
)
HARM_INSTR = "Which FDA event type is this medical device adverse event report?"
HARM_CRITERIA = {  # from 21 CFR 803.3 (serious injury, malfunction) and the openFDA field docs
    "Death": "The patient died, and the device may have caused or contributed to the death",
    "Injury": (
        "A serious injury and no death: the device may have caused or contributed to an injury "
        "or illness that was life-threatening, resulted in permanent impairment of a body "
        "function or permanent damage to a body structure, or needed medical or surgical "
        "intervention to preclude such impairment or damage"
    ),
    "Malfunction": (
        "The device failed to meet its performance specifications or otherwise perform as "
        "intended, and no death or serious injury is reported"
    ),
    "Other": "Another serious or important medical event that is not a death, a serious "
    "injury or a device malfunction",
}
SEVERITY_INSTR = (
    "How severe was the harm to the patient in this medical device adverse event report?"
)
SEVERITY_LEVELS = [
    "no harm to the patient",
    "minor harm that needed no medical treatment",
    "harm that needed medical or surgical treatment",
    "life-threatening harm or permanent impairment",
    "the patient died",
]


def load_ids() -> dict[str, Any]:
    if not IDS_FILE.exists():
        raise SystemExit("run `bench/maude.py prepare` first")
    return load_json(IDS_FILE)


def load_criteria() -> dict[str, Any]:
    return load_json(CRITERIA_FILE)


def problem_options(rnd: Round, code: str, ids: dict[str, Any]) -> list[str]:
    """The problem Choice's options for a code, catch-all last, before any reordering."""
    if rnd.vocabulary == "global":
        terms = [t for t, _ in ids["global_options"]]
    elif rnd.candidates == "train":
        if "options_train" not in ids["codes"][code]:
            raise SystemExit("run `bench/maude.py freeze` first")
        terms = [t for t, _ in ids["codes"][code]["options_train"]]
    else:
        terms = [t for t, _ in ids["codes"][code]["options"]]
    return terms[: MAX_OPTIONS - 1] + [CATCH_ALL]


def vocab_name(term: str, crit: dict[str, Any]) -> str:
    """The vocabulary's name for an openFDA label string (FDA renamed some terms)."""
    return crit.get("aliases", {}).get(term, term)


def neighbours(term: str, vocab: dict[str, dict[str, Any]], crit: dict[str, Any]) -> list[str]:
    """The confusable neighbours of a term in the hierarchy, as openFDA labels: its parent and
    children, and its siblings when they share a level-2 parent (a top-level category groups
    problems as unlike as a battery fault and loss of capture)."""
    label = {v: k for k, v in crit.get("aliases", {}).items()}
    t = vocab.get(vocab_name(term, crit))
    if t is None:
        return []
    out = []
    parent = vocab.get(t["parent"] or "")
    if parent:
        if parent["level"] >= 2:
            out = [v["term"] for v in vocab.values() if v["parent"] == t["parent"] and v is not t]
        out.append(parent["term"])
    out += [v["term"] for v in vocab.values() if v["parent"] == t["term"]]
    return [label.get(n, n) for n in out]


def option_description(
    term: str,
    rnd: Round,
    options: list[str],
    vocab: dict[str, dict[str, Any]],
    crit: dict[str, Any],
) -> Any:
    """One option's criteria entry: null (bare), {what}, or {what, not_for, examples}."""
    if rnd.criteria == "none":
        return None
    entry = vocab.get(vocab_name(term, crit))
    what = entry["definition"] if entry else crit["glosses"][term]
    if rnd.criteria == "what":
        return {"what": what}
    extra = crit["terms"].get(term, {})
    out: dict[str, Any] = {"what": what}
    near = neighbours(term, vocab, crit) + extra.get("also_not_for", [])
    named = [n for n in dict.fromkeys(near) if n in options and n != term]
    if named:
        short = crit.get("short", {})
        out["not_for"] = "; ".join(f"{short[n]} ({n})" if n in short else n for n in named)
    if extra.get("example"):
        out["examples"] = [extra["example"]]
    return out


def harm_description(harm: str, rnd: Round, crit: dict[str, Any]) -> Any:
    """The event type's gloss; from R2 on, with its `not_for` and one example."""
    if rnd.criteria != "full":
        return HARM_CRITERIA[harm]
    extra = crit.get("harm", {}).get(harm, {})
    out: dict[str, Any] = {"what": HARM_CRITERIA[harm]}
    if extra.get("not_for"):
        out["not_for"] = extra["not_for"]
    if extra.get("example"):
        out["examples"] = [extra["example"]]
    return out


def _ordered(items: list[Any], rnd: Round) -> list[Any]:
    return list(reversed(items)) if rnd.order == "reversed" else list(items)


def questions_for(
    rnd: Round,
    code: str,
    ids: dict[str, Any] | None = None,
    vocab: dict[str, dict[str, Any]] | None = None,
    crit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The three questions one report is asked, in the round's configuration."""
    ids = ids if ids is not None else load_ids()
    vocab = vocab if vocab is not None else terms_by_name()
    crit = crit if crit is not None else (load_criteria() if rnd.criteria != "none" else {})
    options = problem_options(rnd, code, ids)
    own = problem_options(replace(rnd, vocabulary="code"), code, ids)

    def describe(t: str) -> Any:
        # R5: the code's own options exactly as in the base round; the rest of the vocabulary
        # carries its official definition, since no not_for or example was written for it
        if t in own or rnd.criteria == "none":
            return option_description(t, rnd, own, vocab, crit)
        return option_description(t, replace(rnd, criteria="what"), options, vocab, crit)

    return {
        "problem": {
            "type": "choice",
            "instructions": PROBLEM_INSTR,
            "criteria": {t: describe(t) for t in _ordered(options, rnd)},
        },
        "harm": {
            "type": "choice",
            "instructions": HARM_INSTR,
            "criteria": {h: harm_description(h, rnd, crit) for h in _ordered(list(HARMS), rnd)},
        },
        "severity": {"type": "score", "instructions": SEVERITY_INSTR, "criteria": SEVERITY_LEVELS},
    }


def object_state(narrative: str, brand: str | None, generic: str | None, mfr: str | None) -> str:
    """R3's state: the report as a JSON object, serialized to text. ``jev()`` takes a VARCHAR
    state (as ``jev_pair`` does for pairs), so Jev reads the object as JSON text, not as an
    object-valued state; a native object state needs a package change and a new run."""
    obj: dict[str, str] = {"description of event or problem": narrative}
    if brand:
        obj["brand name"] = brand
    if generic:
        obj["generic name"] = generic
    if mfr:
        obj["additional manufacturer narrative"] = mfr
    return json.dumps(obj, ensure_ascii=False)


# --------------------------------------------------------------------------- SQL

LOAD_SQL = """CREATE OR REPLACE TABLE reports AS
SELECT * FROM read_parquet($paths) ORDER BY product_code, mdr_report_key"""
SAMPLE_SQL = """CREATE OR REPLACE TABLE reports AS
SELECT * FROM reports
QUALIFY row_number() OVER (PARTITION BY product_code
                           ORDER BY md5(mdr_report_key || 'maude-sample')) <= $per_code
ORDER BY product_code, mdr_report_key"""
STATE_SQL = """CREATE OR REPLACE TABLE reports AS
SELECT r.*, s.state_object FROM reports r JOIN _states s USING (product_code, mdr_report_key)
ORDER BY product_code, mdr_report_key"""
STATE_COLUMN = {"text": "narrative", "object": "state_object"}

# One fused request per report, or three separate ones merged into the same answers map.
JUDGE_SQL = {
    True: """CREATE OR REPLACE TABLE {table} AS
SELECT r.product_code, r.mdr_report_key, jev(r.{state}, q.questions) AS a
FROM reports r JOIN questions q USING (product_code)""",
    False: """CREATE OR REPLACE TABLE {table} AS
SELECT product_code, mdr_report_key,
       json_merge_patch(json_merge_patch(a_problem, a_harm), a_severity)::VARCHAR AS a
FROM (SELECT r.product_code, r.mdr_report_key,
             jev(r.{state}, q.q_problem) AS a_problem, jev(r.{state}, q.q_harm) AS a_harm,
             jev(r.{state}, q.q_severity) AS a_severity
      FROM reports r JOIN questions q USING (product_code))""",
}
JUDGE_SQLS: dict[tuple[bool, str, str], str] = {
    (fused, state, table): sql.format(table=table, state=col)
    for fused, sql in JUDGE_SQL.items()
    for state, col in STATE_COLUMN.items()
    for table in ("judged", "judged_rerun")
}
IDENTICAL_SQL = """SELECT count(*) FROM judged j JOIN judged_rerun r
USING (product_code, mdr_report_key) WHERE j.a IS NOT DISTINCT FROM r.a"""

# Every metric query is a literal constant; the product code, the catch-all, the question and
# the confidence column are bound parameters ($code NULL means all codes).
SUMMARY_SQL = """SELECT count(*) AS rows,
  count(*) FILTER (covered) AS covered_rows,
  avg(top1_in_set::INT) FILTER (covered) AS top1_in_set,
  avg(top1_in_set::INT) AS top1_all_rows,
  avg(strict_correct::INT) FILTER (covered AND n_filed = 1) AS strict,
  count(*) FILTER (covered AND n_filed = 1) AS strict_rows,
  avg(set_mass) FILTER (covered) AS set_mass,
  count(*) FILTER (covered_code) AS covered_code_rows,
  avg(top1_in_set::INT) FILTER (covered_code) AS top1_on_code_rows,
  avg((problem = $catch_all)::INT) AS catch_all_share,
  avg(n_filed) AS filed_terms_per_report,
  avg(harm_correct::INT) AS harm_accuracy
FROM scored WHERE ($code IS NULL OR product_code = $code)"""
HARM_SQL = """SELECT event_type, count(*) AS n, avg(harm_correct::INT) AS accuracy,
  avg(severity) AS mean_severity
FROM scored WHERE ($code IS NULL OR product_code = $code)
GROUP BY event_type ORDER BY event_type"""
HARM_MATRIX_SQL = """SELECT event_type, harm, count(*) AS n
FROM scored WHERE ($code IS NULL OR product_code = $code)
GROUP BY ALL ORDER BY event_type, harm"""
# Both Choices as rows of one view, so reliability and deferral take the question as a
# parameter: problem accuracy is top-1 in set over the covered reports, harm over all.
LONG_VIEW_SQL = """CREATE OR REPLACE TEMP VIEW judged_long AS
SELECT product_code, 'problem' AS question, problem_confidence AS confidence,
       problem_top_p AS top_p, top1_in_set AS correct, covered AS eligible FROM scored
UNION ALL
SELECT product_code, 'harm', harm_confidence, harm_top_p, harm_correct, true FROM scored"""
QUESTIONS = ("problem", "harm")
RELIABILITY_SQL = """WITH b AS (
  SELECT least(floor(conf * 10), 9)::INT AS bin, conf, correct FROM (
    SELECT CASE WHEN $by = 'top_p' THEN top_p ELSE confidence END AS conf,
           correct::INT AS correct
    FROM judged_long
    WHERE question = $question AND eligible AND ($code IS NULL OR product_code = $code)))
SELECT bin, count(*) AS n, avg(conf) AS mean_conf, avg(correct) AS accuracy
FROM b GROUP BY bin ORDER BY bin"""
DEFERRAL_SQL = """SELECT confidence, correct::INT FROM judged_long
WHERE question = $question AND eligible AND ($code IS NULL OR product_code = $code)"""
SEVERITY_SQL = """SELECT severity, (event_type IN ('Death', 'Injury'))::INT FROM scored
WHERE event_type IN ('Death', 'Injury', 'Malfunction')
  AND ($code IS NULL OR product_code = $code)"""
CONFUSIONS_SQL = """SELECT array_to_string(filed, ' + ') AS filed, problem AS predicted,
  count(*) AS n
FROM scored
WHERE covered AND NOT top1_in_set AND ($code IS NULL OR product_code = $code)
GROUP BY ALL ORDER BY n DESC, filed, predicted LIMIT $k"""
ERRORS_SQL = """SELECT count(*) FILTER (covered AND NOT top1_in_set) FROM scored
WHERE ($code IS NULL OR product_code = $code)"""
CONFUSION_EXAMPLES_SQL = """SELECT mdr_report_key, problem_top_p, narrative FROM scored
WHERE array_to_string(filed, ' + ') = $f AND problem = $p
  AND ($code IS NULL OR product_code = $code)
ORDER BY problem_top_p DESC LIMIT 4"""
# Per filed term: reports that carry it, reports whose argmax is it, and Σp with its SE (a
# filed term outside the round's options is kept, with no probability mass).
COUNTS_SQL = """WITH filed AS (
  SELECT t AS term, count(*) AS filed_count FROM scored, UNNEST(filed) AS u(t)
  WHERE ($code IS NULL OR product_code = $code) GROUP BY ALL),
hard AS (
  SELECT problem AS term, count(*) AS argmax_count FROM scored
  WHERE ($code IS NULL OR product_code = $code) GROUP BY ALL),
soft AS (
  SELECT e.key AS term, sum(e.value) AS expected, sqrt(sum(e.value * (1 - e.value))) AS se
  FROM scored, UNNEST(map_entries(problem_probs)) AS u(e)
  WHERE ($code IS NULL OR product_code = $code) GROUP BY ALL)
SELECT f.term, f.filed_count, coalesce(h.argmax_count, 0) AS argmax_count,
  coalesce(s.expected, 0.0) AS expected, coalesce(s.se, 0.0) AS se
FROM filed f LEFT JOIN hard h USING (term) LEFT JOIN soft s USING (term)
ORDER BY f.filed_count DESC, f.term LIMIT $k"""

# --------------------------------------------------------------------------- load


def run_tag(split: str, rnd: str, limit: int | None = None, dry_run: bool = False) -> str:
    """Names a run's cache and per-row files; only full live runs get the bare tag."""
    return f"{split}_{rnd}" + (f"_n{limit}" if limit else "") + ("_dry" if dry_run else "")


def load(con: duckdb.DuckDBPyConnection, split: str, limit: int | None) -> int:
    """The split's reports of every code in ``reports``, plus the object state column."""
    paths = [slice_file(code, split) for code in CODES]
    if not all(p.exists() for p in paths):
        raise SystemExit("run `bench/maude.py prepare` first")
    con.execute(LOAD_SQL, {"paths": [str(p) for p in paths]})
    if limit:
        con.execute(SAMPLE_SQL, {"per_code": math.ceil(limit / len(CODES))})
        con.execute(
            "CREATE OR REPLACE TABLE reports AS SELECT * FROM reports "
            "ORDER BY product_code, mdr_report_key LIMIT $n",
            {"n": limit},
        )
    rows = con.execute(
        "SELECT product_code, mdr_report_key, narrative, brand_name, generic_name, "
        "mfr_narrative FROM reports"
    ).fetchall()
    states = pa.table(
        {
            "product_code": [r[0] for r in rows],
            "mdr_report_key": [r[1] for r in rows],
            "state_object": [object_state(*r[2:]) for r in rows],
        }
    )
    con.register("_states", states)
    con.execute(STATE_SQL)
    con.unregister("_states")
    return len(rows)


def install_questions(con: duckdb.DuckDBPyConnection, rnd: Round) -> dict[str, Any]:
    """The per-code questions table the judged SQL joins on; returns the maps by code."""
    ids, vocab = load_ids(), terms_by_name()
    crit = load_criteria() if rnd.criteria != "none" else {}
    qs = {code: questions_for(rnd, code, ids, vocab, crit) for code in CODES}
    table = pa.table(
        {
            "product_code": list(qs),
            "questions": [json.dumps(q) for q in qs.values()],
            "q_problem": [json.dumps({"problem": q["problem"]}) for q in qs.values()],
            "q_harm": [json.dumps({"harm": q["harm"]}) for q in qs.values()],
            "q_severity": [json.dumps({"severity": q["severity"]}) for q in qs.values()],
        }
    )
    con.register("_questions", table)
    con.execute("CREATE OR REPLACE TABLE questions AS SELECT * FROM _questions")
    con.unregister("_questions")
    return qs


def judge_sql(rnd: Round, table: str) -> str:
    return JUDGE_SQLS[(rnd.fused, rnd.state, table)]


# --------------------------------------------------------------------------- dry-run transport

HARM_WORDS = {"Death": ("died", "death"), "Injury": ("injur", "surgery", "hospital")}


def fake_transport() -> httpx.MockTransport:
    """Dry-run stand-in that answers by keyword, so tiny pipelines have fixed numbers: an
    option named in the state gets 0.7, and a pair is a match when its two sides share most
    of their words."""

    def choice(state: str, options: list[str]) -> dict[str, Any]:
        low = state.lower()
        named = [o for o in options if o != CATCH_ALL and o.lower() in low]
        if not named and options[0] in HARM_CRITERIA:
            named = [h for h, words in HARM_WORDS.items() if any(w in low for w in words)]
            named = named or ["Malfunction"]
        top = named[0] if named else (CATCH_ALL if CATCH_ALL in options else options[0])
        rest = 0.3 / max(len(options) - 1, 1)
        probs = {o: (0.7 if o == top else rest) for o in options}
        if len(options) == 1:
            probs[top] = 1.0
        return {
            "type": "choice",
            "choice": top,
            "probabilities": probs,
            "confidence": 0.7 if named else 0.4,
        }

    def score(state: str, levels: list[str]) -> dict[str, Any]:
        low = state.lower()
        top = 0
        if any(w in low for w in HARM_WORDS["Injury"]):
            top = min(2, len(levels) - 1)
        if any(w in low for w in HARM_WORDS["Death"]):
            top = len(levels) - 1
        rest = 0.3 / max(len(levels) - 1, 1)
        probs = [0.7 if i == top else rest for i in range(len(levels))]
        return {
            "type": "score",
            "score": sum(i * p for i, p in enumerate(probs)),
            "legend": {str(i): lv for i, lv in enumerate(levels)},
            "probabilities": {str(i): p for i, p in enumerate(probs)},
            "confidence": 0.7,
        }

    def noul(state: str) -> dict[str, Any]:
        try:
            pair = json.loads(state)
            a, b = (set(re.findall(r"[a-z]{4,}", str(pair[k]).lower())) for k in ("a", "b"))
            p = 0.8 if a and b and len(a & b) / len(a | b) > 0.5 else 0.1
        except (ValueError, KeyError, TypeError):
            p = 0.5
        return {"type": "noul", "noul": p}

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        state = body["state"] if isinstance(body["state"], str) else json.dumps(body["state"])
        answers: dict[str, Any] = {}
        for qid, q in body["questions"].items():
            if q["type"] == "choice":
                answers[qid] = choice(state, list(q["criteria"]))
            elif q["type"] == "score":
                answers[qid] = score(state, list(q["criteria"]))
            else:
                answers[qid] = noul(state)
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "answers": answers,
                "usage": {"input_tokens": 100 + len(request.content) // 4, "output_tokens": 20},
            },
        )

    return httpx.MockTransport(handle)


# --------------------------------------------------------------------------- scoring


def flatten(
    answers: dict[str, Any], filed: list[str], options: set[str], code_options: set[str]
) -> dict[str, Any]:
    """One scored row from a report's answers map and the terms its manufacturer filed.

    ``top1_in_set``: the argmax is one of the filed terms. ``strict_correct``: exact match,
    defined only on rows with one filed term. ``set_mass``: Σp over the filed terms.
    ``covered``: some filed term is an option; uncovered rows are left out of the problem
    accuracy and reported as the option set's coverage.
    """
    p, h, s = answers["problem"], answers["harm"], answers["severity"]
    probs = dict(p["probabilities"])
    hprobs = dict(h["probabilities"])
    return {
        "filed": filed,
        "n_filed": len(filed),
        "covered": any(f in options for f in filed),
        "covered_code": any(f in code_options for f in filed),
        "problem": p["choice"],
        "problem_confidence": p["confidence"],
        "problem_top_p": probs[p["choice"]],
        "problem_probs": list(probs.items()),
        "set_mass": sum(probs.get(f, 0.0) for f in filed),
        "top1_in_set": p["choice"] in filed,
        "strict_correct": p["choice"] == filed[0] if len(filed) == 1 else None,
        "harm": h["choice"],
        "harm_confidence": h["confidence"],
        "harm_top_p": hprobs[h["choice"]],
        "harm_probs": list(hprobs.items()),
        "severity": s["score"],
        "severity_confidence": s["confidence"],
        "severity_probs": list(s["probabilities"].items()),
    }


SCORED_SCHEMA = pa.schema(
    [
        ("product_code", pa.string()),
        ("mdr_report_key", pa.string()),
        ("event_type", pa.string()),
        ("filed", pa.list_(pa.string())),
        ("n_filed", pa.int32()),
        ("covered", pa.bool_()),
        ("covered_code", pa.bool_()),
        ("problem", pa.string()),
        ("problem_confidence", pa.float64()),
        ("problem_top_p", pa.float64()),
        ("problem_probs", pa.map_(pa.string(), pa.float64())),
        ("set_mass", pa.float64()),
        ("top1_in_set", pa.bool_()),
        ("strict_correct", pa.bool_()),
        ("harm", pa.string()),
        ("harm_correct", pa.bool_()),
        ("harm_confidence", pa.float64()),
        ("harm_top_p", pa.float64()),
        ("harm_probs", pa.map_(pa.string(), pa.float64())),
        ("severity", pa.float64()),
        ("severity_confidence", pa.float64()),
        ("severity_probs", pa.map_(pa.string(), pa.float64())),
        ("narrative", pa.string()),
    ]
)


def score(con: duckdb.DuckDBPyConnection, rnd: Round) -> None:
    """Build ``scored`` from ``judged`` and the reports' filed codes."""
    ids = load_ids()
    records = []
    for code, key, event_type, filed, narrative, a in con.execute(
        "SELECT r.product_code, r.mdr_report_key, r.event_type, r.product_problems, "
        "r.narrative, j.a FROM reports r JOIN judged j USING (product_code, mdr_report_key) "
        "ORDER BY r.product_code, r.mdr_report_key"
    ).fetchall():
        options = set(problem_options(rnd, code, ids))
        code_options = set(problem_options(ROUNDS["R0"], code, ids))
        rec = flatten(json.loads(a), filed, options, code_options)
        records.append(
            {
                "product_code": code,
                "mdr_report_key": key,
                "event_type": event_type,
                **rec,
                "harm_correct": rec["harm"] == event_type,
                "narrative": narrative,
            }
        )
    con.register("_scored", pa.Table.from_pylist(records, schema=SCORED_SCHEMA))
    con.execute("CREATE OR REPLACE TABLE scored AS SELECT * FROM _scored")
    con.unregister("_scored")


# --------------------------------------------------------------------------- metrics

DEFER_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def _row_dict(
    con: duckdb.DuckDBPyConnection, sql: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    cur = con.execute(sql, params or {})
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, cur.fetchone(), strict=True))


def _ece(rows: list[list[Any]]) -> float | None:
    n = sum(r[1] for r in rows)
    return sum(r[1] * abs(r[3] - r[2]) for r in rows) / n if n else None


def _se(p: float | None, n: int) -> float | None:
    return math.sqrt(p * (1 - p) / n) if p is not None and n else None


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
            ranks[k] = (i + j + 2) / 2
        i = j + 1
    rank_sum = sum(r for r, (_, y) in zip(ranks, ranked, strict=True) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def deferral(rows: list[tuple[float, int]]) -> list[list[Any]]:
    """Per confidence threshold: the share of rows answered, and their accuracy."""
    out = []
    for t in DEFER_THRESHOLDS:
        kept = [c for conf, c in rows if conf >= t]
        out.append(
            [t, len(kept) / len(rows) if rows else None, sum(kept) / len(kept) if kept else None]
        )
    return out


def metrics_for(con: duckdb.DuckDBPyConnection, code: str | None) -> dict[str, Any]:
    """Every reported metric for one product code (or all of them), from ``scored``."""
    p = {"code": code}
    out = _row_dict(con, SUMMARY_SQL, p | {"catch_all": CATCH_ALL})
    out["coverage"] = out["covered_rows"] / out["rows"] if out["rows"] else None
    out["top1_in_set_se"] = _se(out["top1_in_set"], out["covered_rows"])
    out["harm_accuracy_se"] = _se(out["harm_accuracy"], out["rows"])
    harm = [list(r) for r in con.execute(HARM_SQL, p).fetchall()]
    out["harm_by_type"] = harm
    accs = [r[2] for r in harm if r[1]]
    out["harm_macro"] = sum(accs) / len(accs) if accs else None
    out["harm_matrix"] = [list(r) for r in con.execute(HARM_MATRIX_SQL, p).fetchall()]
    con.execute(LONG_VIEW_SQL)
    for q in QUESTIONS:
        for by in ("confidence", "top_p"):
            params = p | {"question": q, "by": by}
            rel = [list(r) for r in con.execute(RELIABILITY_SQL, params).fetchall()]
            out[f"{q}_reliability_{by}"] = rel
            out[f"{q}_ece_{by}"] = _ece(rel)
        out[f"{q}_deferral"] = deferral(con.execute(DEFERRAL_SQL, p | {"question": q}).fetchall())
    out["severity_auroc"] = _auroc(con.execute(SEVERITY_SQL, p).fetchall())
    out["confusions"] = [list(r) for r in con.execute(CONFUSIONS_SQL, p | {"k": 10}).fetchall()]
    out["counts"] = [list(r) for r in con.execute(COUNTS_SQL, p | {"k": 10}).fetchall()]
    return out


def metrics(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    return {
        "pooled": metrics_for(con, None),
        "codes": {code: metrics_for(con, code) for code in CODES},
    }


# --------------------------------------------------------------------------- run log and spend


def load_runs() -> dict[str, Any]:
    return load_json(RUNS_FILE) if RUNS_FILE.exists() else {}


def save_runs(runs: dict[str, Any]) -> None:
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _write_json(RUNS_FILE, runs)


def ledger(rnd: str) -> tuple[str, float]:
    """The run log's spend list for a round, and the budget it is held to."""
    return ("spend_frozen", FROZEN_BUDGET_USD) if rnd in FROZEN else ("spend", TOTAL_BUDGET_USD)


def total_spend(runs: dict[str, Any], book: str = "spend") -> float:
    return sum(e["usd"] for e in runs.get(book, []))


def record_spend(what: str, use: dict[str, Any], book: str = "spend") -> None:
    """Every live call's cost goes in the run log, pre-flights and failed runs included."""
    runs = load_runs()
    runs.setdefault(book, []).append(
        {
            "what": what,
            "usd": use["est_usd"],
            "requests": use["requests"],
            "input_tokens": use["input_tokens"],
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        }
    )
    save_runs(runs)


def budget_tokens(max_usd: float, dry_run: bool, rnd: str = "R0") -> int:
    """Set the run's input-token guard from its cap and the remaining spend on its ledger.

    duckjev reserves each request's estimate before sending (issue #9), so the guard holds up
    to the estimate error of the requests in flight.
    """
    usd = max_usd
    if not dry_run:
        book, total = ledger(rnd)
        left = total - total_spend(load_runs(), book)
        if left <= 0:
            raise SystemExit(f"the ${total:.2f} budget of ledger {book!r} is spent")
        usd = min(usd, left)
    return int(usd / USD_PER_INPUT_TOKEN)


def _connect(cache_file: Path, args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    if cache_file.exists():
        cache_file.unlink()  # a fresh cache per run, so the timed run pays for every report
    con = duckdb.connect()
    duckjev.register(
        con,
        cache_path=cache_file,
        concurrency=args.concurrency,
        max_input_tokens=budget_tokens(
            args.max_usd, args.dry_run, getattr(args, "round", None) or "R0"
        ),
        transport=fake_transport() if args.dry_run else None,
        api_key="dry-run" if args.dry_run else None,
    )
    return con


def _timed(
    con: duckdb.DuckDBPyConnection, sql: str, params: Any, what: str, dry: bool, book: str = "spend"
):
    """Run one live statement; its usage is recorded as spend even if it fails."""
    duckjev.usage(reset=True)
    t0 = time.perf_counter()
    try:
        con.execute(sql, params or {})
    finally:
        use = duckjev.usage(reset=True)
        if not dry and use["requests"]:
            record_spend(what, use, book)
    return time.perf_counter() - t0, use


# --------------------------------------------------------------------------- run


def _preflighted(rnd: str) -> bool:
    """A live pre-flight of the round ran (a dry run does not count), or the round is recorded."""
    live = [f for f in DATA.glob(f"summary_dev_{rnd}_n*.json") if not f.name.endswith("_dry.json")]
    return f"dev/{rnd}" in load_runs() or bool(live)


def split_size(split: str) -> int:
    return sum(len(load_ids()["codes"][c]["splits"][split]) for c in CODES)


def run(args: argparse.Namespace) -> int:
    rnd = ROUNDS[args.round]
    runs = load_runs()
    if not (args.limit or args.dry_run or _preflighted(args.round)):
        print(f"run `bench/maude.py run {args.round} --split dev --limit 40` first")
        return 2
    frozen = args.round in FROZEN
    if frozen and not FROZEN_FILE.exists():
        print("run `bench/maude.py freeze` first: frozen rounds verify report content")
        return 2
    if args.split == "test" and not args.dry_run:
        if args.limit:
            print("the held-out split takes no pre-flight; the dev pre-flight covers the round")
            return 2
        key = "reading_frozen" if frozen else "reading"
        if key not in runs:
            flag = " --frozen" if frozen else ""
            print(
                f"record the hand-written reading (`bench/maude.py reading{flag} --file ...`) first"
            )
            return 2
        allowed = set(FROZEN) if frozen else {"R0", chosen_round(runs)}
        if args.round not in allowed:
            print(f"the held-out split runs only with {sorted(allowed)}")
            return 2
    DATA.mkdir(parents=True, exist_ok=True)
    tag = run_tag(args.split, args.round, args.limit, args.dry_run)
    label = f"{args.split}/{args.round}" + (f" (n={args.limit})" if args.limit else "")
    book, total = ledger(args.round)
    con = _connect(DATA / f"cache_{tag}.duckdb", args)
    n = load(con, args.split, args.limit)
    if frozen:
        verify_frozen(con)
    install_questions(con, rnd)

    secs, use = _timed(con, judge_sql(rnd, "judged"), None, label, args.dry_run, book)
    rerun_secs, rerun = _timed(con, judge_sql(rnd, "judged_rerun"), None, label, args.dry_run, book)
    identical = con.execute(IDENTICAL_SQL).fetchone()[0]
    duckjev.flush(con)
    score(con, rnd)
    con.table("scored").write_parquet(str(DATA / f"scored_{tag}.parquet"))

    summary = {
        "corpus": "+".join(CODES),
        "round": args.round,
        "split": args.split,
        "limit": args.limit,
        "dry_run": args.dry_run,
        "round_config": asdict(rnd),
        "model": duckjev.client_for(con).model,
        "duckjev": duckjev.__version__,
        "duckdb": duckdb.__version__,
        "python": platform.python_version(),
        "concurrency": args.concurrency,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "seconds": secs,
        "usage": use,
        "reports": n,
        "reports_per_second": n / secs,
        "requests_per_report": use["requests"] / n,
        "input_tokens_per_request": use["input_tokens"] / max(use["requests"], 1),
        "input_tokens_per_report": use["input_tokens"] / n,
        "usd_per_1k_reports": use["est_usd"] / n * 1000,
        "rerun_seconds": rerun_secs,
        "rerun_usage": rerun,
        "rerun_identical_rows": identical,
        **metrics(con),
    }
    if args.limit:
        summary["projected_full_usd"] = use["est_usd"] / n * split_size(args.split)
    print(json.dumps(_brief(summary), indent=2))
    if args.limit is None and not args.dry_run:
        runs = load_runs()
        runs[f"{args.split}/{args.round}"] = summary
        save_runs(runs)
        print(f"recorded {args.split}/{args.round} in {_rel(RUNS_FILE)}")
    else:
        _write_json(DATA / f"summary_{tag}.json", summary)
    spent = total_spend(load_runs(), book)
    print(f"spend so far on ledger {book!r}: ${spent:.4f} of ${total:.2f}")
    return 0


def _r(v: Any, k: int = 4) -> Any:
    return round(v, k) if isinstance(v, float) else v


def _brief(s: dict[str, Any]) -> dict[str, Any]:
    def scope(m: dict[str, Any]) -> dict[str, Any]:
        return {
            "rows": m["rows"],
            "coverage": _r(m["coverage"]),
            "top1_in_set": _r(m["top1_in_set"]),
            "strict": _r(m["strict"]),
            "set_mass": _r(m["set_mass"]),
            "problem_ece": _r(m["problem_ece_confidence"]),
            "harm_accuracy": _r(m["harm_accuracy"]),
            "harm_macro": _r(m["harm_macro"]),
            "harm_ece": _r(m["harm_ece_confidence"]),
            "severity_auroc": _r(m["severity_auroc"]),
            "catch_all_share": _r(m["catch_all_share"]),
        }

    out = {
        "run": f"{s['split']}/{s['round']}" + (f" (n={s['limit']})" if s["limit"] else ""),
        "seconds": round(s["seconds"], 1),
        "requests": s["usage"]["requests"],
        "tokens_per_request": round(s["input_tokens_per_request"]),
        "est_usd": round(s["usage"]["est_usd"], 4),
        "usd_per_1k_reports": round(s["usd_per_1k_reports"], 4),
        "429s": s["usage"]["rate_limited"],
        "pooled": scope(s["pooled"]),
        "codes": {c: scope(m) for c, m in s["codes"].items()},
        "top_confusions": s["pooled"]["confusions"][:5],
        "rerun": {
            "seconds": round(s["rerun_seconds"], 2),
            "requests": s["rerun_usage"]["requests"],
        },
    }
    if "projected_full_usd" in s:
        out["projected_full_usd"] = round(s["projected_full_usd"], 4)
    return out


def _refuse(request: httpx.Request) -> httpx.Response:
    raise RuntimeError("rescore is served from the run's own cache; this request missed it")


def recorded_runs(runs: dict[str, Any]) -> list[str]:
    return [k for k in runs if re.fullmatch(r"(dev|test)/[RF]\d+", k)]


def rescore(args: argparse.Namespace) -> int:
    """Rebuild every recorded run's rows and metrics from that run's Jev answer cache.

    The transport refuses every request, so a rescore can neither spend nor read anything
    but the run's own answers; timing and usage in the run log are kept from the live run.
    """
    runs = load_runs()
    for key in recorded_runs(runs):
        summary = runs[key]
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
        install_questions(con, rnd)
        duckjev.usage(reset=True)
        con.execute(judge_sql(rnd, "judged"))
        assert duckjev.usage()["requests"] == 0
        score(con, rnd)
        if not args.dry_run:
            con.table("scored").write_parquet(str(DATA / f"scored_{tag}.parquet"))
        before = summary["pooled"]["top1_in_set"]
        summary.update(metrics(con))
        print(
            f"rescored {key}: top-1 in set {before:.4f} -> {summary['pooled']['top1_in_set']:.4f}"
        )
    if not args.dry_run:
        save_runs(runs)
    return 0


def confusions(args: argparse.Namespace) -> int:
    """Top problem confusions of a run from its per-row file, with the reports behind them."""
    path = DATA / f"scored_{run_tag(args.split, args.round, args.limit, args.dry_run)}.parquet"
    con = duckdb.connect()
    con.execute("CREATE TABLE scored AS SELECT * FROM read_parquet(?)", [str(path)])
    p = {"code": args.code}
    rows = con.execute(CONFUSIONS_SQL, p | {"k": args.top}).fetchall()
    errors = con.execute(ERRORS_SQL, p).fetchone()[0]
    print(f"{_rel(path)}{' ' + args.code if args.code else ''}: {errors} errors; top pairs")
    for filed, pred, n in rows:
        print(f"  {n:3d}  {filed}  ->  {pred}")
    for filed, pred, _ in rows[: args.show]:
        print(f"\n== {filed} -> {pred}")
        for key, conf, text in con.execute(
            CONFUSION_EXAMPLES_SQL, p | {"f": filed, "p": pred}
        ).fetchall():
            print(f"  {key} {conf:.2f}  {text[:300]!r}")
    return 0


def reading(args: argparse.Namespace) -> int:
    """Record the hand-written "Reading the numbers" section; only before the held-out run.

    ``--frozen`` records the frozen re-evaluation's own reading (issue #10), before its
    held-out runs (test/F0, test/F3); PR #8's reading and runs are left as they are.
    """
    runs = load_runs()
    frozen = getattr(args, "frozen", False)
    prefixes = tuple(f"test/{r}" for r in FROZEN) if frozen else ("test/R",)
    if any(k.startswith(prefixes) for k in runs):
        print("the held-out split has run; the reading must be written before it")
        return 2
    text = Path(args.file).read_text(encoding="utf-8").strip()
    if not text:
        print(f"{args.file} is empty; the reading must say something before the held-out run")
        return 2
    entry = {"written": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), "text": text}
    if args.dry_run:
        print(json.dumps(entry, indent=1))
        return 0
    runs["reading_frozen" if frozen else "reading"] = entry
    save_runs(runs)
    print(f"recorded the reading ({len(text):,} characters) in {_rel(RUNS_FILE)}")
    return 0


# --------------------------------------------------------------------------- demo

DEMO_SAMPLE_SQL = """CREATE OR REPLACE TABLE reports AS
SELECT * FROM reports WHERE product_code = $code
ORDER BY md5(mdr_report_key || 'maude-demo') LIMIT $n"""
ANSWER_STRUCT = pa.struct(
    [
        ("choice", pa.string()),
        ("confidence", pa.float64()),
        ("probabilities", pa.map_(pa.string(), pa.float64())),
    ]
)
DEMO_SCHEMA = pa.schema(
    [
        ("mdr_report_key", pa.string()),
        ("product_code", pa.string()),
        ("date_received", pa.date32()),
        ("date_of_event", pa.date32()),
        ("brand_name", pa.string()),
        ("event_type", pa.string()),
        ("product_problems", pa.list_(pa.string())),
        ("remedial_action", pa.list_(pa.string())),
        ("narrative", pa.string()),
        ("problem", ANSWER_STRUCT),
        ("harm", ANSWER_STRUCT),
        (
            "severity",
            pa.struct(
                [
                    ("score", pa.float64()),
                    ("confidence", pa.float64()),
                    ("probabilities", pa.map_(pa.string(), pa.float64())),
                ]
            ),
        ),
    ]
)

# 1. Expected reports per problem and month, with error bars: the trend table.
TREND_SQL = """CREATE OR REPLACE TABLE trend AS
SELECT product_code, e.key AS problem, date_trunc('month', date_received)::DATE AS month,
       expected_count(e.value) AS expected, expected_count_stderr(e.value) AS se,
       count(*) FILTER (WHERE jev_argmax(problem) = e.key) AS argmax_count,
       count(*) FILTER (WHERE list_contains(product_problems, e.key)) AS filed_count
FROM judged, UNNEST(map_entries(problem.probabilities)) AS u(e)
GROUP BY ALL ORDER BY product_code, problem, month"""
# 2. Signal months: expected count more than two SE above the trailing three-month mean.
SIGNAL_SQL = """SELECT product_code, problem, month, expected, se, trailing_mean, trailing_se
FROM (
  SELECT *, avg(expected) OVER w AS trailing_mean,
         sqrt(sum(se * se) OVER w) / count(*) OVER w AS trailing_se,
         count(*) OVER w AS trailing_months
  FROM trend
  WINDOW w AS (PARTITION BY product_code, problem ORDER BY month
               ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING))
WHERE trailing_months = 3
  AND expected > trailing_mean + 2 * sqrt(se * se + trailing_se * trailing_se)
ORDER BY expected - trailing_mean DESC"""
# 3. One event, one row: dedup within the code and event date, at most $k other reports of
# the code's pool per block beside the sampled ones.
DEDUP_TABLE_SQL = """CREATE OR REPLACE TABLE dd AS
SELECT mdr_report_key, product_code || '/' || date_of_event::VARCHAR AS dedup_block, narrative,
       'sample' AS origin
FROM judged WHERE date_of_event IS NOT NULL
UNION ALL
(SELECT mdr_report_key, product_code || '/' || date_of_event::VARCHAR AS dedup_block, narrative,
        'pool' AS origin
 FROM read_parquet($pool)
 WHERE date_of_event IS NOT NULL
   AND product_code || '/' || date_of_event::VARCHAR IN
       (SELECT product_code || '/' || date_of_event::VARCHAR FROM judged)
   AND mdr_report_key NOT IN (SELECT mdr_report_key FROM judged)
 QUALIFY row_number() OVER (PARTITION BY date_of_event
                            ORDER BY md5(mdr_report_key || 'maude-demo')) <= $k)"""
DEDUP_Q = (
    "Do these two adverse event reports describe the same incident with the same device on "
    "the same occasion?"
)
DUPS_SQL = """CREATE OR REPLACE TABLE dups AS
SELECT * FROM sem_dups('dd', 'mdr_report_key', 'dedup_block', 'narrative', $q, 0.0)"""
DEDUP_SQL = """CREATE OR REPLACE TABLE deduped AS
SELECT * FROM sem_dedup('dd', 'mdr_report_key', 'dedup_block', 'narrative', $q, 0.5)"""
DEDUP_METRICS_SQL = """SELECT (SELECT count(*) FROM dd) AS rows_in,
  (SELECT count(*) FILTER (origin = 'sample') FROM dd) AS sampled_rows_in,
  (SELECT count(DISTINCT dedup_block) FROM dd) AS blocks,
  (SELECT count(*) FROM dups) AS pairs_judged,
  (SELECT count(*) FILTER (p >= 0.5) FROM dups) AS duplicate_pairs,
  (SELECT expected_count(p) FROM dups) AS expected_pairs,
  (SELECT expected_count_stderr(p) FROM dups) AS expected_pairs_se"""
DUP_EXAMPLES_SQL = """SELECT d.id, d.duplicate_id, d.p,
  left(a.narrative, 160), left(b.narrative, 160)
FROM dups d JOIN dd a ON a.mdr_report_key = d.id JOIN dd b ON b.mdr_report_key = d.duplicate_id
WHERE d.p >= 0.5 ORDER BY d.p DESC, d.id, d.duplicate_id LIMIT 5"""
# 4. Recall coverage: which reports describe a defect a recall already addresses.
RECALLS_SQL = """CREATE OR REPLACE TABLE recalls AS
SELECT * FROM read_parquet($path) WHERE product_code = $code ORDER BY product_res_number"""
JOIN_Q = "Does this adverse event report describe the defect that this recall addresses?"
JOIN_SQL = """CREATE OR REPLACE TABLE joined AS
SELECT left_row.mdr_report_key AS mdr_report_key,
       right_row.product_res_number AS product_res_number, p
FROM sem_join('judged', 'recalls', 'product_code', 'narrative', 'recall_text', $q, 0.0)"""
JOIN_METRICS_SQL = """SELECT (SELECT count(*) FROM recalls) AS recalls,
  count(*) AS pairs, count(*) FILTER (p >= 0.5) AS matched_pairs,
  expected_count(p) AS expected_pairs, expected_count_stderr(p) AS expected_pairs_se,
  count(DISTINCT mdr_report_key) FILTER (p >= 0.5) AS reports_matched,
  count(DISTINCT product_res_number) FILTER (p >= 0.5) AS recalls_matched
FROM joined"""
JOIN_FLAG_SQL = """WITH per AS (SELECT mdr_report_key, max(p) AS max_p FROM joined GROUP BY ALL)
SELECT list_contains(r.remedial_action, 'Recall') AS flagged_recall, count(*) AS reports,
  avg((max_p >= 0.5)::INT) AS match_rate, avg(max_p) AS mean_max_p
FROM judged r JOIN per USING (mdr_report_key) GROUP BY ALL ORDER BY flagged_recall DESC"""
JOIN_TOP_SQL = """SELECT j.product_res_number, r.event_date_initiated,
  left(replace(r.recall_text, chr(10), ' '), 200) AS recall,
  count(*) FILTER (p >= 0.5) AS reports_matched, expected_count(p) AS expected
FROM joined j JOIN recalls r USING (product_res_number)
GROUP BY ALL ORDER BY expected DESC, j.product_res_number LIMIT 5"""
# ... and the complement: the problem clusters no recall on file addresses.
UNCOVERED_SQL = """WITH per AS (SELECT mdr_report_key, max(p) AS max_p FROM joined GROUP BY ALL)
SELECT jev_argmax(problem) AS problem, count(*) AS reports,
  count(*) FILTER (max_p < 0.5) AS without_recall
FROM judged JOIN per USING (mdr_report_key)
GROUP BY ALL ORDER BY without_recall DESC, problem LIMIT 10"""
# 5. What a reviewer reads first.
TOPK_SQL = """SELECT mdr_report_key, brand_name, event_type, score, confidence,
  left(narrative, 160) AS narrative
FROM sem_topk('judged', 'narrative', $instr, $levels, 20)"""


def build_judged(con: duckdb.DuckDBPyConnection) -> None:
    """``judged`` for the demo queries: one row per report, the three answers as structs."""
    rows = []
    for rec in con.execute(
        "SELECT r.mdr_report_key, r.product_code, r.date_received, r.date_of_event, "
        "r.brand_name, r.event_type, r.product_problems, r.remedial_action, r.narrative, j.a "
        "FROM reports r JOIN answers j USING (product_code, mdr_report_key) "
        "ORDER BY r.mdr_report_key"
    ).fetchall():
        a = json.loads(rec[-1])
        cols = [d.name for d in DEMO_SCHEMA][:9]
        row = dict(zip(cols, rec[:-1], strict=True))
        for q in ("problem", "harm"):
            row[q] = {
                "choice": a[q]["choice"],
                "confidence": a[q]["confidence"],
                "probabilities": list(a[q]["probabilities"].items()),
            }
        s = a["severity"]
        row["severity"] = {
            "score": s["score"],
            "confidence": s["confidence"],
            "probabilities": list(s["probabilities"].items()),
        }
        rows.append(row)
    con.register("_judged", pa.Table.from_pylist(rows, schema=DEMO_SCHEMA))
    con.execute("CREATE OR REPLACE TABLE judged AS SELECT * FROM _judged")
    con.unregister("_judged")


def _rows(con: duckdb.DuckDBPyConnection, sql: str, params: Any = None) -> list[list[Any]]:
    return [list(r) for r in con.execute(sql, params or {}).fetchall()]


def demo(args: argparse.Namespace) -> int:
    """The demo queries of MAUDE.md §4, live, on sampled test reports of one code."""
    runs = load_runs()
    has_dev = any(k.startswith("dev/") for k in runs)
    name = args.round or (chosen_round(runs) if has_dev else "R0")
    rnd = ROUNDS[name]
    code = args.code
    tag = f"demo_{code}" + ("_dry" if args.dry_run else "")
    con = _connect(DATA / f"cache_{tag}.duckdb", args)
    load(con, "test", None)
    con.execute(DEMO_SAMPLE_SQL, {"code": code, "n": args.n})
    n = con.execute("SELECT count(*) FROM reports").fetchone()[0]
    install_questions(con, rnd)
    what = f"demo/{code}"
    out: dict[str, Any] = {
        "code": code,
        "round": name,
        "sample_rows": n,
        "dry_run": args.dry_run,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }

    judge = judge_sql(rnd, "judged").replace("TABLE judged AS", "TABLE answers AS", 1)
    out["judge_seconds"], out["judge_usage"] = _timed(con, judge, None, what, args.dry_run)
    build_judged(con)

    con.execute(TREND_SQL)
    out["trend_rows"] = con.execute("SELECT count(*) FROM trend").fetchone()[0]
    out["trend_top"] = _rows(
        con,
        "WITH top AS (SELECT problem FROM trend GROUP BY ALL ORDER BY sum(expected) DESC, "
        "problem LIMIT 5) SELECT problem, month, expected, se, argmax_count, filed_count "
        "FROM trend JOIN top USING (problem) ORDER BY problem, month",
    )
    out["signals"] = _rows(con, SIGNAL_SQL)

    con.execute(DEDUP_TABLE_SQL, {"pool": str(pool_file(code)), "k": DEMO_OTHERS_PER_BLOCK})
    q = {"q": DEDUP_Q}
    out["dedup_seconds"], out["dedup_usage"] = _timed(con, DUPS_SQL, q, what, args.dry_run)
    out["dedup"] = _row_dict(con, DEDUP_METRICS_SQL)
    out["dedup_rerun_usage"] = _timed(con, DEDUP_SQL, q, what, args.dry_run)[1]
    out["dedup"]["survivors"] = con.execute("SELECT count(*) FROM deduped").fetchone()[0]
    out["dedup_examples"] = _rows(con, DUP_EXAMPLES_SQL)

    con.execute(RECALLS_SQL, {"path": str(recalls_file()), "code": code})
    q = {"q": JOIN_Q}
    out["join_seconds"], out["join_usage"] = _timed(con, JOIN_SQL, q, what, args.dry_run)
    out["join"] = _row_dict(con, JOIN_METRICS_SQL)
    out["join_by_flag"] = _rows(con, JOIN_FLAG_SQL)
    out["join_top_recalls"] = _rows(con, JOIN_TOP_SQL)
    out["uncovered"] = _rows(con, UNCOVERED_SQL)

    params = {"instr": SEVERITY_INSTR, "levels": json.dumps(SEVERITY_LEVELS)}
    duckjev.usage(reset=True)
    t0 = time.perf_counter()
    try:
        out["topk"] = _rows(con, TOPK_SQL, params)
    finally:
        use = duckjev.usage(reset=True)
        if not args.dry_run and use["requests"]:
            record_spend(what, use)
    out["topk_seconds"], out["topk_usage"] = time.perf_counter() - t0, use
    duckjev.flush(con)
    usages = ("judge_usage", "dedup_usage", "dedup_rerun_usage", "join_usage", "topk_usage")
    out["usd"] = sum(out[k]["est_usd"] for k in usages)
    print(json.dumps(out, indent=1, default=str)[:6000])
    if not args.dry_run:
        runs = load_runs()
        runs[what] = out
        save_runs(runs)
        print(f"recorded {what} in {_rel(RUNS_FILE)}")
    else:
        _write_json(DATA / f"summary_{tag}.json", out)
    print(f"benchmark spend so far: ${total_spend(load_runs()):.4f} of ${TOTAL_BUDGET_USD:.2f}")
    return 0


# --------------------------------------------------------------------------- report


def chosen_round(runs: dict[str, Any]) -> str:
    """The selectable round with the best pooled dev top-1 in set; ties to fewer tokens."""
    dev = [
        (r["pooled"]["top1_in_set"], -r["input_tokens_per_request"], r["round"])
        for k, r in runs.items()
        if k.startswith("dev/") and ROUNDS[r["round"]].selectable
    ]
    if not dev:
        raise SystemExit("no dev round has run")
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


def _cell(s: Any, width: int = 160) -> str:
    return str(s)[:width].replace("|", "\\|").replace("\n", " ")


def _pm(v: float | None, se: float | None) -> str:
    return "–" if v is None else f"{v:.3f} ± {se:.3f}" if se is not None else f"{v:.3f}"


def _usd_stream(r: dict[str, Any]) -> float:
    return r["input_tokens_per_report"] * STREAM_2026 * USD_PER_INPUT_TOKEN


HEADLINE_ROWS: list[tuple[str, Any]] = [
    ("reports", lambda r, m: f"{m['rows']:,}"),
    ("option-set coverage", lambda r, m: f"{m['coverage']:.3f}"),
    ("problem: top-1 in set", lambda r, m: f"**{_pm(m['top1_in_set'], m['top1_in_set_se'])}**"),
    ("problem: strict (one filed term)", lambda r, m: f"{m['strict']:.3f} ({m['strict_rows']:,})"),
    ("problem: mean set mass", lambda r, m: f"{m['set_mass']:.3f}"),
    ("problem: top-1 over all rows", lambda r, m: f"{m['top1_all_rows']:.3f}"),
    ("harm: accuracy", lambda r, m: f"**{_pm(m['harm_accuracy'], m['harm_accuracy_se'])}**"),
    ("harm: macro average over event types", lambda r, m: f"{m['harm_macro']:.3f}"),
    ("severity: AUROC, Death or Injury vs Malfunction", lambda r, m: _fmt(m["severity_auroc"])),
    (
        "ECE, problem, `confidence` / top-1 p",
        lambda r, m: f"{m['problem_ece_confidence']:.3f} / {m['problem_ece_top_p']:.3f}",
    ),
    (
        "ECE, harm, `confidence` / top-1 p",
        lambda r, m: f"{m['harm_ece_confidence']:.3f} / {m['harm_ece_top_p']:.3f}",
    ),
    (
        "input tokens per request; requests per report",
        lambda r, m: f"{r['input_tokens_per_request']:,.0f}; {r['requests_per_report']:.0f}",
    ),
    ("cost", lambda r, m: f"**${r['usd_per_1k_reports']:.3f} per 1,000 reports**"),
    (
        "scenario: 2.5M reports at this sample's mean input tokens",
        lambda r, m: f"${_usd_stream(r):,.0f}",
    ),
    (
        "throughput",
        lambda r, m: (
            f"{r['reports_per_second']:.0f} reports/s ({r['usage']['rate_limited']:,} "
            f"× 429, concurrency {r['concurrency']})"
        ),
    ),
    (
        "cache re-run",
        lambda r, m: (
            f"{r['rerun_seconds']:.2f} s, {r['rerun_usage']['requests']} requests, "
            f"{r['rerun_identical_rows']:,} / {r['reports']:,} rows identical"
        ),
    ),
]


def _frozen_section(runs: dict[str, Any], ids: dict[str, Any], best: str) -> list[str]:
    """Issue #10: R0 and the chosen round again, on frozen content with train-only options."""
    dev = [runs[f"dev/{r}"] for r in FROZEN if f"dev/{r}" in runs]
    test = [runs[k] for k in ("test/R0", f"test/{best}", "test/F0", "test/F3") if k in runs]
    if not dev and "reading_frozen" not in runs:
        return []
    frozen = load_json(FROZEN_FILE) if FROZEN_FILE.exists() else {}
    changed = {
        c: [t for t, _ in ids["codes"][c]["options_train"]]
        != [t for t, _ in ids["codes"][c]["options"]]
        for c in CODES
    }
    parts = [
        "## Frozen re-evaluation (issue #10)",
        "",
        "The runs above use candidate lists counted over the whole eligible pool, test labels "
        "included, and report content as openFDA served it at each pull. This section reruns "
        f"R0 and {best} as F0 and F3 with two changes and nothing else: each code's options "
        "are counted over its eligible pool without the test reports (`options_train` in "
        "`bench/maude_ids.json`), and every split report's content is frozen by hash "
        f"(`bench/maude_frozen.json`, {frozen.get('frozen_at', '?')}); a frozen run refuses a "
        "report whose description, narratives, device names or filed labels differ. Options: "
        + ", ".join(
            f"{len(ids['codes'][c]['options_train'])} for {c}"
            + ("" if changed[c] else " (same list)")
            for c in CODES
        )
        + ". F3 is fixed as R3's configuration in advance, not reselected; its own reading was "
        "recorded before its held-out runs, and its spend is on a separate ledger "
        f"(`spend_frozen`, ${total_spend(runs, 'spend_frozen'):.3f} of "
        f"${FROZEN_BUDGET_USD:.2f}). The PR #8 numbers stay as recorded: the original, "
        "test-aware run.",
        "",
    ]
    if dev:
        parts += [
            "Dev, top-1 in set:",
            "",
            _table(
                ["round", "pooled", *CODES, "harm accuracy", "tokens / request"],
                [
                    [
                        r["round"],
                        f"{r['pooled']['top1_in_set']:.3f}",
                        *[f"{r['codes'][c]['top1_in_set']:.3f}" for c in CODES],
                        f"{r['pooled']['harm_accuracy']:.3f}",
                        f"{r['input_tokens_per_request']:,.0f}",
                    ]
                    for r in [runs[k] for k in ("dev/R0", f"dev/{best}") if k in runs] + dev
                ],
            ),
            "",
        ]
    if "reading_frozen" in runs:
        rd = runs["reading_frozen"]
        parts += [
            f"Reading, recorded by hand on {rd['written']}, before the frozen held-out runs:",
            "",
            rd["text"],
            "",
        ]
    if any(r["round"] in FROZEN for r in test):
        parts += [
            f"Held-out test split, pooled, PR #8's R0 and {best} beside F0 and F3:",
            "",
            _headline_table(test),
            "",
            "Per code, top-1 in set:",
            "",
            _table(
                ["round", *CODES],
                [
                    [
                        r["round"],
                        *[
                            _pm(r["codes"][c]["top1_in_set"], r["codes"][c]["top1_in_set_se"])
                            for c in CODES
                        ],
                    ]
                    for r in test
                ],
            ),
            "",
        ]
    return parts


def _headline_table(runs: list[dict[str, Any]], scope: str | None = None) -> str:
    def m(r: dict[str, Any]) -> dict[str, Any]:
        return r["pooled"] if scope is None else r["codes"][scope]

    header = ["metric"] + [f"{r['round']}: {r['round_config']['note'].split(':')[0]}" for r in runs]
    return _table(header, [[label] + [f(r, m(r)) for r in runs] for label, f in HEADLINE_ROWS])


def _codes_table(r: dict[str, Any], base: dict[str, Any] | None) -> str:
    rows = []
    for code in [*CODES, None]:
        m = r["pooled"] if code is None else r["codes"][code]
        b = None if base is None else base["pooled"] if code is None else base["codes"][code]
        rows.append(
            [
                code or "pooled",
                f"{m['rows']:,}",
                m["coverage"],
                "–" if b is None else f"{b['top1_in_set']:.3f}",
                f"**{m['top1_in_set']:.3f}**",
                m["strict"],
                m["set_mass"],
                m["harm_accuracy"],
                m["harm_macro"],
                m["severity_auroc"],
                m["problem_ece_confidence"],
                m["harm_ece_confidence"],
            ]
        )
    return _table(
        [
            "code",
            "reports",
            "coverage",
            "R0 top-1 in set",
            f"{r['round']} top-1 in set",
            "strict",
            "set mass",
            "harm acc.",
            "harm macro",
            "severity AUROC",
            "problem ECE",
            "harm ECE",
        ],
        rows,
    )


def _rounds_table(runs: list[dict[str, Any]]) -> str:
    rows = []
    for r in runs:
        m = r["pooled"]
        rows.append(
            [
                r["round"] + ("" if r["round_config"]["selectable"] else " (check)"),
                f"**{m['top1_in_set']:.3f}**",
                *[f"{r['codes'][c]['top1_in_set']:.3f}" for c in CODES],
                m["strict"],
                m["set_mass"],
                m["coverage"],
                m["harm_accuracy"],
                m["harm_macro"],
                m["problem_ece_confidence"],
                m["harm_ece_confidence"],
                f"{r['input_tokens_per_report']:,.0f}",
                f"${r['usd_per_1k_reports']:.3f}",
                f"{r['reports_per_second']:.0f}",
            ]
        )
    return _table(
        [
            "round",
            "top-1 in set",
            *[f"{c}" for c in CODES],
            "strict",
            "set mass",
            "coverage",
            "harm acc.",
            "harm macro",
            "problem ECE",
            "harm ECE",
            "tokens / report",
            "$ / 1k",
            "reports / s",
        ],
        rows,
    )


def _reliability_table(rows: list[list[Any]]) -> str:
    return _table(
        ["bin", "n", "mean confidence", "accuracy"],
        [[f"{b / 10:.1f}–{(b + 1) / 10:.1f}", n, mc, acc] for b, n, mc, acc in rows],
    )


def _deferral_table(m: dict[str, Any]) -> str:
    rows = []
    for (t, pa_, aa), (_, ph, ah) in zip(m["problem_deferral"], m["harm_deferral"], strict=True):
        rows.append([f"{t:.2f}", pa_, aa, ph, ah])
    return _table(
        [
            "confidence ≥",
            "problem: share answered",
            "problem: top-1 in set",
            "harm: share answered",
            "harm: accuracy",
        ],
        rows,
    )


def _harm_table(r: dict[str, Any]) -> str:
    rows = []
    for code in CODES:
        for event_type, n, acc, sev in r["codes"][code]["harm_by_type"]:
            rows.append([code, event_type, n, acc, sev])
    return _table(["code", "filed event type", "reports", "harm accuracy", "mean severity"], rows)


def _matrix_table(m: dict[str, Any]) -> str:
    cells = {(g, p): n for g, p, n in m["harm_matrix"]}
    golds = [h for h in HARMS if any(g == h for g, _ in cells)]
    return _table(
        ["filed \\ answered", *HARMS],
        [[g, *[cells.get((g, h), 0) for h in HARMS]] for g in golds],
    )


SIGNALS_SHOWN = 8


def _trend_pivot(rows: list[list[Any]]) -> str:
    """Problems as rows, months as columns: expected ± SE [argmax, filed]."""
    months = sorted({str(r[1])[:7] for r in rows})
    cells: dict[str, dict[str, str]] = {}
    for p, mo, e, se, a, f in rows:
        cells.setdefault(p, {})[str(mo)[:7]] = f"{e:.1f} ± {se:.1f} [{a}, {f}]"
    return _table(
        ["problem", *months],
        [[p, *[c.get(m, "–") for m in months]] for p, c in cells.items()],
    )


def _demo_section(runs: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    for key in sorted(k for k in runs if k.startswith("demo/")):
        d = runs[key]
        code = d["code"]
        dd, j = d["dedup"], d["join"]
        signals = d["signals"]
        parts += [
            f"## The demo queries live: {code}, {d['sample_rows']} sampled test reports",
            "",
            f"Round {d['round']}'s questions, one fused request per report "
            f"({d['judge_usage']['requests']} requests, ${d['judge_usage']['est_usd']:.4f}, "
            f"{d['judge_seconds']:.1f} s); the answers unpack into `problem`, `harm` and "
            "`severity` struct columns of `judged`. Every query below is the SQL of MAUDE.md §4 "
            "over that table.",
            "",
            "### 1. The trend table: expected reports per problem and month",
            "",
            f"`expected_count` and `expected_count_stderr` over the exploded problem "
            f"distribution, {d['trend_rows']} (problem, month) rows. The five problems with the "
            "most expected reports, per month: expected ± SE, and in brackets the argmax count "
            "and the count the manufacturers filed.",
            "",
            _trend_pivot(d["trend_top"]),
            "",
            "### 2. Signal months",
            "",
            (
                f"{len(signals)} (problem, month) rows have an expected count more than two "
                "standard errors above the trailing three-month mean; the largest excesses:"
                if signals
                else "No month's expected count sits more than two standard errors above its "
                "trailing three-month mean on this sample."
            ),
            "",
        ]
        if signals:
            parts += [
                _table(
                    ["problem", "month", "expected ± SE", "trailing mean ± SE"],
                    [
                        [p, str(mo)[:7], f"{e:.1f} ± {se:.1f}", f"{t:.1f} ± {ts:.1f}"]
                        for _, p, mo, e, se, t, ts in signals[:SIGNALS_SHOWN]
                    ],
                ),
                "",
                "The standard error is the Bernoulli error of the calibrated count: it covers "
                "the uncertainty in each report's coding, not the month-to-month variation in "
                "how many reports arrive. On a sample this size a month that goes from no "
                "reports of a problem to one confident report also clears the bar; the "
                "excesses worth a reviewer's time are the large ones at the top.",
                "",
            ]
        parts += [
            "### 3. One event, one row: `sem_dedup`",
            "",
            f"Blocks are `product_code || '/' || date_of_event`: the {dd['sampled_rows_in']} "
            f"sampled reports with an event date plus at most {DEMO_OTHERS_PER_BLOCK} other "
            f"reports of the same code and event date from the pool, {dd['rows_in']} rows in "
            f"{dd['blocks']} blocks. Every pair inside a block is judged: {dd['pairs_judged']:,} "
            f"pairs, {dd['duplicate_pairs']} at p ≥ 0.5, Σp = {dd['expected_pairs']:.1f} ± "
            f"{dd['expected_pairs_se']:.1f}; `sem_dedup` keeps {dd['survivors']} of "
            f"{dd['rows_in']} rows. {d['dedup_usage']['requests']:,} requests, "
            f"${d['dedup_usage']['est_usd']:.4f} (the `sem_dedup` pass after `sem_dups` made "
            f"{d['dedup_rerun_usage']['requests']} requests: the cache served every pair).",
            "",
        ]
        if d["dedup_examples"]:
            parts += [
                "The pairs judged most likely the same incident:",
                "",
                _table(
                    ["p", "report", "description (cut)", "report", "description (cut)"],
                    [[p, a, _cell(ta), b, _cell(tb)] for a, b, p, ta, tb in d["dedup_examples"]],
                ),
                "",
            ]
        parts += [
            "### 4. Recall coverage: `sem_join` to the recalls on file",
            "",
            f"Every sampled report against every one of the {j['recalls']} recall records "
            f"openFDA holds for {code}: {j['pairs']:,} pairs, {j['matched_pairs']} at p ≥ 0.5 "
            f"covering {j['reports_matched']} reports and {j['recalls_matched']} recalls; the "
            f"calibrated size of the covered set is Σp = {j['expected_pairs']:.1f} ± "
            f"{j['expected_pairs_se']:.1f} pairs. {d['join_usage']['requests']:,} requests, "
            f"${d['join_usage']['est_usd']:.4f}, {d['join_seconds']:.1f} s.",
            "",
            "The one check the data offers for free: reports whose `remedial_action` includes "
            "Recall against the rest.",
            "",
            _table(
                ["report flags Recall", "reports", "matched a recall at 0.5", "mean max p"],
                [[f, n, mr, mp] for f, n, mr, mp in d["join_by_flag"]],
            ),
            "",
            "The recalls with the most expected matching reports:",
            "",
            _table(
                ["recall", "initiated", "recall text (cut)", "reports at 0.5", "Σp"],
                [[r, i, _cell(t, 120), m, e] for r, i, t, m, e in d["join_top_recalls"]],
            ),
            "",
            "The complement, problem clusters no recall on file addresses (argmax problem, "
            "reports, reports with no recall match at 0.5):",
            "",
            _table(["problem", "reports", "without a recall"], d["uncovered"]),
            "",
            "### 5. What a reviewer reads first: `sem_topk` by severity",
            "",
            f"{d['topk_usage']['requests']} requests, ${d['topk_usage']['est_usd']:.4f}. The "
            "twenty highest expected severity levels (0 = no harm, 4 = death):",
            "",
            _table(
                [
                    "report",
                    "brand",
                    "filed event type",
                    "severity",
                    "confidence",
                    "description (cut)",
                ],
                [[k, _cell(b, 40), e, s, c, _cell(t, 120)] for k, b, e, s, c, t in d["topk"]],
            ),
            "",
            f"The demo cost ${d['usd']:.4f} in all.",
            "",
        ]
    return parts


def report(args: argparse.Namespace) -> int:
    runs = load_runs()
    ids = load_ids()
    dev = [runs[k] for k in sorted(runs) if k.startswith("dev/R")]
    if not dev:
        raise SystemExit("no dev round has run")
    best = chosen_round(runs)
    base, final = runs.get("test/R0"), runs.get(f"test/{best}")
    pull = ids.get("pull", {})
    first = dev[0]
    parts = [
        "# MAUDE: coded complaint surveillance on FDA device adverse events",
        "",
        "Generated by `bench/maude.py report` from the run log, committed IDs and "
        "criteria; do not edit "
        f"by hand. Measured numbers below are from live runs against `{first['model']}`; "
        "the whole-stream cost is an extrapolation.",
        "",
        "**Corpus.** Medical device adverse event reports from openFDA `device/event` "
        f"(MAUDE), `date_received` {WINDOW[0][:4]}-{WINDOW[0][4:6]}-{WINDOW[0][6:]} to "
        f"{WINDOW[1][:4]}-{WINDOW[1][4:6]}-{WINDOW[1][6:]}, for three product codes: "
        + "; ".join(
            f"{c} ({CODES[c].name}, "
            + ("the 8th and 22nd of each month, " if CODES[c].sampled else "")
            + f"{pull.get(c, {}).get('records', 0):,} reports pulled, "
            f"{ids['codes'][c]['pool']:,} eligible)"
            for c in CODES
        )
        + ". A report is eligible when its Description of Event or Problem has at least "
        f"{MIN_DESCRIPTION} characters and its event type is one of {', '.join(HARMS)}. Per "
        f"code, dev is {SPLIT_SIZES['dev']} reports and test {SPLIT_SIZES['test']:,}, drawn by "
        "a fixed hash rule stratified by event type with a floor of "
        f"{FLOOR} per event type where that many remain; a disjoint gloss slice of "
        f"{SPLIT_SIZES['gloss']} per code is the only source of the criteria's examples. "
        "The keys are committed in `bench/maude_ids.json`.",
        "The option frequencies and order were computed from the full eligible pool before "
        "the split, including the test reports' filed terms. The reported test rows were "
        "held out from round selection, but the candidate lists are test-aware. The R2 "
        "phrases and examples were written from the gloss slice; which confusions they "
        "target was read from the dev rounds, and the manufacturer conventions behind them "
        "were cross-checked against pool-wide term counts that include the test reports.",
        "",
        "**What is asked.** One fused `jev(description, $questions)` request per report: "
        "`problem`, a Choice over the product code's option set (its "
        f"{OPTIONS_PER_CODE} most frequent filed device-problem terms with at least "
        f"{MIN_TERM_REPORTS} reports in the pool, plus a catch-all `{CATCH_ALL}`: "
        + ", ".join(f"{len(ids['codes'][c]['options'])} for {c}" for c in CODES)
        + "); `harm`, a Choice over FDA's event types with glosses from 21 CFR 803.3; and "
        "`severity`, a Score over five levels from no harm to death. R3's object state is a JSON "
        "object sent as text: `jev()` takes a VARCHAR state, so Jev reads the serialized object.",
        "",
        "**How it is measured.** Against the codes the manufacturers filed. *Top-1 in set*: "
        "the argmax is one of the report's filed problem terms (a report may carry several), "
        "over the reports that have a filed term in the option set; *coverage* is the share "
        "that do. *Strict*: exact match on reports with exactly one filed term. *Set mass*: "
        "the mean Σp over the filed terms, which tracks top-1 in set when the distribution is "
        "calibrated. *Harm accuracy*: argmax equals the filed event type; the *macro average* "
        "weighs each event type present equally, since Malfunction dominates QBJ and Injury "
        "dominates FTR. *Severity AUROC* ranks Death or Injury above Malfunction by the "
        "expected severity level. ECE uses 10 equal-width bins. The chosen round is the "
        "selectable round with the best pooled dev top-1 in set, ties to fewer tokens per "
        f"request: **{best}**. The test split ran only with R0 and {best}.",
        "",
        "The whole-stream cost row applies the three-code test sample's average input tokens "
        "per report to all 2.5 million 2026 MAUDE reports. It is a scaling scenario, not "
        "a measured cost estimate for untested product codes.",
        "",
    ]
    if final and base:
        parts += [
            "## Held-out test split",
            "",
            f"Pooled over the three codes, {final['pooled']['rows']:,} reports:",
            "",
            _headline_table([base, final]),
            "",
            f"### Per product code, round {best}",
            "",
            _codes_table(final, base),
            "",
            f"### Harm by filed event type, round {best}",
            "",
            _harm_table(final),
            "",
            "Pooled, filed event type against the answer:",
            "",
            _matrix_table(final["pooled"]),
            "",
            f"### Calibration, round {best}, pooled",
            "",
            "Problem (over covered reports; accuracy is top-1 in set), by `confidence`:",
            "",
            _reliability_table(final["pooled"]["problem_reliability_confidence"]),
            "",
            "Harm, by `confidence`:",
            "",
            _reliability_table(final["pooled"]["harm_reliability_confidence"]),
            "",
            f"### Deferral, round {best}, pooled",
            "",
            "Answer only above a confidence threshold and route the rest to a person:",
            "",
            _deferral_table(final["pooled"]),
            "",
            f"### What round {best} still gets wrong",
            "",
            "The most frequent problem confusions on covered reports, filed term(s) against "
            "the argmax, per code:",
            "",
            _table(
                ["code", "filed", "answered", "reports"],
                [[c, f, p, n] for c in CODES for f, p, n in final["codes"][c]["confusions"][:5]],
            ),
            "",
            f"### Expected against filed counts, round {best}",
            "",
            "Per code, the ten most filed terms: reports that carry the term, reports whose "
            "argmax is the term, and Σp with its Bernoulli standard error.",
            "",
            _table(
                ["code", "term", "filed", "argmax", "Σp ± SE"],
                [
                    [c, t, f, a, f"{e:.1f} ± {se:.1f}"]
                    for c in CODES
                    for t, f, a, e, se in final["codes"][c]["counts"]
                ],
            ),
            "",
        ]
    parts += [
        "## Tuning rounds on the dev split",
        "",
        f"{first['pooled']['rows']:,} dev reports; top-1 in set pooled and per code, the "
        "rest pooled.",
        "",
        _rounds_table(dev),
        "",
        *[f"- **{r['round']}**: {r['round_config']['note']}." for r in dev],
        "",
        "The historical R2 note says one example per option; the actual R2 question "
        "maps carry examples in "
        + ", ".join(
            f"{c} {sum('examples' in d for d in q.values())}/{len(q)}"
            for c in CODES
            for q in [questions_for(ROUNDS["R2"], c)["problem"]["criteria"]]
        )
        + " problem options (including the catch-all).",
        "",
    ]
    r5 = runs.get("dev/R5")
    if r5:
        parts += [
            "R5's option set is every term seen on the corpus, so its coverage and its "
            "covered rows differ from the other rounds; on the rows the per-code set covers, "
            f"R5's top-1 in set is {r5['pooled']['top1_on_code_rows']:.3f}.",
            "",
        ]
    if "reading" in runs:
        rd = runs["reading"]
        parts += [
            "## Reading the numbers",
            "",
            f"Written by hand from the dev rounds and recorded in the run log on {rd['written']}, "
            "before the held-out runs: `run --split test` refuses to start until the reading "
            "is recorded, and `reading` refuses once a held-out run exists. Reproduced here "
            "unchanged.",
            "",
            rd["text"],
            "",
        ]
    parts += _frozen_section(runs, ids, best)
    parts += _demo_section(runs)
    ex = questions_for(ROUNDS[best], "QBJ")
    parts += [
        "## The questions",
        "",
        f"Problem instructions: `{PROBLEM_INSTR}`",
        "",
        f"Harm instructions: `{HARM_INSTR}`; options and glosses:",
        "",
        *[f"- `{h}`: {g}" for h, g in HARM_CRITERIA.items()],
        "",
        f"Severity instructions: `{SEVERITY_INSTR}`; levels 0 to 4: "
        + "; ".join(f"`{lv}`" for lv in SEVERITY_LEVELS)
        + ".",
        "",
        f"Round {best}'s first three QBJ problem options as sent:",
        "",
        "```json",
        json.dumps(dict(list(ex["problem"]["criteria"].items())[:3]), indent=1),
        "```",
        "",
        "The judged SQL of a fused round (`questions` holds each product code's question map):",
        "",
        "```sql",
        JUDGE_SQL[True].format(table="judged", state=STATE_COLUMN[ROUNDS[best].state]),
        "```",
        "",
        "## Historical run sequence",
        "",
        "These commands describe the original paid runs. They cannot be replayed against "
        "the committed run log: its spend ledger already consumes most of the configured "
        "budget, and a new ledger would also require fresh pre-flights and a new reading "
        "before test runs. Do not execute them without a separate ledger and spend approval.",
        "",
        "```bash",
        "uv run python bench/maude.py prepare",
        *[
            f"uv run python bench/maude.py run {runs[k]['round']} --split {runs[k]['split']}"
            for k in sorted(recorded_runs(runs), key=lambda k: runs[k]["timestamp"])
        ],
        *[
            f"uv run python bench/maude.py demo --code {k.split('/')[1]}"
            for k in sorted(runs)
            if k.startswith("demo/")
        ],
        "uv run python bench/maude.py report",
        "```",
        "",
        "## Spend",
        "",
        f"Total live spend for the benchmark: **${total_spend(runs):.3f}** over "
        f"{len(runs.get('spend', []))} live calls (pre-flights, dev rounds, held-out runs and "
        f"the demo), against a configured ${TOTAL_BUDGET_USD:.2f} token guard. "
        "Concurrent in-flight requests can overshoot that guard.",
        "",
    ]
    text = "\n".join(parts)
    if args.dry_run:
        print(text)
    else:
        RESULTS.write_text(text)
        print(f"wrote {_rel(RESULTS)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    fz = sub.add_parser("freeze")
    fz.add_argument("--refreeze", action="store_true", help="rewrite the frozen hashes and options")
    pr = sub.add_parser("prepare")
    pr.add_argument("--from-fixture", help="openFDA-shaped JSON instead of the network (tests)")
    pr.add_argument("--resample", action="store_true", help="redraw bench/maude_ids.json")
    pr.add_argument("--dry-run", action="store_true", help="count what a pull would fetch")
    r = sub.add_parser("run")
    r.add_argument("round", choices=list(ROUNDS))
    r.add_argument("--split", default="dev", choices=SPLITS)
    r.add_argument("--limit", type=int, default=None, help="pre-flight sample size")
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--max-usd", type=float, default=0.50, help="input-token guard for this run")
    r.add_argument("--dry-run", action="store_true", help="fake transport, no network")
    c = sub.add_parser("confusions")
    c.add_argument("round", choices=list(ROUNDS))
    c.add_argument("--split", default="dev", choices=SPLITS)
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--code", default=None, choices=list(CODES))
    c.add_argument("--top", type=int, default=15)
    c.add_argument("--show", type=int, default=0, help="print the reports behind the top N")
    c.add_argument("--dry-run", action="store_true", help="read the dry run's rows")
    rd = sub.add_parser("reading")
    rd.add_argument("--file", required=True)
    rd.add_argument("--frozen", action="store_true", help="the frozen re-evaluation's reading")
    rd.add_argument("--dry-run", action="store_true", help="print, record nothing")
    d = sub.add_parser("demo")
    d.add_argument("--code", default="QBJ", choices=list(CODES))
    d.add_argument("--n", type=int, default=DEMO_ROWS, help="sampled test reports")
    d.add_argument("--round", default=None, choices=list(ROUNDS), help="default: the chosen one")
    d.add_argument("--concurrency", type=int, default=16)
    d.add_argument("--max-usd", type=float, default=0.50)
    d.add_argument("--dry-run", action="store_true")
    rs = sub.add_parser("rescore", help="rebuild recorded runs from their answer caches")
    rs.add_argument("--dry-run", action="store_true", help="print, write nothing")
    rp = sub.add_parser("report")
    rp.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = p.parse_args(argv)
    commands = {
        "prepare": prepare,
        "freeze": freeze,
        "run": run,
        "confusions": confusions,
        "reading": reading,
        "demo": demo,
        "rescore": rescore,
        "report": report,
    }
    return commands[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
