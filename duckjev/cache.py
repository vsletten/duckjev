"""Content-addressed answer cache.

The cache lives in its own DuckDB file on its own connection, never on the
user's query connection: a UDF must not write to the connection that is
running the query that called it. Reads are served from an in-process dict;
writes go to both the dict and the file under a lock.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

DEFAULT_CACHE_PATH = Path.home() / ".cache" / "duckjev" / "cache.duckdb"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jev_cache (
    key VARCHAR PRIMARY KEY,
    model VARCHAR,
    answers VARCHAR,
    input_tokens BIGINT,
    output_tokens BIGINT,
    created_at DOUBLE
)
"""


def canonical_json(obj: Any) -> str:
    """Compact JSON in the order the objects were built, which is the order sent to Jev.

    Keys are deliberately not sorted: option order in a Choice, level order in a Score
    and field order in an object state all reach the model and can change its answer,
    so two requests that differ only in order must not share a cache entry.
    """
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def cache_key(model: str, state: Any, questions: dict[str, Any]) -> str:
    """sha256 of the canonical JSON of (model, state, questions), order included."""
    payload = canonical_json({"model": model, "state": state, "questions": questions})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CacheEntry(dict):
    """``{"model": str, "answers": {...}, "usage": {...}, "ts": float}``."""


class AnswerCache:
    """Answer cache backed by a DuckDB file, or memory only when ``path`` is None."""

    def __init__(self, path: str | Path | None = DEFAULT_CACHE_PATH) -> None:
        self.path = Path(path).expanduser() if path is not None else None
        self._lock = threading.Lock()
        self._mem: dict[str, CacheEntry] = {}
        self._con: duckdb.DuckDBPyConnection | None = None
        self._opened = False

    # The file is opened lazily so register() never touches the filesystem.
    def _open(self) -> None:
        if self._opened:
            return
        self._opened = True
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            con = duckdb.connect(str(self.path))
            con.execute(_SCHEMA)
            rows = con.execute(
                "SELECT key, model, answers, input_tokens, output_tokens, created_at FROM jev_cache"
            ).fetchall()
        except (duckdb.Error, OSError) as exc:
            warnings.warn(
                f"duckjev: cache file {self.path} unavailable ({exc}); using memory only",
                stacklevel=3,
            )
            return
        self._con = con
        for key, model, answers, tin, tout, ts in rows:
            self._mem[key] = CacheEntry(
                model=model,
                answers=json.loads(answers),
                usage={"input_tokens": tin, "output_tokens": tout},
                ts=ts,
            )

    def get_many(self, keys: Iterable[str]) -> dict[str, CacheEntry]:
        with self._lock:
            self._open()
            return {k: self._mem[k] for k in keys if k in self._mem}

    def put_many(self, entries: dict[str, tuple[str, dict[str, Any], dict[str, Any]]]) -> None:
        """Store ``key -> (model, answers, usage)``."""
        if not entries:
            return
        now = time.time()
        with self._lock:
            self._open()
            for key, (model, answers, usage) in entries.items():
                self._mem[key] = CacheEntry(model=model, answers=answers, usage=usage, ts=now)
            if self._con is None:
                return
            table = pa.table(
                {
                    "key": list(entries),
                    "model": [m for m, _, _ in entries.values()],
                    "answers": [canonical_json(a) for _, a, _ in entries.values()],
                    "input_tokens": [int(u.get("input_tokens", 0)) for _, _, u in entries.values()],
                    "output_tokens": [
                        int(u.get("output_tokens", 0)) for _, _, u in entries.values()
                    ],
                    "created_at": [now] * len(entries),
                }
            )
            self._con.register("_duckjev_new", table)
            try:
                self._con.execute("INSERT OR REPLACE INTO jev_cache SELECT * FROM _duckjev_new")
            finally:
                self._con.unregister("_duckjev_new")

    def __len__(self) -> int:
        with self._lock:
            self._open()
            return len(self._mem)

    def flush(self) -> None:
        with self._lock:
            if self._con is not None:
                self._con.execute("CHECKPOINT")

    def to_arrow(self) -> pa.Table:
        with self._lock:
            self._open()
            items = list(self._mem.items())
        return pa.table(
            {
                "key": pa.array([k for k, _ in items], pa.string()),
                "model": pa.array([e["model"] for _, e in items], pa.string()),
                "answers": pa.array([canonical_json(e["answers"]) for _, e in items], pa.string()),
                "input_tokens": pa.array(
                    [int(e["usage"].get("input_tokens", 0)) for _, e in items], pa.int64()
                ),
                "output_tokens": pa.array(
                    [int(e["usage"].get("output_tokens", 0)) for _, e in items], pa.int64()
                ),
                "created_at": pa.array([float(e["ts"]) for _, e in items], pa.float64()),
            }
        )

    def close(self) -> None:
        with self._lock:
            if self._con is not None:
                self._con.close()
                self._con = None
