"""duckjev: semantic operators for DuckDB backed by TypeSafe's Jev.

Usage: ``duckjev.register(con)`` installs ``jev()``, ``jev_noul()``,
``jev_choice()``, ``jev_score()``, ``jev_extract()``, the ``sem_where`` /
``expected_count`` / candidate-builder macros and the answer cache on a DuckDB
connection.
"""

from __future__ import annotations

import threading
from importlib.resources import files
from pathlib import Path
from typing import Any

import duckdb
import httpx

from .cache import DEFAULT_CACHE_PATH, AnswerCache, cache_key
from .client import (
    DEFAULT_MODEL,
    JevAPIError,
    JevAuthError,
    JevBudgetExceeded,
    JevClient,
    JevError,
    JevTransportError,
)
from .client import USAGE as _USAGE
from .functions import register_udfs
from .marshal import JevQuestionError

__version__ = "0.2.0"

__all__ = [
    "JevAPIError",
    "JevAuthError",
    "JevBudgetExceeded",
    "JevClient",
    "JevError",
    "JevQuestionError",
    "JevTransportError",
    "cache_key",
    "cache_table",
    "client_for",
    "flush",
    "register",
    "usage",
]

_lock = threading.Lock()
_clients: dict[int, JevClient] = {}
_caches: dict[str, AnswerCache] = {}


def _macro_statements() -> list[str]:
    text = files("duckjev").joinpath("macros.sql").read_text(encoding="utf-8")
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("--"))
    return [s.strip() for s in body.split(";") if s.strip()]


def _shared_cache(path: str | Path) -> AnswerCache:
    """One AnswerCache (and one DuckDB connection) per cache file in this process."""
    resolved = str(Path(path).expanduser().resolve())
    with _lock:
        if resolved not in _caches:
            _caches[resolved] = AnswerCache(resolved)
        return _caches[resolved]


def register(
    con: duckdb.DuckDBPyConnection,
    *,
    model: str = DEFAULT_MODEL,
    concurrency: int = 16,
    cache: bool = True,
    cache_path: str | Path | None = None,
    max_input_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> JevClient:
    """Install the Jev SQL functions, macros and answer cache on ``con``.

    The API key defaults to ``$TYPESAFE_API_KEY`` and is only required at the
    first call that actually needs the network, so registering and running
    macros or fully cached queries work offline. ``transport`` is for tests.
    Returns the :class:`JevClient` backing this connection.
    """
    answer_cache = _shared_cache(cache_path or DEFAULT_CACHE_PATH) if cache else None
    client = JevClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        concurrency=concurrency,
        cache=answer_cache,
        max_input_tokens=max_input_tokens,
        transport=transport,
    )
    register_udfs(con, client)
    for stmt in _macro_statements():
        con.execute(stmt)
    with _lock:
        old = _clients.pop(id(con), None)
        _clients[id(con)] = client
    if old is not None:
        old.close()
    return client


def client_for(con: duckdb.DuckDBPyConnection) -> JevClient:
    """The client ``register()`` installed on ``con``."""
    try:
        return _clients[id(con)]
    except KeyError:
        raise JevError("duckjev.register(con) has not been called on this connection") from None


def usage(reset: bool = False) -> dict[str, Any]:
    """Process-wide usage: requests, tokens, cache hits/misses, 429s, est_usd."""
    return _USAGE.snapshot(reset=reset)


def flush(con: duckdb.DuckDBPyConnection | None = None) -> None:
    """Checkpoint the answer cache file. Safe to call at any time, including with no cache."""
    if con is None:
        clients = list(_clients.values())
    else:
        clients = [c] if (c := _clients.get(id(con))) is not None else []
    for c in clients:
        if c.cache is not None:
            c.cache.flush()


def cache_table(con: duckdb.DuckDBPyConnection, name: str = "jev_cache") -> str:
    """Copy the answer cache into table ``name`` on ``con`` for SQL inspection."""
    client = client_for(con)
    if client.cache is None:
        raise JevError("the answer cache is disabled on this connection")
    arrow = client.cache.to_arrow()
    con.register("_duckjev_cache_view", arrow)
    try:
        quoted = '"' + name.replace('"', '""') + '"'
        con.execute(
            f"CREATE OR REPLACE TABLE {quoted} AS "
            "SELECT key, model, answers::JSON AS answers, input_tokens, output_tokens, "
            "to_timestamp(created_at) AS created_at FROM _duckjev_cache_view"
        )
    finally:
        con.unregister("_duckjev_cache_view")
    return name
