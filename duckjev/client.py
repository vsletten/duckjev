"""Jev HTTP client: in-batch dedupe, cache, bounded concurrent fan-out, backoff, usage.

The one public entry point is :meth:`JevClient.judge`, which is synchronous and
safe to call from a DuckDB UDF on any thread. The async fan-out runs on a
dedicated event loop in a helper thread, so it works whether or not the host
process already has a running loop (Jupyter, for example).
"""

from __future__ import annotations

import asyncio
import os
import random
import threading
from collections.abc import Awaitable, Sequence
from concurrent.futures import Future
from typing import Any

import httpx

from .cache import AnswerCache, cache_key

DEFAULT_MODEL = "jev-1.13.0"
DEFAULT_BASE_URL = "https://api.typesafe.ai"
ENDPOINT = "/v1/systemone"
API_KEY_ENV = "TYPESAFE_API_KEY"

#: $42 per billion input tokens; output tokens are free.
USD_PER_INPUT_TOKEN = 42 / 1e9

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})

Answers = dict[str, dict[str, Any]]
Item = tuple[Any, dict[str, Any]]


class JevError(Exception):
    """Base class for duckjev errors."""


class JevAuthError(JevError):
    """No API key, or the API rejected it."""


class JevAPIError(JevError):
    """A non-retryable API error (for example 422 validation)."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"Jev API returned {status}: {detail}")
        self.status = status


class JevTransportError(JevError):
    """Retries were exhausted on 429/529/5xx or network errors."""


class JevBudgetExceeded(JevError):
    """Cumulative billed input tokens passed ``max_input_tokens``."""


class Usage:
    """Thread-safe usage counters, shared process-wide by default."""

    FIELDS = (
        "rows",
        "deduped",
        "cache_hits",
        "cache_misses",
        "requests",
        "input_tokens",
        "output_tokens",
        "retries",
        "rate_limited",
        "overloaded",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._c = dict.fromkeys(self.FIELDS, 0)

    def add(self, **counts: int) -> None:
        with self._lock:
            for k, v in counts.items():
                self._c[k] += v

    def snapshot(self, reset: bool = False) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = dict(self._c)
            if reset:
                self._c = dict.fromkeys(self.FIELDS, 0)
        out["est_usd"] = out["input_tokens"] * USD_PER_INPUT_TOKEN
        return out


USAGE = Usage()


class _LoopThread:
    """A private asyncio loop running forever in a daemon thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="duckjev-loop", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run[T](self, coro: Awaitable[T]) -> T:
        fut: Future[T] = asyncio.run_coroutine_threadsafe(coro, self.loop)  # type: ignore[arg-type]
        return fut.result()

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)


class JevClient:
    """Batching Jev client. ``judge(batch)`` returns one ``answers`` map per item."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        concurrency: int = 16,
        cache: AnswerCache | None = None,
        max_input_tokens: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        usage: Usage | None = None,
        timeout: float = 30.0,
        max_attempts: int = 6,
        backoff_base: float = 0.5,
        backoff_factor: float = 2.0,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.concurrency = concurrency
        self.cache = cache
        self.max_input_tokens = max_input_tokens
        self.usage = usage if usage is not None else USAGE
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_factor = backoff_factor
        self._api_key = api_key
        self._transport = transport
        self._billed = 0
        self._billed_lock = threading.Lock()
        self._loop: _LoopThread | None = None
        self._loop_lock = threading.Lock()
        self._http: httpx.AsyncClient | None = None
        self._sem: asyncio.Semaphore | None = None

    # ------------------------------------------------------------------ public

    @property
    def billed_input_tokens(self) -> int:
        return self._billed

    def key_for(self, state: Any, questions: dict[str, Any]) -> str:
        return cache_key(self.model, state, questions)

    def judge(self, batch: Sequence[Item]) -> list[Answers]:
        """Judge ``(state, questions)`` pairs; returns answers in input order.

        Identical pairs are sent once; cached pairs are not sent. If any request
        fails after retries the whole call raises (no partial fill), but answers
        already received are cached so a retry does not pay for them again.
        """
        if not batch:
            return []
        keys = [self.key_for(s, q) for s, q in batch]
        unique: dict[str, Item] = {}
        for k, item in zip(keys, batch, strict=True):
            unique.setdefault(k, item)
        self.usage.add(rows=len(batch), deduped=len(batch) - len(unique))

        found: dict[str, Answers] = {}
        if self.cache is not None:
            found = {k: e["answers"] for k, e in self.cache.get_many(unique).items()}
        misses = {k: v for k, v in unique.items() if k not in found}
        self.usage.add(cache_hits=len(found), cache_misses=len(misses))

        if misses:
            self._check_budget()
            api_key = self._resolve_key()
            fetched, error = self._loop_thread().run(self._fan_out(misses, api_key))
            if self.cache is not None and fetched:
                self.cache.put_many(
                    {k: (self.model, ans, use) for k, (ans, use) in fetched.items()}
                )
            if error is not None:
                raise error
            found.update({k: ans for k, (ans, _) in fetched.items()})
            self._check_budget()
        return [found[k] for k in keys]

    def close(self) -> None:
        if self._loop is not None:
            if self._http is not None:
                self._loop.run(self._http.aclose())
                self._http = None
            self._loop.stop()
            self._loop = None

    # ------------------------------------------------------------------ internals

    def _resolve_key(self) -> str:
        key = self._api_key or os.environ.get(API_KEY_ENV)
        if not key:
            raise JevAuthError(
                f"{API_KEY_ENV} is not set: export it or pass api_key= to duckjev.register()"
            )
        return key

    def _check_budget(self) -> None:
        if self.max_input_tokens is not None and self._billed > self.max_input_tokens:
            raise JevBudgetExceeded(
                f"billed input tokens {self._billed} passed max_input_tokens "
                f"{self.max_input_tokens}"
            )

    def _loop_thread(self) -> _LoopThread:
        with self._loop_lock:
            if self._loop is None:
                self._loop = _LoopThread()
            return self._loop

    async def _fan_out(
        self, misses: dict[str, Item], api_key: str
    ) -> tuple[dict[str, tuple[Answers, dict[str, Any]]], BaseException | None]:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self._transport,
                limits=httpx.Limits(
                    max_connections=self.concurrency, max_keepalive_connections=self.concurrency
                ),
            )
            self._sem = asyncio.Semaphore(self.concurrency)
        fetched: dict[str, tuple[Answers, dict[str, Any]]] = {}

        async def one(key: str, state: Any, questions: dict[str, Any]) -> None:
            fetched[key] = await self._post(state, questions, api_key)

        tasks = [asyncio.ensure_future(one(k, s, q)) for k, (s, q) in misses.items()]
        error: BaseException | None = None
        try:
            await asyncio.gather(*tasks)
        except BaseException as exc:  # re-raised by judge() after caching what arrived
            error = exc
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return fetched, error

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        delay = self.backoff_base * self.backoff_factor**attempt
        delay *= 0.5 + random.random()  # jitter in [0.5x, 1.5x)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        return delay

    async def _post(
        self, state: Any, questions: dict[str, Any], api_key: str
    ) -> tuple[Answers, dict[str, Any]]:
        assert self._http is not None and self._sem is not None
        payload = {"state": state, "model": self.model, "questions": questions}
        headers = {"Authorization": f"Bearer {api_key}"}
        last = "no attempt made"
        for attempt in range(self.max_attempts):
            self._check_budget()
            retry_after: str | None = None
            async with self._sem:
                try:
                    resp = await self._http.post(ENDPOINT, json=payload, headers=headers)
                except httpx.TransportError as exc:
                    last = f"{type(exc).__name__}: {exc}"
                    resp = None
            if resp is not None:
                if resp.status_code == 200:
                    return self._accept(resp, questions)
                if resp.status_code in (401, 403):
                    raise JevAuthError(f"Jev API rejected the API key ({resp.status_code})")
                if resp.status_code not in RETRY_STATUSES:
                    raise JevAPIError(resp.status_code, resp.text[:500])
                if resp.status_code == 429:
                    self.usage.add(rate_limited=1)
                elif resp.status_code == 529:
                    self.usage.add(overloaded=1)
                last = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("retry-after")
            if attempt + 1 < self.max_attempts:
                self.usage.add(retries=1)
                await asyncio.sleep(self._delay(attempt, retry_after))
        raise JevTransportError(f"Jev request failed after {self.max_attempts} attempts ({last})")

    def _accept(
        self, resp: httpx.Response, questions: dict[str, Any]
    ) -> tuple[Answers, dict[str, Any]]:
        body = resp.json()
        answers = body.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise JevAPIError(resp.status_code, "response answers do not match question ids")
        usage = body.get("usage") or {}
        tin = int(usage.get("input_tokens", 0))
        tout = int(usage.get("output_tokens", 0))
        with self._billed_lock:
            self._billed += tin
        self.usage.add(requests=1, input_tokens=tin, output_tokens=tout)
        return answers, {"input_tokens": tin, "output_tokens": tout}
