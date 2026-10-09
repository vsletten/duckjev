"""Jev HTTP client: in-batch dedupe, cache, bounded concurrent fan-out, backoff, usage.

The one public entry point is :meth:`JevClient.judge`, which is synchronous and
safe to call from a DuckDB UDF on any thread. The async fan-out runs on a
dedicated event loop in a helper thread, so it works whether or not the host
process already has a running loop (Jupyter, for example).
"""

from __future__ import annotations

import asyncio
import json
import math
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

#: Seconds to wait for an answer. Long fused requests take tens of seconds under load, and a
#: request that times out after it was sent may still be billed, so the retry can pay twice.
DEFAULT_TIMEOUT = 120.0
#: Seconds to wait for a connection. Nothing has been sent yet, so a retry costs nothing.
CONNECT_TIMEOUT = 10.0
#: Tokens added to the UTF-8 byte count of a request body for the first estimate. The API
#: wraps the body in its own framing, so bytes alone undercount; see ``JevClient`` budget.
ESTIMATE_FRAMING_TOKENS = 64

# Failures raised before the request left this process: retrying them never pays twice.
_NOT_SENT = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.UnsupportedProtocol,
    httpx.LocalProtocolError,
    httpx.ProxyError,
)

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
    """The next request would not fit in what is left of ``max_input_tokens``."""


class _Stopped(Exception):
    """A request left unsent because another request in its batch failed."""


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
        "lost_responses",
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
    """Batching Jev client. ``judge(batch)`` returns one ``answers`` map per item.

    ``max_input_tokens`` is enforced before a request is sent. Each attempt reserves an
    estimate of its input tokens and is sent only if the tokens already billed, the
    reservations of requests in flight and its own estimate fit under the limit; otherwise it
    waits for the requests in flight to settle and raises :class:`JevBudgetExceeded` if it
    still does not fit. A response settles its reservation to the reported usage. An error
    status releases it. A response lost or unusable after the request was sent (a read
    timeout, a dropped connection, a malformed answer) keeps the estimate as billed, since
    the API may have charged for it; those are counted in ``usage()["lost_responses"]``.
    ``usage()["input_tokens"]`` and its estimated cost include these estimates. A request
    that cannot be built is never reserved or sent.

    The first failure in a batch (the budget, an API error, retries exhausted) stops it:
    requests not yet sent are dropped, requests already in flight finish and are settled and
    cached, then ``judge()`` raises that first failure.

    The estimate is the UTF-8 byte count of the request body plus
    ``ESTIMATE_FRAMING_TOKENS``, scaled by the largest ratio of reported to estimated tokens
    seen so far on this client. Overshoot is therefore bounded by how far the requests in
    flight exceed their estimates, which shrinks once the first responses arrive. Without a
    tokenizer for the API's framing this is not an exact ceiling.
    """

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
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = 6,
        backoff_base: float = 0.5,
        backoff_factor: float = 2.0,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if max_input_tokens is not None and max_input_tokens < 0:
            raise ValueError("max_input_tokens must be >= 0")
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
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
        # Budget reservations. Touched only on the loop thread, so the condition is enough.
        self._reserved = 0
        self._estimate_ratio = 1.0
        self._budget: asyncio.Condition | None = None
        self._loop: _LoopThread | None = None
        self._loop_lock = threading.Lock()
        self._http: httpx.AsyncClient | None = None
        self._sem: asyncio.Semaphore | None = None

    # ------------------------------------------------------------------ public

    @property
    def billed_input_tokens(self) -> int:
        """Reported input tokens, plus the estimates of requests whose response was lost."""
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
        if self.max_input_tokens is not None and self._billed >= self.max_input_tokens:
            raise JevBudgetExceeded(
                f"billed input tokens {self._billed} reached max_input_tokens "
                f"{self.max_input_tokens}"
            )

    def _estimate(self, body: bytes) -> int:
        return math.ceil((len(body) + ESTIMATE_FRAMING_TOKENS) * self._estimate_ratio)

    async def _reserve(self, body: bytes) -> int:
        """Wait for room, refreshing the learned estimate before each admission check."""
        if self.max_input_tokens is None:
            return self._estimate(body)
        assert self._budget is not None
        async with self._budget:
            while True:
                estimate = self._estimate(body)
                if self._billed + self._reserved + estimate <= self.max_input_tokens:
                    self._reserved += estimate
                    return estimate
                if self._reserved == 0:
                    raise JevBudgetExceeded(
                        f"a request estimated at {estimate} input tokens does not fit: "
                        f"{self._billed} of max_input_tokens {self.max_input_tokens} "
                        "already billed"
                    )
                await self._budget.wait()

    async def _settle(self, estimate: int, spent: int, *, lost_response: bool = False) -> None:
        """Replace a reservation with what it cost: reported tokens, the estimate, or 0."""
        with self._billed_lock:
            self._billed += spent
        self.usage.add(input_tokens=spent, lost_responses=int(lost_response))
        if self.max_input_tokens is None:
            return
        assert self._budget is not None
        async with self._budget:
            self._reserved -= estimate
            self._budget.notify_all()

    def _loop_thread(self) -> _LoopThread:
        with self._loop_lock:
            if self._loop is None:
                self._loop = _LoopThread()
            return self._loop

    async def _fan_out(
        self, misses: dict[str, Item], api_key: str
    ) -> tuple[dict[str, tuple[Answers, dict[str, Any]]], BaseException | None]:
        if self._http is None:
            # Read from self.timeout here, not in __init__, so a caller can still change it
            # after register() and before the first request.
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=min(self.timeout, CONNECT_TIMEOUT)),
                transport=self._transport,
                limits=httpx.Limits(
                    max_connections=self.concurrency, max_keepalive_connections=self.concurrency
                ),
            )
            self._sem = asyncio.Semaphore(self.concurrency)
            self._budget = asyncio.Condition()
        fetched: dict[str, tuple[Answers, dict[str, Any]]] = {}
        # The first failure stops the batch: nothing more is sent, but requests already in
        # flight finish, so what they cost is settled and their answers are cached.
        stop = asyncio.Event()
        errors: list[BaseException] = []

        async def one(key: str, state: Any, questions: dict[str, Any]) -> None:
            try:
                fetched[key] = await self._post(state, questions, api_key, stop)
            except _Stopped:
                pass
            except BaseException as exc:  # re-raised by judge() after caching what arrived
                errors.append(exc)
                stop.set()

        await asyncio.gather(*(one(k, s, q) for k, (s, q) in misses.items()))
        return fetched, (errors[0] if errors else None)

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
        self, state: Any, questions: dict[str, Any], api_key: str, stop: asyncio.Event
    ) -> tuple[Answers, dict[str, Any]]:
        assert self._http is not None and self._sem is not None
        payload = {"state": state, "model": self.model, "questions": questions}
        # Encoded here, as httpx 0.28 encodes json= (compact, UTF-8, no NaN), so the estimate
        # measures the bytes sent and a body that cannot be encoded fails before any reservation.
        body = json.dumps(
            payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        # URL/header encoding can fail locally too. Build before reserving so those failures
        # never count as sent; transport-level protocol/proxy failures use _NOT_SENT below.
        request = self._http.build_request("POST", ENDPOINT, content=body, headers=headers)
        last = "no attempt made"
        for attempt in range(self.max_attempts):
            retry_after: str | None = None
            if stop.is_set():
                raise _Stopped
            async with self._sem:
                base = len(body) + ESTIMATE_FRAMING_TOKENS
                estimate = await self._reserve(body)
                if stop.is_set():  # the batch failed while this one waited for room
                    await self._settle(estimate, 0)
                    raise _Stopped
                spent = 0
                lost_response = False
                try:
                    resp = await self._http.send(request)
                except _NOT_SENT as exc:
                    last = f"{type(exc).__name__}: {exc}"
                    resp = None
                except httpx.TransportError as exc:
                    # Sent, but the answer never arrived: the API may have billed it.
                    last = f"{type(exc).__name__}: {exc}"
                    resp = None
                    spent = estimate
                    lost_response = True
                except BaseException:
                    # Cancelled or failed mid-request, possibly after sending: count it.
                    await self._settle(estimate, estimate, lost_response=True)
                    raise
                if resp is not None and resp.status_code == 200:
                    try:
                        result = self._accept(resp, questions)
                    except BaseException:
                        await self._settle(estimate, estimate, lost_response=True)
                        raise
                    spent = result[1]["input_tokens"]
                    self._estimate_ratio = max(self._estimate_ratio, spent / base)
                    await self._settle(estimate, spent)
                    return result
                await self._settle(estimate, spent, lost_response=lost_response)
            if resp is not None:
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
                # Wake early if the batch stops: the next attempt would not be sent anyway.
                try:
                    await asyncio.wait_for(stop.wait(), self._delay(attempt, retry_after))
                except TimeoutError:
                    pass
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
        self.usage.add(requests=1, output_tokens=tout)
        return answers, {"input_tokens": tin, "output_tokens": tout}
