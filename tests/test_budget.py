"""Issue #9: max_input_tokens is reserved before a request is sent, under concurrency."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import duckjev
from duckjev.cache import AnswerCache, canonical_json
from duckjev.client import ESTIMATE_FRAMING_TOKENS, JevBudgetExceeded, JevClient, Usage

NOUL = {"q": {"type": "noul", "instructions": "about a card?"}}


def body_tokens(state: str) -> int:
    """The estimate's base for one NOUL request: UTF-8 body bytes plus framing."""
    payload = {"state": state, "model": "jev-1.13.0", "questions": NOUL}
    return len(canonical_json(payload).encode("utf-8")) + ESTIMATE_FRAMING_TOKENS


class SlowTransport(httpx.MockTransport):
    """Answers after a delay, reporting ``tokens(state)`` input tokens; tracks concurrency."""

    def __init__(self, tokens, delay: float = 0.02, fail: dict[str, list] | None = None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(self._handle)
        self.tokens = tokens
        self.delay = delay
        self.fail = fail or {}
        self.sent: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        state = json.loads(request.content)["state"]
        self.sent.append(state)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            if self.fail.get(state):
                outcome = self.fail[state].pop(0)
                if isinstance(outcome, int):
                    return httpx.Response(outcome, json={"error": "injected"})
                raise outcome
            return httpx.Response(
                200,
                json={
                    "answers": {"q": {"type": "noul", "noul": 0.5}},
                    "usage": {"input_tokens": self.tokens(state), "output_tokens": 1},
                },
            )
        finally:
            self.in_flight -= 1


def make(transport: httpx.MockTransport, **kw) -> JevClient:  # type: ignore[no-untyped-def]
    kw.setdefault("api_key", "test-key")
    kw.setdefault("backoff_base", 0.0)
    kw.setdefault("usage", Usage())
    return JevClient(transport=transport, **kw)


def test_concurrent_requests_cannot_all_pass_the_same_balance() -> None:
    # The race in #9: 16 slots, ten misses, room for two. Before, all ten were sent.
    est = body_tokens("s0")
    t = SlowTransport(lambda s: est)
    c = make(t, concurrency=16, max_input_tokens=2 * est + est // 2)
    with pytest.raises(JevBudgetExceeded):
        c.judge([(f"s{i}", NOUL) for i in range(10)])
    assert len(t.sent) == 2
    assert c.billed_input_tokens == 2 * est <= c.max_input_tokens
    c.close()


def test_queued_requests_wait_for_in_flight_ones_to_settle() -> None:
    # Room for three estimates at once; reported usage is a quarter of the estimate. The
    # budget, not the 8 slots, bounds what is in flight, and the waiting requests go out as
    # earlier ones settle, so the whole batch completes under the limit.
    est = body_tokens("s0")
    t = SlowTransport(lambda s: est // 4)
    c = make(t, concurrency=8, max_input_tokens=3 * est)
    out = c.judge([(f"s{i}", NOUL) for i in range(8)])
    assert len(out) == 8 and len(t.sent) == 8
    assert t.max_in_flight == 3
    assert c.billed_input_tokens == 8 * (est // 4) <= c.max_input_tokens
    c.close()


def test_estimate_learns_from_reported_usage() -> None:
    # The API reports twice the byte estimate. The first request can overshoot by its own
    # error; once it settles the estimate doubles and the rest stay under the limit.
    t = SlowTransport(lambda s: 2 * body_tokens(s))
    limit = 7 * body_tokens("s0")
    c = make(t, concurrency=1, max_input_tokens=limit)
    with pytest.raises(JevBudgetExceeded):
        c.judge([(f"s{i}", NOUL) for i in range(10)])
    assert len(t.sent) == 3  # 2 + 2 + 2 estimates of 7, then 6 + 2 > 7
    assert c.billed_input_tokens == 6 * body_tokens("s0") <= limit
    c.close()


def test_lost_response_counts_as_billed_and_the_retry_is_reserved_again() -> None:
    est = body_tokens("s0")
    t = SlowTransport(lambda s: est, fail={"s0": [httpx.ReadTimeout("slow")]})
    c = make(t, concurrency=1, max_input_tokens=10 * est)
    c.judge([("s0", NOUL)])
    assert t.sent == ["s0", "s0"]
    assert c.billed_input_tokens == 2 * est  # the lost attempt at its estimate, plus the answer
    u = c.usage.snapshot()
    assert u["lost_responses"] == 1 and u["input_tokens"] == est and u["retries"] == 1
    c.close()


def test_lost_response_with_no_room_left_for_the_retry_raises() -> None:
    est = body_tokens("s0")
    t = SlowTransport(lambda s: est, fail={"s0": [httpx.ReadTimeout("slow")]})
    c = make(t, concurrency=1, max_input_tokens=est + est // 2)
    with pytest.raises(JevBudgetExceeded):
        c.judge([("s0", NOUL)])
    assert t.sent == ["s0"] and c.billed_input_tokens == est
    c.close()


def test_error_statuses_and_unsent_requests_release_their_reservation() -> None:
    est = body_tokens("s0")
    fail = {"s0": [429, 529, httpx.ConnectError("refused")]}
    t = SlowTransport(lambda s: est, fail=fail)
    c = make(t, concurrency=1, max_input_tokens=est)  # room for exactly one billed request
    c.judge([("s0", NOUL)])
    assert len(t.sent) == 4
    assert c.billed_input_tokens == est
    assert c.usage.snapshot()["lost_responses"] == 0
    c.close()


def test_a_request_larger_than_the_whole_budget_is_never_sent() -> None:
    t = SlowTransport(lambda s: 1)
    c = make(t, max_input_tokens=body_tokens("s0") - 1)
    with pytest.raises(JevBudgetExceeded, match="does not fit"):
        c.judge([("s0", NOUL)])
    assert t.sent == []
    c.close()


def test_answers_received_before_the_budget_stops_are_cached() -> None:
    est = body_tokens("s0")
    t = SlowTransport(lambda s: est)
    cache = AnswerCache(None)
    c = make(t, concurrency=4, max_input_tokens=3 * est, cache=cache)
    with pytest.raises(JevBudgetExceeded):
        c.judge([(f"s{i}", NOUL) for i in range(6)])
    assert len(cache) == 3 == len(t.sent)
    c.close()


def test_no_budget_means_no_waiting() -> None:
    t = SlowTransport(lambda s: 10**9)
    c = make(t, concurrency=8)
    c.judge([(f"s{i}", NOUL) for i in range(8)])
    assert t.max_in_flight == 8
    c.close()


def test_register_passes_the_timeout_and_defaults_to_two_minutes() -> None:
    import duckdb

    con = duckdb.connect()
    assert duckjev.register(con, cache=False).timeout == 120.0
    assert duckjev.register(con, cache=False, timeout=7.5).timeout == 7.5
    with pytest.raises(ValueError):
        duckjev.register(con, cache=False, timeout=0)
    duckjev.client_for(con).close()


def test_timeout_reaches_the_http_client_with_a_short_connect_cap() -> None:
    t = SlowTransport(lambda s: 1)
    c = make(t, timeout=45.0)
    c.judge([("s0", NOUL)])
    assert c._http is not None
    assert c._http.timeout.read == 45.0 and c._http.timeout.connect == 10.0
    c.close()
