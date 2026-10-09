"""Issue #9: max_input_tokens is reserved before a request is sent, under concurrency."""

from __future__ import annotations

import asyncio
import json

import httpx
import pyarrow as pa
import pytest

import duckjev
from duckjev.cache import AnswerCache, canonical_json
from duckjev.client import (
    ESTIMATE_FRAMING_TOKENS,
    USD_PER_INPUT_TOKEN,
    JevAPIError,
    JevBudgetExceeded,
    JevClient,
    Usage,
)
from duckjev.functions import make_udfs

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
    assert u["lost_responses"] == 1 and u["input_tokens"] == 2 * est and u["retries"] == 1
    assert u["est_usd"] == pytest.approx(c.billed_input_tokens * USD_PER_INPUT_TOKEN)
    c.close()


def test_lost_response_with_no_room_left_for_the_retry_raises() -> None:
    est = body_tokens("s0")
    t = SlowTransport(lambda s: est, fail={"s0": [httpx.ReadTimeout("slow")]})
    c = make(t, concurrency=1, max_input_tokens=est + est // 2)
    with pytest.raises(JevBudgetExceeded):
        c.judge([("s0", NOUL)])
    assert t.sent == ["s0"] and c.billed_input_tokens == est
    assert c.usage.snapshot()["input_tokens"] == est
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


def test_nothing_is_sent_after_the_batch_stops_at_the_limit() -> None:
    # A big row cannot fit once the first is in flight and raises; a smaller row queued
    # behind it would fit. Before, it was sent, billed and thrown away (cold review).
    small, big, smaller = "a", "x" * 300, "y" * 20
    t = SlowTransport(lambda s: 1)
    for concurrency in (1, 3):
        t.sent.clear()
        c = make(
            t,
            concurrency=concurrency,
            cache=AnswerCache(None),
            max_input_tokens=body_tokens(small) + body_tokens(smaller) - 1,
        )
        with pytest.raises(JevBudgetExceeded):
            c.judge([(small, NOUL), (big, NOUL), (smaller, NOUL)])
        assert t.sent == [small]
        assert len(c.cache) == 1 and c.billed_input_tokens == 1
        c.close()


def test_a_failure_lets_requests_in_flight_finish_and_caches_them() -> None:
    # "bad" fails while the others are still in flight. Before, they were cancelled: billed
    # at the API, never cached.
    sent: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        state = json.loads(request.content)["state"]
        sent.append(state)
        if state == "bad":
            await asyncio.sleep(0.01)  # after all four were sent, before the others answer
            return httpx.Response(422, json={"error": "injected"})
        await asyncio.sleep(0.05)
        return httpx.Response(
            200,
            json={"answers": {"q": {"type": "noul", "noul": 0.5}}, "usage": {"input_tokens": 7}},
        )

    c = make(httpx.MockTransport(handle), concurrency=4, cache=AnswerCache(None))
    with pytest.raises(JevAPIError):
        c.judge([("bad", NOUL), ("ok1", NOUL), ("ok2", NOUL), ("ok3", NOUL)])
    assert sorted(sent) == ["bad", "ok1", "ok2", "ok3"]
    assert len(c.cache) == 3 and c.billed_input_tokens == 21
    c.close()


def test_a_body_that_cannot_be_encoded_is_neither_sent_nor_charged() -> None:
    t = SlowTransport(lambda s: 1)
    c = make(t, max_input_tokens=10_000)
    nan_q = {"q": {"type": "noul", "instructions": "x", "criteria": {"w": float("nan")}}}
    with pytest.raises(ValueError):
        c.judge([("s0", nan_q)])
    assert t.sent == [] and c.billed_input_tokens == 0
    c.close()


def test_a_malformed_200_counts_its_estimate() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {"other": {}}, "usage": {}})

    c = make(httpx.MockTransport(handle), max_input_tokens=10_000)
    with pytest.raises(JevAPIError, match="do not match"):
        c.judge([("s0", NOUL)])
    assert c.billed_input_tokens == body_tokens("s0")
    assert c.usage.snapshot()["input_tokens"] == c.billed_input_tokens
    assert c.usage.snapshot()["lost_responses"] == 1
    c.close()


@pytest.mark.parametrize("failure", [httpx.LocalProtocolError, httpx.ProxyError])
def test_local_protocol_and_proxy_failures_are_not_billed(
    failure: type[httpx.TransportError],
) -> None:
    est = body_tokens("s0")
    t = SlowTransport(lambda s: est, fail={"s0": [failure("not forwarded")]})
    c = make(t, concurrency=1, max_input_tokens=est)
    try:
        c.judge([("s0", NOUL)])
        assert t.sent == ["s0", "s0"]
        assert c.billed_input_tokens == est and c._reserved == 0
        assert c.usage.snapshot()["lost_responses"] == 0
    finally:
        c.close()


def test_headers_that_cannot_be_encoded_are_neither_sent_nor_charged() -> None:
    t = SlowTransport(lambda s: 1)
    c = make(t, api_key="invalid-\N{SNOWMAN}", max_input_tokens=10_000)
    try:
        with pytest.raises(UnicodeEncodeError):
            c.judge([("s0", NOUL)])
        assert t.sent == [] and c.billed_input_tokens == 0 and c._reserved == 0
        assert c.usage.snapshot()["lost_responses"] == 0
    finally:
        c.close()


def test_an_unexpected_failure_after_sending_keeps_estimated_cost() -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("failed while receiving the answer")

    c = make(httpx.MockTransport(handle), max_input_tokens=10_000)
    try:
        with pytest.raises(RuntimeError, match="receiving the answer"):
            c.judge([("s0", NOUL)])
        assert c.billed_input_tokens == body_tokens("s0") and c._reserved == 0
        assert c.usage.snapshot()["input_tokens"] == c.billed_input_tokens
        assert c.usage.snapshot()["lost_responses"] == 1
    finally:
        c.close()


@pytest.mark.parametrize(
    "reported_usage",
    [None, {}, {"input_tokens": -1}, {"input_tokens": False}, {"input_tokens": 1.5}],
)
def test_missing_or_invalid_usage_keeps_the_estimate_and_stops_the_batch(
    reported_usage: dict | None,
) -> None:
    sent: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["state"])
        return httpx.Response(
            200,
            json={"answers": {"q": {"type": "noul", "noul": 0.5}}, "usage": reported_usage},
        )

    c = make(
        httpx.MockTransport(handle), concurrency=1, max_input_tokens=10_000, cache=AnswerCache(None)
    )
    try:
        with pytest.raises(JevAPIError, match="usage"):
            c.judge([(f"s{i}", NOUL) for i in range(3)])
        assert sent == ["s0"] and len(c.cache) == 0 and c._reserved == 0
        assert c.billed_input_tokens == body_tokens("s0")
        assert c.usage.snapshot()["input_tokens"] == c.billed_input_tokens
        assert c.usage.snapshot()["lost_responses"] == 1
    finally:
        c.close()


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        (NOUL["q"], {}),
        (NOUL["q"], {"noul": "not numeric"}),
        ({"type": "choice", "criteria": {"a": None}}, {"choice": "a"}),
        ({"type": "choice", "criteria": {"a": None}}, {"choice": "a", "probabilities": []}),
        ({"type": "score", "criteria": ["low", "high"]}, {"probabilities": {"0": 1.0}}),
        ({"type": "score", "criteria": ["low", "high"]}, {"score": 0.5, "probabilities": []}),
    ],
)
def test_unusable_typed_answers_stop_before_more_sends_or_cache_writes(
    question: dict,
    answer: dict,
) -> None:
    sent: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["state"])
        return httpx.Response(200, json={"answers": {"q": answer}, "usage": {"input_tokens": 1}})

    c = make(httpx.MockTransport(handle), concurrency=1, cache=AnswerCache(None))
    questions = {"q": question}
    try:
        with pytest.raises(JevAPIError, match="unusable"):
            c.judge([(f"s{i}", questions) for i in range(3)])
        assert sent == ["s0"] and len(c.cache) == 0
        payload = {"state": "s0", "model": c.model, "questions": questions}
        estimate = len(canonical_json(payload).encode("utf-8")) + ESTIMATE_FRAMING_TOKENS
        assert c.billed_input_tokens == estimate
        assert c.usage.snapshot()["input_tokens"] == estimate
        assert c.usage.snapshot()["lost_responses"] == 1
    finally:
        c.close()


def test_sql_rejects_an_unusable_answer_before_judging_the_next_row() -> None:
    sent: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["state"])
        return httpx.Response(200, json={"answers": {"q": {}}, "usage": {"input_tokens": 1}})

    c = make(httpx.MockTransport(handle), concurrency=1, cache=AnswerCache(None))
    try:
        with pytest.raises(JevAPIError, match="unusable"):
            make_udfs(c)["jev_noul2"](pa.array(["s0", "s1", "s2"]), pa.array(["about a card?"]))
        assert sent == ["s0"] and len(c.cache) == 0
        assert c.billed_input_tokens == body_tokens("s0")
    finally:
        c.close()


def test_budget_waiters_use_the_latest_learned_estimate() -> None:
    # Three requests fit at first. The first answer doubles the estimate and the other
    # two release enough room for the old estimate, but not for the newly learned one.
    est = body_tokens("s0")
    sent: list[str] = []
    first_answered = asyncio.Event()

    async def handle(request: httpx.Request) -> httpx.Response:
        state = json.loads(request.content)["state"]
        sent.append(state)
        if state == "s0":
            await asyncio.sleep(0.01)  # let the fourth request enter the budget wait
            first_answered.set()
            tokens = 2 * est
        else:
            await first_answered.wait()
            tokens = 2 * est if state == "s3" else 0
        return httpx.Response(
            200,
            json={
                "answers": {"q": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": tokens},
            },
        )

    c = make(httpx.MockTransport(handle), concurrency=4, max_input_tokens=3 * est)
    try:
        with pytest.raises(JevBudgetExceeded):
            c.judge([(f"s{i}", NOUL) for i in range(4)])
        assert sent == ["s0", "s1", "s2"]
        assert c.billed_input_tokens == 2 * est and c._reserved == 0
    finally:
        c.close()


def test_threads_judging_at_once_share_one_budget() -> None:
    import threading

    est = body_tokens("t0-00")
    t = SlowTransport(lambda s: est, delay=0.005)
    c = make(t, concurrency=8, max_input_tokens=20 * est)
    raised: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            c.judge([(f"t{i}-{j:02d}", NOUL) for j in range(10)])
        except JevBudgetExceeded as exc:
            raised.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(t.sent) == 20 and c.billed_input_tokens == 20 * est
    assert raised and c._reserved == 0
    c.close()


def test_a_stopped_batch_does_not_sleep_out_a_retry_backoff() -> None:
    # One request backs off for Retry-After: 5 while a sibling fails with 422. The backoff
    # ends when the batch stops, since the retry would not be sent (cold review: 5 s before).
    import time

    async def handle(request: httpx.Request) -> httpx.Response:
        state = json.loads(request.content)["state"]
        if state == "busy":
            return httpx.Response(529, headers={"retry-after": "5"}, json={})
        await asyncio.sleep(0.05)
        return httpx.Response(422, json={"error": "injected"})

    c = make(httpx.MockTransport(handle), concurrency=2)
    started = time.monotonic()
    with pytest.raises(JevAPIError):
        c.judge([("busy", NOUL), ("bad", NOUL)])
    assert time.monotonic() - started < 1.0
    c.close()
