from __future__ import annotations

import pytest
from fake import FakeTransport

from duckjev.cache import AnswerCache, cache_key
from duckjev.client import (
    JevAPIError,
    JevAuthError,
    JevBudgetExceeded,
    JevClient,
    JevTransportError,
    Usage,
)

NOUL = {"q": {"type": "noul", "instructions": "about a card?"}}


def make(transport: FakeTransport, **kw) -> JevClient:  # type: ignore[no-untyped-def]
    kw.setdefault("api_key", "test-key")
    kw.setdefault("backoff_base", 0.0)
    kw.setdefault("usage", Usage())
    return JevClient(transport=transport, **kw)


def test_one_request_per_unique_pair_in_order() -> None:
    t = FakeTransport()
    c = make(t)
    batch = [("my card", NOUL), ("hello", NOUL), ("my card", NOUL), ("hello", NOUL)]
    out = c.judge(batch)
    assert [a["q"]["noul"] for a in out] == [0.8, 0.1, 0.8, 0.1]
    assert len(t.requests) == 2
    assert {r["state"] for r in t.requests} == {"my card", "hello"}
    assert all(r["model"] == "jev-1.13.0" for r in t.requests)
    assert t.headers[0]["authorization"] == "Bearer test-key"
    u = c.usage.snapshot()
    assert u["rows"] == 4 and u["deduped"] == 2 and u["requests"] == 2
    c.close()


def test_usage_accounting_and_reset() -> None:
    t = FakeTransport(input_tokens=1_000)
    c = make(t)
    c.judge([(f"s{i}", NOUL) for i in range(5)])
    u = c.usage.snapshot(reset=True)
    assert u["input_tokens"] == 5_000
    assert u["output_tokens"] == 100
    assert u["est_usd"] == pytest.approx(5_000 * 42 / 1e9)
    assert u["cache_misses"] == 5
    assert c.usage.snapshot()["requests"] == 0
    c.close()


def test_backoff_retries_then_succeeds_and_counts_429() -> None:
    t = FakeTransport(fail_statuses=[429, 529, 429])
    c = make(t)
    out = c.judge([("card", NOUL)])
    assert out[0]["q"]["noul"] == 0.8
    u = c.usage.snapshot()
    assert u["rate_limited"] == 2 and u["overloaded"] == 1 and u["retries"] == 3
    assert t.attempts == 4
    c.close()


def test_backoff_exhausted_raises_without_partial_fill() -> None:
    t = FakeTransport(fail_statuses=[429] * 6)
    c = make(t, max_attempts=6)
    with pytest.raises(JevTransportError, match="6 attempts"):
        c.judge([("card", NOUL)])
    assert t.attempts == 6
    c.close()


def test_non_retryable_status_raises() -> None:
    c = make(FakeTransport(fail_statuses=[422]))
    with pytest.raises(JevAPIError) as exc:
        c.judge([("card", NOUL)])
    assert exc.value.status == 422
    c.close()


def test_rejected_key_raises_auth_error_without_leaking_key() -> None:
    c = make(FakeTransport(fail_statuses=[401]), api_key="sekrit-value")
    with pytest.raises(JevAuthError) as exc:
        c.judge([("card", NOUL)])
    assert "sekrit-value" not in str(exc.value)
    c.close()


def test_missing_key_raises_clear_error() -> None:
    t = FakeTransport()
    c = make(t, api_key=None)
    with pytest.raises(JevAuthError, match="TYPESAFE_API_KEY"):
        c.judge([("card", NOUL)])
    assert t.attempts == 0
    c.close()


def test_budget_exceeded_raises() -> None:
    t = FakeTransport(input_tokens=100)
    c = make(t, max_input_tokens=250, concurrency=1)
    with pytest.raises(JevBudgetExceeded):
        c.judge([(f"s{i}", NOUL) for i in range(10)])
    assert c.billed_input_tokens <= 300
    with pytest.raises(JevBudgetExceeded):  # stays exceeded
        c.judge([("another", NOUL)])
    c.close()


def test_cache_hit_skips_network_and_key() -> None:
    t = FakeTransport()
    cache = AnswerCache(None)
    c = make(t, cache=cache)
    c.judge([("card", NOUL)])
    c2 = make(t, cache=cache, api_key=None)  # no key needed when everything is cached
    out = c2.judge([("card", NOUL), ("card", NOUL)])
    assert out[0]["q"]["noul"] == 0.8
    assert len(t.requests) == 1
    assert c2.usage.snapshot()["cache_hits"] == 1
    c.close()
    c2.close()


def test_answers_received_before_failure_are_cached() -> None:
    t = FakeTransport()
    cache = AnswerCache(None)
    c = make(t, cache=cache, concurrency=1, max_attempts=1)
    c.judge([("first", NOUL)])
    t.fail_statuses = [529]
    with pytest.raises(JevTransportError):
        c.judge([("first", NOUL), ("second", NOUL)])
    assert len(cache) == 1
    c.close()


def test_cache_key_includes_model() -> None:
    assert cache_key("jev-1.13.0", "s", NOUL) != cache_key("jev-1.14.0", "s", NOUL)
    # order is part of the key: it is what the model sees
    assert cache_key("m", "s", {"a": 1, "b": 2}) != cache_key("m", "s", {"b": 2, "a": 1})
