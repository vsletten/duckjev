from __future__ import annotations

from pathlib import Path

from fake import FakeTransport

from duckjev.cache import AnswerCache, cache_key
from duckjev.client import JevClient, Usage

NOUL = {"q": {"type": "noul", "instructions": "about a card?"}}


def client(t: FakeTransport, cache: AnswerCache | None, model: str = "jev-1.13.0") -> JevClient:
    return JevClient(api_key="k", transport=t, cache=cache, model=model, usage=Usage())


def test_hit_and_miss(tmp_path: Path) -> None:
    t = FakeTransport()
    c = client(t, AnswerCache(tmp_path / "c.duckdb"))
    c.judge([("card", NOUL)])
    c.judge([("card", NOUL), ("other", NOUL)])
    u = c.usage.snapshot()
    assert (u["cache_hits"], u["cache_misses"]) == (1, 2)
    assert len(t.requests) == 2
    c.close()


def test_model_change_invalidates(tmp_path: Path) -> None:
    t = FakeTransport()
    cache = AnswerCache(tmp_path / "c.duckdb")
    client(t, cache).judge([("card", NOUL)])
    client(t, cache, model="jev-latest").judge([("card", NOUL)])
    assert [r["model"] for r in t.requests] == ["jev-1.13.0", "jev-latest"]


def test_disabled_cache_always_sends() -> None:
    t = FakeTransport()
    c = client(t, None)
    c.judge([("card", NOUL)])
    c.judge([("card", NOUL)])
    assert len(t.requests) == 2
    c.close()


def test_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "c.duckdb"
    t = FakeTransport()
    first = AnswerCache(path)
    c1 = client(t, first)
    c1.judge([("card", NOUL), ("other", NOUL)])
    c1.close()
    first.close()

    second = AnswerCache(path)
    assert len(second) == 2
    c2 = client(t, second)
    out = c2.judge([("card", NOUL)])
    assert out[0]["q"]["noul"] == 0.8
    assert len(t.requests) == 2  # nothing new was sent
    table = second.to_arrow()
    assert table.num_rows == 2
    assert set(table.column("model").to_pylist()) == {"jev-1.13.0"}
    assert table.column("input_tokens").to_pylist() == [100, 100]
    c2.close()
    second.close()


def test_unopenable_file_falls_back_to_memory(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    cache = AnswerCache(blocker / "c.duckdb")
    t = FakeTransport()
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client(t, cache).judge([("card", NOUL)])
    assert any("memory only" in str(w.message) for w in caught)
    assert len(cache) == 1


CHOICE = {"q": {"type": "choice", "instructions": "which?", "criteria": {"a": "A", "b": "B"}}}
CHOICE_REVERSED = {
    "q": {"type": "choice", "instructions": "which?", "criteria": {"b": "B", "a": "A"}}
}


def test_key_preserves_option_order() -> None:
    assert cache_key("m", "s", CHOICE) == cache_key("m", "s", dict(CHOICE))
    assert cache_key("m", "s", CHOICE) != cache_key("m", "s", CHOICE_REVERSED)
    assert cache_key("m", {"x": 1, "y": 2}, CHOICE) != cache_key("m", {"y": 2, "x": 1}, CHOICE)


def test_reordered_options_are_not_served_from_cache(tmp_path: Path) -> None:
    t = FakeTransport()
    c = client(t, AnswerCache(tmp_path / "c.duckdb"))
    c.judge([("card", CHOICE)])
    c.judge([("card", CHOICE_REVERSED)])
    c.judge([("card", CHOICE)])
    assert [list(r["questions"]["q"]["criteria"]) for r in t.requests] == [["a", "b"], ["b", "a"]]
    u = c.usage.snapshot()
    assert (u["cache_hits"], u["cache_misses"]) == (1, 2)
    c.close()
