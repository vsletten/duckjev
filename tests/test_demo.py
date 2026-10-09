"""The keyless demo replays its shipped answers: no key, no sockets, no requests."""

from __future__ import annotations

import io

import duckdb
import pytest
from fake import FakeTransport

import duckjev
from duckjev import demo


def test_demo_runs_offline_from_the_shipped_answers(capsys: pytest.CaptureFixture[str]) -> None:
    assert demo.main([]) == 0
    out = capsys.readouterr().out
    assert "Replaying 138 recorded Jev answers" in out
    assert "usage: 0 requests, 0 input tokens, 138 cache hits, $0.0000" in out
    # step 3: the missed-alert count, filtered and expected, as recorded on 2026-10-09
    assert "150      88        87.1      2.6" in out
    # step 4: the reports filed as Injury mostly say the patient got care
    assert "Injury       10       0.79" in out


def test_every_shipped_answer_is_used() -> None:
    con = demo.connect(live=False)
    client = duckjev.client_for(con)
    shipped = len(client.cache)
    demo.run(con, out=io.StringIO())
    assert client.usage.snapshot()["cache_hits"] == shipped
    client.close()
    con.close()


def test_a_changed_question_says_to_run_live(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(demo.PARAMS, "care_q", demo.CARE_Q + " Say yes only if stated.")
    assert demo.main([]) == 1
    assert "rerun with --live" in capsys.readouterr().err


def test_live_without_a_key_stops_before_any_request(capsys: pytest.CaptureFixture[str]) -> None:
    assert demo.main(["--live"]) == 1
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err


def test_demo_preserves_other_clients_usage(capsys: pytest.CaptureFixture[str]) -> None:
    con = duckdb.connect()
    other = duckjev.register(con, cache=False, api_key="test-key", transport=FakeTransport())
    try:
        con.execute("SELECT jev_noul('my card', 'about a card?')").fetchall()
        before = duckjev.usage()
        assert before["requests"] == 1
        assert demo.main([]) == 0
        assert duckjev.usage() == before
        assert (
            "usage: 0 requests, 0 input tokens, 138 cache hits, $0.0000" in capsys.readouterr().out
        )
    finally:
        other.close()
        con.close()


def test_failed_live_run_reports_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeTransport(input_tokens=1_000)
    register = duckjev.register

    def fake_register(con, **kwargs):
        return register(con, **(kwargs | {"api_key": "test-key", "transport": fake}))

    monkeypatch.setattr(duckjev, "register", fake_register)
    monkeypatch.setattr(
        demo,
        "STEPS",
        [
            ("One billed request", "SELECT jev_noul('my card', 'about a card?')"),
            ("SQL failure after billing", "SELECT * FROM missing_table"),
        ],
    )
    assert demo.main(["--live"]) == 1
    captured = capsys.readouterr()
    assert "missing_table" in captured.err
    assert len(fake.requests) == 1
    assert "usage: 1 requests, 1,000 input tokens, 0 cache hits, $0.0000" in captured.out
