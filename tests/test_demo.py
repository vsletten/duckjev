"""The keyless demo replays its shipped answers: no key, no sockets, no requests."""

from __future__ import annotations

import io

import pytest

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
    assert duckjev.usage()["cache_hits"] == shipped
    client.close()


def test_a_changed_question_says_to_run_live(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(demo.PARAMS, "care_q", demo.CARE_Q + " Say yes only if stated.")
    assert demo.main([]) == 1
    assert "rerun with --live" in capsys.readouterr().err


def test_live_without_a_key_stops_before_any_request(capsys: pytest.CaptureFixture[str]) -> None:
    assert demo.main(["--live"]) == 1
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err
