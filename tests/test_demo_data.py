"""Regeneration uses a fake live transport and verifies the newly recorded answers offline."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pytest
from fake import FakeTransport

import duckjev
from bench import demo_data
from duckjev import demo


@pytest.fixture
def recording(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, FakeTransport]:
    answers = tmp_path / "answers.jsonl.gz"
    monkeypatch.setattr(demo_data, "ANSWERS", answers)
    monkeypatch.setattr(demo_data, "ROOT", tmp_path)
    data_file = demo.data_file
    monkeypatch.setattr(
        demo, "data_file", lambda name: answers if name == "answers.jsonl.gz" else data_file(name)
    )
    fake = FakeTransport(input_tokens=1_000)
    register = duckjev.register

    def fake_register(con, **kwargs):
        if "transport" not in kwargs:
            kwargs |= {"api_key": "test-key", "transport": fake}
        return register(con, **kwargs)

    monkeypatch.setattr(duckjev, "register", fake_register)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    return answers, fake


def test_record_preserves_other_clients_usage_and_replays_new_answers(
    recording: tuple[Path, FakeTransport], capsys: pytest.CaptureFixture[str]
) -> None:
    answers, fake = recording
    con = duckdb.connect()
    other = duckjev.register(con, cache=False)
    try:
        con.execute("SELECT jev_noul('my card', 'about a card?')").fetchall()
        before = duckjev.usage()
        assert before["requests"] == 1
        assert demo_data.record(argparse.Namespace()) == 0
        assert duckjev.usage() == before
        assert answers.is_file()
        assert len(fake.requests) == 139  # one other-client request, 138 recorded, zero on replay
        out = capsys.readouterr().out
        assert "live: 138 requests, 138,000 input tokens" in out
        assert "offline replay: 0 requests, 138 cache hits" in out
    finally:
        other.close()
        con.close()


def test_failed_record_reports_spend_and_keeps_previous_answers(
    recording: tuple[Path, FakeTransport],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    answers, fake = recording
    answers.write_bytes(b"previous answers")
    monkeypatch.setattr(
        demo,
        "STEPS",
        [
            ("One billed request", "SELECT jev_noul('my card', 'about a card?')"),
            ("SQL failure after billing", "SELECT * FROM missing_table"),
        ],
    )
    assert demo_data.record(argparse.Namespace()) == 1
    captured = capsys.readouterr()
    assert "missing_table" in captured.err
    assert "live: 1 requests, 1,000 input tokens" in captured.out
    assert len(fake.requests) == 1
    assert answers.read_bytes() == b"previous answers"
