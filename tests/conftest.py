from __future__ import annotations

import socket

import pytest

import duckjev
from duckjev.client import API_KEY_ENV


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never touch the network: no real key, and sockets refuse to connect."""
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("network access in tests is forbidden")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    duckjev.usage(reset=True)
