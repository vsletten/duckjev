"""A fake Jev transport: canned answers keyed by question type, request log, fault injection."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

import httpx

ChoiceFn = Callable[[str, list[str]], dict[str, float]]


def default_choice_probs(state: str, options: list[str]) -> dict[str, float]:
    """Deterministic distribution: the option named in the state (else the first) gets 0.7."""
    top = next((o for o in options if o in state), options[0])
    rest = 0.3 / (len(options) - 1) if len(options) > 1 else 0.0
    return {o: (0.7 if o == top else rest) if len(options) > 1 else 1.0 for o in options}


class FakeTransport(httpx.MockTransport):
    """Records requests; serves canned answers; optionally fails with a status sequence first."""

    def __init__(
        self,
        *,
        noul: Callable[[str], float] | None = None,
        choice: ChoiceFn | None = None,
        fail_statuses: list[int] | None = None,
        input_tokens: int = 100,
    ) -> None:
        super().__init__(self._handle)
        self.noul = noul or (lambda state: 0.8 if "card" in state else 0.1)
        self.choice = choice or default_choice_probs
        self.fail_statuses = list(fail_statuses or [])
        self.input_tokens = input_tokens
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []
        self.attempts = 0
        self._lock = threading.Lock()

    def _answer(self, state: str, q: dict[str, Any]) -> dict[str, Any]:
        if q["type"] == "noul":
            return {"type": "noul", "noul": self.noul(state)}
        if q["type"] == "choice":
            probs = self.choice(state, list(q["criteria"]))
            top = max(probs, key=probs.__getitem__)
            return {"type": "choice", "choice": top, "probabilities": probs, "confidence": 0.5}
        levels = q["criteria"]
        n = len(levels)
        probs = {str(i): (1.0 if i == n - 1 else 0.0) for i in range(n)}
        return {
            "type": "score",
            "score": float(n - 1),
            "legend": {str(i): lv for i, lv in enumerate(levels)},
            "probabilities": probs,
            "confidence": 1.0,
        }

    def _handle(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.attempts += 1
            if self.fail_statuses:
                return httpx.Response(self.fail_statuses.pop(0), json={"error": "injected"})
            body = json.loads(request.content)
            self.requests.append(body)
            self.headers.append(request.headers)
        answers = {qid: self._answer(body["state"], q) for qid, q in body["questions"].items()}
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "answers": answers,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": 20},
            },
        )
