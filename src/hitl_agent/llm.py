"""Minimal LLM interface so the agent never depends on a concrete provider."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Protocol

DEFAULT_MODEL = "claude-sonnet-5-5"


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


class AnthropicClient:
    """Claude via langchain-anthropic; the only place the network is touched."""

    def __init__(self, model: str | None = None, api_key: str | None = None, max_tokens: int = 600):
        from langchain_anthropic import ChatAnthropic  # lazy: offline use never needs a key

        self.model = model or os.environ.get("HITL_MODEL", DEFAULT_MODEL)
        kwargs = {"api_key": api_key} if api_key else {}
        self._chat = ChatAnthropic(model=self.model, max_tokens=max_tokens, **kwargs)

    def complete(self, system: str, prompt: str) -> str:
        msg = self._chat.invoke([("system", system), ("human", prompt)])
        content = msg.content
        if isinstance(content, str):
            return content
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


class FakeLLMClient:
    """Deterministic offline client: replays canned replies or calls a function."""

    def __init__(self, replies: Sequence[str] | Callable[[str, str], str]):
        self._replies = replies
        self._i = 0
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        if callable(self._replies):
            return self._replies(system, prompt)
        reply = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        if isinstance(reply, Exception):  # lets tests simulate provider failures
            raise reply
        return reply


def load_env() -> None:
    """Load a gitignored .env if python-dotenv is present; real env vars win."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(override=False)
