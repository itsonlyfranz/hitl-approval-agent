from types import SimpleNamespace

import anthropic

from hitl_agent.llm import DEFAULT_MODEL, AnthropicClient


def test_anthropic_client_uses_configured_model_without_network(monkeypatch):
    seen = {}

    class FakeMessages:
        def create(self, **kw):
            seen.update(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")])

    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key=None: SimpleNamespace(messages=FakeMessages()))
    monkeypatch.setenv("HITL_MODEL", "custom-model")
    assert AnthropicClient(api_key="x").complete("sys", "prompt") == "hi"
    assert seen["model"] == "custom-model" and seen["system"] == "sys"


def test_default_model(monkeypatch):
    monkeypatch.delenv("HITL_MODEL", raising=False)
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key=None: SimpleNamespace())
    assert AnthropicClient(api_key="x").model == DEFAULT_MODEL == "claude-sonnet-5-5"
