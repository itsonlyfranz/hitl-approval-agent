from types import SimpleNamespace

import langchain_anthropic

from hitl_agent.llm import DEFAULT_MODEL, AnthropicClient


class FakeChat:
    def __init__(self, **kw):
        self.kw = kw

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=[{"type": "text", "text": "hi"}])


def test_anthropic_client_uses_configured_model_without_network(monkeypatch):
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", FakeChat)
    monkeypatch.setenv("HITL_MODEL", "custom-model")
    client = AnthropicClient(api_key="x")
    assert client.complete("sys", "prompt") == "hi"
    assert client._chat.kw["model"] == "custom-model"
    assert client._chat.messages[0] == ("system", "sys")


def test_default_model(monkeypatch):
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", FakeChat)
    monkeypatch.delenv("HITL_MODEL", raising=False)
    assert AnthropicClient(api_key="x").model == DEFAULT_MODEL == "claude-sonnet-5-5"
