from types import SimpleNamespace

from gullivers_router.config import ModelConfig
from gullivers_router.inference.base import Provider, Role, system_and_user_message
from gullivers_router.inference.openai_compat import OpenAICompatChat


def _response(content, usage=None):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def _chat_with_responses(monkeypatch, responses, **config_overrides):
    config = ModelConfig(provider=Provider.FIREWORKS, model="m", api_key="k", base_url="u", **config_overrides)
    chat = OpenAICompatChat(config)
    completions = FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(chat, "_get_client", lambda: client)
    return chat, completions


def test_system_and_user_message_orders_system_before_user():
    messages = system_and_user_message("be concise", "why is the sky blue?")

    assert [message.role for message in messages] == [Role.SYSTEM, Role.USER]
    assert messages[0].content == "be concise"
    assert messages[1].content == "why is the sky blue?"


def test_complete_accumulates_usage_across_calls(monkeypatch):
    chat, _ = _chat_with_responses(
        monkeypatch,
        [
            _response("first", SimpleNamespace(prompt_tokens=10, completion_tokens=4)),
            _response("second", SimpleNamespace(prompt_tokens=5, completion_tokens=2)),
        ],
    )

    chat.complete(system_and_user_message("s", "a"))
    chat.complete(system_and_user_message("s", "b"))

    assert chat.usage.prompt_tokens == 15
    assert chat.usage.completion_tokens == 6
    assert chat.usage.total_tokens == 21


def test_complete_tolerates_missing_usage(monkeypatch):
    chat, _ = _chat_with_responses(monkeypatch, [_response("answer", usage=None)])

    chat.complete(system_and_user_message("s", "a"))

    assert chat.usage.total_tokens == 0


def test_disabled_thinking_sends_reasoning_effort_none(monkeypatch):
    chat, completions = _chat_with_responses(monkeypatch, [_response("answer")], enable_thinking=False)

    chat.complete(system_and_user_message("s", "a"))

    assert completions.calls[0]["extra_body"] == {"reasoning_effort": "none"}


def test_enabled_thinking_sends_reasoning_effort_true(monkeypatch):
    chat, completions = _chat_with_responses(monkeypatch, [_response("answer")], enable_thinking=True)

    chat.complete(system_and_user_message("s", "a"))

    assert completions.calls[0]["extra_body"] == {"reasoning_effort": True}


def test_reasoning_effort_overrides_thinking_boolean(monkeypatch):
    chat, completions = _chat_with_responses(
        monkeypatch,
        [_response("answer")],
        enable_thinking=True,
        reasoning_effort="adaptive",
    )

    chat.complete(system_and_user_message("s", "a"))

    assert completions.calls[0]["extra_body"] == {"reasoning_effort": "adaptive"}


def test_unset_thinking_leaves_reasoning_untouched(monkeypatch):
    chat, completions = _chat_with_responses(monkeypatch, [_response("answer")])

    chat.complete(system_and_user_message("s", "a"))

    assert completions.calls[0]["extra_body"] == {}
