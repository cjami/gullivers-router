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

    def create(self, **_kwargs):
        return self._responses.pop(0)


def _chat_with_responses(monkeypatch, responses):
    chat = OpenAICompatChat(ModelConfig(provider=Provider.FIREWORKS, model="m", api_key="k", base_url="u"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(responses)))
    monkeypatch.setattr(chat, "_get_client", lambda: client)
    return chat


def test_system_and_user_message_orders_system_before_user():
    messages = system_and_user_message("be concise", "why is the sky blue?")

    assert [message.role for message in messages] == [Role.SYSTEM, Role.USER]
    assert messages[0].content == "be concise"
    assert messages[1].content == "why is the sky blue?"


def test_complete_accumulates_usage_across_calls(monkeypatch):
    chat = _chat_with_responses(
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
    chat = _chat_with_responses(monkeypatch, [_response("answer", usage=None)])

    chat.complete(system_and_user_message("s", "a"))

    assert chat.usage.total_tokens == 0
