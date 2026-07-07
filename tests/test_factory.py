import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from gullivers_router.config import ModelConfig
from gullivers_router.inference import factory
from gullivers_router.inference.base import DEFAULT_INFERENCE_SEED, Message, Provider, Role
from gullivers_router.inference.llama_cpp import LlamaCppChat, LlamaCppEmbedder
from gullivers_router.inference.openai_compat import OpenAICompatChat


class StructuredReply(BaseModel):
    answer: str


def _cfg(provider, **kwargs):
    return ModelConfig(provider=provider, **kwargs)


def test_build_chat_model_local():
    assert isinstance(factory.build_chat_model(_cfg(Provider.LLAMA)), LlamaCppChat)


def test_build_chat_model_openai_compatible():
    cfg = _cfg(Provider.OPENAI, api_key="k", model="m")
    assert isinstance(factory.build_chat_model(cfg), OpenAICompatChat)


def test_build_embedding_model_local():
    assert isinstance(factory.build_embedding_model(_cfg(Provider.LLAMA)), LlamaCppEmbedder)


def test_build_embedding_model_rejects_cloud_provider():
    cfg = _cfg(Provider.FIREWORKS, api_key="k", model="m")
    with pytest.raises(ValueError, match="embedding"):
        factory.build_embedding_model(cfg)


def test_openai_compatible_chat_uses_global_seed():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    chat = OpenAICompatChat(_cfg(Provider.OPENAI, api_key="k", model="m"))
    chat._client = cast(Any, SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))

    assert chat.complete([Message(Role.USER, "hello")]) == "ok"
    assert captured["seed"] == DEFAULT_INFERENCE_SEED


def test_openai_compatible_chat_sends_json_schema_response_format():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content='{"answer": "ok"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    chat = OpenAICompatChat(_cfg(Provider.FIREWORKS, api_key="k", model="m"))
    chat._client = cast(Any, SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))

    result = chat.complete_structured([Message(Role.USER, "hello")], StructuredReply)

    assert result == StructuredReply(answer="ok")
    assert captured["seed"] == DEFAULT_INFERENCE_SEED
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["name"] == "StructuredReply"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["response_format"]["json_schema"]["schema"]["properties"]["answer"]["type"] == "string"


def test_llama_chat_loads_with_global_seed(monkeypatch):
    captured = {}

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **kwargs) -> "FakeLlama":
            captured.update(kwargs)
            return cls()

        def create_chat_completion(self, messages, seed):
            captured["completion_seed"] = seed
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    chat = LlamaCppChat(_cfg(Provider.LLAMA, repo_id="repo", filename="model.gguf"))

    assert chat.complete([Message(Role.USER, "hello")]) == "ok"
    assert captured["seed"] == DEFAULT_INFERENCE_SEED
    assert captured["completion_seed"] == DEFAULT_INFERENCE_SEED


def test_llama_chat_sends_schema_response_format(monkeypatch):
    captured = {}

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **kwargs) -> "FakeLlama":
            captured.update(kwargs)
            return cls()

        def create_chat_completion(self, messages, seed, response_format):
            captured["completion_seed"] = seed
            captured["response_format"] = response_format
            return {"choices": [{"message": {"content": '{"answer": "ok"}'}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    chat = LlamaCppChat(_cfg(Provider.LLAMA, repo_id="repo", filename="model.gguf"))

    result = chat.complete_structured([Message(Role.USER, "hello")], StructuredReply)

    assert result == StructuredReply(answer="ok")
    assert captured["completion_seed"] == DEFAULT_INFERENCE_SEED
    assert captured["response_format"]["type"] == "json_object"
    assert captured["response_format"]["schema"]["properties"]["answer"]["type"] == "string"
