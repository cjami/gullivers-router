import pytest

from gullivers_router.config import ModelConfig
from gullivers_router.inference import factory
from gullivers_router.inference.base import Provider
from gullivers_router.inference.llama_cpp import LlamaCppChat, LlamaCppEmbedder
from gullivers_router.inference.openai_compat import OpenAICompatChat


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
