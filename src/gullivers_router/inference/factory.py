"""Build the right backend for a role's configured provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gullivers_router.inference.base import BatchChatModel, ChatModel, EmbeddingModel, Provider
from gullivers_router.inference.fireworks_batch import FireworksBatchChat
from gullivers_router.inference.llama_cpp import LlamaCppChat, LlamaCppEmbedder
from gullivers_router.inference.openai_compat import OpenAICompatChat

if TYPE_CHECKING:
    from gullivers_router.config import ModelConfig


def _unsupported(role: str, config: ModelConfig) -> ValueError:
    msg = f"provider {config.provider} is not supported for {role}"
    return ValueError(msg)


def build_chat_model(config: ModelConfig) -> ChatModel:
    """Single-call chat model for the runtime path."""
    match config.provider:
        case Provider.LLAMA:
            return LlamaCppChat(config)
        case Provider.OPENAI | Provider.FIREWORKS:
            return OpenAICompatChat(config)
        case _:
            error = _unsupported("chat", config)
            raise error


def build_embedding_model(config: ModelConfig) -> EmbeddingModel:
    """Embedding model for query preprocessing."""
    match config.provider:
        case Provider.LLAMA:
            return LlamaCppEmbedder(config)
        case _:
            error = _unsupported("embedding", config)
            raise error


def build_batch_chat_model(config: ModelConfig) -> BatchChatModel:
    """Batch chat model for the training path.

    Fireworks uses the dedicated Batch API; other providers fall back to their
    sequential ``complete_batch``.
    """
    match config.provider:
        case Provider.FIREWORKS:
            return FireworksBatchChat(config)
        case Provider.LLAMA:
            return LlamaCppChat(config)
        case Provider.OPENAI:
            return OpenAICompatChat(config)
        case _:
            error = _unsupported("batch chat", config)
            raise error
