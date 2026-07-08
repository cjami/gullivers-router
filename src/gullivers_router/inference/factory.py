"""Build the right backend for a role's configured provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gullivers_router.inference.base import ChatModel, EmbeddingModel, Provider
from gullivers_router.inference.llama_cpp import (
    DEFAULT_CHAT_CONTEXT,
    DEFAULT_ENABLE_THINKING,
    DEFAULT_MODEL_ROOT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    OFFLOAD_ALL_LAYERS,
    LlamaCppChat,
    LlamaCppEmbedder,
)
from gullivers_router.inference.openai_compat import OpenAICompatChat
from gullivers_router.inference.truncation import EMBEDDING_CONTEXT_LIMIT

if TYPE_CHECKING:
    from gullivers_router.config import ModelConfig


def _unsupported(role: str, config: ModelConfig) -> ValueError:
    msg = f"provider {config.provider} is not supported for {role}"
    return ValueError(msg)


def build_chat_model(config: ModelConfig) -> ChatModel:
    """Single-call chat model for the runtime path."""
    match config.provider:
        case Provider.LLAMA:
            return LlamaCppChat(
                config,
                n_ctx=config.n_ctx if config.n_ctx is not None else DEFAULT_CHAT_CONTEXT,
                n_gpu_layers=config.n_gpu_layers if config.n_gpu_layers is not None else OFFLOAD_ALL_LAYERS,
                flash_attn=config.flash_attn if config.flash_attn is not None else True,
                enable_thinking=config.enable_thinking
                if config.enable_thinking is not None
                else DEFAULT_ENABLE_THINKING,
                temperature=config.temperature if config.temperature is not None else DEFAULT_TEMPERATURE,
                top_p=config.top_p if config.top_p is not None else DEFAULT_TOP_P,
                top_k=config.top_k if config.top_k is not None else DEFAULT_TOP_K,
                n_threads=config.n_threads,
                model_root=config.model_root or DEFAULT_MODEL_ROOT,
            )
        case Provider.OPENAI | Provider.FIREWORKS:
            return OpenAICompatChat(config)
        case _:
            error = _unsupported("chat", config)
            raise error


def build_embedding_model(config: ModelConfig) -> EmbeddingModel:
    """Embedding model for query preprocessing."""
    match config.provider:
        case Provider.LLAMA:
            return LlamaCppEmbedder(
                config,
                n_ctx=config.n_ctx if config.n_ctx is not None else EMBEDDING_CONTEXT_LIMIT,
                n_gpu_layers=config.n_gpu_layers if config.n_gpu_layers is not None else OFFLOAD_ALL_LAYERS,
                n_threads=config.n_threads,
                model_root=config.model_root or DEFAULT_MODEL_ROOT,
            )
        case _:
            error = _unsupported("embedding", config)
            raise error
