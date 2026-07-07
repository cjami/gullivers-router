"""Provider-agnostic inference clients for Gulliver's Router."""

from gullivers_router.inference.base import (
    ChatModel,
    EmbeddingModel,
    Message,
    Provider,
    Role,
    StructuredChatModel,
    user_message,
)
from gullivers_router.inference.factory import build_chat_model, build_embedding_model
from gullivers_router.inference.structured import complete_structured
from gullivers_router.inference.truncation import truncate_head_tail

__all__ = [
    "ChatModel",
    "EmbeddingModel",
    "Message",
    "Provider",
    "Role",
    "StructuredChatModel",
    "build_chat_model",
    "build_embedding_model",
    "complete_structured",
    "truncate_head_tail",
    "user_message",
]
