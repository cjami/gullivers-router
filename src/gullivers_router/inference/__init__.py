"""Provider-agnostic inference clients for Gulliver's Router."""

from gullivers_router.inference.base import (
    BatchChatModel,
    ChatModel,
    EmbeddingModel,
    Message,
    Provider,
    Role,
    user_message,
)
from gullivers_router.inference.factory import (
    build_batch_chat_model,
    build_chat_model,
    build_embedding_model,
)
from gullivers_router.inference.truncation import truncate_head_tail

__all__ = [
    "BatchChatModel",
    "ChatModel",
    "EmbeddingModel",
    "Message",
    "Provider",
    "Role",
    "build_batch_chat_model",
    "build_chat_model",
    "build_embedding_model",
    "truncate_head_tail",
    "user_message",
]
