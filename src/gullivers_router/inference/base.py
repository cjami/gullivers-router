"""Provider-agnostic contracts shared by every inference backend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_INFERENCE_SEED = 1


class Provider(StrEnum):
    """Source that backs a model role."""

    LLAMA = "llama"
    OPENAI = "openai"
    FIREWORKS = "fireworks"


class Role(StrEnum):
    """Chat message author."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    """A single chat turn."""

    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        """Render to the ``{"role", "content"}`` shape used by chat APIs."""
        return {"role": self.role.value, "content": self.content}


@runtime_checkable
class ChatModel(Protocol):
    """A model that answers one prompt at a time (runtime path)."""

    def complete(self, messages: Sequence[Message]) -> str:
        """Return the model's answer to a single prompt."""
        ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """A model that maps text to a dense vector."""

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for ``text``."""
        ...


def user_message(text: str) -> list[Message]:
    """Wrap raw prompt text as a single-turn user message list."""
    return [Message(Role.USER, text)]
