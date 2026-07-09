"""Provider-agnostic contracts shared by every inference backend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import BaseModel

DEFAULT_INFERENCE_SEED = 1
StructuredOutput = TypeVar("StructuredOutput", bound="BaseModel")


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


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Cumulative token counts reported by a provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class ChatModel(Protocol):
    """A model that answers one prompt at a time (runtime path)."""

    def complete(self, messages: Sequence[Message]) -> str:
        """Return the model's answer to a single prompt."""
        ...


@runtime_checkable
class UsageReporting(Protocol):
    """A model that reports cumulative token usage across its calls."""

    @property
    def usage(self) -> TokenUsage:
        """Return the tokens consumed so far."""
        ...


@runtime_checkable
class Closeable(Protocol):
    """A backend that can release its underlying model and memory."""

    def close(self) -> None:
        """Release any resources the backend holds."""
        ...


@runtime_checkable
class StructuredChatModel(ChatModel, Protocol):
    """A chat model that can constrain output to a Pydantic response model."""

    def complete_structured(
        self,
        messages: Sequence[Message],
        response_model: type[StructuredOutput],
    ) -> StructuredOutput:
        """Return the model's answer parsed as ``response_model``."""
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


def system_and_user_message(system: str, text: str) -> list[Message]:
    """Wrap a system instruction followed by the user prompt."""
    return [Message(Role.SYSTEM, system), Message(Role.USER, text)]
