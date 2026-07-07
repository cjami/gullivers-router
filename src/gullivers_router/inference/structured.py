"""Shared helpers for provider-specific structured chat completions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gullivers_router.inference.base import StructuredChatModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import BaseModel

    from gullivers_router.inference.base import ChatModel, Message, StructuredOutput


def complete_structured(
    model: ChatModel,
    messages: Sequence[Message],
    response_model: type[StructuredOutput],
) -> StructuredOutput:
    """Use native structured output when available, else validate plain JSON locally."""
    if isinstance(model, StructuredChatModel):
        return model.complete_structured(messages, response_model)
    return response_model.model_validate_json(model.complete(messages))


def openai_json_schema_response_format(response_model: type[BaseModel]) -> dict:
    """Build the OpenAI-compatible JSON Schema response format."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": response_model.model_json_schema(),
            "strict": True,
        },
    }


def llama_cpp_json_schema_response_format(response_model: type[BaseModel]) -> dict:
    """Build llama-cpp-python's JSON grammar response format."""
    return {
        "type": "json_object",
        "schema": response_model.model_json_schema(),
    }
