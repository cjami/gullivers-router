"""Chat backend for any OpenAI-compatible endpoint (OpenAI, Fireworks, …)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gullivers_router.config import ModelConfig
    from gullivers_router.inference.base import Message, StructuredOutput

from gullivers_router.inference.base import DEFAULT_INFERENCE_SEED
from gullivers_router.inference.structured import openai_json_schema_response_format


class OpenAICompatChat:
    """Single-call chat completion over an OpenAI-compatible HTTP API."""

    def __init__(self, config: ModelConfig) -> None:
        """Configure the client; requires an API key."""
        if not config.api_key:
            msg = f"provider {config.provider} requires an API key"
            raise ValueError(msg)
        self._config = config
        self._client = None

    def _get_client(self):  # noqa: ANN202
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._config.api_key, base_url=self._config.base_url)
        return self._client

    def complete(self, messages: Sequence[Message]) -> str:
        """Generate a response for a single prompt."""
        response = self._get_client().chat.completions.create(
            model=self._config.model,
            messages=[m.as_dict() for m in messages],
            seed=DEFAULT_INFERENCE_SEED,
        )
        return response.choices[0].message.content or ""

    def complete_structured(
        self,
        messages: Sequence[Message],
        response_model: type[StructuredOutput],
    ) -> StructuredOutput:
        """Generate a response constrained to a Pydantic model schema."""
        response = self._get_client().chat.completions.create(
            model=self._config.model,
            messages=[m.as_dict() for m in messages],
            seed=DEFAULT_INFERENCE_SEED,
            response_format=openai_json_schema_response_format(response_model),
        )
        content = response.choices[0].message.content or ""
        return response_model.model_validate_json(content)
