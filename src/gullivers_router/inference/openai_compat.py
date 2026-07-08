"""Chat backend for any OpenAI-compatible endpoint (OpenAI, Fireworks, …)."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gullivers_router.config import ModelConfig
    from gullivers_router.inference.base import Message, StructuredOutput

from gullivers_router.inference.base import DEFAULT_INFERENCE_SEED, TokenUsage
from gullivers_router.inference.structured import openai_json_schema_response_format


class OpenAICompatChat:
    """Single-call chat completion over an OpenAI-compatible HTTP API."""

    def __init__(self, config: ModelConfig) -> None:
        """Configure the client; requires an API key."""
        if not config.api_key:
            msg = f"provider {config.provider} requires an API key"
            raise ValueError(msg)
        self._config = config
        self._extra_body = _reasoning_extra_body(config)
        self._client = None
        self._usage_lock = Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def usage(self) -> TokenUsage:
        """Cumulative tokens consumed across every call on this client."""
        with self._usage_lock:
            return TokenUsage(self._prompt_tokens, self._completion_tokens)

    def _record_usage(self, response) -> None:  # noqa: ANN001 - provider response object
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        with self._usage_lock:
            self._prompt_tokens += prompt
            self._completion_tokens += completion

    def _get_client(self):  # noqa: ANN202
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
                timeout=self._config.timeout_seconds,
            )
        return self._client

    def complete(self, messages: Sequence[Message]) -> str:
        """Generate a response for a single prompt."""
        response = self._get_client().chat.completions.create(
            model=self._config.model,
            messages=[m.as_dict() for m in messages],
            seed=DEFAULT_INFERENCE_SEED,
            extra_body=self._extra_body,
        )
        self._record_usage(response)
        return _completion_content(response)

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
            extra_body=self._extra_body,
        )
        self._record_usage(response)
        content = _completion_content(response)
        return response_model.model_validate_json(content)


def _reasoning_extra_body(config: ModelConfig) -> dict:
    """Disable thinking on hybrid reasoning models; Fireworks reads ``reasoning_effort='none'``.

    Only applied when a role explicitly opts out, leaving reasoning-required models untouched.
    """
    if config.enable_thinking is False:
        return {"reasoning_effort": "none"}
    return {}


def _completion_content(response) -> str:  # noqa: ANN001 - OpenAI-compatible clients return provider objects.
    content = response.choices[0].message.content
    if not content or not content.strip():
        msg = "chat completion returned empty content"
        raise RuntimeError(msg)
    return content
