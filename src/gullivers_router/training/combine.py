"""Combine local and cloud batch generations into aligned pair records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gullivers_router.inference.base import user_message

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gullivers_router.inference.base import BatchChatModel
    from gullivers_router.training.dataset import Prompt


@dataclass(frozen=True, slots=True)
class ResponsePair:
    """A prompt with the local and cloud responses to be judged."""

    prompt: Prompt
    local_response: str
    cloud_response: str


def generate_pairwise(
    prompts: Sequence[Prompt],
    local: BatchChatModel,
    cloud: BatchChatModel,
) -> list[ResponsePair]:
    """Run one prompt list through both models and zip results by index.

    Depends only on the ``BatchChatModel`` protocol, so the local (looped) and
    cloud (Fireworks Batch API) backends stay fully decoupled.
    """
    requests = [user_message(prompt.text) for prompt in prompts]
    local_responses = local.complete_batch(requests)
    cloud_responses = cloud.complete_batch(requests)
    return [
        ResponsePair(prompt=prompt, local_response=local_response, cloud_response=cloud_response)
        for prompt, local_response, cloud_response in zip(prompts, local_responses, cloud_responses, strict=True)
    ]
