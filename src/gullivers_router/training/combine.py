"""Align local and cloud generations into judge-ready pairs by prompt id."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from gullivers_router.training.dataset import Prompt


@dataclass(frozen=True, slots=True)
class ResponsePair:
    """A prompt with the local and cloud responses to be judged."""

    prompt: Prompt
    local_response: str
    cloud_response: str


def align_pairs(
    prompts: Sequence[Prompt],
    local_responses: Mapping[str, str],
    cloud_responses: Mapping[str, str],
) -> list[ResponsePair]:
    """Join responses to prompts by id, keeping only prompts answered by both models.

    Aligning by id (not position) means a missing or failed generation drops a single pair
    without shifting the others, so partial results from a resumed run stay consistent.
    """
    return [
        ResponsePair(
            prompt=prompt,
            local_response=local_responses[prompt.id],
            cloud_response=cloud_responses[prompt.id],
        )
        for prompt in prompts
        if prompt.id in local_responses and prompt.id in cloud_responses
    ]
