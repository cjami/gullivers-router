"""Turn judge outputs into router training rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gullivers_router.training import store

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from gullivers_router.training.dataset import Prompt
    from gullivers_router.training.judge import Judgement


def build_labels(
    prompts: Sequence[Prompt],
    judgements: Sequence[Judgement],
    out: Path,
) -> None:
    """Write router training rows, skipping unscored judgements and already-written ids."""
    prompt_by_id = {prompt.id: prompt for prompt in prompts}
    done = store.completed_ids(out)
    for judgement in judgements:
        prompt = prompt_by_id.get(judgement.id)
        if prompt is None or judgement.id in done:
            continue
        if judgement.local_score is None or judgement.cloud_score is None:
            continue
        store.append(
            out,
            {
                "id": judgement.id,
                "category": prompt.category.value,
                "local_score": judgement.local_score,
                "cloud_score": judgement.cloud_score,
                "local_rationale": judgement.local_rationale,
                "cloud_rationale": judgement.cloud_rationale,
                "local_quality": judgement.local_quality,
                "cloud_quality": judgement.cloud_quality,
                "preferred_source": judgement.preferred_source,
            },
        )
