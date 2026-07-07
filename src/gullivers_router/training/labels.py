"""Turn judge scores into 0/1 routing labels (outline §2, Label Engineering)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gullivers_router.training import store

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from gullivers_router.training.dataset import Prompt
    from gullivers_router.training.judge import Judgement

LOCAL_LABEL = 0
CLOUD_LABEL = 1
DEFAULT_MARGIN = 2


def label(local_score: int, cloud_score: int, margin: int = DEFAULT_MARGIN) -> int:
    """Return 1 (route to cloud) when cloud beats local by ``margin``, else 0 (local)."""
    return CLOUD_LABEL if cloud_score - local_score >= margin else LOCAL_LABEL


def build_labels(
    prompts: Sequence[Prompt],
    judgements: Sequence[Judgement],
    out: Path,
    margin: int = DEFAULT_MARGIN,
) -> None:
    """Write final training rows, skipping unscored judgements and already-labelled ids."""
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
                "label": label(judgement.local_score, judgement.cloud_score, margin),
            },
        )
