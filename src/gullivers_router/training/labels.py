"""Turn judge outputs into router training rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gullivers_router.training import store
from gullivers_router.training.judge import PREFERRED_SOURCES, QUALITY_VALUES

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from gullivers_router.training.dataset import Prompt
    from gullivers_router.training.judge import Judgement


def _numeric(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _valid_label(record: dict) -> bool:
    return (
        _numeric(record.get("local_score"))
        and _numeric(record.get("cloud_score"))
        and record.get("local_quality") in QUALITY_VALUES
        and record.get("cloud_quality") in QUALITY_VALUES
        and isinstance(record.get("local_rationale"), str)
        and bool(record["local_rationale"].strip())
        and isinstance(record.get("cloud_rationale"), str)
        and bool(record["cloud_rationale"].strip())
        and record.get("preferred_source") in PREFERRED_SOURCES
    )


def _completed_valid_ids(path: Path) -> set[str]:
    return {record["id"] for record in store.read_records(path) if _valid_label(record)}


def _complete_judgement(judgement: Judgement) -> bool:
    return _valid_label(_judgement_record("", judgement))


def _judgement_record(category: str, judgement: Judgement) -> dict:
    return {
        "id": judgement.id,
        "category": category,
        "local_score": judgement.local_score,
        "cloud_score": judgement.cloud_score,
        "local_rationale": judgement.local_rationale,
        "cloud_rationale": judgement.cloud_rationale,
        "local_quality": judgement.local_quality,
        "cloud_quality": judgement.cloud_quality,
        "preferred_source": judgement.preferred_source,
    }


def build_labels(
    prompts: Sequence[Prompt],
    judgements: Sequence[Judgement],
    out: Path,
) -> None:
    """Write router training rows, skipping unscored judgements and already-written ids."""
    prompt_by_id = {prompt.id: prompt for prompt in prompts}
    done = _completed_valid_ids(out)
    for judgement in judgements:
        prompt = prompt_by_id.get(judgement.id)
        if prompt is None or judgement.id in done:
            continue
        if not _complete_judgement(judgement):
            continue
        store.append(out, _judgement_record(prompt.category.value, judgement))
