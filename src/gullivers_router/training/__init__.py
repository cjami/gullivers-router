"""Matrix-factorization training pipeline (offline, batch-based)."""

from __future__ import annotations

from pathlib import Path

from gullivers_router.training.combine import ResponsePair, align_pairs
from gullivers_router.training.dataset import SAMPLES_PER_CATEGORY, Category, Prompt, load_prompts
from gullivers_router.training.generate import DEFAULT_CONCURRENCY
from gullivers_router.training.labels import DEFAULT_MARGIN
from gullivers_router.training.pipeline import DEFAULT_OUT, STAGES, run_pipeline

__all__ = [
    "Category",
    "Prompt",
    "ResponsePair",
    "align_pairs",
    "load_prompts",
    "train",
]


def train(
    samples_per_category: int = SAMPLES_PER_CATEGORY,
    out: str = DEFAULT_OUT,
    margin: int = DEFAULT_MARGIN,
    stages: tuple[str, ...] = STAGES,
    workers: int = DEFAULT_CONCURRENCY,
) -> None:
    """Build the labelled training dataset (select -> generate -> judge -> label)."""
    run_pipeline(samples_per_category, Path(out), margin=margin, stages=stages, workers=workers)
