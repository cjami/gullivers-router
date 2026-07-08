"""Offline training pipeline: select -> generate -> judge -> label -> embed -> train router."""

from __future__ import annotations

from pathlib import Path

from gullivers_router.training.combine import ResponsePair, align_pairs
from gullivers_router.training.dataset import SAMPLES_PER_CATEGORY, Category, Prompt, load_prompts
from gullivers_router.training.generate import DEFAULT_CONCURRENCY
from gullivers_router.training.pipeline import DEFAULT_OUT, STAGES, run_pipeline
from gullivers_router.training.router import (
    DEFAULT_ACCURACY_GATE,
    DEFAULT_QUALITY_FLOOR,
    DEFAULT_SEED,
    DEFAULT_TARGET_MARGIN,
    DEFAULT_VAL_FRACTION,
    train_router,
)

__all__ = [
    "DEFAULT_ACCURACY_GATE",
    "DEFAULT_QUALITY_FLOOR",
    "DEFAULT_SEED",
    "DEFAULT_TARGET_MARGIN",
    "DEFAULT_VAL_FRACTION",
    "Category",
    "Prompt",
    "ResponsePair",
    "align_pairs",
    "load_prompts",
    "train",
    "train_router",
]


def train(
    samples_per_category: int = SAMPLES_PER_CATEGORY,
    out: str = DEFAULT_OUT,
    stages: tuple[str, ...] = STAGES,
    workers: int = DEFAULT_CONCURRENCY,
) -> None:
    """Build the router training dataset (select -> generate -> judge -> targets)."""
    run_pipeline(samples_per_category, Path(out), stages=stages, workers=workers)
