"""Matrix-factorization training pipeline (offline, batch-based)."""

from __future__ import annotations

from gullivers_router.training.combine import ResponsePair, generate_pairwise
from gullivers_router.training.dataset import Category, Prompt, load_prompts

__all__ = [
    "Category",
    "Prompt",
    "ResponsePair",
    "generate_pairwise",
    "load_prompts",
    "train",
]


def train() -> None:
    """Run the dataset-build and matrix-factorization training pipeline."""
    print("Training pipeline is not implemented yet.")
