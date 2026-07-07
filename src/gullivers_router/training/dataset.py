"""Source and shape training prompts from OpenLeecher/lmsys_chat_1m_clean.

Rows in the dataset are effectively shuffled (not sorted by id or category), so a single
streaming pass taking the first matches per bucket is unbiased and deterministic. Buckets
fed by several source categories are split with an equal quota per source, backfilled from
the remaining sources if one runs dry.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from gullivers_router.training import store

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

DATASET_ID = "OpenLeecher/lmsys_chat_1m_clean"
SAMPLES_PER_CATEGORY = 1000
QUALITY_FLAW = "normal"


class Category(StrEnum):
    """The eight routing categories (outline §2)."""

    FACTUAL_KNOWLEDGE = "factual_knowledge"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    SENTIMENT_CLASSIFICATION = "sentiment_classification"
    TEXT_SUMMARISATION = "text_summarisation"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    CODE_DEBUGGING = "code_debugging"
    LOGICAL_REASONING = "logical_reasoning"
    CODE_GENERATION = "code_generation"


@dataclass(frozen=True, slots=True)
class Prompt:
    """A single training prompt carrying its stable id and routing category."""

    id: str
    category: Category
    text: str


# Each routing category is fed by one or more raw dataset categories (verified against the
# dataset's actual `category` values, not the outline's aspirational names).
_SOURCES: dict[Category, tuple[str, ...]] = {
    Category.FACTUAL_KNOWLEDGE: ("explanation", "trivia"),
    Category.MATHEMATICAL_REASONING: ("math",),
    Category.SENTIMENT_CLASSIFICATION: ("sentiment analysis", "text classification"),
    Category.TEXT_SUMMARISATION: ("summarization",),
    Category.NAMED_ENTITY_RECOGNITION: ("information extraction",),
    Category.CODE_DEBUGGING: ("debugging",),
    Category.LOGICAL_REASONING: ("logical reasoning",),
    Category.CODE_GENERATION: ("coding",),
}

_CATEGORY_BY_RAW: dict[str, Category] = {raw: category for category, raws in _SOURCES.items() for raw in raws}


def extract_prompt_text(conversations: Sequence[Mapping[str, str]]) -> str | None:
    """Return the human turn's text from a two-turn conversation, if present."""
    for turn in conversations:
        if turn.get("from") == "human":
            return turn.get("value")
    return None


def classify(raw_category: str) -> Category | None:
    """Map a raw dataset category to a routing category, or None to skip it."""
    return _CATEGORY_BY_RAW.get(raw_category.strip().lower())


def _source_quotas(category: Category, total: int) -> dict[str, int]:
    """Split ``total`` samples evenly across a category's source categories."""
    sources = _SOURCES[category]
    base, remainder = divmod(total, len(sources))
    return {source: base + (1 if index < remainder else 0) for index, source in enumerate(sources)}


class _BalancedSampler:
    """Fill each category to ``target`` with an equal per-source quota and backfill."""

    def __init__(self, target: int) -> None:
        self._target = target
        self._quotas = {category: _source_quotas(category, target) for category in Category}
        self._selected: dict[Category, list[Prompt]] = {category: [] for category in Category}
        self._overflow: dict[Category, list[Prompt]] = {category: [] for category in Category}
        self._source_counts: dict[Category, dict[str, int]] = {category: defaultdict(int) for category in Category}

    def offer(self, raw_category: str, prompt: Prompt) -> None:
        """Accept ``prompt`` toward its source quota, or hold it for backfill."""
        category = prompt.category
        raw = raw_category.strip().lower()
        if self._source_counts[category][raw] < self._quotas[category][raw]:
            self._selected[category].append(prompt)
            self._source_counts[category][raw] += 1
        elif len(self._overflow[category]) < self._target:
            self._overflow[category].append(prompt)

    def is_complete(self) -> bool:
        """Report whether every category can now be filled to ``target``."""
        return all(
            len(self._selected[category]) + len(self._overflow[category]) >= self._target for category in Category
        )

    def result(self) -> list[Prompt]:
        """Combine quota picks with backfill, capped at ``target`` per category."""
        prompts: list[Prompt] = []
        for category in Category:
            selected = self._selected[category]
            shortfall = self._target - len(selected)
            prompts.extend(selected)
            if shortfall > 0:
                prompts.extend(self._overflow[category][:shortfall])
        return prompts


def load_prompts(samples_per_category: int = SAMPLES_PER_CATEGORY, split: str = "train") -> list[Prompt]:
    """Load a balanced sample of prompts, one bucket per routing category."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split=split, streaming=True)
    sampler = _BalancedSampler(samples_per_category)
    for row in dataset:
        if (row.get("flaw") or "").strip().lower() != QUALITY_FLAW:
            continue
        category = classify(row["category"] or "")
        if category is None:
            continue
        text = extract_prompt_text(row["conversations"])
        if not text:
            continue
        sampler.offer(row["category"], Prompt(id=row["id"], category=category, text=text))
        if sampler.is_complete():
            break
    return sampler.result()


def save_prompts(prompts: Sequence[Prompt], path: Path) -> None:
    """Freeze the sampled prompts to JSONL so downstream stages share one input."""
    for prompt in prompts:
        store.append(path, {"id": prompt.id, "category": prompt.category.value, "text": prompt.text})


def load_prompts_file(path: Path) -> list[Prompt]:
    """Read prompts previously frozen by :func:`save_prompts`."""
    return [
        Prompt(id=record["id"], category=Category(record["category"]), text=record["text"])
        for record in store.read_records(path)
    ]
