"""Source and shape training prompts from OpenLeecher/lmsys_chat_1m_clean."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DATASET_ID = "OpenLeecher/lmsys_chat_1m_clean"
SAMPLES_PER_CATEGORY = 1000


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


_DEBUG_PATTERN = re.compile(r"\b(bug|fix|fixed|error|debug|broken|exception|traceback|crash)\b", re.IGNORECASE)

_CATEGORY_BY_RAW: dict[str, Category] = {
    "trivia": Category.FACTUAL_KNOWLEDGE,
    "explanation": Category.FACTUAL_KNOWLEDGE,
    "math": Category.MATHEMATICAL_REASONING,
    "classification": Category.SENTIMENT_CLASSIFICATION,
    "text analysis": Category.SENTIMENT_CLASSIFICATION,
    "summarization": Category.TEXT_SUMMARISATION,
    "extraction": Category.NAMED_ENTITY_RECOGNITION,
    "reasoning": Category.LOGICAL_REASONING,
    "logic": Category.LOGICAL_REASONING,
}


def extract_prompt_text(conversations: Sequence[Mapping[str, str]]) -> str | None:
    """Return the human turn's text from a two-turn conversation, if present."""
    for turn in conversations:
        if turn.get("from") == "human":
            return turn.get("value")
    return None


def classify(raw_category: str, prompt_text: str) -> Category | None:
    """Map a raw dataset category to a routing category, or None to skip it."""
    key = raw_category.strip().lower()
    if key == "coding":
        return Category.CODE_DEBUGGING if _DEBUG_PATTERN.search(prompt_text) else Category.CODE_GENERATION
    return _CATEGORY_BY_RAW.get(key)


def load_prompts(samples_per_category: int = SAMPLES_PER_CATEGORY, split: str = "train") -> list[Prompt]:
    """Load a balanced sample of prompts, one bucket per routing category."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split=split)
    counts: dict[Category, int] = defaultdict(int)
    prompts: list[Prompt] = []
    for row in dataset:
        text = extract_prompt_text(row["conversations"])
        if not text:
            continue
        category = classify(row["category"], text)
        if category is None or counts[category] >= samples_per_category:
            continue
        counts[category] += 1
        prompts.append(Prompt(id=row["id"], category=category, text=text))
        if all(counts[category] >= samples_per_category for category in Category):
            break
    return prompts
