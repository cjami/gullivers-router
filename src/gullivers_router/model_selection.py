"""Select a cloud model from the harness-provided allowlist by family preference."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_MODEL_PREFERENCE: tuple[str, ...] = ("minimax", "glm", "deepseek", "gemma", "kimi", "qwen")


def select_model(allowed: Sequence[str], preference: Sequence[str] = DEFAULT_MODEL_PREFERENCE) -> str:
    """Pick the allowed model whose family appears earliest in ``preference``.

    Families are matched as case-insensitive substrings of the model id. Among models sharing the
    winning family, the first in ``allowed`` order wins. Falls back to the first allowed model when
    no family matches, so any non-empty allowlist yields a usable choice.
    """
    for family in preference:
        for model in allowed:
            if family in model.lower():
                return model
    return allowed[0]
