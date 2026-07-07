"""Head-and-tail truncation for the embedding context window (outline §3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

EMBEDDING_CONTEXT_LIMIT = 2048
HEAD_TOKENS = 1024
TAIL_TOKENS = 1024


def truncate_head_tail(
    tokens: Sequence[int],
    limit: int = EMBEDDING_CONTEXT_LIMIT,
    head: int = HEAD_TOKENS,
    tail: int = TAIL_TOKENS,
) -> list[int]:
    """Keep the leading ``head`` and trailing ``tail`` tokens when over ``limit``.

    Prompts within the limit pass through unchanged; longer prompts retain their
    instruction context and final constraints while dropping the middle.
    """
    if len(tokens) <= limit:
        return list(tokens)
    return [*tokens[:head], *tokens[len(tokens) - tail :]]
