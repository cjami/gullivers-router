"""Resumable embedding of the training prompts (outline §3).

Embeddings are produced once and cached keyed by prompt id, so the router-training stage can
join them against the score-delta rows without re-running the embedder. Generation is sequential (a single
local GGUF cannot batch-decode) and appends each vector as it lands, so a crash costs at most the
in-flight item and a rerun resumes from the gap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gullivers_router.training import store

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from gullivers_router.inference.base import EmbeddingModel
    from gullivers_router.training.dataset import Prompt


def run_embed(prompts: Sequence[Prompt], model: EmbeddingModel, out: Path) -> None:
    """Embed each prompt one at a time, resuming past whatever is already done."""
    from tqdm import tqdm

    done = store.completed_ids(out)
    remaining = [prompt for prompt in prompts if prompt.id not in done]
    for prompt in tqdm(remaining, desc="embedding"):
        embedding = model.embed(prompt.text)
        store.append(out, {"id": prompt.id, "embedding": embedding})
