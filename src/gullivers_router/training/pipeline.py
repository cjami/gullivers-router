"""Orchestrate the offline dataset build: select -> generate -> judge -> label.

Every stage persists its output keyed by prompt id and short-circuits once complete, so the
whole run is idempotent and resumable: rerun after any interruption and it continues from the
first unfinished item. Stages can also be run in isolation via ``stages`` once their inputs
exist on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gullivers_router.config import Settings
from gullivers_router.inference.factory import build_chat_model
from gullivers_router.training import store
from gullivers_router.training.combine import align_pairs
from gullivers_router.training.dataset import load_prompts, load_prompts_file, save_prompts
from gullivers_router.training.generate import DEFAULT_CONCURRENCY, run_cloud, run_local
from gullivers_router.training.judge import load_judgements, run_judge
from gullivers_router.training.labels import DEFAULT_MARGIN, build_labels

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from gullivers_router.training.dataset import Prompt

DEFAULT_OUT = "artifacts/training"
STAGES = ("local", "cloud", "judge", "labels")


@dataclass(frozen=True, slots=True)
class Artifacts:
    """Filesystem locations for every pipeline stage under one output root."""

    root: Path

    @property
    def prompts(self) -> Path:
        """The frozen prompt sample shared by every stage."""
        return self.root / "prompts.jsonl"

    @property
    def local(self) -> Path:
        """Local model responses keyed by prompt id."""
        return self.root / "local.jsonl"

    @property
    def cloud(self) -> Path:
        """Cloud model responses keyed by prompt id."""
        return self.root / "cloud.jsonl"

    @property
    def judge(self) -> Path:
        """Judge scores keyed by prompt id."""
        return self.root / "judge.jsonl"

    @property
    def labels(self) -> Path:
        """Final labelled training rows."""
        return self.root / "labels.jsonl"


def _select(artifacts: Artifacts, samples_per_category: int) -> list[Prompt]:
    """Load the frozen prompt sample, creating it on first run."""
    if artifacts.prompts.exists():
        return load_prompts_file(artifacts.prompts)
    prompts = load_prompts(samples_per_category)
    save_prompts(prompts, artifacts.prompts)
    return prompts


def run_pipeline(
    samples_per_category: int,
    out: Path,
    *,
    margin: int = DEFAULT_MARGIN,
    stages: Sequence[str] = STAGES,
    workers: int = DEFAULT_CONCURRENCY,
) -> None:
    """Build the labelled training dataset, resuming any unfinished stage."""
    settings = Settings.from_env()
    artifacts = Artifacts(out)

    prompts = _select(artifacts, samples_per_category)
    print(f"selected {len(prompts)} prompts -> {artifacts.prompts}")

    if "local" in stages:
        run_local(prompts, build_chat_model(settings.local), artifacts.local)

    if "cloud" in stages:
        run_cloud(prompts, build_chat_model(settings.cloud), artifacts.cloud, max_workers=workers)

    if "judge" in stages:
        local = {prompt_id: str(response) for prompt_id, response in store.read_map(artifacts.local).items()}
        cloud = {prompt_id: str(response) for prompt_id, response in store.read_map(artifacts.cloud).items()}
        pairs = align_pairs(prompts, local, cloud)
        run_judge(pairs, build_chat_model(settings.judge), artifacts.judge, max_workers=workers)

    if "labels" in stages:
        build_labels(prompts, load_judgements(artifacts.judge), artifacts.labels, margin)
        print(f"wrote labels -> {artifacts.labels}")
