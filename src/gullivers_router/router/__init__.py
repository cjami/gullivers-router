"""Runtime router: batch routing between local and cloud models."""

from __future__ import annotations

import json
import sys
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from gullivers_router.config import Settings
from gullivers_router.inference.base import user_message
from gullivers_router.inference.factory import build_chat_model, build_embedding_model
from gullivers_router.router.model import load_numpy, probabilities
from gullivers_router.training.generate import DEFAULT_CONCURRENCY, complete_with_retry

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gullivers_router.config import ModelConfig
    from gullivers_router.inference.base import ChatModel, EmbeddingModel

DEFAULT_INPUT = Path("/input/tasks.json")
DEFAULT_OUTPUT = Path("/output/results.json")
DEFAULT_ROUTER_WEIGHTS = Path("artifacts/training/router.npz")
LOCAL_ROUTE = "local"
CLOUD_ROUTE = "cloud"

__all__ = [
    "CLOUD_ROUTE",
    "DEFAULT_INPUT",
    "DEFAULT_OUTPUT",
    "DEFAULT_ROUTER_WEIGHTS",
    "LOCAL_ROUTE",
    "RuntimeContext",
    "RuntimeOptions",
    "Task",
    "run",
    "run_with_context",
]


@dataclass(frozen=True, slots=True)
class Task:
    """One runtime task."""

    task_id: str
    prompt: str


@dataclass(frozen=True, slots=True)
class _Decision:
    task: Task
    route: str
    risk: float
    threshold: float
    model: str


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    """Runtime file paths and execution switches."""

    input_path: Path = DEFAULT_INPUT
    output_path: Path = DEFAULT_OUTPUT
    router_weights: Path = DEFAULT_ROUTER_WEIGHTS
    workers: int = DEFAULT_CONCURRENCY
    classify_only: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Runtime dependencies."""

    settings: Settings
    chat_factory: Callable[[ModelConfig], ChatModel]
    embedding_factory: Callable[[ModelConfig], EmbeddingModel]


def run(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    router_weights: Path = DEFAULT_ROUTER_WEIGHTS,
    workers: int = DEFAULT_CONCURRENCY,
    classify_only: bool = False,
) -> None:
    """Run the batch router."""
    run_with_context(
        RuntimeOptions(
            input_path=input_path,
            output_path=output_path,
            router_weights=router_weights,
            workers=workers,
            classify_only=classify_only,
        ),
        RuntimeContext(
            settings=Settings.from_env(),
            chat_factory=build_chat_model,
            embedding_factory=build_embedding_model,
        ),
    )


def run_with_context(options: RuntimeOptions, context: RuntimeContext) -> None:
    """Run the batch router with explicit dependencies."""
    tasks = load_tasks(options.input_path)
    _log(f"loaded {len(tasks)} tasks <- {options.input_path}")
    local_model = _model_name(context.settings.local)
    cloud_model = _model_name(context.settings.cloud)
    _log(f"routing with local={local_model} cloud={cloud_model} weights={options.router_weights}")

    embedder = context.embedding_factory(context.settings.embedding)
    decisions = classify_tasks(
        tasks,
        embedder,
        load_numpy(options.router_weights),
        local_model=local_model,
        cloud_model=cloud_model,
    )

    if options.classify_only:
        _log(f"classify-only: writing {len(decisions)} route diagnostics -> {options.output_path}")
        write_results(options.output_path, [classification_record(decision) for decision in decisions])
        return

    local = context.chat_factory(context.settings.local)
    cloud = context.chat_factory(context.settings.cloud)
    answers = answer_tasks(decisions, local, cloud, workers=options.workers)
    _log(f"writing {len(answers)} answers -> {options.output_path}")
    write_results(options.output_path, [{"task_id": task_id, "answer": answer} for task_id, answer in answers])


def load_tasks(path: Path) -> list[Task]:
    """Read and validate tasks."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        msg = f"failed to read input file {path}: {error}"
        raise RuntimeError(msg) from error
    except json.JSONDecodeError as error:
        msg = f"input file {path} is not valid JSON: {error}"
        raise ValueError(msg) from error

    if not isinstance(raw, list):
        msg = "input must be a JSON array"
        raise TypeError(msg)

    tasks: list[Task] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            msg = f"task at index {index} must be an object"
            raise TypeError(msg)
        task_id = item.get("task_id")
        prompt = item.get("prompt")
        if not isinstance(task_id, str) or not task_id:
            msg = f"task at index {index} must have a non-empty string task_id"
            raise ValueError(msg)
        if not isinstance(prompt, str) or not prompt:
            msg = f"task {task_id} must have a non-empty string prompt"
            raise ValueError(msg)
        tasks.append(Task(task_id=task_id, prompt=prompt))
    return tasks


def classify_tasks(
    tasks: Sequence[Task],
    embedder: EmbeddingModel,
    weights: dict[str, np.ndarray],
    *,
    local_model: str,
    cloud_model: str,
) -> list[_Decision]:
    """Classify tasks into local or cloud routes."""
    if not tasks:
        return []

    _log(f"embedding {len(tasks)} prompts")
    embeddings = np.asarray([embedder.embed(task.prompt) for task in tasks], dtype=np.float64)
    risks = probabilities(weights, embeddings)
    threshold = float(weights["alpha"])
    _log(f"classifying against threshold={threshold:.3f} (route to cloud when risk >= threshold)")
    decisions: list[_Decision] = []
    for task, risk in zip(tasks, risks, strict=True):
        route = CLOUD_ROUTE if risk >= threshold else LOCAL_ROUTE
        model = cloud_model if route == CLOUD_ROUTE else local_model
        _log(f"[classify] {task.task_id}: risk={float(risk):.3f} -> {route} ({model})")
        decisions.append(
            _Decision(
                task=task,
                route=route,
                risk=float(risk),
                threshold=threshold,
                model=model,
            )
        )
    local_count = sum(1 for decision in decisions if decision.route == LOCAL_ROUTE)
    _log(f"[classify] routed {local_count} -> local, {len(decisions) - local_count} -> cloud")
    return decisions


def classification_record(decision: _Decision) -> dict[str, object]:
    """Render a classifier diagnostic row."""
    return {
        "task_id": decision.task.task_id,
        "route": decision.route,
        "risk": decision.risk,
        "threshold": decision.threshold,
        "model": decision.model,
    }


def answer_tasks(
    decisions: Sequence[_Decision],
    local: ChatModel,
    cloud: ChatModel,
    *,
    workers: int,
) -> list[tuple[str, str]]:
    """Generate answers for routed tasks, preserving input order."""
    local_decisions = [decision for decision in decisions if decision.route == LOCAL_ROUTE]
    cloud_decisions = [decision for decision in decisions if decision.route == CLOUD_ROUTE]
    _log(f"answering {len(local_decisions)} local (sequential), {len(cloud_decisions)} cloud ({workers} workers)")

    answers: dict[str, str] = {}
    for index, decision in enumerate(local_decisions, start=1):
        _log(f"[local {index}/{len(local_decisions)}] {decision.task.task_id}")
        answers[decision.task.task_id] = local.complete(user_message(decision.task.prompt))

    if cloud_decisions:
        answers.update(_answer_cloud(cloud_decisions, cloud, workers=workers))

    return [(decision.task.task_id, answers[decision.task.task_id]) for decision in decisions]


def write_results(path: Path, records: Sequence[dict[str, object]]) -> None:
    """Write the final JSON result array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), ensure_ascii=False), encoding="utf-8")


def _answer_cloud(decisions: Sequence[_Decision], cloud: ChatModel, *, workers: int) -> dict[str, str]:
    lock = Lock()
    answers: dict[str, str] = {}
    max_workers = max(1, workers)
    total = len(decisions)
    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        submitted = {
            pool.submit(complete_with_retry, cloud, user_message(decision.task.prompt)): decision.task.task_id
            for decision in decisions
        }
        for completed, future in enumerate(futures.as_completed(submitted), start=1):
            task_id = submitted[future]
            answer = future.result()
            with lock:
                answers[task_id] = answer
            _log(f"[cloud {completed}/{total}] {task_id} done")
    return answers


def _log(message: str) -> None:
    print(f"[router] {message}", file=sys.stderr, flush=True)


def _model_name(config: object) -> str:
    model = getattr(config, "model", None)
    repo_id = getattr(config, "repo_id", None)
    return str(model or repo_id or "unknown")
