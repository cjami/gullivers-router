"""Runtime router: batch routing between local and cloud models."""

from __future__ import annotations

import gc
import json
import os
import sys
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gullivers_router.config import Settings
from gullivers_router.inference.base import Closeable, UsageReporting, system_and_user_message
from gullivers_router.inference.factory import build_chat_model, build_embedding_model
from gullivers_router.router.model import category_thresholds, load_numpy, predict_categories, probabilities
from gullivers_router.training.generate import DEFAULT_CONCURRENCY, complete_with_retry

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gullivers_router.config import ModelConfig
    from gullivers_router.inference.base import ChatModel, EmbeddingModel

DEFAULT_INPUT = Path("examples/practice_tasks.json")
DEFAULT_OUTPUT = Path("outputs/results.json")
DEFAULT_ROUTER_WEIGHTS = Path("artifacts/training/router.npz")
LOCAL_ROUTE = "local"
CLOUD_ROUTE = "cloud"

_CONCISE_SYSTEM_PROMPT = "Answer accurately and concisely. Follow requested format constraints; skip filler."
_CATEGORY_SYSTEM_HINTS = {
    "code_debugging": "For debugging: identify the bug and provide corrected implementation.",
    "code_generation": (
        "For code: use the requested language; write a concise complete function; no examples unless asked."
    ),
    "factual_knowledge": "For facts: answer the requested concept, definition, or mechanism; avoid uncertain extras.",
    "logical_reasoning": "For logic: satisfy every condition before answering.",
    "mathematical_reasoning": "For math: show brief calculations, then the final answer.",
    "named_entity_recognition": (
        "For NER: scan the full text for people, organizations, locations, and date/time expressions; label each type."
    ),
    "sentiment_classification": "For sentiment: label positive, negative, or mixed; justify only if asked.",
    "text_summarisation": "For summaries: preserve who does what and obey format or length constraints.",
}

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
    category: str | None


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

    _log_rss("after classification")
    _release_embedder(embedder)
    _log_rss("after releasing embedder")

    local = context.chat_factory(context.settings.local)
    cloud = context.chat_factory(context.settings.cloud)
    answers = answer_tasks(decisions, local, cloud, workers=options.workers)
    _log_rss("after answering")
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
    categories, thresholds = _thresholds(weights, embeddings, len(tasks))
    _log("classifying with per-category thresholds (route to cloud when risk >= threshold)")
    decisions: list[_Decision] = []
    for task, risk, threshold, category in zip(tasks, risks, thresholds, categories, strict=True):
        route = CLOUD_ROUTE if risk >= threshold else LOCAL_ROUTE
        model = cloud_model if route == CLOUD_ROUTE else local_model
        _log(f"[classify] {task.task_id}: risk={float(risk):.3f} thr={threshold:.3f} {category} -> {route} ({model})")
        decisions.append(
            _Decision(
                task=task,
                route=route,
                risk=float(risk),
                threshold=float(threshold),
                model=model,
                category=category,
            )
        )
    local_count = sum(1 for decision in decisions if decision.route == LOCAL_ROUTE)
    _log(f"[classify] routed {local_count} -> local, {len(decisions) - local_count} -> cloud")
    return decisions


def _thresholds(
    weights: dict[str, np.ndarray],
    embeddings: np.ndarray,
    count: int,
) -> tuple[list[str | None], np.ndarray]:
    """Per-task decision thresholds, using the category head when the bundle carries one."""
    predicted = predict_categories(weights, embeddings)
    if predicted is None:
        return [None] * count, np.full(count, float(weights["alpha"]))
    return [str(category) for category in predicted], category_thresholds(weights, predicted)


def classification_record(decision: _Decision) -> dict[str, object]:
    """Render a classifier diagnostic row."""
    return {
        "task_id": decision.task.task_id,
        "route": decision.route,
        "risk": decision.risk,
        "threshold": decision.threshold,
        "category": decision.category,
        "model": decision.model,
    }


def answer_tasks(
    decisions: Sequence[_Decision],
    local: ChatModel,
    cloud: ChatModel,
    *,
    workers: int,
) -> list[tuple[str, str]]:
    """Generate answers for routed tasks, preserving input order.

    Cloud requests are network-bound and dispatched to a thread pool, while the local model is
    CPU- and memory-bound and runs one prompt at a time in this thread. Both proceed at once, so
    local inference (and its cold-start load) overlaps the in-flight cloud latency instead of
    waiting behind it.
    """
    local_decisions = [decision for decision in decisions if decision.route == LOCAL_ROUTE]
    cloud_decisions = [decision for decision in decisions if decision.route == CLOUD_ROUTE]
    _log(f"answering {len(local_decisions)} local (sequential), {len(cloud_decisions)} cloud ({workers} workers)")

    answers: dict[str, str] = {}
    with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        cloud_futures = {
            pool.submit(
                complete_with_retry, cloud, system_and_user_message(_system_prompt(decision), decision.task.prompt)
            ): decision.task.task_id
            for decision in cloud_decisions
        }
        for index, decision in enumerate(local_decisions, start=1):
            _log(f"[local {index}/{len(local_decisions)}] {decision.task.task_id}")
            answers[decision.task.task_id] = local.complete(
                system_and_user_message(_system_prompt(decision), decision.task.prompt)
            )
        for completed, future in enumerate(futures.as_completed(cloud_futures), start=1):
            task_id = cloud_futures[future]
            answers[task_id] = future.result()
            _log(f"[cloud {completed}/{len(cloud_decisions)}] {task_id} done")

    if cloud_decisions:
        _log_cloud_usage(cloud)

    return [(decision.task.task_id, answers[decision.task.task_id]) for decision in decisions]


def _system_prompt(decision: _Decision) -> str:
    hint = _CATEGORY_SYSTEM_HINTS.get(decision.category or "")
    if hint is None:
        return _CONCISE_SYSTEM_PROMPT
    return f"{_CONCISE_SYSTEM_PROMPT} {hint}"


def write_results(path: Path, records: Sequence[dict[str, object]]) -> None:
    """Write the final JSON result array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), ensure_ascii=False), encoding="utf-8")


def _log_cloud_usage(cloud: ChatModel) -> None:
    if not isinstance(cloud, UsageReporting):
        return
    usage = cloud.usage
    _log(f"cloud tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")


def _release_embedder(embedder: EmbeddingModel) -> None:
    """Free the embedder's model before the local GGUF loads.

    On the memory-constrained submission host, holding the embedder resident while the local
    model loads can push the process into swap and stall generation. The embedder is unused past
    classification, so releasing it reclaims that headroom.
    """
    if isinstance(embedder, Closeable):
        embedder.close()
    gc.collect()


def _log_rss(label: str) -> None:
    """Log current resident memory on Linux; a no-op where ``/proc`` or ``sysconf`` is unavailable."""
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return
    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
    except (OSError, IndexError, ValueError):
        return
    mib = resident_pages * sysconf("SC_PAGE_SIZE") / 1024 / 1024
    _log(f"rss {label}: {mib:.0f} MiB")


def _log(message: str) -> None:
    print(f"[router] {message}", file=sys.stderr, flush=True)


def _model_name(config: object) -> str:
    model = getattr(config, "model", None)
    repo_id = getattr(config, "repo_id", None)
    return str(model or repo_id or "unknown")
