"""Runtime router: batch routing between local and cloud models."""

from __future__ import annotations

import gc
import json
import os
import sys
from concurrent import futures
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field

from gullivers_router.config import Settings
from gullivers_router.inference.base import (
    Closeable,
    Message,
    Role,
    ThreadAdjustable,
    TokenUsage,
    UsageReporting,
    system_and_user_message,
)
from gullivers_router.inference.factory import build_chat_model, build_embedding_model, build_named_entity_model
from gullivers_router.inference.structured import complete_structured
from gullivers_router.router.deterministic_math import deterministic_math_answer
from gullivers_router.router.model import category_thresholds, load_numpy, predict_categories, probabilities
from gullivers_router.router.ner import answer_named_entities
from gullivers_router.training.generate import DEFAULT_CONCURRENCY, call_with_retry, complete_with_retry

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gullivers_router.config import ModelConfig
    from gullivers_router.inference.base import ChatModel, EmbeddingModel, NamedEntityModel

DEFAULT_INPUT = Path("examples/practice_tasks.json")
DEFAULT_OUTPUT = Path("outputs/results.json")
DEFAULT_ROUTER_WEIGHTS = Path("artifacts/training/router.npz")
DEFAULT_CASCADE_MARGIN = 0.10
LOCAL_ROUTE = "local"
CLOUD_ROUTE = "cloud"
DETERMINISTIC_MATH_ROUTE = "deterministic_math"

_CONCISE_SYSTEM_PROMPT = "Answer correctly in the fewest words. No filler."
_CATEGORY_SYSTEM_HINTS = {
    "code_debugging": "Identify bug; provide corrected implementation only.",
    "code_generation": "Use requested language; complete function only; no examples unless asked.",
    "logical_reasoning": "Show brief deductions and final answer.",
    "mathematical_reasoning": "Show brief calculations and final answer.",
    "named_entity_recognition": "Find all people, organizations, locations, full dates/times; label type.",
    "sentiment_classification": "Label positive, negative, or neutral; briefly justify.",
    "text_summarisation": "Preserve all facts; obey length/format.",
}
_FAST_CLOUD_CATEGORIES = {
    "code_debugging",
    "code_generation",
    "factual_knowledge",
    "logical_reasoning",
    "mathematical_reasoning",
    "named_entity_recognition",
    "sentiment_classification",
    "text_summarisation",
}
_CASCADE_CATEGORIES = {
    "code_debugging",
    "code_generation",
    "factual_knowledge",
    "logical_reasoning",
    "mathematical_reasoning",
}
_CLOUD_FIRST_CATEGORIES = {
    "code_debugging",
    "code_generation",
}
_LOCAL_FIRST_CATEGORIES: set[str] = set()
_SPECIALIST_FIRST_CATEGORIES = {
    "text_summarisation",
}
_NER_FIRST_CATEGORIES = {"named_entity_recognition"}
_LOCAL_SELF_CHECK_SYSTEM_PROMPT = (
    "You are a strict verifier for a small local model. Given the original task and the local answer, decide "
    "whether the answer is safe to return or should be escalated to a stronger cloud model. Escalate for likely "
    "factual errors, unsupported claims, missed constraints, flawed reasoning, code risks, or format failures. "
    "Do not revise the answer."
)

__all__ = [
    "CLOUD_ROUTE",
    "DEFAULT_INPUT",
    "DEFAULT_OUTPUT",
    "DEFAULT_ROUTER_WEIGHTS",
    "DETERMINISTIC_MATH_ROUTE",
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
    answer: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    """Runtime file paths and execution switches."""

    input_path: Path = DEFAULT_INPUT
    output_path: Path = DEFAULT_OUTPUT
    router_weights: Path = DEFAULT_ROUTER_WEIGHTS
    workers: int = DEFAULT_CONCURRENCY
    classify_only: bool = False
    local_cascade: bool = False
    cascade_margin: float = DEFAULT_CASCADE_MARGIN


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Runtime dependencies."""

    settings: Settings
    chat_factory: Callable[[ModelConfig], ChatModel]
    embedding_factory: Callable[[ModelConfig], EmbeddingModel]
    ner_factory: Callable[[ModelConfig], NamedEntityModel]


def run(  # noqa: PLR0913 - CLI options are passed through explicitly.
    *,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    router_weights: Path = DEFAULT_ROUTER_WEIGHTS,
    workers: int = DEFAULT_CONCURRENCY,
    classify_only: bool = False,
    local_cascade: bool = False,
    cascade_margin: float = DEFAULT_CASCADE_MARGIN,
) -> None:
    """Run the batch router."""
    run_with_context(
        RuntimeOptions(
            input_path=input_path,
            output_path=output_path,
            router_weights=router_weights,
            workers=workers,
            classify_only=classify_only,
            local_cascade=local_cascade,
            cascade_margin=cascade_margin,
        ),
        RuntimeContext(
            settings=Settings.from_env(),
            chat_factory=build_chat_model,
            embedding_factory=build_embedding_model,
            ner_factory=build_named_entity_model,
        ),
    )


def run_with_context(options: RuntimeOptions, context: RuntimeContext) -> None:
    """Run the batch router with explicit dependencies."""
    tasks = load_tasks(options.input_path)
    _log(f"loaded {len(tasks)} tasks <- {options.input_path}")
    local_model = _model_name(context.settings.local)
    specialist_model = _model_name(context.settings.specialist)
    ner_model = _model_name(context.settings.ner)
    cloud_model = _model_name(context.settings.cloud)
    _log(
        f"routing with local={local_model} specialist={specialist_model} ner={ner_model} "
        f"cloud={cloud_model} weights={options.router_weights}"
    )

    embedder = context.embedding_factory(context.settings.embedding)
    decisions = classify_tasks(
        tasks,
        embedder,
        load_numpy(options.router_weights),
        local_model=local_model,
        specialist_model=specialist_model,
        ner_model=ner_model,
        cloud_model=cloud_model,
    )

    if options.classify_only:
        _log(f"classify-only: writing {len(decisions)} route diagnostics -> {options.output_path}")
        write_results(options.output_path, [classification_record(decision) for decision in decisions])
        return

    _log_rss("after classification")
    _release_embedder(embedder)
    _log_rss("after releasing embedder")

    needs_cloud = any(decision.route == CLOUD_ROUTE for decision in decisions)
    cascade_candidates = [decision for decision in decisions if _uses_local_cascade(decision, options.cascade_margin)]
    needs_cascade_cloud = options.local_cascade and bool(cascade_candidates)
    fast_cloud_candidates = [
        decision
        for decision in decisions
        if decision.route == CLOUD_ROUTE or (options.local_cascade and decision in cascade_candidates)
    ]
    cloud = context.chat_factory(context.settings.cloud) if needs_cloud or needs_cascade_cloud else None
    cloud_fast = (
        context.chat_factory(_fast_cloud_config(context.settings.cloud))
        if (needs_cloud or needs_cascade_cloud)
        and any(_uses_fast_cloud_category(decision) for decision in fast_cloud_candidates)
        else None
    )
    has_general_lane = any(
        decision.route == LOCAL_ROUTE and not _uses_specialist_model(decision) and not _uses_ner_model(decision)
        for decision in decisions
    )
    has_specialist_lane = any(_uses_specialist_model(decision) or _uses_ner_model(decision) for decision in decisions)
    local_threads = context.settings.local.n_threads or 1
    initial_local_threads = 1 if has_general_lane and has_specialist_lane else local_threads
    answers = answer_tasks(
        decisions,
        lambda: context.chat_factory(replace(context.settings.local, n_threads=initial_local_threads)),
        lambda: context.chat_factory(_single_threaded(context.settings.specialist)),
        lambda: context.ner_factory(_single_threaded(context.settings.ner)),
        cloud,
        cloud_fast=cloud_fast,
        workers=options.workers,
        local_cascade=options.local_cascade,
        cascade_margin=options.cascade_margin,
        local_threads_after_specialist=local_threads,
    )
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


def classify_tasks(  # noqa: PLR0913 - classifier diagnostics need explicit model labels.
    tasks: Sequence[Task],
    embedder: EmbeddingModel,
    weights: dict[str, np.ndarray],
    *,
    local_model: str,
    specialist_model: str | None = None,
    ner_model: str | None = None,
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
        answer = deterministic_math_answer(task.prompt, category)
        if answer is not None:
            route = DETERMINISTIC_MATH_ROUTE
            model = DETERMINISTIC_MATH_ROUTE
        elif category in _CLOUD_FIRST_CATEGORIES:
            route = CLOUD_ROUTE
            model = cloud_model
        elif category in _NER_FIRST_CATEGORIES:
            route = LOCAL_ROUTE
            model = ner_model or specialist_model or local_model
        elif category in _SPECIALIST_FIRST_CATEGORIES:
            route = LOCAL_ROUTE
            model = specialist_model or local_model
        elif category in _LOCAL_FIRST_CATEGORIES:
            route = LOCAL_ROUTE
            model = local_model
        _log(f"[classify] {task.task_id}: risk={float(risk):.3f} thr={threshold:.3f} {category} -> {route} ({model})")
        decisions.append(
            _Decision(
                task=task,
                route=route,
                risk=float(risk),
                threshold=float(threshold),
                model=model,
                category=category,
                answer=answer,
            )
        )
    local_count = sum(1 for decision in decisions if decision.route == LOCAL_ROUTE)
    cloud_count = sum(1 for decision in decisions if decision.route == CLOUD_ROUTE)
    direct_count = len(decisions) - local_count - cloud_count
    _log(f"[classify] routed {local_count} -> local, {cloud_count} -> cloud, {direct_count} -> deterministic")
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
    record = {
        "task_id": decision.task.task_id,
        "route": decision.route,
        "risk": decision.risk,
        "threshold": decision.threshold,
        "category": decision.category,
        "model": decision.model,
    }
    if decision.answer is not None:
        record["answer"] = decision.answer
    return record


type _CascadeFailureMode = Literal[
    "none",
    "missing_information",
    "reasoning_uncertain",
    "format_risk",
    "factual_uncertain",
    "math_or_code_risk",
]


class _LocalSelfCheck(BaseModel):
    should_escalate: bool
    confidence: float = Field(ge=0.0, le=1.0)
    failure_mode: _CascadeFailureMode
    rationale: str


def answer_tasks(  # noqa: PLR0913 - orchestration wires distinct runtime dependencies.
    decisions: Sequence[_Decision],
    local_factory: Callable[[], ChatModel],
    specialist_factory: Callable[[], ChatModel],
    ner_factory: Callable[[], NamedEntityModel],
    cloud: ChatModel | None,
    *,
    cloud_fast: ChatModel | None = None,
    workers: int,
    local_cascade: bool = False,
    cascade_margin: float = DEFAULT_CASCADE_MARGIN,
    local_threads_after_specialist: int = 1,
) -> list[tuple[str, str]]:
    """Generate answers for routed tasks, preserving input order.

    Cloud requests are network-bound and dispatched to a thread pool. Two single-threaded local
    lanes run concurrently: the general model in this thread, and NER followed by the summary
    specialist in a worker. NER is released before the specialist loads so no more than two local
    models are resident at once.
    """
    ner_decisions = [decision for decision in decisions if _uses_ner_model(decision)]
    specialist_decisions = [decision for decision in decisions if _uses_specialist_model(decision)]
    local_decisions = [
        decision
        for decision in decisions
        if decision.route == LOCAL_ROUTE and not _uses_specialist_model(decision) and not _uses_ner_model(decision)
    ]
    cloud_decisions = [decision for decision in decisions if decision.route == CLOUD_ROUTE]
    direct_decisions = [decision for decision in decisions if decision.route == DETERMINISTIC_MATH_ROUTE]
    cascade_count = sum(1 for decision in local_decisions if _uses_local_cascade(decision, cascade_margin))
    cascade_label = f", {cascade_count} cascade-eligible" if local_cascade else ""
    _log(
        f"answering {len(ner_decisions)} ner, {len(specialist_decisions)} specialist, {len(local_decisions)} local "
        f"({cascade_label.removeprefix(', ') or 'general lane'}), "
        f"{len(cloud_decisions)} cloud ({workers} workers), "
        f"{len(direct_decisions)} deterministic"
    )

    answers = {decision.task.task_id: str(decision.answer) for decision in direct_decisions}
    with (
        futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool,
        futures.ThreadPoolExecutor(max_workers=1) as specialist_pool,
    ):
        cloud_futures = {
            pool.submit(
                complete_with_retry,
                _cloud_model(decision, cloud, cloud_fast),
                system_and_user_message(_system_prompt(decision), decision.task.prompt),
            ): decision.task.task_id
            for decision in cloud_decisions
        }
        specialist_future = specialist_pool.submit(
            _answer_specialist_lane,
            ner_decisions,
            specialist_decisions,
            ner_factory,
            specialist_factory,
        )

        local = local_factory() if local_decisions else None
        local_threads_promoted = False
        try:
            for index, decision in enumerate(local_decisions, start=1):
                _log(f"[local {index}/{len(local_decisions)}] {decision.task.task_id}")
                if local is None:
                    msg = "local model is required for local-routed tasks"
                    raise RuntimeError(msg)
                if not local_threads_promoted and specialist_future.done():
                    _set_threads(local, local_threads_after_specialist)
                    local_threads_promoted = True
                    _log(f"[local] specialist lane complete; promoted to {local_threads_after_specialist} threads")
                local_answer = local.complete(system_and_user_message(_system_prompt(decision), decision.task.prompt))
                if (
                    local_cascade
                    and _uses_local_cascade(decision, cascade_margin)
                    and _should_escalate_local_answer(decision, local_answer, local)
                ):
                    _log(f"[cascade] {decision.task.task_id}: self-check escalated to cloud")
                    cloud_futures[
                        pool.submit(
                            complete_with_retry,
                            _cloud_model(decision, cloud, cloud_fast),
                            system_and_user_message(_system_prompt(decision), decision.task.prompt),
                        )
                    ] = decision.task.task_id
                    continue
                answers[decision.task.task_id] = local_answer
        finally:
            _release_chat_model(local)

        answers.update(specialist_future.result())

        total_cloud = len(cloud_futures)
        for completed, future in enumerate(futures.as_completed(cloud_futures), start=1):
            task_id = cloud_futures[future]
            answers[task_id] = future.result()
            _log(f"[cloud {completed}/{total_cloud}] {task_id} done")

    if cloud_futures:
        _log_cloud_usage(cloud, cloud_fast)

    return [(decision.task.task_id, answers[decision.task.task_id]) for decision in decisions]


def _answer_specialist_lane(
    ner_decisions: Sequence[_Decision],
    specialist_decisions: Sequence[_Decision],
    ner_factory: Callable[[], NamedEntityModel],
    specialist_factory: Callable[[], ChatModel],
) -> dict[str, str]:
    answers: dict[str, str] = {}
    ner = ner_factory() if ner_decisions else None
    try:
        for index, decision in enumerate(ner_decisions, start=1):
            _log(f"[ner {index}/{len(ner_decisions)}] {decision.task.task_id}")
            if ner is None:
                msg = "NER model is required for NER-routed tasks"
                raise RuntimeError(msg)
            answers[decision.task.task_id] = answer_named_entities(decision.task.prompt, ner)
    finally:
        _release_named_entity_model(ner)
    if ner_decisions:
        _log_rss("after releasing ner")

    specialist = specialist_factory() if specialist_decisions else None
    try:
        for index, decision in enumerate(specialist_decisions, start=1):
            _log(f"[specialist {index}/{len(specialist_decisions)}] {decision.task.task_id}")
            if specialist is None:
                msg = "specialist model is required for specialist-routed tasks"
                raise RuntimeError(msg)
            answers[decision.task.task_id] = specialist.complete(_specialist_messages(decision))
    finally:
        _release_chat_model(specialist)
    if specialist_decisions:
        _log_rss("after releasing specialist")
    return answers


def _single_threaded(config: ModelConfig) -> ModelConfig:
    return replace(config, n_threads=1)


def _set_threads(model: ChatModel, n_threads: int) -> None:
    if isinstance(model, ThreadAdjustable):
        model.set_threads(n_threads)


def _cloud_model(decision: _Decision, cloud: ChatModel | None, cloud_fast: ChatModel | None) -> ChatModel:
    if cloud_fast is not None and _uses_fast_cloud_category(decision):
        return cloud_fast
    if cloud is None:
        msg = "cloud model is required for cloud-routed tasks"
        raise RuntimeError(msg)
    return cloud


def _uses_fast_cloud_category(decision: _Decision) -> bool:
    return decision.category in _FAST_CLOUD_CATEGORIES


def _uses_specialist_model(decision: _Decision) -> bool:
    return decision.route == LOCAL_ROUTE and decision.category in _SPECIALIST_FIRST_CATEGORIES


def _uses_ner_model(decision: _Decision) -> bool:
    return decision.route == LOCAL_ROUTE and decision.category in _NER_FIRST_CATEGORIES


def _uses_local_cascade(decision: _Decision, margin: float) -> bool:
    if decision.route != LOCAL_ROUTE:
        return False
    if _uses_specialist_model(decision) or _uses_ner_model(decision):
        return False
    return decision.category in _CASCADE_CATEGORIES or decision.risk >= decision.threshold - margin


def _should_escalate_local_answer(decision: _Decision, answer: str, local: ChatModel) -> bool:
    try:
        result = call_with_retry(
            lambda: complete_structured(local, _self_check_messages(decision, answer), _LocalSelfCheck)
        )
    except Exception as error:  # noqa: BLE001 - a failed confidence check should fail closed.
        _log(f"[cascade] {decision.task.task_id}: self-check failed ({error}); escalating")
        return True
    _log(
        f"[cascade] {decision.task.task_id}: self-check escalate={result.should_escalate} "
        f"confidence={result.confidence:.2f} mode={result.failure_mode}"
    )
    return result.should_escalate


def _self_check_messages(decision: _Decision, answer: str) -> list[Message]:
    return [
        Message(Role.SYSTEM, _LOCAL_SELF_CHECK_SYSTEM_PROMPT),
        Message(Role.USER, f"{_task_context(decision)}\n\n[Local answer]\n{answer}"),
    ]


def _task_context(decision: _Decision) -> str:
    return (
        f"[Predicted category]\n{decision.category or 'unknown'}\n\n"
        f"[Router risk]\n{decision.risk:.3f}\n\n"
        f"[Router threshold]\n{decision.threshold:.3f}\n\n"
        f"[Task]\n{decision.task.prompt}"
    )


def _fast_cloud_config(config: ModelConfig) -> ModelConfig:
    return replace(config, enable_thinking=False, reasoning_effort=None, temperature=0.0)


def _specialist_messages(decision: _Decision) -> list[Message]:
    return system_and_user_message(_system_prompt(decision), decision.task.prompt)


def _system_prompt(decision: _Decision) -> str:
    hint = _CATEGORY_SYSTEM_HINTS.get(decision.category or "")
    if hint is None:
        return _CONCISE_SYSTEM_PROMPT
    return f"{_CONCISE_SYSTEM_PROMPT} {hint}"


def write_results(path: Path, records: Sequence[dict[str, object]]) -> None:
    """Write the final JSON result array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), ensure_ascii=False), encoding="utf-8")


def _log_cloud_usage(cloud: ChatModel | None, cloud_fast: ChatModel | None = None) -> None:
    usage_by_label = {
        label: usage
        for label, model in (("regular", cloud), ("fast", cloud_fast))
        if (usage := _usage_for(model)) is not None
    }
    if not usage_by_label:
        return
    if len(usage_by_label) > 1:
        for label, usage in usage_by_label.items():
            _log(
                f"cloud {label} tokens: prompt={usage.prompt_tokens} "
                f"completion={usage.completion_tokens} total={usage.total_tokens}"
            )
    usage = TokenUsage(
        prompt_tokens=sum(usage.prompt_tokens for usage in usage_by_label.values()),
        completion_tokens=sum(usage.completion_tokens for usage in usage_by_label.values()),
    )
    _log(f"cloud tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")


def _usage_for(model: ChatModel | None) -> TokenUsage | None:
    if not isinstance(model, UsageReporting):
        return None
    return model.usage


def _release_embedder(embedder: EmbeddingModel) -> None:
    """Free the embedder's model before the local GGUF loads.

    On the memory-constrained submission host, holding the embedder resident while the local
    model loads can push the process into swap and stall generation. The embedder is unused past
    classification, so releasing it reclaims that headroom.
    """
    if isinstance(embedder, Closeable):
        embedder.close()
    gc.collect()


def _release_chat_model(model: ChatModel | None) -> None:
    """Release a chat model before loading the next local GGUF."""
    if isinstance(model, Closeable):
        model.close()
    gc.collect()


def _release_named_entity_model(model: NamedEntityModel | None) -> None:
    """Release the NER model before loading the summary specialist."""
    if isinstance(model, Closeable):
        model.close()
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
