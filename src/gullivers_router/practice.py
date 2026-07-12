"""LLM-judged scoring for local practice task answers."""

from __future__ import annotations

import json
import sys
from concurrent import futures
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from gullivers_router.config import Settings
from gullivers_router.inference.base import Message, Role
from gullivers_router.inference.factory import build_chat_model
from gullivers_router.inference.structured import complete_structured
from gullivers_router.router import DEFAULT_INPUT, Task, load_tasks
from gullivers_router.training.generate import call_with_retry

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gullivers_router.config import ModelConfig
    from gullivers_router.inference.base import ChatModel

DEFAULT_RESULTS = Path("outputs/results.json")
DEFAULT_ANSWER_SET = Path("examples/practice_answer_set.json")
DEFAULT_SCORE_OUTPUT = Path("outputs/practice_score.json")
DEFAULT_PRACTICE_WORKERS = 8
DEFAULT_JUDGE_TIMEOUT_SECONDS = 30.0
DEFAULT_JUDGE_ATTEMPTS = 3

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
GradeQuality = Literal["pass", "fail"]

_SYSTEM_PROMPT = (
    "You are an impartial grader for a practice benchmark. Compare the submitted answer to the "
    "reference answer and any scoring notes. Reward semantic correctness and following explicit "
    "constraints in the user prompt. Accept equivalent wording, harmless formatting differences, "
    "and alternate correct code. Pass only when the answer satisfies every material requirement; "
    "otherwise fail. Do not award partial credit. Respond only with JSON."
)

_USER_TEMPLATE = (
    "[User prompt]\n{prompt}\n\n[Reference answer]\n{reference_answer}\n\n[Scoring notes]\n{scoring_notes}\n\n"
    "[Submitted answer]\n{submitted_answer}\n\nGrade the submitted answer."
)

__all__ = [
    "DEFAULT_ANSWER_SET",
    "DEFAULT_RESULTS",
    "DEFAULT_SCORE_OUTPUT",
    "AnswerSetItem",
    "CandidateAnswer",
    "GradeRecord",
    "PracticeContext",
    "PracticeOptions",
    "PracticeReport",
    "PracticeSummary",
    "score_practice",
    "score_practice_with_context",
]


@dataclass(frozen=True, slots=True)
class AnswerSetItem:
    """Reference answer and optional rubric for one practice task."""

    task_id: str
    answer: str
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateAnswer:
    """Submitted answer for one practice task."""

    task_id: str
    answer: str


@dataclass(frozen=True, slots=True)
class PracticeOptions:
    """Practice scoring file paths and concurrency."""

    tasks_path: Path = DEFAULT_INPUT
    results_path: Path = DEFAULT_RESULTS
    answer_set_path: Path = DEFAULT_ANSWER_SET
    output_path: Path = DEFAULT_SCORE_OUTPUT
    workers: int = DEFAULT_PRACTICE_WORKERS
    timeout_seconds: float = DEFAULT_JUDGE_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_JUDGE_ATTEMPTS


@dataclass(frozen=True, slots=True)
class PracticeContext:
    """Practice scoring dependencies."""

    settings: Settings
    chat_factory: Callable[[ModelConfig], ChatModel]


class GradeRecord(TypedDict):
    """JSON report row for one practice answer."""

    task_id: str
    quality: GradeQuality
    score: float
    rationale: str


class PracticeSummary(TypedDict):
    """Aggregate practice scoring metrics."""

    tasks: int
    mean_score: float
    percent_score: float
    passed: int
    failed: int


class PracticeReport(TypedDict):
    """Practice scoring JSON report."""

    summary: PracticeSummary
    grades: list[GradeRecord]


class PracticeGrade(BaseModel):
    """Structured judge result for one submitted practice answer."""

    model_config = ConfigDict(extra="forbid")

    quality: GradeQuality
    rationale: ShortText

    @field_validator("quality", mode="before")
    @classmethod
    def _normalise_quality(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


def score_practice(  # noqa: PLR0913 - paths and judge controls are independent CLI options.
    *,
    tasks_path: Path = DEFAULT_INPUT,
    results_path: Path = DEFAULT_RESULTS,
    answer_set_path: Path = DEFAULT_ANSWER_SET,
    output_path: Path = DEFAULT_SCORE_OUTPUT,
    workers: int = DEFAULT_PRACTICE_WORKERS,
    timeout_seconds: float = DEFAULT_JUDGE_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_JUDGE_ATTEMPTS,
) -> PracticeReport:
    """Score routed answers against a reference answer set using the configured judge model."""
    return score_practice_with_context(
        PracticeOptions(
            tasks_path=tasks_path,
            results_path=results_path,
            answer_set_path=answer_set_path,
            output_path=output_path,
            workers=workers,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
        ),
        PracticeContext(settings=Settings.from_env(), chat_factory=build_chat_model),
    )


def score_practice_with_context(options: PracticeOptions, context: PracticeContext) -> PracticeReport:
    """Score practice answers with explicit dependencies."""
    tasks = load_tasks(options.tasks_path)
    references = _load_answer_set(options.answer_set_path)
    candidates = _load_candidate_answers(options.results_path)
    _validate_task_coverage(tasks, references, candidates)

    judge_config = replace(context.settings.judge, timeout_seconds=options.timeout_seconds)
    judge = context.chat_factory(judge_config)
    records = _grade_tasks(
        tasks,
        references,
        candidates,
        judge,
        workers=options.workers,
        max_attempts=options.max_attempts,
    )
    report = _report(records)
    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    options.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _log_summary(report, options.output_path)
    return report


def _load_answer_set(path: Path) -> dict[str, AnswerSetItem]:
    raw = _read_json_array(path, "answer set")
    items: dict[str, AnswerSetItem] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            msg = f"answer set entry at index {index} must be an object"
            raise TypeError(msg)
        task_id = item.get("task_id")
        answer = item.get("answer")
        notes = item.get("notes")
        if not isinstance(task_id, str) or not task_id:
            msg = f"answer set entry at index {index} must have a non-empty string task_id"
            raise ValueError(msg)
        if not isinstance(answer, str) or not answer:
            msg = f"answer set entry {task_id} must have a non-empty string answer"
            raise ValueError(msg)
        if notes is not None and not isinstance(notes, str):
            msg = f"answer set entry {task_id} notes must be a string when present"
            raise TypeError(msg)
        if task_id in items:
            msg = f"duplicate answer set task_id: {task_id}"
            raise ValueError(msg)
        items[task_id] = AnswerSetItem(task_id=task_id, answer=answer, notes=notes)
    return items


def _load_candidate_answers(path: Path) -> dict[str, CandidateAnswer]:
    raw = _read_json_array(path, "results")
    items: dict[str, CandidateAnswer] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            msg = f"result at index {index} must be an object"
            raise TypeError(msg)
        task_id = item.get("task_id")
        answer = item.get("answer")
        if not isinstance(task_id, str) or not task_id:
            msg = f"result at index {index} must have a non-empty string task_id"
            raise ValueError(msg)
        if not isinstance(answer, str):
            msg = f"result {task_id} must have a string answer"
            raise TypeError(msg)
        if task_id in items:
            msg = f"duplicate result task_id: {task_id}"
            raise ValueError(msg)
        items[task_id] = CandidateAnswer(task_id=task_id, answer=answer)
    return items


def _read_json_array(path: Path, label: str) -> list[object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        msg = f"failed to read {label} file {path}: {error}"
        raise RuntimeError(msg) from error
    except json.JSONDecodeError as error:
        msg = f"{label} file {path} is not valid JSON: {error}"
        raise ValueError(msg) from error
    if not isinstance(raw, list):
        msg = f"{label} file must be a JSON array"
        raise TypeError(msg)
    return raw


def _validate_task_coverage(
    tasks: Sequence[Task],
    references: dict[str, AnswerSetItem],
    candidates: dict[str, CandidateAnswer],
) -> None:
    task_ids = {task.task_id for task in tasks}
    missing_references = sorted(task_id for task_id in task_ids if task_id not in references)
    if missing_references:
        msg = f"answer set is missing task ids: {', '.join(missing_references)}"
        raise ValueError(msg)
    extra_results = sorted(task_id for task_id in candidates if task_id not in task_ids)
    if extra_results:
        msg = f"results include unknown task ids: {', '.join(extra_results)}"
        raise ValueError(msg)


def _grade_tasks(  # noqa: PLR0913 - grading requires aligned task data and execution controls.
    tasks: Sequence[Task],
    references: dict[str, AnswerSetItem],
    candidates: dict[str, CandidateAnswer],
    judge: ChatModel,
    *,
    workers: int,
    max_attempts: int,
) -> list[GradeRecord]:
    records_by_id: dict[str, GradeRecord] = {}
    submitted = [task for task in tasks if _submitted_answer(candidates.get(task.task_id))]

    _log(f"grading {len(submitted)} submitted answers ({max(1, workers)} workers)")
    with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        lock = Lock()
        submitted_futures = {
            pool.submit(
                _grade_one,
                task,
                references[task.task_id],
                candidates[task.task_id],
                judge,
                max_attempts=max_attempts,
            ): task.task_id
            for task in submitted
        }
        for completed, future in enumerate(futures.as_completed(submitted_futures), start=1):
            task_id = submitted_futures[future]
            record = future.result()
            with lock:
                records_by_id[task_id] = record
            _log(f"[grade {completed}/{len(submitted_futures)}] {task_id} -> {record['quality']}")

    for task in tasks:
        if task.task_id in records_by_id:
            continue
        records_by_id[task.task_id] = {
            "task_id": task.task_id,
            "quality": "fail",
            "score": 0.0,
            "rationale": "No submitted answer was provided.",
        }
    return [records_by_id[task.task_id] for task in tasks]


def _submitted_answer(candidate: CandidateAnswer | None) -> bool:
    return candidate is not None and bool(candidate.answer.strip())


def _grade_one(
    task: Task,
    reference: AnswerSetItem,
    candidate: CandidateAnswer,
    judge: ChatModel,
    *,
    max_attempts: int,
) -> GradeRecord:
    messages = [
        Message(Role.SYSTEM, _SYSTEM_PROMPT),
        Message(
            Role.USER,
            _USER_TEMPLATE.format(
                prompt=task.prompt,
                reference_answer=reference.answer,
                scoring_notes=reference.notes or "No extra notes.",
                submitted_answer=candidate.answer,
            ),
        ),
    ]
    grade = call_with_retry(
        lambda: complete_structured(judge, messages, PracticeGrade),
        max_attempts=max_attempts,
    )
    return {
        "task_id": task.task_id,
        "quality": grade.quality,
        "score": 1.0 if grade.quality == "pass" else 0.0,
        "rationale": grade.rationale,
    }


def _report(records: Sequence[GradeRecord]) -> PracticeReport:
    count = len(records)
    scores = [float(record["score"]) for record in records]
    passed = sum(1 for record in records if record["quality"] == "pass")
    failed = count - passed
    mean_score = sum(scores) / count if count else 0.0
    return {
        "summary": {
            "tasks": count,
            "mean_score": mean_score,
            "percent_score": mean_score * 100,
            "passed": passed,
            "failed": failed,
        },
        "grades": list(records),
    }


def _log_summary(report: PracticeReport, output_path: Path) -> None:
    summary = report["summary"]
    _log(
        "practice score: "
        f"{summary['percent_score']:.1f}% "
        f"({summary['passed']} passed, {summary['failed']} failed) "
        f"-> {output_path}"
    )


def _log(message: str) -> None:
    print(f"[practice] {message}", file=sys.stderr, flush=True)
