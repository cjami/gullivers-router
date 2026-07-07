"""Evaluate blinded local/cloud response pairs with the judge model over serverless calls.

The judge sees the prompt and two anonymized responses in a deterministic, balanced order. It
returns word-only quality labels and a simple A/B/tie preference; numeric training values are
derived in code. Malformed output is retried and left unwritten if the judge keeps returning an
invalid judgement, so a later resume can try again.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from gullivers_router.inference.base import Message, Role
from gullivers_router.inference.structured import complete_structured
from gullivers_router.training import store
from gullivers_router.training.generate import DEFAULT_CONCURRENCY, call_with_retry

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from gullivers_router.inference.base import ChatModel
    from gullivers_router.training.combine import ResponsePair

QUALITY_VALUES = {
    "unacceptable": 1.0,
    "poor": 2.0,
    "adequate": 3.0,
    "good": 4.0,
    "excellent": 5.0,
}
TIE_BREAK_BONUS = 0.25
QualityLabel = Literal["unacceptable", "poor", "adequate", "good", "excellent"]
PreferredResponse = Literal["response_a", "response_b", "tie"]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

_SYSTEM_PROMPT = (
    "You are an impartial evaluator. The two assistant responses are anonymized and may appear "
    "in either order. Judge accuracy, completeness, instruction-following, and helpfulness. "
    "Choose each quality label from: unacceptable, poor, adequate, good, excellent. Choose the "
    "preferred response from: response_a, response_b, tie. Give concise rationales. Respond with "
    "only a JSON object of the form "
    '{"response_a_quality": "<quality label>", "response_a_rationale": "<short reason>", '
    '"response_b_quality": "<quality label>", "response_b_rationale": "<short reason>", '
    '"preferred_response": "<preferred response>"}.'
)

_USER_TEMPLATE = (
    "[User prompt]\n{prompt}\n\n[Response A]\n{response_a}\n\n[Response B]\n{response_b}\n\n"
    "Evaluate both responses now."
)

_LOCAL = "local"
_CLOUD = "cloud"
PREFERRED_SOURCES = {_LOCAL, _CLOUD, "tie"}


def _normalised_token(value: object) -> object:
    return value.strip().lower() if isinstance(value, str) else value


class JudgeResult(BaseModel):
    """Structured judge result for anonymized response A/B."""

    model_config = ConfigDict(extra="forbid")

    response_a_quality: QualityLabel
    response_a_rationale: ShortText
    response_b_quality: QualityLabel
    response_b_rationale: ShortText
    preferred_response: PreferredResponse

    @field_validator("response_a_quality", "response_b_quality", "preferred_response", mode="before")
    @classmethod
    def _normalise_labels(cls, value: object) -> object:
        return _normalised_token(value)


@dataclass(frozen=True, slots=True)
class Judgement:
    """The judge's mapped scores for one prompt; ``None`` when parsing failed."""

    id: str
    local_score: float | None
    cloud_score: float | None
    local_rationale: str | None = None
    cloud_rationale: str | None = None
    local_quality: str | None = None
    cloud_quality: str | None = None
    preferred_source: str | None = None


def _judge_messages(pair: ResponsePair, order: tuple[str, str]) -> list[Message]:
    user = _USER_TEMPLATE.format(
        prompt=pair.prompt.text,
        response_a=_response(pair, order[0]),
        response_b=_response(pair, order[1]),
    )
    return [Message(Role.SYSTEM, _SYSTEM_PROMPT), Message(Role.USER, user)]


def _response(pair: ResponsePair, source: str) -> str:
    return pair.local_response if source == _LOCAL else pair.cloud_response


def _short_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _quality_label(value: object) -> str | None:
    text = _short_text(value)
    if text is None:
        return None
    label = text.lower()
    return label if label in QUALITY_VALUES else None


def _primary_order(item_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    return (_LOCAL, _CLOUD) if digest[0] % 2 == 0 else (_CLOUD, _LOCAL)


def _mapped_scores(result: JudgeResult, order: tuple[str, str]) -> dict[str, float | bool | str | None]:
    mapped: dict[str, float | bool | str | None] = {}
    scored = _scored_labels(result)
    for source, score, quality, rationale in (
        (order[0], scored[0], result.response_a_quality, result.response_a_rationale),
        (order[1], scored[1], result.response_b_quality, result.response_b_rationale),
    ):
        mapped[f"{source}_score"] = score
        mapped[f"{source}_quality"] = quality
        mapped[f"{source}_rationale"] = rationale
    mapped["preferred_source"] = _preferred_source(result.preferred_response, order)
    mapped["preference_consistent"] = _preference_consistent(result)
    return mapped


def _scored_labels(result: JudgeResult) -> tuple[float, float]:
    score_a = QUALITY_VALUES[result.response_a_quality]
    score_b = QUALITY_VALUES[result.response_b_quality]
    if score_a == score_b and result.preferred_response == "response_a":
        score_a += TIE_BREAK_BONUS
    elif score_a == score_b and result.preferred_response == "response_b":
        score_b += TIE_BREAK_BONUS
    return score_a, score_b


def _preferred_source(preferred_response: PreferredResponse, order: tuple[str, str]) -> str:
    if preferred_response == "response_a":
        return order[0]
    if preferred_response == "response_b":
        return order[1]
    return preferred_response


def _preference_consistent(result: JudgeResult) -> bool:
    score_a = QUALITY_VALUES[result.response_a_quality]
    score_b = QUALITY_VALUES[result.response_b_quality]
    if score_a == score_b:
        return True
    if score_a > score_b:
        return result.preferred_response == "response_a"
    return result.preferred_response == "response_b"


def _record(pair: ResponsePair, result: JudgeResult) -> dict:
    primary_order = _primary_order(pair.prompt.id)
    primary = _mapped_scores(result, primary_order)
    return {
        "id": pair.prompt.id,
        "local_score": primary.get("local_score"),
        "cloud_score": primary.get("cloud_score"),
        "local_rationale": primary.get("local_rationale"),
        "cloud_rationale": primary.get("cloud_rationale"),
        "local_quality": primary.get("local_quality"),
        "cloud_quality": primary.get("cloud_quality"),
        "preferred_source": primary.get("preferred_source"),
        "primary_order": f"{primary_order[0]}_first",
    }


def _judge_pair(pair: ResponsePair, model: ChatModel) -> dict:
    primary_order = _primary_order(pair.prompt.id)
    messages = _judge_messages(pair, primary_order)
    result = call_with_retry(lambda: complete_structured(model, messages, JudgeResult))
    return _record(pair, result)


def _numeric(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _valid_record(record: dict) -> bool:
    return (
        _numeric(record.get("local_score"))
        and _numeric(record.get("cloud_score"))
        and _quality_label(record.get("local_quality")) is not None
        and _quality_label(record.get("cloud_quality")) is not None
        and _short_text(record.get("local_rationale")) is not None
        and _short_text(record.get("cloud_rationale")) is not None
        and _short_text(record.get("preferred_source")) in PREFERRED_SOURCES
    )


def _completed_valid_ids(path: Path) -> set[str]:
    return {record["id"] for record in store.read_records(path) if _valid_record(record)}


def _legacy_record(record: dict) -> Judgement:
    local_score = record["local_score"]
    cloud_score = record["cloud_score"]
    return Judgement(
        id=record["id"],
        local_score=float(local_score) if local_score is not None else None,
        cloud_score=float(cloud_score) if cloud_score is not None else None,
        local_rationale=_short_text(record.get("local_rationale")),
        cloud_rationale=_short_text(record.get("cloud_rationale")),
        local_quality=_quality_label(record.get("local_quality")),
        cloud_quality=_quality_label(record.get("cloud_quality")),
        preferred_source=_short_text(record.get("preferred_source")),
    )


def run_judge(
    pairs: Sequence[ResponsePair],
    model: ChatModel,
    out: Path,
    *,
    max_workers: int = DEFAULT_CONCURRENCY,
) -> None:
    """Judge every response pair over concurrent serverless calls, resuming done ids."""
    from concurrent import futures
    from threading import Lock

    from tqdm import tqdm

    done = _completed_valid_ids(out)
    pending = [pair for pair in pairs if pair.prompt.id not in done]
    if not pending:
        return

    lock = Lock()
    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool, tqdm(total=len(pending), desc="judging") as bar:
        submitted = {pool.submit(_judge_pair, pair, model): pair.prompt.id for pair in pending}
        for future in futures.as_completed(submitted):
            item_id = submitted[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 - isolate one request's failure from the batch
                bar.write(f"judging: {item_id} failed after retries ({error}); will retry on resume")
            else:
                with lock:
                    store.append(out, record)
            bar.update(1)


def load_judgements(path: Path) -> list[Judgement]:
    """Read judgements previously written by :func:`run_judge`."""
    return [_legacy_record(record) for record in store.read_records(path)]
