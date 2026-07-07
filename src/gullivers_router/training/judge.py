"""Evaluate blinded local/cloud response pairs with the judge model over serverless calls.

The judge sees the prompt and two anonymized responses in a deterministic, balanced order. It
returns word-only quality labels and a simple A/B/tie preference; numeric training values are
derived in code. Parsing is defensive: malformed output yields null scores so the row is still
recorded and skipped by downstream training-row construction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gullivers_router.inference.base import Message, Role
from gullivers_router.training import store
from gullivers_router.training.generate import DEFAULT_CONCURRENCY, complete_with_retry

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
PREFERRED_RESPONSES = {"response_a", "response_b", "tie"}
TIE_BREAK_BONUS = 0.25

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


@dataclass(frozen=True, slots=True)
class ParsedScores:
    """Word-only qualities, preference, and rationales for anonymized response A/B."""

    response_a_quality: str | None
    response_b_quality: str | None
    preferred_response: str | None
    response_a_rationale: str | None = None
    response_b_rationale: str | None = None


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


def _preferred_response(value: object) -> str | None:
    text = _short_text(value)
    if text is None:
        return None
    label = text.lower()
    return label if label in PREFERRED_RESPONSES else None


def _extract_json_object(content: str) -> dict | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        return payload if isinstance(payload, dict) else None
    return None


def parse_scores(content: str) -> ParsedScores:
    """Extract anonymized A/B quality labels and preference from judge output."""
    payload = _extract_json_object(content)
    if payload is None:
        return ParsedScores(None, None, None)
    return ParsedScores(
        response_a_quality=_quality_label(payload.get("response_a_quality")),
        response_b_quality=_quality_label(payload.get("response_b_quality")),
        preferred_response=_preferred_response(payload.get("preferred_response")),
        response_a_rationale=_short_text(payload.get("response_a_rationale")),
        response_b_rationale=_short_text(payload.get("response_b_rationale")),
    )


def _primary_order(item_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    return (_LOCAL, _CLOUD) if digest[0] % 2 == 0 else (_CLOUD, _LOCAL)


def _mapped_scores(parsed: ParsedScores, order: tuple[str, str]) -> dict[str, float | bool | str | None]:
    mapped: dict[str, float | bool | str | None] = {}
    scored = _scored_labels(parsed)
    for source, score, quality, rationale in (
        (order[0], scored[0], parsed.response_a_quality, parsed.response_a_rationale),
        (order[1], scored[1], parsed.response_b_quality, parsed.response_b_rationale),
    ):
        mapped[f"{source}_score"] = score
        mapped[f"{source}_quality"] = quality
        mapped[f"{source}_rationale"] = rationale
    mapped["preferred_source"] = _preferred_source(parsed.preferred_response, order)
    mapped["preference_consistent"] = _preference_consistent(parsed)
    return mapped


def _scored_labels(parsed: ParsedScores) -> tuple[float | None, float | None]:
    if parsed.response_a_quality is None or parsed.response_b_quality is None:
        return None, None
    score_a = QUALITY_VALUES[parsed.response_a_quality]
    score_b = QUALITY_VALUES[parsed.response_b_quality]
    if score_a == score_b and parsed.preferred_response == "response_a":
        score_a += TIE_BREAK_BONUS
    elif score_a == score_b and parsed.preferred_response == "response_b":
        score_b += TIE_BREAK_BONUS
    return score_a, score_b


def _preferred_source(preferred_response: str | None, order: tuple[str, str]) -> str | None:
    if preferred_response == "response_a":
        return order[0]
    if preferred_response == "response_b":
        return order[1]
    return preferred_response


def _preference_consistent(parsed: ParsedScores) -> bool | None:
    if parsed.response_a_quality is None or parsed.response_b_quality is None or parsed.preferred_response is None:
        return None
    score_a = QUALITY_VALUES[parsed.response_a_quality]
    score_b = QUALITY_VALUES[parsed.response_b_quality]
    if score_a == score_b:
        return True
    if score_a > score_b:
        return parsed.preferred_response == "response_a"
    return parsed.preferred_response == "response_b"


def _record(pair: ResponsePair, content: str) -> dict:
    primary_order = _primary_order(pair.prompt.id)
    primary = _mapped_scores(parse_scores(content), primary_order)
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
    content = complete_with_retry(model, _judge_messages(pair, primary_order))
    return _record(pair, content)


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

    done = store.completed_ids(out)
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
