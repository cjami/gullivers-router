"""Score local vs. cloud responses with the judge model (GLM 5.2) over serverless calls.

The judge sees the prompt and both responses and returns a 1-10 score for each as JSON.
Parsing is defensive: malformed output yields null scores so the row is still recorded
(and skipped at labelling) rather than re-judged forever.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gullivers_router.inference.base import Message, Role
from gullivers_router.training import store
from gullivers_router.training.generate import DEFAULT_CONCURRENCY, run_concurrent

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from gullivers_router.inference.base import ChatModel
    from gullivers_router.training.combine import ResponsePair

SCORE_MIN = 1
SCORE_MAX = 10

_SYSTEM_PROMPT = (
    "You are an impartial evaluator. Score how well each assistant response answers the "
    "user prompt on a scale from 1 (useless) to 10 (excellent). Judge accuracy, "
    "completeness, and helpfulness. Respond with only a JSON object of the form "
    '{"local_score": <int>, "cloud_score": <int>}.'
)

_USER_TEMPLATE = (
    "[User prompt]\n{prompt}\n\n[Response LOCAL]\n{local}\n\n[Response CLOUD]\n{cloud}\n\nScore both responses now."
)

_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class Judgement:
    """The judge's scores for one prompt; ``None`` when the output was unparseable."""

    id: str
    local_score: int | None
    cloud_score: int | None


def _judge_messages(pair: ResponsePair) -> list[Message]:
    user = _USER_TEMPLATE.format(
        prompt=pair.prompt.text,
        local=pair.local_response,
        cloud=pair.cloud_response,
    )
    return [Message(Role.SYSTEM, _SYSTEM_PROMPT), Message(Role.USER, user)]


def _clamp_score(value: object) -> int | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return max(SCORE_MIN, min(SCORE_MAX, int(value)))


def parse_scores(content: str) -> tuple[int | None, int | None]:
    """Extract ``(local_score, cloud_score)`` from judge output, or ``(None, None)``."""
    match = _JSON_OBJECT.search(content)
    if match is None:
        return None, None
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return None, None
    return _clamp_score(payload.get("local_score")), _clamp_score(payload.get("cloud_score"))


def run_judge(
    pairs: Sequence[ResponsePair],
    model: ChatModel,
    out: Path,
    *,
    max_workers: int = DEFAULT_CONCURRENCY,
) -> None:
    """Judge every response pair over concurrent serverless calls, resuming past done ids."""
    items = {pair.prompt.id: _judge_messages(pair) for pair in pairs}
    run_concurrent(
        model,
        items,
        out,
        label="judging",
        max_workers=max_workers,
        to_record=_score_record,
    )


def _score_record(item_id: str, content: str) -> dict:
    local_score, cloud_score = parse_scores(content)
    return {"id": item_id, "local_score": local_score, "cloud_score": cloud_score}


def load_judgements(path: Path) -> list[Judgement]:
    """Read judgements previously written by :func:`run_judge`."""
    return [
        Judgement(id=record["id"], local_score=record["local_score"], cloud_score=record["cloud_score"])
        for record in store.read_records(path)
    ]
