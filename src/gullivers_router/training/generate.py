"""Resumable generation of local and cloud responses for the training prompts.

Local generation runs sequentially (a single GPU cannot batch-decode). Cloud generation
runs many serverless calls in parallel, since the chosen cloud model is serverless-only and
not compatible with the Fireworks Batch API. Both stages append each answer keyed by prompt
id as it lands, so a crash costs at most the in-flight work and a rerun resumes from the gap.
"""

from __future__ import annotations

import random
import time
from concurrent import futures
from threading import Lock
from typing import TYPE_CHECKING, TypeVar

from gullivers_router.inference.base import user_message
from gullivers_router.training import store

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from gullivers_router.inference.base import ChatModel, Message
    from gullivers_router.training.dataset import Prompt

DEFAULT_CONCURRENCY = 24
MAX_ATTEMPTS = 6
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 30.0
RetryResult = TypeVar("RetryResult")


def run_local(prompts: Sequence[Prompt], model: ChatModel, out: Path) -> None:
    """Generate local responses one at a time, resuming past whatever is already done."""
    from tqdm import tqdm

    done = store.completed_ids(out)
    remaining = [prompt for prompt in prompts if prompt.id not in done]
    for prompt in tqdm(remaining, desc="local generation"):
        response = model.complete(user_message(prompt.text))
        store.append(out, {"id": prompt.id, "response": response})


def run_cloud(
    prompts: Sequence[Prompt],
    model: ChatModel,
    out: Path,
    *,
    max_workers: int = DEFAULT_CONCURRENCY,
) -> None:
    """Generate cloud responses concurrently over the serverless endpoint."""
    items = {prompt.id: user_message(prompt.text) for prompt in prompts}
    run_concurrent(
        model,
        items,
        out,
        label="cloud generation",
        max_workers=max_workers,
        to_record=lambda item_id, response: {"id": item_id, "response": response},
    )


def run_concurrent(  # noqa: PLR0913 - each argument is a distinct orchestration input
    model: ChatModel,
    items: Mapping[str, Sequence[Message]],
    out: Path,
    *,
    label: str,
    max_workers: int,
    to_record: Callable[[str, str], dict],
) -> None:
    """Complete every request with bounded concurrency, appending results as they finish.

    Items already present in ``out`` are skipped. A request that still fails after retries is
    left out so a later run retries it, rather than poisoning the output with a blank.
    """
    from tqdm import tqdm

    done = store.completed_ids(out)
    pending = [item_id for item_id in items if item_id not in done]
    if not pending:
        return

    lock = Lock()
    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool, tqdm(total=len(pending), desc=label) as bar:
        submitted = {pool.submit(complete_with_retry, model, items[item_id]): item_id for item_id in pending}
        for future in futures.as_completed(submitted):
            item_id = submitted[future]
            try:
                response = future.result()
            except Exception as error:  # noqa: BLE001 - isolate one request's failure from the batch
                bar.write(f"{label}: {item_id} failed after retries ({error}); will retry on resume")
            else:
                with lock:
                    store.append(out, to_record(item_id, response))
            bar.update(1)


def complete_with_retry(model: ChatModel, messages: Sequence[Message]) -> str:
    """Call the model, retrying transient failures (e.g. 429s) with jittered backoff."""
    return call_with_retry(lambda: model.complete(messages))


def call_with_retry(operation: Callable[[], RetryResult], *, max_attempts: int = MAX_ATTEMPTS) -> RetryResult:
    """Run an operation with jittered backoff between transient failures."""
    if max_attempts < 1:
        msg = "max_attempts must be at least 1"
        raise ValueError(msg)
    for attempt in range(max_attempts):
        try:
            return operation()
        except Exception:
            if attempt + 1 == max_attempts:
                raise
            delay = min(_BACKOFF_BASE_SECONDS * 2**attempt, _BACKOFF_CAP_SECONDS)
            time.sleep(delay + random.uniform(0, delay))  # noqa: S311 - jitter, not security
    message = "retry loop exited without returning"
    raise RuntimeError(message)  # pragma: no cover
