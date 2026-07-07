"""Training-time batch chat via the Fireworks Batch API.

The runtime path uses the OpenAI-compatible endpoint; offline training generation runs
here instead because the Batch API is roughly half the cost for large jobs. The lifecycle
is: upload a JSONL dataset, create a batch job, poll until it completes, then download and
realign the results with the inputs by ``custom_id``.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gullivers_router.config import ModelConfig
    from gullivers_router.inference.base import Message

POLL_INTERVAL_SECONDS = 15
_FAILURE_STATES = {"FAILED", "CANCELLED", "EXPIRED", "FAILED_CLEANING_UP"}


def _custom_id(index: int) -> str:
    return f"request-{index}"


def _index_of(custom_id: str) -> int:
    return int(custom_id.rsplit("-", 1)[-1])


def _input_row(index: int, messages: Sequence[Message]) -> dict:
    """One JSONL request line (OpenAI batch shape; model is set on the job)."""
    return {
        "custom_id": _custom_id(index),
        "body": {"messages": [m.as_dict() for m in messages]},
    }


def _parse_output_row(row: dict) -> tuple[int, str]:
    """Map an output line back to its input index and completion text."""
    body = row.get("response", row).get("body", row.get("response", row))
    content = body["choices"][0]["message"].get("content") or ""
    return _index_of(row["custom_id"]), content


class FireworksBatchChat:
    """Batch chat completion over the Fireworks Batch API."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        inference_parameters: dict | None = None,
    ) -> None:
        """Configure the batch job; requires an API key and model id."""
        if not config.api_key:
            msg = f"provider {config.provider} requires an API key"
            raise ValueError(msg)
        if not config.model:
            msg = "batch inference requires a model id"
            raise ValueError(msg)
        self._provider = config.provider
        self._api_key = config.api_key
        self._model = config.model
        self._poll_interval = poll_interval
        self._inference_parameters = inference_parameters

    def complete_batch(self, requests: Sequence[Sequence[Message]]) -> list[str]:
        """Run all requests as one batch job, returned aligned to input order."""
        from fireworks.batch_inference_job import BatchInferenceJob
        from fireworks.dataset import Dataset
        from fireworks.gateway import Gateway

        rows = [_input_row(index, request) for index, request in enumerate(requests)]
        input_dataset = Dataset.from_list(rows)
        input_dataset.sync()

        job = BatchInferenceJob.create(
            model=self._model,
            input_dataset_id=input_dataset.id,
            inference_parameters=self._inference_parameters,
            api_key=self._api_key,
        )
        account = Gateway(api_key=self._api_key).account_id()
        output_dataset_id = self._await_completion(job.name, account)

        results = Dataset.from_id(output_dataset_id).read().decode("utf-8")
        return self._align(results, len(requests))

    def _await_completion(self, job_name: str, account: str) -> str:
        from fireworks.batch_inference_job import BatchInferenceJob
        from fireworks.control_plane.generated.protos.gateway import JobState

        while True:
            proto = BatchInferenceJob.get(job_name, account, api_key=self._api_key)
            if proto is None:
                msg = f"batch job {job_name} was not found"
                raise RuntimeError(msg)
            state = JobState(proto.state).name
            if state == JobState.COMPLETED.name:
                return proto.output_dataset_id
            if state in _FAILURE_STATES:
                msg = f"batch job {job_name} ended in state {state}"
                raise RuntimeError(msg)
            time.sleep(self._poll_interval)

    @staticmethod
    def _align(results_jsonl: str, count: int) -> list[str]:
        completions = [""] * count
        for line in results_jsonl.splitlines():
            if not line.strip():
                continue
            index, content = _parse_output_row(json.loads(line))
            completions[index] = content
        return completions
