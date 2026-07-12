"""In-process continuous batching for llama-cpp-python chat models."""

# ruff: noqa: ANN401

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gullivers_router.inference.base import DEFAULT_INFERENCE_SEED

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gullivers_router.inference.base import Message

DEFAULT_BATCH_SLOTS = 2


@dataclass
class _SequenceState:
    request_index: int
    prompt: list[int]
    stop: tuple[str, ...]
    sampler: Any
    generated: list[int]


def complete_chat_batch(  # noqa: C901, PLR0912, PLR0913, PLR0915 - sequence scheduler and sampling controls.
    model: Any,
    message_batches: Sequence[Sequence[Message]],
    *,
    enable_thinking: bool | None,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int | None,
    sequence_context: int,
    slots: int = DEFAULT_BATCH_SLOTS,
) -> list[str]:
    """Continuously decode chat requests over a fixed number of llama.cpp sequences."""
    if not message_batches:
        return []

    from llama_cpp import llama_cpp
    from llama_cpp._internals import LlamaBatch

    requests = [_format_request(model, messages, enable_thinking) for messages in message_batches]
    for prompt, _stop in requests:
        if len(prompt) >= sequence_context:
            msg = f"prompt has {len(prompt)} tokens but sequence context is {sequence_context}"
            raise ValueError(msg)
    answers = [""] * len(requests)
    batch = LlamaBatch(n_tokens=model.n_batch, embd=0, n_seq_max=slots, verbose=False)
    active: dict[int, _SequenceState] = {}
    next_request = 0
    model._ctx.kv_cache_clear()  # noqa: SLF001 - llama-cpp-python has no public multi-sequence API.

    try:
        while next_request < len(requests) or active:
            for seq_id in range(slots):
                if seq_id in active or next_request >= len(requests):
                    continue
                prompt, stop = requests[next_request]
                _prefill_prefix(model, batch, prompt, seq_id)
                active[seq_id] = _SequenceState(
                    request_index=next_request,
                    prompt=prompt,
                    stop=stop,
                    sampler=_sampler(model, temperature, top_p, top_k),
                    generated=[],
                )
                next_request += 1

            batch.reset()
            row_for_sequence: dict[int, int] = {}
            for seq_id, state in active.items():
                if state.generated:
                    token = state.generated[-1]
                    position = len(state.prompt) + len(state.generated) - 1
                else:
                    token = state.prompt[-1]
                    position = len(state.prompt) - 1
                row_for_sequence[seq_id] = batch.n_tokens()
                _add_token(batch, token, position, seq_id, logits=True)
            model._ctx.decode(batch)  # noqa: SLF001

            completed: list[int] = []
            for seq_id, state in active.items():
                token = state.sampler.sample(model._ctx, row_for_sequence[seq_id])  # noqa: SLF001
                if llama_cpp.llama_vocab_is_eog(model._model.vocab, token):  # noqa: SLF001
                    completed.append(seq_id)
                    continue
                state.generated.append(token)
                text, stopped = _decoded_answer(model, state.generated, state.stop)
                token_limit = max_tokens if max_tokens is not None else sequence_context - len(state.prompt)
                if stopped or len(state.generated) >= token_limit:
                    answers[state.request_index] = text
                    completed.append(seq_id)

            for seq_id in completed:
                state = active.pop(seq_id)
                if not answers[state.request_index]:
                    answers[state.request_index] = _decoded_answer(model, state.generated, state.stop)[0]
                state.sampler.close()
                model._ctx.kv_cache_seq_rm(seq_id, 0, -1)  # noqa: SLF001
    finally:
        for state in active.values():
            state.sampler.close()
        batch.close()

    return answers


def _format_request(
    model: Any, messages: Sequence[Message], enable_thinking: bool | None
) -> tuple[list[int], tuple[str, ...]]:
    handler = model._chat_handlers.get(model.chat_format)  # noqa: SLF001
    closure = getattr(handler, "__closure__", None)
    if not closure:
        msg = f"chat format {model.chat_format!r} cannot be batched"
        raise ValueError(msg)
    formatter = closure[0].cell_contents
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    formatted = formatter(messages=[message.as_dict() for message in messages], **kwargs)
    prompt = model.tokenize(formatted.prompt.encode("utf-8"), add_bos=not formatted.added_special, special=True)
    if not prompt:
        msg = "chat template produced an empty prompt"
        raise ValueError(msg)
    raw_stop = formatted.stop
    stop = () if raw_stop is None else (raw_stop,) if isinstance(raw_stop, str) else tuple(raw_stop)
    return prompt, stop


def _sampler(model: Any, temperature: float, top_p: float, top_k: int) -> Any:
    model._seed = DEFAULT_INFERENCE_SEED  # noqa: SLF001
    return model._init_sampler(  # noqa: SLF001
        temp=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=1.1,
    )


def _prefill_prefix(model: Any, batch: Any, prompt: list[int], seq_id: int) -> None:
    prefix = prompt[:-1]
    for offset in range(0, len(prefix), model.n_batch):
        batch.reset()
        for index, token in enumerate(prefix[offset : offset + model.n_batch], start=offset):
            _add_token(batch, token, index, seq_id, logits=False)
        model._ctx.decode(batch)  # noqa: SLF001


def _add_token(batch: Any, token: int, position: int, seq_id: int, *, logits: bool) -> None:
    index = batch.n_tokens()
    batch.batch.n_tokens += 1
    batch.batch.token[index] = token
    batch.batch.pos[index] = position
    batch.batch.seq_id[index][0] = seq_id
    batch.batch.n_seq_id[index] = 1
    batch.batch.logits[index] = logits


def _decoded_answer(model: Any, tokens: list[int], stop: tuple[str, ...]) -> tuple[str, bool]:
    text = model.detokenize(tokens).decode("utf-8", errors="ignore")
    stop_positions = [position for marker in stop if (position := text.find(marker)) >= 0]
    if not stop_positions:
        return text, False
    return text[: min(stop_positions)], True
