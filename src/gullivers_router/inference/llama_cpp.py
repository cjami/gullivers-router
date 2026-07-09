"""Local GGUF backends via llama-cpp-python."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, cast

from gullivers_router.inference.base import DEFAULT_INFERENCE_SEED
from gullivers_router.inference.structured import llama_cpp_json_schema_response_format, strip_thinking_sections
from gullivers_router.inference.truncation import EMBEDDING_CONTEXT_LIMIT, truncate_head_tail

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gullivers_router.config import ModelConfig
    from gullivers_router.inference.base import Message, StructuredOutput

DEFAULT_CHAT_CONTEXT = 2048
OFFLOAD_ALL_LAYERS = -1
DEFAULT_MODEL_ROOT = Path("models")
DEFAULT_ENABLE_THINKING = False
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64
_POOLING_TYPES = {
    "none": 0,
    "mean": 1,
    "cls": 2,
    "last": 3,
    "rank": 4,
}
type ChatCompletionDict = dict[str, object]
type ChatCompletionHandler = Callable[..., ChatCompletionDict]


def _require_repo_id(config: ModelConfig) -> str:
    if config.repo_id is None:
        msg = "local model requires a repo_id"
        raise ValueError(msg)
    return config.repo_id


def _local_model_path(config: ModelConfig, model_root: Path) -> Path | None:
    if config.filename is None:
        return None

    repo_parts = _require_repo_id(config).split("/")
    repo_dir = model_root.joinpath(*repo_parts)
    search_dirs = (repo_dir, model_root)
    for directory in search_dirs:
        if _has_glob(config.filename):
            matches = sorted(directory.glob(config.filename))
        else:
            matches = [directory / config.filename]
        existing = [path for path in matches if path.is_file()]
        if len(existing) == 1:
            return existing[0]
        if len(existing) > 1:
            msg = f"multiple local models match {config.filename} in {directory}"
            raise ValueError(msg)
    return None


def _has_glob(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")


def _pooling_type(name: str | None) -> int:
    if name is None:
        return -1
    try:
        return _POOLING_TYPES[name.lower()]
    except KeyError as error:
        choices = ", ".join(_POOLING_TYPES)
        msg = f"unsupported pooling type {name!r}; expected one of: {choices}"
        raise ValueError(msg) from error


class LlamaCppChat:
    """Chat completion from a locally hosted GGUF model."""

    def __init__(  # noqa: PLR0913
        self,
        config: ModelConfig,
        *,
        n_ctx: int = DEFAULT_CHAT_CONTEXT,
        n_gpu_layers: int = OFFLOAD_ALL_LAYERS,
        flash_attn: bool = True,
        enable_thinking: bool | None = DEFAULT_ENABLE_THINKING,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        max_tokens: int | None = None,
        n_threads: int | None = None,
        model_root: Path = DEFAULT_MODEL_ROOT,
    ) -> None:
        """Configure the model; the GGUF loads lazily on first use.

        ``n_gpu_layers`` defaults to offloading every layer for local training. Submission
        builds override it to zero for CPU-only inference. ``max_tokens`` defaults to unset
        (unbounded) so training generation is never truncated; runtime builds cap it as a
        fail-safe against runaway generation.
        """
        self._config = config
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._flash_attn = flash_attn
        self._enable_thinking = enable_thinking
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._max_tokens = max_tokens
        self._n_threads = n_threads
        self._model_root = model_root
        self._model = None

    def _load(self):  # noqa: ANN202
        if self._model is None:
            from llama_cpp import Llama

            local_path = _local_model_path(self._config, self._model_root)
            if local_path is not None:
                self._model = Llama(
                    model_path=str(local_path),
                    n_ctx=self._n_ctx,
                    n_gpu_layers=self._n_gpu_layers,
                    flash_attn=self._flash_attn,
                    n_threads=self._n_threads,
                    seed=DEFAULT_INFERENCE_SEED,
                    verbose=False,
                )
            else:
                self._model = Llama.from_pretrained(
                    repo_id=_require_repo_id(self._config),
                    filename=self._config.filename,
                    n_ctx=self._n_ctx,
                    n_gpu_layers=self._n_gpu_layers,
                    flash_attn=self._flash_attn,
                    n_threads=self._n_threads,
                    seed=DEFAULT_INFERENCE_SEED,
                    verbose=False,
                )
        return self._model

    def complete(self, messages: Sequence[Message]) -> str:
        """Generate a response for a single prompt."""
        result = self._create_chat_completion({"messages": [m.as_dict() for m in messages]})
        return _completion_content(result)

    def complete_structured(
        self,
        messages: Sequence[Message],
        response_model: type[StructuredOutput],
    ) -> StructuredOutput:
        """Generate a response constrained to a Pydantic model schema."""
        result = self._create_chat_completion(
            {
                "messages": [m.as_dict() for m in messages],
                "response_format": llama_cpp_json_schema_response_format(response_model),
            }
        )
        content = _completion_content(result)
        return response_model.model_validate_json(content)

    def _create_chat_completion(self, payload: ChatCompletionDict) -> ChatCompletionDict:
        model = self._load()
        payload = {**payload, **self._sampling_payload()}
        if self._enable_thinking is None or not self._template_supports_enable_thinking(model):
            return cast(ChatCompletionDict, model.create_chat_completion(**payload))

        handler = self._chat_completion_handler(model)
        if handler is None:
            return cast(ChatCompletionDict, model.create_chat_completion(**payload))

        return handler(llama=model, **payload, enable_thinking=self._enable_thinking)

    def _sampling_payload(self) -> ChatCompletionDict:
        return {
            "seed": DEFAULT_INFERENCE_SEED,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "max_tokens": self._max_tokens,
        }

    def _template_supports_enable_thinking(self, model: object) -> bool:
        metadata = getattr(model, "metadata", {})
        if not isinstance(metadata, dict):
            return False
        template = metadata.get("tokenizer.chat_template")
        return isinstance(template, str) and "enable_thinking" in template

    def _chat_completion_handler(self, model: object) -> ChatCompletionHandler | None:
        handler = getattr(model, "chat_handler", None)
        if callable(handler):
            return cast(ChatCompletionHandler, handler)

        chat_format = getattr(model, "chat_format", None)
        chat_handlers = getattr(model, "_chat_handlers", {})
        if isinstance(chat_handlers, dict) and chat_format in chat_handlers:
            return cast(ChatCompletionHandler, chat_handlers[chat_format])
        if chat_format is None:
            return None

        try:
            chat_format_module = import_module("llama_cpp.llama_chat_format")
            return cast(ChatCompletionHandler, chat_format_module.get_chat_completion_handler(chat_format))
        except (ImportError, KeyError, TypeError, ValueError):
            return None


def _completion_content(result: ChatCompletionDict) -> str:
    choices = cast("list[dict[str, object]]", result["choices"])
    message = cast("dict[str, str | None]", choices[0]["message"])
    return strip_thinking_sections(message.get("content") or "")


class LlamaCppEmbedder:
    """Dense embeddings from a locally hosted GGUF model."""

    def __init__(  # noqa: PLR0913 - backend controls are independently configurable
        self,
        config: ModelConfig,
        *,
        n_ctx: int = EMBEDDING_CONTEXT_LIMIT,
        n_gpu_layers: int = OFFLOAD_ALL_LAYERS,
        n_threads: int | None = None,
        model_root: Path = DEFAULT_MODEL_ROOT,
        pooling_type: str | None = None,
        input_prefix: str = "",
    ) -> None:
        """Configure the embedder; the GGUF loads lazily on first use."""
        self._config = config
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._n_threads = n_threads
        self._model_root = model_root
        self._pooling_type = _pooling_type(pooling_type)
        self._input_prefix = input_prefix
        self._model = None

    def _load(self):  # noqa: ANN202
        if self._model is None:
            from llama_cpp import Llama

            local_path = _local_model_path(self._config, self._model_root)
            if local_path is not None:
                self._model = Llama(
                    model_path=str(local_path),
                    n_ctx=self._n_ctx,
                    n_gpu_layers=self._n_gpu_layers,
                    n_threads=self._n_threads,
                    embedding=True,
                    pooling_type=self._pooling_type,
                    seed=DEFAULT_INFERENCE_SEED,
                    verbose=False,
                )
            else:
                self._model = Llama.from_pretrained(
                    repo_id=_require_repo_id(self._config),
                    filename=self._config.filename,
                    n_ctx=self._n_ctx,
                    n_gpu_layers=self._n_gpu_layers,
                    n_threads=self._n_threads,
                    embedding=True,
                    pooling_type=self._pooling_type,
                    seed=DEFAULT_INFERENCE_SEED,
                    verbose=False,
                )
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed text, applying head-and-tail truncation past the context window."""
        model = self._load()
        input_text = f"{self._input_prefix}{text}"
        tokens = model.tokenize(input_text.encode("utf-8"), add_bos=True)
        truncated_tokens = truncate_head_tail(tokens, limit=self._n_ctx)
        truncated_text = model.detokenize(truncated_tokens).decode("utf-8", errors="ignore")
        return model.embed(truncated_text)

    def close(self) -> None:
        """Release the underlying GGUF model, freeing its memory."""
        if self._model is not None:
            self._model.close()
            self._model = None
