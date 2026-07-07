"""Local GGUF backends via llama-cpp-python."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gullivers_router.inference.base import DEFAULT_INFERENCE_SEED
from gullivers_router.inference.truncation import EMBEDDING_CONTEXT_LIMIT, truncate_head_tail

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gullivers_router.config import ModelConfig
    from gullivers_router.inference.base import Message

DEFAULT_CHAT_CONTEXT = 4096
OFFLOAD_ALL_LAYERS = -1


def _require_repo_id(config: ModelConfig) -> str:
    if config.repo_id is None:
        msg = "local model requires a repo_id"
        raise ValueError(msg)
    return config.repo_id


class LlamaCppChat:
    """Chat completion from a locally hosted GGUF model."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        n_ctx: int = DEFAULT_CHAT_CONTEXT,
        n_gpu_layers: int = OFFLOAD_ALL_LAYERS,
        flash_attn: bool = True,
    ) -> None:
        """Configure the model; the GGUF loads lazily on first use.

        ``n_gpu_layers`` defaults to offloading every layer to the GPU (llama.cpp's own
        default is CPU-only, orders of magnitude slower for a 31B model). ``flash_attn``
        keeps Gemma's sliding-window KV cache compact so weights plus cache fit in VRAM.
        """
        self._config = config
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._flash_attn = flash_attn
        self._model = None

    def _load(self):  # noqa: ANN202
        if self._model is None:
            from llama_cpp import Llama

            self._model = Llama.from_pretrained(
                repo_id=_require_repo_id(self._config),
                filename=self._config.filename,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                flash_attn=self._flash_attn,
                seed=DEFAULT_INFERENCE_SEED,
                verbose=False,
            )
        return self._model

    def complete(self, messages: Sequence[Message]) -> str:
        """Generate a response for a single prompt."""
        result = self._load().create_chat_completion(
            messages=[m.as_dict() for m in messages],
            seed=DEFAULT_INFERENCE_SEED,
        )
        return result["choices"][0]["message"].get("content") or ""


class LlamaCppEmbedder:
    """Dense embeddings from a locally hosted GGUF model."""

    def __init__(self, config: ModelConfig, *, n_ctx: int = EMBEDDING_CONTEXT_LIMIT) -> None:
        """Configure the embedder; the GGUF loads lazily on first use."""
        self._config = config
        self._n_ctx = n_ctx
        self._model = None

    def _load(self):  # noqa: ANN202
        if self._model is None:
            from llama_cpp import Llama

            self._model = Llama.from_pretrained(
                repo_id=_require_repo_id(self._config),
                filename=self._config.filename,
                n_ctx=self._n_ctx,
                embedding=True,
                seed=DEFAULT_INFERENCE_SEED,
                verbose=False,
            )
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed text, applying head-and-tail truncation past the context window."""
        model = self._load()
        tokens = model.tokenize(text.encode("utf-8"), add_bos=True)
        truncated_tokens = truncate_head_tail(tokens, limit=self._n_ctx)
        truncated_text = model.detokenize(truncated_tokens).decode("utf-8", errors="ignore")
        return model.embed(truncated_text)
