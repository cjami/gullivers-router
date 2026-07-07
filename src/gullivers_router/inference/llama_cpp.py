"""Local GGUF backends via llama-cpp-python."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gullivers_router.inference.truncation import EMBEDDING_CONTEXT_LIMIT, truncate_head_tail

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gullivers_router.config import ModelConfig
    from gullivers_router.inference.base import Message

DEFAULT_CHAT_CONTEXT = 8192


def _require_repo_id(config: ModelConfig) -> str:
    if config.repo_id is None:
        msg = "local model requires a repo_id"
        raise ValueError(msg)
    return config.repo_id


class LlamaCppChat:
    """Chat completion from a locally hosted GGUF model."""

    def __init__(self, config: ModelConfig, *, n_ctx: int = DEFAULT_CHAT_CONTEXT) -> None:
        """Configure the model; the GGUF loads lazily on first use."""
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
                verbose=False,
            )
        return self._model

    def complete(self, messages: Sequence[Message]) -> str:
        """Generate a response for a single prompt."""
        result = self._load().create_chat_completion(messages=[m.as_dict() for m in messages])
        return result["choices"][0]["message"].get("content") or ""

    def complete_batch(self, requests: Sequence[Sequence[Message]]) -> list[str]:
        """Generate responses sequentially; a single GPU cannot batch decode."""
        from tqdm import tqdm

        return [self.complete(request) for request in tqdm(requests, desc="local generation")]


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
