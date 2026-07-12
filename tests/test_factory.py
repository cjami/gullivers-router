import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from gullivers_router.config import ModelConfig
from gullivers_router.inference import factory
from gullivers_router.inference.base import (
    DEFAULT_INFERENCE_SEED,
    InferenceDeadlineExceededError,
    Message,
    Provider,
    Role,
)
from gullivers_router.inference.llama_cpp import LlamaCppChat, LlamaCppEmbedder, LlamaCppNamedEntity
from gullivers_router.inference.openai_compat import OpenAICompatChat


class StructuredReply(BaseModel):
    answer: str


def _cfg(provider, **kwargs):
    return ModelConfig(provider=provider, **kwargs)


def test_build_chat_model_local():
    chat = factory.build_chat_model(_cfg(Provider.LLAMA))

    assert isinstance(chat, LlamaCppChat)
    assert chat._enable_thinking is False
    assert chat._temperature == 0.0
    assert chat._top_p == 0.95
    assert chat._top_k == 64


def test_build_chat_model_openai_compatible():
    cfg = _cfg(Provider.OPENAI, api_key="k", model="m")
    assert isinstance(factory.build_chat_model(cfg), OpenAICompatChat)


def test_build_embedding_model_local():
    assert isinstance(factory.build_embedding_model(_cfg(Provider.LLAMA)), LlamaCppEmbedder)


def test_build_named_entity_model_local():
    model = factory.build_named_entity_model(_cfg(Provider.LLAMA))

    assert isinstance(model, LlamaCppNamedEntity)


def test_build_named_entity_model_rejects_cloud_provider():
    cfg = _cfg(Provider.FIREWORKS, api_key="k", model="m")
    with pytest.raises(ValueError, match="named entity extraction"):
        factory.build_named_entity_model(cfg)


def test_factory_passes_named_entity_runtime_options(tmp_path):
    model = factory.build_named_entity_model(
        _cfg(
            Provider.LLAMA,
            n_ctx=1024,
            n_gpu_layers=0,
            max_tokens=256,
            n_threads=2,
            model_root=tmp_path,
        )
    )

    assert isinstance(model, LlamaCppNamedEntity)
    assert model._n_ctx == 1024
    assert model._n_gpu_layers == 0
    assert model._max_tokens == 256
    assert model._n_threads == 2
    assert model._model_root == tmp_path


def test_factory_passes_llama_chat_runtime_options(tmp_path):
    chat = factory.build_chat_model(
        _cfg(
            Provider.LLAMA,
            n_ctx=2048,
            n_gpu_layers=0,
            flash_attn=False,
            enable_thinking=False,
            temperature=0.7,
            top_p=0.8,
            top_k=32,
            max_tokens=1024,
            n_threads=2,
            model_root=tmp_path,
        )
    )

    assert isinstance(chat, LlamaCppChat)
    assert chat._n_ctx == 2048
    assert chat._n_gpu_layers == 0
    assert chat._flash_attn is False
    assert chat._enable_thinking is False
    assert chat._temperature == 0.7
    assert chat._top_p == 0.8
    assert chat._top_k == 32
    assert chat._max_tokens == 1024
    assert chat._n_threads == 2
    assert chat._model_root == tmp_path


def test_factory_passes_llama_embedder_runtime_options(tmp_path):
    embedder = factory.build_embedding_model(
        _cfg(
            Provider.LLAMA,
            n_ctx=2048,
            n_gpu_layers=-1,
            n_threads=4,
            model_root=tmp_path,
            pooling_type="last",
            input_prefix="classify: ",
        )
    )

    assert isinstance(embedder, LlamaCppEmbedder)
    assert embedder._n_ctx == 2048
    assert embedder._n_gpu_layers == -1
    assert embedder._n_threads == 4
    assert embedder._model_root == tmp_path
    assert embedder._pooling_type == 3
    assert embedder._input_prefix == "classify: "


def test_build_embedding_model_rejects_cloud_provider():
    cfg = _cfg(Provider.FIREWORKS, api_key="k", model="m")
    with pytest.raises(ValueError, match="embedding"):
        factory.build_embedding_model(cfg)


def test_openai_compatible_chat_uses_global_seed():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    chat = OpenAICompatChat(_cfg(Provider.OPENAI, api_key="k", model="m"))
    chat._client = cast(Any, SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))

    assert chat.complete([Message(Role.USER, "hello")]) == "ok"
    assert captured["seed"] == DEFAULT_INFERENCE_SEED


@pytest.mark.parametrize("content", [None, "   "])
def test_openai_compatible_chat_rejects_empty_content(content):
    def create(**_kwargs):
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    chat = OpenAICompatChat(_cfg(Provider.OPENAI, api_key="k", model="m"))
    chat._client = cast(Any, SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))

    with pytest.raises(RuntimeError, match="empty content"):
        chat.complete([Message(Role.USER, "hello")])


def test_openai_compatible_chat_sends_json_schema_response_format():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content='{"answer": "ok"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    chat = OpenAICompatChat(_cfg(Provider.FIREWORKS, api_key="k", model="m"))
    chat._client = cast(Any, SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))

    result = chat.complete_structured([Message(Role.USER, "hello")], StructuredReply)

    assert result == StructuredReply(answer="ok")
    assert captured["seed"] == DEFAULT_INFERENCE_SEED
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["name"] == "StructuredReply"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["response_format"]["json_schema"]["schema"]["properties"]["answer"]["type"] == "string"


def test_llama_chat_loads_with_global_seed(monkeypatch):
    captured = {}

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **kwargs) -> "FakeLlama":
            captured.update(kwargs)
            return cls()

        def create_chat_completion(self, messages, **kwargs):
            captured["messages"] = messages
            captured.update(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    chat = LlamaCppChat(_cfg(Provider.LLAMA, repo_id="repo", filename="model.gguf"))

    assert chat.complete([Message(Role.USER, "hello")]) == "ok"
    assert captured["seed"] == DEFAULT_INFERENCE_SEED
    assert captured["temperature"] == 0.0
    assert captured["top_p"] == 0.95
    assert captured["top_k"] == 64


def test_llama_chat_stops_generation_at_deadline(monkeypatch):
    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **_kwargs) -> "FakeLlama":
            return cls()

        def create_chat_completion(self, *, stopping_criteria, **_kwargs):
            stopping_criteria(None, None)
            return {"choices": [{"message": {"content": "partial"}}]}

    class FakeStoppingCriteriaList(list):
        def __call__(self, input_ids, logits):
            return any(criterion(input_ids, logits) for criterion in self)

    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(Llama=FakeLlama, StoppingCriteriaList=FakeStoppingCriteriaList),
    )
    times = iter([0.0, 2.0])
    monkeypatch.setattr("gullivers_router.inference.llama_cpp.time.monotonic", lambda: next(times))
    chat = LlamaCppChat(_cfg(Provider.LLAMA, repo_id="repo", filename="model.gguf"))

    with pytest.raises(InferenceDeadlineExceededError):
        chat.complete_before([Message(Role.USER, "hello")], deadline=1.0)


def test_llama_chat_updates_threads_after_loading(monkeypatch):
    captured = {}
    thread_changes = []

    class FakeLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.ctx = object()

        @classmethod
        def from_pretrained(cls, **kwargs) -> "FakeLlama":
            return cls(**kwargs)

        def create_chat_completion(self, **_kwargs):
            return {"choices": [{"message": {"content": "ok"}}]}

    low_level = SimpleNamespace(
        llama_set_n_threads=lambda _ctx, n_threads, n_threads_batch: thread_changes.append((n_threads, n_threads_batch))
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama, llama_cpp=low_level))
    chat = LlamaCppChat(
        _cfg(Provider.LLAMA, repo_id="repo", filename="model.gguf"),
        n_threads=1,
    )

    assert chat.complete([Message(Role.USER, "hello")]) == "ok"
    chat.set_threads(2)

    assert captured["n_threads"] == 1
    assert captured["n_threads_batch"] == 1
    assert thread_changes == [(2, 2)]


def test_llama_model_construction_is_serialized_across_roles(monkeypatch):
    state_lock = Lock()
    active_loads = 0
    max_active_loads = 0

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **_kwargs) -> "FakeLlama":
            nonlocal active_loads, max_active_loads
            with state_lock:
                active_loads += 1
                max_active_loads = max(max_active_loads, active_loads)
            time.sleep(0.02)
            with state_lock:
                active_loads -= 1
            return cls()

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setattr("gullivers_router.inference.llama_cpp._SERIALIZE_MODEL_LOADS", True)
    config = _cfg(Provider.LLAMA, repo_id="repo", filename="model.gguf")
    chat = LlamaCppChat(config)
    ner = LlamaCppNamedEntity(config)

    with ThreadPoolExecutor(max_workers=2) as pool:
        chat_future = pool.submit(chat._load)
        ner_future = pool.submit(ner._load)
        chat_future.result()
        ner_future.result()

    assert max_active_loads == 1


def test_llama_chat_rejects_invalid_thread_count():
    chat = LlamaCppChat(_cfg(Provider.LLAMA, repo_id="repo", filename="model.gguf"))

    with pytest.raises(ValueError, match="at least 1"):
        chat.set_threads(0)


def test_llama_named_entity_model_uses_direct_deterministic_completion(monkeypatch):
    captured = {}

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **kwargs) -> "FakeLlama":
            captured.update(kwargs)
            return cls()

        def create_completion(self, **kwargs):
            captured.update(kwargs)
            return {"choices": [{"text": '{"PER":["Ada Lovelace"]}'}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    model = LlamaCppNamedEntity(
        _cfg(Provider.LLAMA, repo_id="repo", filename="model.gguf"),
        max_tokens=256,
    )

    assert model.extract("Ada Lovelace wrote a program.") == '{"PER":["Ada Lovelace"]}'
    assert "Input: Ada Lovelace wrote a program." in captured["prompt"]
    assert "DATE" not in captured["prompt"]
    assert captured["max_tokens"] == 256
    assert captured["temperature"] == 0.1
    assert captured["top_p"] == 1.0
    assert captured["top_k"] == 1
    assert captured["seed"] == DEFAULT_INFERENCE_SEED


def test_llama_chat_uses_local_model_before_hugging_face(monkeypatch, tmp_path):
    captured = {}
    local_model = tmp_path / "google" / "gemma"
    local_model.mkdir(parents=True)
    model_path = local_model / "model.gguf"
    model_path.write_text("model", encoding="utf-8")

    class FakeLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        @classmethod
        def from_pretrained(cls, **_kwargs) -> "FakeLlama":
            raise AssertionError

        def create_chat_completion(self, messages, **kwargs):
            captured["messages"] = messages
            captured.update(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    chat = LlamaCppChat(_cfg(Provider.LLAMA, repo_id="google/gemma", filename="model.gguf"), model_root=tmp_path)

    assert chat.complete([Message(Role.USER, "hello")]) == "ok"
    assert captured["model_path"] == str(model_path)
    assert captured["seed"] == DEFAULT_INFERENCE_SEED


def test_llama_chat_sends_schema_response_format(monkeypatch):
    captured = {}

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **kwargs) -> "FakeLlama":
            captured.update(kwargs)
            return cls()

        def create_chat_completion(self, messages, **kwargs):
            captured["messages"] = messages
            captured.update(kwargs)
            return {"choices": [{"message": {"content": '{"answer": "ok"}'}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    chat = LlamaCppChat(_cfg(Provider.LLAMA, repo_id="repo", filename="model.gguf"))

    result = chat.complete_structured([Message(Role.USER, "hello")], StructuredReply)

    assert result == StructuredReply(answer="ok")
    assert captured["seed"] == DEFAULT_INFERENCE_SEED
    assert captured["response_format"]["type"] == "json_object"
    assert captured["response_format"]["schema"]["properties"]["answer"]["type"] == "string"
    assert captured["temperature"] == 0.0
    assert captured["top_p"] == 0.95
    assert captured["top_k"] == 64


def test_llama_chat_strips_thinking_sections(monkeypatch):
    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **_kwargs) -> "FakeLlama":
            return cls()

        def create_chat_completion(self, **_kwargs):
            return {"choices": [{"message": {"content": "<think>private</think>\nanswer"}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    chat = LlamaCppChat(_cfg(Provider.LLAMA, repo_id="repo", filename="model.gguf"))

    assert chat.complete([Message(Role.USER, "hello")]) == "answer"


def test_llama_chat_strips_gemma_thought_channel(monkeypatch):
    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **_kwargs) -> "FakeLlama":
            return cls()

        def create_chat_completion(self, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "<|channel|>thought hidden reasoning<channel|>\nfinal answer",
                        }
                    }
                ]
            }

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    chat = LlamaCppChat(_cfg(Provider.LLAMA, repo_id="repo", filename="model.gguf"))

    assert chat.complete([Message(Role.USER, "hello")]) == "final answer"


def test_llama_structured_chat_strips_thinking_before_parsing(monkeypatch):
    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **_kwargs) -> "FakeLlama":
            return cls()

        def create_chat_completion(self, **_kwargs):
            return {"choices": [{"message": {"content": '<think>private</think>\n{"answer": "ok"}'}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    chat = LlamaCppChat(_cfg(Provider.LLAMA, repo_id="repo", filename="model.gguf"))

    assert chat.complete_structured([Message(Role.USER, "hello")], StructuredReply) == StructuredReply(answer="ok")


def test_llama_chat_enables_thinking_for_compatible_template(monkeypatch):
    captured = {}

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **_kwargs) -> "FakeLlama":
            return cls()

        def __init__(self):
            self.metadata = {"tokenizer.chat_template": "{% if enable_thinking %}<think>{% endif %}"}
            self.chat_handler = None
            self.chat_format = "chat_template.default"
            self._chat_handlers = {"chat_template.default": self._handler}

        def _handler(self, **kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": "thinking"}}]}

        def create_chat_completion(self, **_kwargs):
            raise AssertionError

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    chat = LlamaCppChat(_cfg(Provider.LLAMA, repo_id="repo", filename="model.gguf"), enable_thinking=True)

    assert chat.complete([Message(Role.USER, "hello")]) == "thinking"
    assert captured["enable_thinking"] is True
    assert captured["temperature"] == 0.0
    assert captured["top_p"] == 0.95
    assert captured["top_k"] == 64
    assert captured["llama"] is chat._model


def test_llama_chat_ignores_thinking_for_unsupported_template(monkeypatch):
    captured = {}

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, **_kwargs) -> "FakeLlama":
            return cls()

        def __init__(self):
            self.metadata = {"tokenizer.chat_template": "{{ messages[0]['content'] }}"}

        def create_chat_completion(self, **kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": "plain"}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    chat = LlamaCppChat(_cfg(Provider.LLAMA, repo_id="repo", filename="model.gguf"))

    assert chat.complete([Message(Role.USER, "hello")]) == "plain"
    assert "enable_thinking" not in captured


def test_llama_embedder_uses_qwen_pooling_and_instruction(monkeypatch, tmp_path):
    captured = {}
    local_model = tmp_path / "Qwen" / "Qwen3-Embedding-0.6B-GGUF"
    local_model.mkdir(parents=True)
    model_path = local_model / "Qwen3-Embedding-0.6B-Q8_0.gguf"
    model_path.write_text("model", encoding="utf-8")

    class FakeLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        @classmethod
        def from_pretrained(cls, **_kwargs) -> "FakeLlama":
            raise AssertionError

        def tokenize(self, text, add_bos):
            captured["add_bos"] = add_bos
            captured["input_text"] = text.decode()
            return [1, 2, 3]

        def detokenize(self, tokens):
            captured["tokens"] = tokens
            return b"hello"

        def embed(self, text):
            captured["text"] = text
            return [0.1, 0.2]

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    embedder = LlamaCppEmbedder(
        _cfg(
            Provider.LLAMA,
            repo_id="Qwen/Qwen3-Embedding-0.6B-GGUF",
            filename="Qwen3-Embedding-0.6B-Q8_0.gguf",
        ),
        model_root=tmp_path,
        pooling_type="last",
        input_prefix="classify: ",
    )

    assert embedder.embed("hello") == [0.1, 0.2]
    assert captured["model_path"] == str(model_path)
    assert captured["embedding"] is True
    assert captured["pooling_type"] == 3
    assert captured["input_text"] == "classify: hello"


def test_llama_embedder_rejects_unknown_pooling_type():
    with pytest.raises(ValueError, match="unsupported pooling type"):
        LlamaCppEmbedder(_cfg(Provider.LLAMA), pooling_type="mystery")
