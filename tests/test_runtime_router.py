import json
import time

import numpy as np
import pytest

from gullivers_router.config import ModelConfig, Settings
from gullivers_router.inference.base import Provider, Role, TokenUsage
from gullivers_router.router import (
    CLOUD_ROUTE,
    LOCAL_ROUTE,
    RuntimeContext,
    RuntimeOptions,
    Task,
    classify_tasks,
    load_tasks,
    run_with_context,
)


class FakeEmbedder:
    def embed(self, text):
        return [1.0] if "cloud" in text else [-1.0]


class FakeChat:
    def __init__(self, prefix, *, delay=False):
        self.prefix = prefix
        self.delay = delay
        self.calls = []

    def complete(self, messages):
        self.calls.append(list(messages))
        prompt = messages[-1].content
        if self.delay and "slow" in prompt:
            time.sleep(0.02)
        return f"{self.prefix}: {prompt}"


def _settings():
    return Settings(
        hf_token=None,
        local=ModelConfig(provider=Provider.LLAMA, repo_id="local-model"),
        embedding=ModelConfig(provider=Provider.LLAMA, repo_id="embedding-model"),
        cloud=ModelConfig(
            provider=Provider.FIREWORKS,
            model="cloud-model",
            api_key="key",
            base_url="https://proxy.example/v1",
        ),
        judge=ModelConfig(provider=Provider.FIREWORKS, model="judge-model"),
    )


def _weights(path, *, alpha=0.5):
    np.savez(path, weights=np.array([1.0]), bias=np.float64(0.0), alpha=np.float64(alpha), normalize=True)


class UnexpectedChatBuildError(AssertionError):
    pass


def _context(*, chats=None, chat_factory=None):
    if chat_factory is None:
        assert chats is not None

        def chat_factory(config):
            return chats[config.provider]

    return RuntimeContext(
        settings=_settings(),
        embedding_factory=lambda config: FakeEmbedder(),
        chat_factory=chat_factory,
    )


def test_run_writes_results_in_input_order(tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    weights_path = tmp_path / "router.npz"
    _weights(weights_path)
    input_path.write_text(
        json.dumps(
            [
                {"task_id": "a", "prompt": "local factual question"},
                {"task_id": "b", "prompt": "cloud hard reasoning"},
            ]
        ),
        encoding="utf-8",
    )
    chats = {
        Provider.LLAMA: FakeChat("local"),
        Provider.FIREWORKS: FakeChat("cloud"),
    }

    run_with_context(
        RuntimeOptions(input_path=input_path, output_path=output_path, router_weights=weights_path),
        _context(chats=chats),
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        {"task_id": "a", "answer": "local: local factual question"},
        {"task_id": "b", "answer": "cloud: cloud hard reasoning"},
    ]


def test_classify_only_writes_diagnostics_and_skips_chat_models(tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "routes.json"
    weights_path = tmp_path / "router.npz"
    _weights(weights_path)
    input_path.write_text(
        json.dumps(
            [
                {"task_id": "a", "prompt": "local short task"},
                {"task_id": "b", "prompt": "cloud dense task"},
            ]
        ),
        encoding="utf-8",
    )

    def fail_chat_factory(config):
        raise UnexpectedChatBuildError

    run_with_context(
        RuntimeOptions(
            input_path=input_path,
            output_path=output_path,
            router_weights=weights_path,
            classify_only=True,
        ),
        _context(chat_factory=fail_chat_factory),
    )

    records = json.loads(output_path.read_text(encoding="utf-8"))
    assert records[0]["task_id"] == "a"
    assert records[0]["route"] == LOCAL_ROUTE
    assert records[0]["model"] == "local-model"
    assert records[1]["task_id"] == "b"
    assert records[1]["route"] == CLOUD_ROUTE
    assert records[1]["model"] == "cloud-model"
    assert records[1]["risk"] > records[0]["risk"]
    assert records[0]["threshold"] == 0.5


def test_classify_tasks_routes_from_exported_weights(tmp_path):
    weights_path = tmp_path / "router.npz"
    _weights(weights_path, alpha=0.5)
    weights = dict(np.load(weights_path))

    decisions = classify_tasks(
        [Task(task_id="local", prompt="local task"), Task(task_id="cloud", prompt="cloud task")],
        FakeEmbedder(),
        weights,
        local_model="local-model",
        cloud_model="cloud-model",
    )

    assert [decision.route for decision in decisions] == [LOCAL_ROUTE, CLOUD_ROUTE]


def test_cloud_answers_preserve_input_order(tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    weights_path = tmp_path / "router.npz"
    _weights(weights_path)
    input_path.write_text(
        json.dumps(
            [
                {"task_id": "slow", "prompt": "cloud slow task"},
                {"task_id": "fast", "prompt": "cloud fast task"},
            ]
        ),
        encoding="utf-8",
    )
    chats = {
        Provider.LLAMA: FakeChat("local"),
        Provider.FIREWORKS: FakeChat("cloud", delay=True),
    }

    run_with_context(
        RuntimeOptions(input_path=input_path, output_path=output_path, router_weights=weights_path, workers=2),
        _context(chats=chats),
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        {"task_id": "slow", "answer": "cloud: cloud slow task"},
        {"task_id": "fast", "answer": "cloud: cloud fast task"},
    ]


class FakeUsageChat(FakeChat):
    def __init__(self, prefix, usage):
        super().__init__(prefix)
        self._usage = usage

    @property
    def usage(self):
        return self._usage


def _run_two_tasks(tmp_path, chats, *, output_name="results.json"):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / output_name
    weights_path = tmp_path / "router.npz"
    _weights(weights_path)
    input_path.write_text(
        json.dumps(
            [
                {"task_id": "a", "prompt": "local factual question"},
                {"task_id": "b", "prompt": "cloud hard reasoning"},
            ]
        ),
        encoding="utf-8",
    )
    run_with_context(
        RuntimeOptions(input_path=input_path, output_path=output_path, router_weights=weights_path),
        _context(chats=chats),
    )


def test_cloud_call_prepends_concise_system_prompt(tmp_path):
    chats = {
        Provider.LLAMA: FakeChat("local"),
        Provider.FIREWORKS: FakeChat("cloud"),
    }

    _run_two_tasks(tmp_path, chats)

    cloud_messages = chats[Provider.FIREWORKS].calls[0]
    assert cloud_messages[0].role == Role.SYSTEM
    assert cloud_messages[0].content
    assert cloud_messages[-1].role == Role.USER
    assert cloud_messages[-1].content == "cloud hard reasoning"

    local_messages = chats[Provider.LLAMA].calls[0]
    assert [message.role for message in local_messages] == [Role.USER]


def test_cloud_token_usage_is_logged(tmp_path, capsys):
    chats = {
        Provider.LLAMA: FakeChat("local"),
        Provider.FIREWORKS: FakeUsageChat("cloud", TokenUsage(prompt_tokens=12, completion_tokens=3)),
    }

    _run_two_tasks(tmp_path, chats)

    assert "cloud tokens: prompt=12 completion=3 total=15" in capsys.readouterr().err


def test_load_tasks_rejects_malformed_input(tmp_path):
    input_path = tmp_path / "tasks.json"
    input_path.write_text(json.dumps([{"task_id": "missing-prompt"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="prompt"):
        load_tasks(input_path)
