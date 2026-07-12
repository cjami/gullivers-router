import json
import time
from threading import Barrier

import numpy as np
import pytest

from gullivers_router.config import ModelConfig, Settings
from gullivers_router.inference.base import Provider, Role, TokenUsage
from gullivers_router.router import (
    CLOUD_ROUTE,
    DETERMINISTIC_MATH_ROUTE,
    LOCAL_ROUTE,
    RuntimeContext,
    RuntimeOptions,
    Task,
    _answer_ner_lane,
    _Decision,
    classify_tasks,
    load_tasks,
    run_with_context,
)


class FakeEmbedder:
    def embed(self, text):
        return [1.0] if "cloud" in text else [-1.0]


class FastAndRegularCloudEmbedder:
    def embed(self, text):
        return [1.0] if "fast" in text else [-1.0]


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


class CloseableFakeChat(FakeChat):
    def __init__(self, prefix, events):
        super().__init__(prefix)
        self.events = events

    def close(self):
        self.events.append(f"close_{self.prefix}")


class FakeNamedEntityModel:
    def __init__(self, response='{"PER": [], "ORG": [], "LOC": [], "MISC": []}', events=None):
        self.response = response
        self.events = events
        self.calls = []

    def extract(self, text):
        self.calls.append(text)
        return self.response

    def close(self):
        if self.events is not None:
            self.events.append("close_ner")


class FakeCascadeChat(FakeChat):
    def __init__(self, prefix, *, should_escalate=False):
        super().__init__(prefix)
        self.should_escalate = should_escalate
        self.structured_calls = []

    def complete_structured(self, messages, response_model):
        self.structured_calls.append(list(messages))
        return response_model(
            should_escalate=self.should_escalate,
            confidence=0.9,
            failure_mode="reasoning_uncertain" if self.should_escalate else "none",
            rationale="test",
        )


def _settings():
    return Settings(
        hf_token=None,
        local=ModelConfig(provider=Provider.LLAMA, repo_id="local-model", n_threads=2),
        embedding=ModelConfig(provider=Provider.LLAMA, repo_id="embedding-model"),
        ner=ModelConfig(provider=Provider.LLAMA, repo_id="ner-model"),
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


def _context(*, chats=None, chat_factory=None, ner_factory=None):
    if chat_factory is None:
        assert chats is not None

        def chat_factory(config):
            return chats[config.provider]

    if ner_factory is None:

        def ner_factory(config):
            return FakeNamedEntityModel()

    return RuntimeContext(
        settings=_settings(),
        embedding_factory=lambda config: FakeEmbedder(),
        chat_factory=chat_factory,
        ner_factory=ner_factory,
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
    assert [decision.category for decision in decisions] == [None, None]


def _category_weights(path):
    np.savez(
        path,
        weights=np.array([1.0]),
        bias=np.float64(0.0),
        alpha=np.float64(0.5),
        normalize=True,
        cat_weights=np.array([[-1.0], [1.0]]),
        cat_bias=np.array([0.0, 0.0]),
        cat_classes=np.array(["easy", "hard"]),
        cat_alpha=np.array([0.9, 0.1]),
    )


def _known_category_weights(path):
    np.savez(
        path,
        weights=np.array([1.0]),
        bias=np.float64(0.0),
        alpha=np.float64(0.5),
        normalize=True,
        cat_weights=np.array([[-1.0], [1.0]]),
        cat_bias=np.array([0.0, 0.0]),
        cat_classes=np.array(["factual_knowledge", "mathematical_reasoning"]),
        cat_alpha=np.array([0.5, 0.5]),
    )


def _always_math_weights(path):
    np.savez(
        path,
        weights=np.array([1.0]),
        bias=np.float64(0.0),
        alpha=np.float64(0.5),
        normalize=True,
        cat_weights=np.array([[0.0]]),
        cat_bias=np.array([1.0]),
        cat_classes=np.array(["mathematical_reasoning"]),
        cat_alpha=np.array([0.5]),
    )


def _cloud_fast_category_weights(path):
    np.savez(
        path,
        weights=np.array([1.0]),
        bias=np.float64(0.0),
        alpha=np.float64(0.5),
        normalize=True,
        cat_weights=np.array([[1.0], [-1.0]]),
        cat_bias=np.array([0.0, 0.0]),
        cat_classes=np.array(["factual_knowledge", "mathematical_reasoning"]),
        cat_alpha=np.array([0.1, 0.1]),
    )


def _mixed_cloud_usage_weights(path):
    np.savez(
        path,
        weights=np.array([1.0]),
        bias=np.float64(0.0),
        alpha=np.float64(0.1),
        normalize=True,
        cat_weights=np.array([[1.0], [-1.0]]),
        cat_bias=np.array([0.0, 0.0]),
        cat_classes=np.array(["factual_knowledge", "other"]),
        cat_alpha=np.array([0.1, 0.1]),
    )


def _sentiment_summary_category_weights(path, *, summary_alpha=0.1):
    np.savez(
        path,
        weights=np.array([1.0]),
        bias=np.float64(0.0),
        alpha=np.float64(0.1),
        normalize=True,
        cat_weights=np.array([[-1.0], [1.0]]),
        cat_bias=np.array([0.0, 0.0]),
        cat_classes=np.array(["sentiment_classification", "text_summarisation"]),
        cat_alpha=np.array([0.5, summary_alpha]),
    )


def _sentiment_ner_category_weights(path):
    np.savez(
        path,
        weights=np.array([1.0]),
        bias=np.float64(0.0),
        alpha=np.float64(0.1),
        normalize=True,
        cat_weights=np.array([[-1.0], [1.0]]),
        cat_bias=np.array([0.0, 0.0]),
        cat_classes=np.array(["sentiment_classification", "named_entity_recognition"]),
        cat_alpha=np.array([0.5, 0.1]),
    )


def _ner_category_weights(path):
    np.savez(
        path,
        weights=np.array([1.0]),
        bias=np.float64(0.0),
        alpha=np.float64(0.1),
        normalize=True,
        cat_weights=np.array([[1.0]]),
        cat_bias=np.array([1.0]),
        cat_classes=np.array(["named_entity_recognition"]),
        cat_alpha=np.array([0.1]),
    )


def _cloud_first_category_weights(path):
    np.savez(
        path,
        weights=np.array([-1.0]),
        bias=np.float64(0.0),
        alpha=np.float64(0.9),
        normalize=True,
        cat_weights=np.array([[-1.0], [1.0]]),
        cat_bias=np.array([0.0, 0.0]),
        cat_classes=np.array(["code_debugging", "code_generation"]),
        cat_alpha=np.array([0.9, 0.9]),
    )


def test_classify_tasks_applies_per_category_thresholds(tmp_path):
    weights_path = tmp_path / "router.npz"
    _category_weights(weights_path)
    weights = dict(np.load(weights_path))

    decisions = classify_tasks(
        [Task(task_id="l", prompt="local task"), Task(task_id="c", prompt="cloud task")],
        FakeEmbedder(),
        weights,
        local_model="local-model",
        cloud_model="cloud-model",
    )

    assert [decision.category for decision in decisions] == ["easy", "hard"]
    assert [decision.threshold for decision in decisions] == [0.9, 0.1]
    assert [decision.route for decision in decisions] == [LOCAL_ROUTE, CLOUD_ROUTE]


def test_summary_category_respects_cloud_threshold(tmp_path):
    weights_path = tmp_path / "router.npz"
    _sentiment_summary_category_weights(weights_path)
    weights = dict(np.load(weights_path))

    decisions = classify_tasks(
        [Task(task_id="sentiment", prompt="local sentiment"), Task(task_id="summary", prompt="cloud summary")],
        FakeEmbedder(),
        weights,
        local_model="local-model",
        cloud_model="cloud-model",
    )

    assert [decision.category for decision in decisions] == ["sentiment_classification", "text_summarisation"]
    assert [decision.route for decision in decisions] == [LOCAL_ROUTE, CLOUD_ROUTE]
    assert [decision.model for decision in decisions] == ["local-model", "cloud-model"]


def test_ner_category_routes_to_dedicated_local_model_first(tmp_path):
    weights_path = tmp_path / "router.npz"
    _ner_category_weights(weights_path)
    weights = dict(np.load(weights_path))

    decisions = classify_tasks(
        [Task(task_id="ner", prompt="cloud ner")],
        FakeEmbedder(),
        weights,
        local_model="local-model",
        ner_model="ner-model",
        cloud_model="cloud-model",
    )

    assert decisions[0].category == "named_entity_recognition"
    assert decisions[0].route == LOCAL_ROUTE
    assert decisions[0].model == "ner-model"


def test_code_tasks_route_cloud_first(tmp_path):
    weights_path = tmp_path / "router.npz"
    _cloud_first_category_weights(weights_path)
    weights = dict(np.load(weights_path))

    decisions = classify_tasks(
        [Task(task_id="debug", prompt="local debug"), Task(task_id="generate", prompt="cloud generate")],
        FakeEmbedder(),
        weights,
        local_model="local-model",
        cloud_model="cloud-model",
    )

    assert [decision.category for decision in decisions] == ["code_debugging", "code_generation"]
    assert [decision.route for decision in decisions] == [CLOUD_ROUTE, CLOUD_ROUTE]
    assert [decision.model for decision in decisions] == ["cloud-model", "cloud-model"]


def test_predicted_math_expression_routes_to_deterministic_answer(tmp_path):
    weights_path = tmp_path / "router.npz"
    _always_math_weights(weights_path)
    weights = dict(np.load(weights_path))

    decisions = classify_tasks(
        [Task(task_id="math", prompt="Calculate 3 + 3.")],
        FakeEmbedder(),
        weights,
        local_model="local-model",
        cloud_model="cloud-model",
    )

    assert decisions[0].route == DETERMINISTIC_MATH_ROUTE
    assert decisions[0].model == DETERMINISTIC_MATH_ROUTE
    assert decisions[0].answer == "6"


def test_deterministic_math_requires_predicted_math_category(tmp_path):
    weights_path = tmp_path / "router.npz"
    _known_category_weights(weights_path)
    weights = dict(np.load(weights_path))

    decisions = classify_tasks(
        [Task(task_id="factual", prompt="Calculate 3 + 3.")],
        FakeEmbedder(),
        weights,
        local_model="local-model",
        cloud_model="cloud-model",
    )

    assert decisions[0].category == "factual_knowledge"
    assert decisions[0].route == LOCAL_ROUTE
    assert decisions[0].answer is None


def test_percentage_math_phrase_does_not_false_positive(tmp_path):
    weights_path = tmp_path / "router.npz"
    _always_math_weights(weights_path)
    weights = dict(np.load(weights_path))

    decisions = classify_tasks(
        [Task(task_id="percent", prompt="What is 20% more than 50?")],
        FakeEmbedder(),
        weights,
        local_model="local-model",
        cloud_model="cloud-model",
    )

    assert decisions[0].route == LOCAL_ROUTE
    assert decisions[0].answer is None


def test_all_deterministic_answers_skip_chat_models(tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    weights_path = tmp_path / "router.npz"
    _always_math_weights(weights_path)
    input_path.write_text(json.dumps([{"task_id": "math", "prompt": "What is 21 plus 5?"}]), encoding="utf-8")

    def fail_chat_factory(config):
        raise UnexpectedChatBuildError

    run_with_context(
        RuntimeOptions(input_path=input_path, output_path=output_path, router_weights=weights_path),
        _context(chat_factory=fail_chat_factory),
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == [{"task_id": "math", "answer": "26"}]


def test_local_cascade_accepts_local_answer_when_self_check_passes(tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    weights_path = tmp_path / "router.npz"
    _always_math_weights(weights_path)
    input_path.write_text(json.dumps([{"task_id": "a", "prompt": "local proof question"}]), encoding="utf-8")
    local = FakeCascadeChat("local", should_escalate=False)
    cloud = FakeChat("cloud")

    run_with_context(
        RuntimeOptions(
            input_path=input_path,
            output_path=output_path,
            router_weights=weights_path,
            local_cascade=True,
        ),
        _context(chats={Provider.LLAMA: local, Provider.FIREWORKS: cloud}),
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        {"task_id": "a", "answer": "local: local proof question"}
    ]
    assert len(local.structured_calls) == 1
    assert len(local.calls) == 1
    assert cloud.calls == []


def test_local_cascade_self_check_can_escalate_after_local_answer(tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    weights_path = tmp_path / "router.npz"
    _always_math_weights(weights_path)
    input_path.write_text(json.dumps([{"task_id": "a", "prompt": "local proof question"}]), encoding="utf-8")
    local = FakeCascadeChat("local", should_escalate=True)
    cloud = FakeChat("cloud")

    run_with_context(
        RuntimeOptions(
            input_path=input_path,
            output_path=output_path,
            router_weights=weights_path,
            local_cascade=True,
        ),
        _context(chats={Provider.LLAMA: local, Provider.FIREWORKS: cloud}),
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        {"task_id": "a", "answer": "cloud: local proof question"}
    ]
    assert len(local.structured_calls) == 1
    assert len(local.calls) == 1


def test_answer_prompts_include_category_hints(tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    weights_path = tmp_path / "router.npz"
    _sentiment_summary_category_weights(weights_path, summary_alpha=0.9)
    input_path.write_text(
        json.dumps(
            [
                {"task_id": "sentiment", "prompt": "local sentiment question"},
                {"task_id": "summary", "prompt": "cloud summary"},
            ]
        ),
        encoding="utf-8",
    )
    chats = {
        "local-model": FakeChat("local"),
        "cloud-model": FakeChat("cloud"),
    }

    def chat_factory(config):
        return chats[config.repo_id or config.model]

    run_with_context(
        RuntimeOptions(input_path=input_path, output_path=output_path, router_weights=weights_path),
        _context(chat_factory=chat_factory),
    )

    sentiment_system = chats["local-model"].calls[0][0].content
    summary_system = chats["local-model"].calls[1][0].content
    assert (
        sentiment_system == "Answer correctly and concisely. No filler. "
        "Label positive, negative, or neutral; briefly justify."
    )
    assert summary_system == ("Answer correctly and concisely. No filler. Preserve all facts; obey length/format.")
    assert "For " not in sentiment_system
    assert "For " not in summary_system
    assert chats["local-model"].calls[0][-1].content == "local sentiment question"
    assert chats["local-model"].calls[1][-1].content == "cloud summary"
    assert chats["cloud-model"].calls == []


def test_factual_answer_uses_local_verification_system_prompt(tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    weights_path = tmp_path / "router.npz"
    _known_category_weights(weights_path)
    input_path.write_text(
        json.dumps([{"task_id": "fact", "prompt": "local factual question"}]),
        encoding="utf-8",
    )
    local = FakeChat("local")

    run_with_context(
        RuntimeOptions(input_path=input_path, output_path=output_path, router_weights=weights_path),
        _context(chats={Provider.LLAMA: local, Provider.FIREWORKS: FakeChat("cloud")}),
    )

    assert local.calls[0][0].role == Role.SYSTEM
    assert local.calls[0][0].content == (
        "Answer correctly and concisely. No filler. "
        "Answer each part directly. For comparisons, contrast both sides briefly. "
        "Verify facts and include only requested details."
    )


def test_ner_and_local_run_concurrently_then_promote_threads(tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    weights_path = tmp_path / "router.npz"
    _sentiment_ner_category_weights(weights_path)
    input_path.write_text(
        json.dumps(
            [
                {"task_id": "sentiment", "prompt": "local sentiment question"},
                {"task_id": "sentiment-2", "prompt": "second local sentiment question"},
                {"task_id": "ner", "prompt": "cloud ner"},
            ]
        ),
        encoding="utf-8",
    )
    events = []
    barrier = Barrier(2)

    class ConcurrentChat(CloseableFakeChat):
        def complete(self, messages):
            if not self.calls:
                barrier.wait(timeout=1)
                if self.prefix == "local":
                    time.sleep(0.02)
            return super().complete(messages)

        def set_threads(self, n_threads):
            events.append(f"threads_{self.prefix}_{n_threads}")

    class ConcurrentNer(FakeNamedEntityModel):
        def extract(self, text):
            barrier.wait(timeout=1)
            return super().extract(text)

    chats = {"local-model": ConcurrentChat("local", events), "cloud-model": FakeChat("cloud")}
    threads = {}

    def chat_factory(config):
        name = config.repo_id or config.model
        events.append(f"build_{name}")
        threads[name] = config.n_threads
        return chats[name]

    def ner_factory(config):
        events.append("build_ner-model")
        threads["ner-model"] = config.n_threads
        return ConcurrentNer(events=events)

    run_with_context(
        RuntimeOptions(input_path=input_path, output_path=output_path, router_weights=weights_path),
        _context(chat_factory=chat_factory, ner_factory=ner_factory),
    )

    assert events.index("build_local-model") < events.index("close_local")
    assert events.index("build_ner-model") < events.index("close_ner")
    assert threads["local-model"] == 1
    assert threads["ner-model"] == 1
    assert "threads_local_2" in events
    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        {"task_id": "sentiment", "answer": "local: local sentiment question"},
        {"task_id": "sentiment-2", "answer": "local: second local sentiment question"},
        {"task_id": "ner", "answer": ""},
    ]


def test_ner_lane_releases_model_after_answering():
    events = []
    ner_decision = _Decision(
        task=Task("ner", "Extract entities from: 'Ada Lovelace visited London.'"),
        route=LOCAL_ROUTE,
        risk=0.1,
        threshold=0.5,
        model="ner-model",
        category="named_entity_recognition",
    )

    def ner_factory():
        events.append("build_ner")
        return FakeNamedEntityModel('{"PER": ["Ada Lovelace"], "ORG": [], "LOC": ["London"]}', events)

    answers = _answer_ner_lane([ner_decision], ner_factory)

    assert events == ["build_ner", "close_ner"]
    assert answers["ner"] == "Ada Lovelace: PERSON\nLondon: LOCATION"


def test_ner_uses_dedicated_model_and_source_ordered_dates(tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    weights_path = tmp_path / "router.npz"
    _ner_category_weights(weights_path)
    input_path.write_text(
        json.dumps(
            [
                {
                    "task_id": "ner",
                    "prompt": (
                        "Extract all named entities and their types from: "
                        "'On March 15 2023, Sundar Pichai announced Google opened in Zurich.'"
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )
    ner = FakeNamedEntityModel('{"PER":["Sundar Pichai"],"ORG":["Google"],"LOC":["Zurich"],"MISC":[]}')
    ner_threads = []

    def ner_factory(config):
        ner_threads.append(config.n_threads)
        return ner

    run_with_context(
        RuntimeOptions(input_path=input_path, output_path=output_path, router_weights=weights_path),
        _context(chat_factory=lambda config: FakeChat("unused"), ner_factory=ner_factory),
    )

    assert ner.calls == ["On March 15 2023, Sundar Pichai announced Google opened in Zurich."]
    assert ner_threads == [1]
    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        {
            "task_id": "ner",
            "answer": ("March 15 2023: DATE\nSundar Pichai: PERSON\nGoogle: ORGANIZATION\nZurich: LOCATION"),
        }
    ]


def test_all_known_cloud_categories_disable_thinking(tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    weights_path = tmp_path / "router.npz"
    _cloud_fast_category_weights(weights_path)
    input_path.write_text(
        json.dumps(
            [
                {"task_id": "fast", "prompt": "cloud factual question"},
                {"task_id": "regular", "prompt": "local hard math"},
            ]
        ),
        encoding="utf-8",
    )
    built = []
    chats = {
        "regular": FakeChat("regular"),
        "fast": FakeChat("fast"),
        "local": FakeChat("local"),
    }

    def chat_factory(config):
        built.append(config)
        if config.provider == Provider.LLAMA:
            return chats["local"]
        if config.enable_thinking is False and config.reasoning_effort is None and config.temperature == 0.0:
            return chats["fast"]
        return chats["regular"]

    run_with_context(
        RuntimeOptions(input_path=input_path, output_path=output_path, router_weights=weights_path),
        _context(chat_factory=chat_factory),
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        {"task_id": "fast", "answer": "fast: cloud factual question"},
        {"task_id": "regular", "answer": "fast: local hard math"},
    ]
    cloud_configs = [config for config in built if config.provider == Provider.FIREWORKS]
    assert [(config.enable_thinking, config.reasoning_effort, config.temperature) for config in cloud_configs] == [
        (None, None, None),
        (False, None, 0.0),
    ]


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


def test_local_and_cloud_calls_use_separate_system_prompts(tmp_path):
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
    assert local_messages[0].role == Role.SYSTEM
    assert cloud_messages[0].content == "Answer correctly in the fewest words. No filler."
    assert local_messages[0].content == "Answer correctly and concisely. No filler."
    assert local_messages[-1].role == Role.USER
    assert local_messages[-1].content == "local factual question"


def test_cloud_token_usage_is_logged(tmp_path, capsys):
    chats = {
        Provider.LLAMA: FakeChat("local"),
        Provider.FIREWORKS: FakeUsageChat("cloud", TokenUsage(prompt_tokens=12, completion_tokens=3)),
    }

    _run_two_tasks(tmp_path, chats)

    assert "cloud tokens: prompt=12 completion=3 total=15" in capsys.readouterr().err


def test_regular_and_fast_cloud_token_usage_are_logged_together(tmp_path, capsys):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    weights_path = tmp_path / "router.npz"
    _mixed_cloud_usage_weights(weights_path)
    input_path.write_text(
        json.dumps(
            [
                {"task_id": "regular", "prompt": "regular cloud question"},
                {"task_id": "fast", "prompt": "fast cloud question"},
            ]
        ),
        encoding="utf-8",
    )
    chats = {
        "regular": FakeUsageChat("regular", TokenUsage(prompt_tokens=10, completion_tokens=2)),
        "fast": FakeUsageChat("fast", TokenUsage(prompt_tokens=7, completion_tokens=1)),
        "local": FakeChat("local"),
    }

    def chat_factory(config):
        if config.provider == Provider.LLAMA:
            return chats["local"]
        if config.enable_thinking is False and config.reasoning_effort is None and config.temperature == 0.0:
            return chats["fast"]
        return chats["regular"]

    context = RuntimeContext(
        settings=_settings(),
        embedding_factory=lambda config: FastAndRegularCloudEmbedder(),
        chat_factory=chat_factory,
        ner_factory=lambda config: FakeNamedEntityModel(),
    )

    run_with_context(
        RuntimeOptions(input_path=input_path, output_path=output_path, router_weights=weights_path),
        context,
    )

    err = capsys.readouterr().err
    assert "cloud regular tokens: prompt=10 completion=2 total=12" in err
    assert "cloud fast tokens: prompt=7 completion=1 total=8" in err
    assert "cloud tokens: prompt=17 completion=3 total=20" in err


def test_run_releases_embedder_before_building_local_model(tmp_path):
    events = []

    class ClosingEmbedder:
        def embed(self, text):
            return [1.0] if "cloud" in text else [-1.0]

        def close(self):
            events.append("embedder_closed")

    chats = {Provider.LLAMA: FakeChat("local"), Provider.FIREWORKS: FakeChat("cloud")}

    def chat_factory(config):
        events.append(f"build_{config.provider.value}")
        return chats[config.provider]

    context = RuntimeContext(
        settings=_settings(),
        embedding_factory=lambda config: ClosingEmbedder(),
        chat_factory=chat_factory,
        ner_factory=lambda config: FakeNamedEntityModel(),
    )

    input_path = tmp_path / "tasks.json"
    weights_path = tmp_path / "router.npz"
    _weights(weights_path)
    input_path.write_text(
        json.dumps([{"task_id": "a", "prompt": "local q"}, {"task_id": "b", "prompt": "cloud q"}]),
        encoding="utf-8",
    )

    run_with_context(
        RuntimeOptions(input_path=input_path, output_path=tmp_path / "results.json", router_weights=weights_path),
        context,
    )

    assert events.index("embedder_closed") < events.index(f"build_{Provider.LLAMA.value}")


def test_load_tasks_rejects_malformed_input(tmp_path):
    input_path = tmp_path / "tasks.json"
    input_path.write_text(json.dumps([{"task_id": "missing-prompt"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="prompt"):
        load_tasks(input_path)
