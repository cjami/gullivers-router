import collections

from gullivers_router.inference.base import Message, Role
from gullivers_router.training import generate, store
from gullivers_router.training.dataset import Category, Prompt
from gullivers_router.training.generate import complete_with_retry, run_cloud, run_local


class FakeChat:
    def __init__(self):
        self.calls = []

    def complete(self, messages):
        text = messages[0].content
        self.calls.append(text)
        return f"resp:{text}"


class FlakyChat:
    def __init__(self, fail_text, fail_times):
        self.fail_text = fail_text
        self.fail_times = fail_times
        self.attempts = collections.Counter()

    def complete(self, messages):
        text = messages[0].content
        self.attempts[text] += 1
        if text == self.fail_text and self.attempts[text] <= self.fail_times:
            raise RuntimeError("boom")
        return f"resp:{text}"


def _prompts(*ids):
    return [Prompt(id=i, category=Category.MATHEMATICAL_REASONING, text=f"q-{i}") for i in ids]


def test_run_local_resumes_past_completed_ids(tmp_path):
    out = tmp_path / "local.jsonl"
    store.append(out, {"id": "a", "response": "resp:q-a"})
    model = FakeChat()

    run_local(_prompts("a", "b"), model, out)

    assert model.calls == ["q-b"]
    assert store.completed_ids(out) == {"a", "b"}


def test_run_cloud_generates_missing_and_resumes(tmp_path):
    out = tmp_path / "cloud.jsonl"
    store.append(out, {"id": "a", "response": "done"})
    model = FakeChat()

    run_cloud(_prompts("a", "b"), model, out, max_workers=2)

    assert model.calls == ["q-b"]
    assert store.read_map(out) == {"a": "done", "b": "resp:q-b"}


def test_run_cloud_recovers_after_transient_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(generate.time, "sleep", lambda _seconds: None)
    out = tmp_path / "cloud.jsonl"
    model = FlakyChat(fail_text="q-a", fail_times=1)

    run_cloud(_prompts("a", "b"), model, out, max_workers=2)

    assert store.read_map(out) == {"a": "resp:q-a", "b": "resp:q-b"}
    assert model.attempts["q-a"] == 2


def test_run_cloud_skips_permanently_failing_request(tmp_path, monkeypatch):
    monkeypatch.setattr(generate.time, "sleep", lambda _seconds: None)
    out = tmp_path / "cloud.jsonl"
    model = FlakyChat(fail_text="q-a", fail_times=99)

    run_cloud(_prompts("a", "b"), model, out, max_workers=2)

    assert set(store.read_map(out)) == {"b"}
    assert model.attempts["q-a"] == generate.MAX_ATTEMPTS


def test_complete_with_retry_reraises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(generate.time, "sleep", lambda _seconds: None)
    model = FlakyChat(fail_text="x", fail_times=99)

    try:
        complete_with_retry(model, [Message(Role.USER, "x")])
    except RuntimeError:
        pass
    else:
        raise AssertionError

    assert model.attempts["x"] == generate.MAX_ATTEMPTS
