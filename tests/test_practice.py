import json

import pytest

from gullivers_router.config import ModelConfig, Settings
from gullivers_router.inference.base import Provider, Role
from gullivers_router.practice import PracticeContext, PracticeOptions, score_practice_with_context


class FakeJudge:
    def __init__(self):
        self.calls = []

    def complete(self, messages):
        raise AssertionError

    def complete_structured(self, messages, response_model):
        self.calls.append(list(messages))
        submitted_answer = messages[-1].content.split("[Submitted answer]", maxsplit=1)[1]
        if "wrong" in submitted_answer:
            return response_model(quality="incorrect", score=0.0, rationale="Does not match the reference.")
        return response_model(quality="correct", score=1.0, rationale="Matches the reference.")


def _settings():
    return Settings(
        hf_token=None,
        local=ModelConfig(provider=Provider.LLAMA, repo_id="local-model"),
        embedding=ModelConfig(provider=Provider.LLAMA, repo_id="embedding-model"),
        cloud=ModelConfig(provider=Provider.FIREWORKS, model="cloud-model"),
        judge=ModelConfig(provider=Provider.FIREWORKS, model="judge-model"),
    )


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_score_practice_writes_summary_and_grades_missing_answers(tmp_path):
    tasks = tmp_path / "tasks.json"
    results = tmp_path / "results.json"
    answer_set = tmp_path / "answer_set.json"
    report = tmp_path / "score.json"
    judge = FakeJudge()
    _write_json(
        tasks,
        [
            {"task_id": "a", "prompt": "What is 2 + 2?"},
            {"task_id": "b", "prompt": "What is 3 + 3?"},
            {"task_id": "c", "prompt": "What is 4 + 4?"},
        ],
    )
    _write_json(results, [{"task_id": "a", "answer": "4"}, {"task_id": "b", "answer": ""}])
    _write_json(
        answer_set,
        [
            {"task_id": "a", "answer": "4", "notes": "Must be numeric."},
            {"task_id": "b", "answer": "6"},
            {"task_id": "c", "answer": "8"},
        ],
    )

    scored = score_practice_with_context(
        PracticeOptions(
            tasks_path=tasks,
            results_path=results,
            answer_set_path=answer_set,
            output_path=report,
            workers=2,
        ),
        PracticeContext(settings=_settings(), chat_factory=lambda config: judge),
    )

    assert scored == json.loads(report.read_text(encoding="utf-8"))
    assert scored["summary"]["tasks"] == 3
    assert scored["summary"]["mean_score"] == pytest.approx(1 / 3)
    assert scored["summary"]["percent_score"] == pytest.approx(100 / 3)
    assert scored["summary"]["correct"] == 1
    assert scored["summary"]["partially_correct"] == 0
    assert scored["summary"]["incorrect"] == 2
    assert [grade["task_id"] for grade in scored["grades"]] == ["a", "b", "c"]
    assert scored["grades"][1]["rationale"] == "No submitted answer was provided."
    assert scored["grades"][2]["rationale"] == "No submitted answer was provided."
    assert len(judge.calls) == 1
    assert judge.calls[0][0].role == Role.SYSTEM
    assert "What is 2 + 2?" in judge.calls[0][-1].content
    assert "Must be numeric." in judge.calls[0][-1].content


def test_score_practice_rejects_unknown_result_ids(tmp_path):
    tasks = tmp_path / "tasks.json"
    results = tmp_path / "results.json"
    answer_set = tmp_path / "answer_set.json"
    _write_json(tasks, [{"task_id": "a", "prompt": "Question?"}])
    _write_json(results, [{"task_id": "b", "answer": "Answer"}])
    _write_json(answer_set, [{"task_id": "a", "answer": "Answer"}])

    with pytest.raises(ValueError, match="unknown task ids: b"):
        score_practice_with_context(
            PracticeOptions(tasks_path=tasks, results_path=results, answer_set_path=answer_set),
            PracticeContext(settings=_settings(), chat_factory=lambda config: FakeJudge()),
        )
