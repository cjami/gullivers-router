from gullivers_router.training import store
from gullivers_router.training.combine import ResponsePair
from gullivers_router.training.dataset import Category, Prompt
from gullivers_router.training.judge import parse_scores, run_judge


class FakeChat:
    def __init__(self, by_token):
        self.by_token = by_token
        self.seen = []

    def complete(self, messages):
        user = messages[-1].content
        self.seen.append(user)
        for token, reply in self.by_token.items():
            if token in user:
                return reply
        return "broken"


def _pair(pid):
    prompt = Prompt(id=pid, category=Category.MATHEMATICAL_REASONING, text=f"PROMPT_{pid}")
    return ResponsePair(prompt=prompt, local_response="local", cloud_response="cloud")


def test_parse_scores_reads_json_object():
    assert parse_scores('{"local_score": 4, "cloud_score": 9}') == (4, 9)


def test_parse_scores_tolerates_surrounding_text_and_clamps():
    assert parse_scores('Here: {"local_score": 20, "cloud_score": 0} done') == (10, 1)


def test_parse_scores_returns_none_on_malformed_output():
    assert parse_scores("not json at all") == (None, None)
    assert parse_scores('{"local_score": "n/a"}') == (None, None)


def test_run_judge_writes_scores_keyed_by_id(tmp_path):
    pairs = [_pair("a"), _pair("b")]
    model = FakeChat({"PROMPT_a": '{"local_score": 3, "cloud_score": 7}', "PROMPT_b": "broken"})
    out = tmp_path / "judge.jsonl"

    run_judge(pairs, model, out, max_workers=2)

    assert store.read_map(out, value="cloud_score") == {"a": 7, "b": None}


def test_run_judge_skips_already_judged(tmp_path):
    out = tmp_path / "judge.jsonl"
    store.append(out, {"id": "a", "local_score": 1, "cloud_score": 1})
    model = FakeChat({"PROMPT_b": '{"local_score": 2, "cloud_score": 5}'})

    run_judge([_pair("a"), _pair("b")], model, out, max_workers=2)

    assert store.read_map(out, value="cloud_score") == {"a": 1, "b": 5}
    assert all("PROMPT_a" not in seen for seen in model.seen)
