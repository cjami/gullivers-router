from gullivers_router.training import store
from gullivers_router.training.combine import ResponsePair
from gullivers_router.training.dataset import Category, Prompt
from gullivers_router.training.judge import ParsedScores, _judge_messages, _mapped_scores, parse_scores, run_judge


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
    assert parse_scores(
        '{"response_a_quality": "adequate", "response_a_rationale": "thin", '
        '"response_b_quality": "excellent", "response_b_rationale": "complete", '
        '"preferred_response": "response_b"}'
    ) == ParsedScores("adequate", "excellent", "response_b", "thin", "complete")


def test_parse_scores_tolerates_surrounding_text():
    assert parse_scores(
        'Here: {"response_a_quality": "poor", "response_b_quality": "good", "preferred_response": "response_b"} done'
    ) == ParsedScores("poor", "good", "response_b")


def test_parse_scores_returns_none_on_malformed_or_unknown_output():
    assert parse_scores("not json at all") == ParsedScores(None, None, None)
    assert parse_scores(
        '{"response_a_quality": "perfect", "response_b_quality": "good", "preferred_response": "response_a"}'
    ) == ParsedScores(None, "good", "response_a")


def test_judge_prompt_uses_word_options_without_numeric_scale():
    messages = _judge_messages(_pair("a"), ("local", "cloud"))
    prompt = "\n".join(message.content for message in messages)

    assert "unacceptable, poor, adequate, good, excellent" in prompt
    assert "response_a, response_b, tie" in prompt
    assert "score" not in prompt.lower()
    assert "1" not in prompt
    assert "5" not in prompt


def test_equal_quality_preference_breaks_tie():
    mapped = _mapped_scores(ParsedScores("good", "good", "response_a"), ("local", "cloud"))

    assert mapped["local_score"] == 4.25
    assert mapped["cloud_score"] == 4.0
    assert mapped["preference_consistent"] is True


def test_equal_quality_tie_has_no_bonus():
    mapped = _mapped_scores(ParsedScores("good", "good", "tie"), ("local", "cloud"))

    assert mapped["local_score"] == 4.0
    assert mapped["cloud_score"] == 4.0
    assert mapped["preference_consistent"] is True


def test_unequal_quality_ignores_preference_for_scoring():
    mapped = _mapped_scores(ParsedScores("excellent", "good", "response_b"), ("local", "cloud"))

    assert mapped["local_score"] == 5.0
    assert mapped["cloud_score"] == 4.0
    assert mapped["preference_consistent"] is False


def test_run_judge_writes_scores_keyed_by_id(tmp_path):
    pairs = [_pair("a"), _pair("b")]
    model = FakeChat(
        {
            "PROMPT_a": (
                '{"response_a_quality": "good", "response_a_rationale": "solid", '
                '"response_b_quality": "good", "response_b_rationale": "also solid", '
                '"preferred_response": "response_b"}'
            ),
            "PROMPT_b": "broken",
        }
    )
    out = tmp_path / "judge.jsonl"

    run_judge(pairs, model, out, max_workers=2)

    rows = {row["id"]: row for row in store.read_records(out)}
    assert {rows["a"]["cloud_score"], rows["a"]["local_score"]} == {4.0, 4.25}
    assert rows["a"]["local_quality"] == "good"
    assert rows["a"]["cloud_quality"] == "good"
    assert rows["a"]["preferred_source"] in {"local", "cloud"}
    assert rows["a"]["primary_order"] in {"local_first", "cloud_first"}
    assert "primary_local_score" not in rows["a"]
    assert "primary_cloud_score" not in rows["a"]
    assert "score_consistent" not in rows["a"]
    assert "swapped_local_score" not in rows["a"]
    assert rows["b"]["cloud_score"] is None


def test_run_judge_skips_already_judged(tmp_path):
    out = tmp_path / "judge.jsonl"
    store.append(out, {"id": "a", "local_score": 1, "cloud_score": 1})
    model = FakeChat(
        {
            "PROMPT_b": (
                '{"response_a_quality": "poor", "response_b_quality": "good", "preferred_response": "response_b"}'
            )
        }
    )

    run_judge([_pair("a"), _pair("b")], model, out, max_workers=2)

    assert set(store.read_map(out, value="cloud_score")) == {"a", "b"}
    assert all("PROMPT_a" not in seen for seen in model.seen)


def test_run_judge_blinds_response_names_and_uses_single_call(tmp_path):
    model = FakeChat(
        {
            "PROMPT_a": (
                '{"response_a_quality": "adequate", "response_b_quality": "good", "preferred_response": "response_b"}'
            )
        }
    )
    out = tmp_path / "judge.jsonl"

    run_judge([_pair("a")], model, out, max_workers=1)

    [first] = model.seen
    assert "[Response LOCAL]" not in first
    assert "[Response CLOUD]" not in first
    assert first.count("[Response A]") == 1
    assert "local" in first
    assert "cloud" in first
