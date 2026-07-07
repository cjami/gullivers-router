from gullivers_router.training.dataset import extract_prompt_text


def test_extract_prompt_text_returns_human_turn():
    conversations = [
        {"from": "human", "value": "hello"},
        {"from": "gpt", "value": "hi"},
    ]
    assert extract_prompt_text(conversations) == "hello"


def test_extract_prompt_text_missing_human_returns_none():
    conversations = [{"from": "gpt", "value": "hi"}]
    assert extract_prompt_text(conversations) is None
