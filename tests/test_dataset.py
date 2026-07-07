from gullivers_router.training.dataset import (
    Category,
    Prompt,
    _BalancedSampler,
    classify,
    extract_prompt_text,
    load_prompts_file,
    save_prompts,
)


def test_extract_prompt_text_returns_human_turn():
    conversations = [
        {"from": "human", "value": "hello"},
        {"from": "gpt", "value": "hi"},
    ]
    assert extract_prompt_text(conversations) == "hello"


def test_extract_prompt_text_missing_human_returns_none():
    conversations = [{"from": "gpt", "value": "hi"}]
    assert extract_prompt_text(conversations) is None


def test_classify_maps_real_dataset_categories():
    assert classify("explanation") == Category.FACTUAL_KNOWLEDGE
    assert classify("trivia") == Category.FACTUAL_KNOWLEDGE
    assert classify("math") == Category.MATHEMATICAL_REASONING
    assert classify("sentiment analysis") == Category.SENTIMENT_CLASSIFICATION
    assert classify("text classification") == Category.SENTIMENT_CLASSIFICATION
    assert classify("information extraction") == Category.NAMED_ENTITY_RECOGNITION
    assert classify("debugging") == Category.CODE_DEBUGGING
    assert classify("coding") == Category.CODE_GENERATION
    assert classify("logical reasoning") == Category.LOGICAL_REASONING


def test_classify_is_case_insensitive_and_skips_unknown():
    assert classify("  Math ") == Category.MATHEMATICAL_REASONING
    assert classify("creative writing") is None


def _offer_many(sampler, raw_category, category, count, start=0):
    for index in range(start, start + count):
        prompt = Prompt(id=f"{raw_category}-{index}", category=category, text="x")
        sampler.offer(raw_category, prompt)


def test_balanced_sampler_splits_quota_evenly_across_sources():
    sampler = _BalancedSampler(target=4)
    _offer_many(sampler, "explanation", Category.FACTUAL_KNOWLEDGE, 10)
    _offer_many(sampler, "trivia", Category.FACTUAL_KNOWLEDGE, 10)

    result = sampler.result()
    sources = [prompt.id.split("-")[0] for prompt in result]
    assert sources.count("explanation") == 2
    assert sources.count("trivia") == 2


def test_balanced_sampler_backfills_when_a_source_runs_dry():
    sampler = _BalancedSampler(target=4)
    _offer_many(sampler, "explanation", Category.FACTUAL_KNOWLEDGE, 10)
    _offer_many(sampler, "trivia", Category.FACTUAL_KNOWLEDGE, 1)

    result = sampler.result()
    sources = [prompt.id.split("-")[0] for prompt in result]
    assert len(result) == 4
    assert sources.count("trivia") == 1
    assert sources.count("explanation") == 3


def test_save_and_load_prompts_round_trip(tmp_path):
    prompts = [
        Prompt(id="a", category=Category.CODE_GENERATION, text="write a function"),
        Prompt(id="b", category=Category.MATHEMATICAL_REASONING, text="2+2"),
    ]
    path = tmp_path / "prompts.jsonl"

    save_prompts(prompts, path)

    assert load_prompts_file(path) == prompts
