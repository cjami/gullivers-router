from gullivers_router.training.combine import align_pairs
from gullivers_router.training.dataset import Category, Prompt


def _prompt(prompt_id):
    return Prompt(id=prompt_id, category=Category.MATHEMATICAL_REASONING, text=prompt_id)


def test_align_pairs_joins_by_id_regardless_of_order():
    prompts = [_prompt("a"), _prompt("b")]
    local = {"b": "local-b", "a": "local-a"}
    cloud = {"a": "cloud-a", "b": "cloud-b"}

    pairs = align_pairs(prompts, local, cloud)

    assert [pair.prompt.id for pair in pairs] == ["a", "b"]
    assert pairs[0].local_response == "local-a"
    assert pairs[0].cloud_response == "cloud-a"


def test_align_pairs_drops_prompts_missing_from_either_side():
    prompts = [_prompt("a"), _prompt("b"), _prompt("c")]
    local = {"a": "local-a", "b": "local-b"}
    cloud = {"a": "cloud-a", "c": "cloud-c"}

    pairs = align_pairs(prompts, local, cloud)

    assert [pair.prompt.id for pair in pairs] == ["a"]
