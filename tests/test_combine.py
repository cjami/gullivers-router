from gullivers_router.training.combine import generate_pairwise
from gullivers_router.training.dataset import Category, Prompt


class FakeBatchModel:
    def __init__(self, prefix):
        self.prefix = prefix

    def complete_batch(self, requests):
        return [f"{self.prefix}:{request[0].content}" for request in requests]


def test_generate_pairwise_aligns_and_preserves_prompt():
    prompts = [
        Prompt(id="a", category=Category.MATHEMATICAL_REASONING, text="2+2"),
        Prompt(id="b", category=Category.CODE_GENERATION, text="write fn"),
    ]

    pairs = generate_pairwise(prompts, FakeBatchModel("local"), FakeBatchModel("cloud"))

    assert [pair.prompt.id for pair in pairs] == ["a", "b"]
    assert pairs[0].local_response == "local:2+2"
    assert pairs[0].cloud_response == "cloud:2+2"
    assert pairs[1].local_response == "local:write fn"
    assert pairs[0].prompt.category == Category.MATHEMATICAL_REASONING
