from gullivers_router.training import store
from gullivers_router.training.dataset import Category, Prompt
from gullivers_router.training.judge import Judgement
from gullivers_router.training.labels import build_labels


def test_build_labels_writes_rows_and_skips_unscored(tmp_path):
    prompts = [
        Prompt(id="a", category=Category.CODE_GENERATION, text="x"),
        Prompt(id="b", category=Category.MATHEMATICAL_REASONING, text="y"),
    ]
    judgements = [
        Judgement(
            id="a",
            local_score=3,
            cloud_score=8,
            local_rationale="missed detail",
            cloud_rationale="complete",
            local_quality="adequate",
            cloud_quality="excellent",
            preferred_source="cloud",
        ),
        Judgement(id="b", local_score=None, cloud_score=None),
    ]
    out = tmp_path / "labels.jsonl"

    build_labels(prompts, judgements, out)

    rows = list(store.read_records(out))
    assert rows == [
        {
            "id": "a",
            "category": "code_generation",
            "local_score": 3,
            "cloud_score": 8,
            "local_rationale": "missed detail",
            "cloud_rationale": "complete",
            "local_quality": "adequate",
            "cloud_quality": "excellent",
            "preferred_source": "cloud",
        }
    ]


def test_build_labels_is_resumable(tmp_path):
    prompts = [Prompt(id="a", category=Category.CODE_GENERATION, text="x")]
    judgements = [Judgement(id="a", local_score=3, cloud_score=8)]
    out = tmp_path / "labels.jsonl"

    build_labels(prompts, judgements, out)
    build_labels(prompts, judgements, out)

    assert len(list(store.read_records(out))) == 1
