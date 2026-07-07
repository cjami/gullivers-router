from gullivers_router.training import store
from gullivers_router.training.dataset import Category, Prompt
from gullivers_router.training.judge import Judgement
from gullivers_router.training.labels import CLOUD_LABEL, LOCAL_LABEL, build_labels, label


def test_label_routes_to_cloud_only_past_the_margin():
    assert label(5, 7, margin=2) == CLOUD_LABEL
    assert label(5, 6, margin=2) == LOCAL_LABEL
    assert label(9, 3, margin=2) == LOCAL_LABEL


def test_build_labels_writes_rows_and_skips_unscored(tmp_path):
    prompts = [
        Prompt(id="a", category=Category.CODE_GENERATION, text="x"),
        Prompt(id="b", category=Category.MATHEMATICAL_REASONING, text="y"),
    ]
    judgements = [
        Judgement(id="a", local_score=3, cloud_score=8),
        Judgement(id="b", local_score=None, cloud_score=None),
    ]
    out = tmp_path / "labels.jsonl"

    build_labels(prompts, judgements, out, margin=2)

    rows = list(store.read_records(out))
    assert rows == [{"id": "a", "category": "code_generation", "local_score": 3, "cloud_score": 8, "label": 1}]


def test_build_labels_is_resumable(tmp_path):
    prompts = [Prompt(id="a", category=Category.CODE_GENERATION, text="x")]
    judgements = [Judgement(id="a", local_score=3, cloud_score=8)]
    out = tmp_path / "labels.jsonl"

    build_labels(prompts, judgements, out)
    build_labels(prompts, judgements, out)

    assert len(list(store.read_records(out))) == 1
