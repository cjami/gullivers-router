from gullivers_router.training import store
from gullivers_router.training.dataset import Category, Prompt
from gullivers_router.training.embed import run_embed


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [float(len(text)), 1.0]


def _prompts(*ids):
    return [Prompt(id=i, category=Category.CODE_GENERATION, text=f"q-{i}") for i in ids]


def test_run_embed_resumes_past_completed_ids(tmp_path):
    out = tmp_path / "embeddings.jsonl"
    store.append(out, {"id": "a", "embedding": [0.0, 0.0]})
    model = FakeEmbedder()

    run_embed(_prompts("a", "b"), model, out)

    assert model.calls == ["q-b"]
    assert store.completed_ids(out) == {"a", "b"}


def test_run_embed_writes_id_and_embedding(tmp_path):
    out = tmp_path / "embeddings.jsonl"

    run_embed(_prompts("a"), FakeEmbedder(), out)

    (record,) = store.read_records(out)
    assert record["id"] == "a"
    assert record["embedding"] == [3.0, 1.0]
