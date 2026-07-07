import numpy as np

from gullivers_router.router.model import RouterModel
from gullivers_router.training import store
from gullivers_router.training.pipeline import Artifacts
from gullivers_router.training.router import train_router


def _write_dataset(root, n_per_category=30):
    artifacts = Artifacts(root)
    rng = np.random.default_rng(0)
    for category in ("math", "chat"):
        cloud_wins = category == "math"
        for i in range(n_per_category):
            prompt_id = f"{category}-{i}"
            centre = 1.0 if cloud_wins else -1.0
            store.append(artifacts.embeddings, {"id": prompt_id, "embedding": (rng.normal(centre, 0.3, 4)).tolist()})
            local_score, cloud_score = (3, 9) if cloud_wins else (7, 7)
            store.append(
                artifacts.labels,
                {
                    "id": prompt_id,
                    "category": category,
                    "local_score": local_score,
                    "cloud_score": cloud_score,
                    "label": 1 if cloud_wins else 0,
                },
            )
    return artifacts


def test_train_router_writes_artifacts_and_metrics(tmp_path):
    artifacts = _write_dataset(tmp_path)

    metrics = train_router(tmp_path, val_fraction=0.25, seed=0)

    assert artifacts.router_model.exists()
    assert artifacts.router_weights.exists()
    assert store.read_json(artifacts.router_metrics) == metrics
    assert metrics["selected_variant"] in {"hard", "soft"}
    assert metrics["n_train"] + metrics["n_val"] == metrics["n_total"]


def test_train_router_separates_easy_categories(tmp_path):
    _write_dataset(tmp_path)

    metrics = train_router(tmp_path, val_fraction=0.25, seed=0)

    rates = metrics["per_category_cloud_rate"]
    assert rates["math"] > rates["chat"]


def test_exported_weights_are_loadable(tmp_path):
    artifacts = _write_dataset(tmp_path)
    train_router(tmp_path, val_fraction=0.25, seed=0)

    weights = RouterModel.load(artifacts.router_model)

    assert weights.predict_proba(np.ones((1, 4))).shape == (1,)
