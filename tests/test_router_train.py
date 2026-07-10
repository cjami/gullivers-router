import numpy as np
import pytest

from gullivers_router.router.model import CategoryModel, RouterModel
from gullivers_router.training import store
from gullivers_router.training.pipeline import Artifacts
from gullivers_router.training.router import (
    _apply_hard_route_categories,
    _Dataset,
    _floor_distance_sample_weights,
    _load_dataset,
    _needs_cloud,
    _per_category_alpha,
    _runtime_categories,
    train_router,
)


def test_needs_cloud_only_when_local_fails_and_cloud_passes():
    local_scores = np.array([3.0, 3.0, 4.0, 5.0])
    cloud_scores = np.array([4.0, 3.0, 5.0, 3.0])

    needs_cloud = _needs_cloud(local_scores, cloud_scores, quality_floor=4.0)

    assert needs_cloud.tolist() == [1, 0, 0, 0]


def test_floor_distance_sample_weights_downweight_boundary_rows():
    local_scores = np.array([3.9, 2.0, 4.0])
    cloud_scores = np.array([4.1, 5.0, 5.0])

    weights = _floor_distance_sample_weights(local_scores, cloud_scores, quality_floor=4.0)

    assert weights[0] < weights[1]
    assert weights[2] < weights[1]
    assert np.isclose(np.mean(weights), 1.0)


def _write_dataset(root, n_per_category=30):
    artifacts = Artifacts(root)
    rng = np.random.default_rng(0)
    for category in ("math", "chat"):
        cloud_wins = category == "math"
        for i in range(n_per_category):
            prompt_id = f"{category}-{i}"
            centre = 1.0 if cloud_wins else -1.0
            store.append(artifacts.embeddings, {"id": prompt_id, "embedding": (rng.normal(centre, 0.3, 4)).tolist()})
            local_score, cloud_score = (2, 5) if cloud_wins else (5, 5)
            store.append(
                artifacts.labels,
                {
                    "id": prompt_id,
                    "category": category,
                    "local_score": local_score,
                    "cloud_score": cloud_score,
                },
            )
    return artifacts


def test_train_router_writes_artifacts_and_metrics(tmp_path):
    artifacts = _write_dataset(tmp_path)

    metrics = train_router(tmp_path, val_fraction=0.25, seed=0)

    assert artifacts.router_model.exists()
    assert artifacts.router_weights.exists()
    assert store.read_json(artifacts.router_metrics) == metrics
    assert metrics["selected_regularization"] in {0.01, 0.1, 1.0, 10.0, 100.0}
    assert metrics["n_train"] + metrics["n_calibration"] + metrics["n_test"] == metrics["n_total"]
    assert metrics["n_val"] == metrics["n_calibration"] + metrics["n_test"]
    assert metrics["quality_floor"] == 4.0
    assert metrics["accuracy_gate"] == 0.91
    assert metrics["target_margin"] == 0.025
    assert metrics["target_pass_rate"] == pytest.approx(0.935)
    assert metrics["selected_alpha_source"] == "calibration"
    assert metrics["threshold_category_source"] == "predicted"
    assert metrics["global_operating_point"]["alpha"] == metrics["calibration_operating_point"]["alpha"]
    assert metrics["operating_point"]["policy"] == "per_category"
    assert set(metrics["per_category_alpha"]) == {"math", "chat"}
    assert isinstance(metrics["test_clears_gate"], bool)
    assert "oracle_ceiling" in metrics
    assert "max_achievable_pass_rate" in metrics["oracle_ceiling"]
    assert "sample_weight" in metrics


def test_train_router_separates_easy_categories(tmp_path):
    _write_dataset(tmp_path)

    metrics = train_router(tmp_path, val_fraction=0.25, seed=0)

    rates = metrics["per_category_cloud_rate"]
    assert rates["math"] > rates["chat"]


def _dataset(categories, local_scores, cloud_scores):
    count = len(categories)
    return _Dataset(
        embeddings=np.zeros((count, 4)),
        needs_cloud=np.zeros(count, int),
        local_scores=np.array(local_scores, float),
        cloud_scores=np.array(cloud_scores, float),
        sample_weights=np.ones(count),
        categories=list(categories),
    )


def test_per_category_alpha_keeps_reliable_category_local():
    size = 30
    data = _dataset(["easy"] * size + ["hard"] * size, [5.0] * size + [2.0] * size, [5.0] * (2 * size))
    risk = np.array([0.1] * size + [0.9] * size)

    alphas = _per_category_alpha(
        risk,
        data,
        quality_floor=3.0,
        target_pass_rate=0.83,
        global_alpha=0.5,
    )

    assert alphas["easy"] > alphas["hard"]


def test_per_category_alpha_falls_back_when_too_few_rows():
    data = _dataset(["rare"] * 5, [2.0] * 5, [5.0] * 5)
    risk = np.full(5, 0.9)

    alphas = _per_category_alpha(
        risk,
        data,
        quality_floor=3.0,
        target_pass_rate=0.83,
        global_alpha=0.42,
    )

    assert alphas == {"rare": 0.42}


def test_apply_hard_route_categories_forces_selected_thresholds():
    alphas = _apply_hard_route_categories(
        {"code_debugging": 0.7, "code_generation": 0.6, "math": 0.5, "named_entity_recognition": 0.4},
        ["code_debugging", "code_generation", "math", "named_entity_recognition"],
    )

    assert alphas == {
        "code_debugging": 0.0,
        "code_generation": 0.0,
        "math": 0.5,
        "named_entity_recognition": 0.4,
    }


def test_runtime_categories_use_category_model_predictions(monkeypatch):
    data = _dataset(["true-a", "true-b"], [5.0, 5.0], [5.0, 5.0])
    model = CategoryModel.__new__(CategoryModel)

    def predict(embeddings):
        assert embeddings is data.embeddings
        return np.array(["predicted-b", "predicted-a"])

    monkeypatch.setattr(model, "predict", predict)

    assert _runtime_categories(model, data) == ["predicted-b", "predicted-a"]
    assert _runtime_categories(None, data) == data.categories


def test_train_router_exports_category_head(tmp_path):
    artifacts = _write_dataset(tmp_path)
    train_router(tmp_path, val_fraction=0.25, seed=0)

    weights = dict(np.load(artifacts.router_weights))

    assert set(weights["cat_classes"]) == {"math", "chat"}
    assert weights["cat_alpha"].shape == weights["cat_classes"].shape
    assert weights["cat_weights"].shape[0] == len(weights["cat_classes"])


def test_exported_weights_are_loadable(tmp_path):
    artifacts = _write_dataset(tmp_path)
    train_router(tmp_path, val_fraction=0.25, seed=0)

    weights = RouterModel.load(artifacts.router_model)

    assert weights.predict_proba(np.ones((1, 4))).shape == (1,)


def test_load_dataset_uses_latest_label_for_duplicate_id(tmp_path):
    artifacts = Artifacts(tmp_path)
    store.append(artifacts.embeddings, {"id": "a", "embedding": [0.0, 1.0]})
    store.append(artifacts.labels, {"id": "a", "category": "code", "local_score": 1, "cloud_score": 1})
    store.append(artifacts.labels, {"id": "a", "category": "code", "local_score": 3, "cloud_score": 5})

    data = _load_dataset(artifacts, quality_floor=4.0)

    assert len(data) == 1
    assert data.local_scores.tolist() == [3.0]
    assert data.cloud_scores.tolist() == [5.0]
