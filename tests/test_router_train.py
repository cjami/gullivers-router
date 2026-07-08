import numpy as np

from gullivers_router.router.model import RouterModel
from gullivers_router.training import store
from gullivers_router.training.pipeline import Artifacts
from gullivers_router.training.router import _floor_distance_sample_weights, _load_dataset, _needs_cloud, train_router


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
            local_score, cloud_score = (3, 5) if cloud_wins else (5, 5)
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
    assert metrics["target_pass_rate"] == 0.98
    assert metrics["selected_alpha_source"] == "calibration"
    assert metrics["operating_point"]["alpha"] == metrics["calibration_operating_point"]["alpha"]
    assert "oracle_ceiling" in metrics
    assert "max_achievable_pass_rate" in metrics["oracle_ceiling"]
    assert "sample_weight" in metrics


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


def test_load_dataset_uses_latest_label_for_duplicate_id(tmp_path):
    artifacts = Artifacts(tmp_path)
    store.append(artifacts.embeddings, {"id": "a", "embedding": [0.0, 1.0]})
    store.append(artifacts.labels, {"id": "a", "category": "code", "local_score": 1, "cloud_score": 1})
    store.append(artifacts.labels, {"id": "a", "category": "code", "local_score": 3, "cloud_score": 5})

    data = _load_dataset(artifacts, quality_floor=4.0)

    assert len(data) == 1
    assert data.local_scores.tolist() == [3.0]
    assert data.cloud_scores.tolist() == [5.0]
