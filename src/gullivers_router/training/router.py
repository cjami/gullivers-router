"""Train and evaluate the routing model from cached judge scores and embeddings.

Reads the judged prompts and their embeddings, fits a classifier that predicts when the local
model is likely to miss the quality floor while cloud can rescue the answer, then chooses the
cheapest threshold that satisfies the target pass rate on a calibration split and reports it on
a held-out test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gullivers_router.inference.base import DEFAULT_INFERENCE_SEED
from gullivers_router.router.model import RouterModel
from gullivers_router.training import evaluate, store
from gullivers_router.training.pipeline import Artifacts

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = DEFAULT_INFERENCE_SEED
DEFAULT_QUALITY_FLOOR = 4.0
DEFAULT_TARGET_PASS_RATE = 0.98
_REGULARIZATION_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
_CV_SPLITS = 5


@dataclass(frozen=True, slots=True)
class _Dataset:
    embeddings: np.ndarray
    needs_cloud: np.ndarray
    local_scores: np.ndarray
    cloud_scores: np.ndarray
    sample_weights: np.ndarray
    categories: list[str]

    def __len__(self) -> int:
        return len(self.needs_cloud)


def _load_dataset(artifacts: Artifacts, quality_floor: float) -> _Dataset:
    """Join judge scores to embeddings by id, keeping only prompts present in both."""
    embeddings = store.read_map(artifacts.embeddings, value="embedding")
    labels_by_id = {row["id"]: row for row in store.read_records(artifacts.labels)}
    rows = [row for item_id, row in labels_by_id.items() if item_id in embeddings]
    if not rows:
        msg = "no rows with both judge scores and an embedding; run the label and embed stages first"
        raise ValueError(msg)
    local_scores = np.array([row["local_score"] for row in rows], dtype=np.float64)
    cloud_scores = np.array([row["cloud_score"] for row in rows], dtype=np.float64)
    return _Dataset(
        embeddings=np.array([embeddings[row["id"]] for row in rows], dtype=np.float64),
        needs_cloud=_needs_cloud(local_scores, cloud_scores, quality_floor),
        local_scores=local_scores,
        cloud_scores=cloud_scores,
        sample_weights=_floor_distance_sample_weights(local_scores, cloud_scores, quality_floor),
        categories=[row["category"] for row in rows],
    )


def _needs_cloud(local_scores: np.ndarray, cloud_scores: np.ndarray, quality_floor: float) -> np.ndarray:
    return ((local_scores < quality_floor) & (cloud_scores >= quality_floor)).astype(int)


def _floor_distance_sample_weights(
    local_scores: np.ndarray,
    cloud_scores: np.ndarray,
    quality_floor: float,
) -> np.ndarray:
    boundary_distance = np.minimum(np.abs(local_scores - quality_floor), np.abs(cloud_scores - quality_floor))
    max_distance = float(np.max(boundary_distance)) if len(boundary_distance) else 0.0
    if max_distance == 0.0:
        return np.ones_like(boundary_distance, dtype=np.float64)
    weights = 0.25 + (0.75 * boundary_distance / max_distance)
    return weights / np.mean(weights)


def _best_regularization(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, seed: int) -> float:
    """Pick regularization by cross-validated AUC, defaulting when infeasible."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    if len(np.unique(y)) < 2:  # noqa: PLR2004 - two classes required
        return 1.0
    splits = min(_CV_SPLITS, int(np.min(np.bincount(y))))
    if splits < 2:  # noqa: PLR2004 - two folds minimum
        return 1.0
    folds = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    scored: dict[float, float] = {}
    for regularization in _REGULARIZATION_GRID:
        aucs = []
        for train_idx, test_idx in folds.split(x, y):
            model = RouterModel(regularization=regularization).fit(
                x[train_idx],
                y[train_idx],
                sample_weight[train_idx],
            )
            aucs.append(roc_auc_score(y[test_idx], model.predict_proba(x[test_idx])))
        scored[regularization] = float(np.mean(aucs))
    return max(scored, key=lambda candidate: scored[candidate])


def _choose_alpha(curve: list[evaluate.OperatingPoint], target_pass_rate: float) -> evaluate.OperatingPoint:
    """Return the cheapest point reaching the target pass rate, else the best pass rate."""
    reached = evaluate.cloud_fraction_for_pass_rate(curve, target_pass_rate)
    if reached is not None:
        return reached
    best_pass_rate = max(point.pass_rate for point in curve)
    best_points = [point for point in curve if point.pass_rate == best_pass_rate]
    return min(best_points, key=lambda point: point.cloud_fraction)


def _baselines(data: _Dataset, quality_floor: float) -> dict[str, float]:
    all_local = float(np.mean(data.local_scores))
    all_cloud = float(np.mean(data.cloud_scores))
    oracle = float(np.mean(np.maximum(data.local_scores, data.cloud_scores)))
    oracle_routes = (data.local_scores < quality_floor) & (data.cloud_scores >= quality_floor)
    both_fail_rate = float(np.mean((data.local_scores < quality_floor) & (data.cloud_scores < quality_floor)))
    oracle_pass_rate = 1.0 - both_fail_rate
    return {
        "all_local_quality": all_local,
        "all_cloud_quality": all_cloud,
        "oracle_quality": oracle,
        "all_local_pass_rate": float(np.mean(data.local_scores >= quality_floor)),
        "all_cloud_pass_rate": float(np.mean(data.cloud_scores >= quality_floor)),
        "oracle_pass_rate": oracle_pass_rate,
        "max_achievable_pass_rate": oracle_pass_rate,
        "oracle_cloud_fraction": float(np.mean(oracle_routes)),
        "both_fail_rate": both_fail_rate,
    }


def _operating_metrics(model: RouterModel, val: _Dataset, alpha: float) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    risk = model.predict_proba(val.embeddings)
    routes = (risk >= alpha).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        val.needs_cloud,
        routes,
        average="binary",
        zero_division=0,
    )
    return {
        "needs_cloud_rate": float(np.mean(val.needs_cloud)),
        "needs_cloud_auc": float(roc_auc_score(val.needs_cloud, risk)) if len(np.unique(val.needs_cloud)) > 1 else None,
        "needs_cloud_accuracy": float(accuracy_score(val.needs_cloud, routes)),
        "needs_cloud_precision": float(precision),
        "needs_cloud_recall": float(recall),
        "needs_cloud_f1": float(f1),
        "per_category_cloud_rate": evaluate.per_category_cloud_rate(routes, val.categories),
    }


def train_router(
    out: Path,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
    quality_floor: float = DEFAULT_QUALITY_FLOOR,
    target_pass_rate: float = DEFAULT_TARGET_PASS_RATE,
) -> dict:
    """Fit, select, and persist the router; return the metrics report."""
    artifacts = Artifacts(out)
    data = _load_dataset(artifacts, quality_floor)
    indices = np.arange(len(data))
    train_idx, holdout_idx = _split_indices(data, indices, test_size=val_fraction, seed=seed)
    holdout = _subset(data, holdout_idx)
    holdout_indices = np.arange(len(holdout))
    calibration_idx, test_idx = _split_indices(holdout, holdout_indices, test_size=0.5, seed=seed)
    train, calibration, test = _subset(data, train_idx), _subset(holdout, calibration_idx), _subset(holdout, test_idx)

    regularization = _best_regularization(train.embeddings, train.needs_cloud, train.sample_weights, seed)
    model = RouterModel(regularization=regularization).fit(
        train.embeddings,
        train.needs_cloud,
        train.sample_weights,
    )
    calibration_risk = model.predict_proba(calibration.embeddings)
    calibration_curve = evaluate.cost_pass_curve(
        calibration_risk,
        calibration.local_scores,
        calibration.cloud_scores,
        quality_floor,
    )
    calibration_point = _choose_alpha(calibration_curve, target_pass_rate)
    test_risk = model.predict_proba(test.embeddings)
    test_curve = evaluate.cost_pass_curve(test_risk, test.local_scores, test.cloud_scores, quality_floor)
    operating_point = evaluate.operating_point_at_alpha(
        test_risk,
        test.local_scores,
        test.cloud_scores,
        quality_floor,
        calibration_point.alpha,
    )

    model.save(artifacts.router_model)
    model.to_numpy(artifacts.router_weights, alpha=calibration_point.alpha)

    metrics = _report(
        data,
        train,
        calibration,
        test,
        model,
        regularization,
        calibration_curve,
        test_curve,
        calibration_point,
        operating_point,
        val_fraction,
        seed,
        quality_floor,
        target_pass_rate,
    )
    store.write_json(artifacts.router_metrics, metrics)
    print(f"wrote router -> {artifacts.router_weights} (alpha={operating_point.alpha:.3f})")
    return metrics


def _subset(data: _Dataset, idx: np.ndarray) -> _Dataset:
    return _Dataset(
        embeddings=data.embeddings[idx],
        needs_cloud=data.needs_cloud[idx],
        local_scores=data.local_scores[idx],
        cloud_scores=data.cloud_scores[idx],
        sample_weights=data.sample_weights[idx],
        categories=[data.categories[i] for i in idx],
    )


def _split_indices(
    data: _Dataset,
    indices: np.ndarray,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    for stratify in _stratification_candidates(data):
        try:
            return train_test_split(indices, test_size=test_size, random_state=seed, stratify=stratify)
        except ValueError:
            continue
    return train_test_split(indices, test_size=test_size, random_state=seed)


def _stratification_candidates(data: _Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    categories = np.array(data.categories)
    labels = data.needs_cloud.astype(str)
    combined = np.array([f"{category}:{label}" for category, label in zip(categories, labels, strict=True)])
    return combined, labels, categories


def _report(  # noqa: PLR0913 - a report gathers every evaluation input
    data: _Dataset,
    train: _Dataset,
    calibration: _Dataset,
    test: _Dataset,
    model: RouterModel,
    regularization: float,
    calibration_curve: list[evaluate.OperatingPoint],
    test_curve: list[evaluate.OperatingPoint],
    calibration_point: evaluate.OperatingPoint,
    operating_point: evaluate.OperatingPoint,
    val_fraction: float,
    seed: int,
    quality_floor: float,
    target_pass_rate: float,
) -> dict:
    return {
        "n_total": len(data),
        "n_train": len(train),
        "n_val": len(calibration) + len(test),
        "n_calibration": len(calibration),
        "n_test": len(test),
        "val_fraction": val_fraction,
        "calibration_fraction": len(calibration) / len(data),
        "test_fraction": len(test) / len(data),
        "seed": seed,
        "quality_floor": quality_floor,
        "target_pass_rate": target_pass_rate,
        "selected_regularization": regularization,
        "selected_alpha_source": "calibration",
        "sample_weight": _sample_weight_summary(train.sample_weights),
        "average_pass_rate": evaluate.average_pass_rate(test_curve),
        "calibration_average_pass_rate": evaluate.average_pass_rate(calibration_curve),
        "calibration_operating_point": {
            "alpha": calibration_point.alpha,
            "cloud_fraction": calibration_point.cloud_fraction,
            "quality": calibration_point.quality,
            "pass_rate": calibration_point.pass_rate,
            "violation_rate": calibration_point.violation_rate,
            "rescue_rate": calibration_point.rescue_rate,
            "unnecessary_cloud_fraction": calibration_point.unnecessary_cloud_fraction,
        },
        "operating_point": {
            "alpha": operating_point.alpha,
            "cloud_fraction": operating_point.cloud_fraction,
            "quality": operating_point.quality,
            "pass_rate": operating_point.pass_rate,
            "violation_rate": operating_point.violation_rate,
            "rescue_rate": operating_point.rescue_rate,
            "unnecessary_cloud_fraction": operating_point.unnecessary_cloud_fraction,
        },
        "baselines": _baselines(test, quality_floor),
        "oracle_ceiling": _oracle_ceiling(test, quality_floor, target_pass_rate),
        **_operating_metrics(model, test, operating_point.alpha),
        "cost_pass_curve": [
            {
                "alpha": point.alpha,
                "cloud_fraction": point.cloud_fraction,
                "pass_rate": point.pass_rate,
                "violation_rate": point.violation_rate,
            }
            for point in test_curve
        ],
        "calibration_cost_pass_curve": [
            {
                "alpha": point.alpha,
                "cloud_fraction": point.cloud_fraction,
                "pass_rate": point.pass_rate,
                "violation_rate": point.violation_rate,
            }
            for point in calibration_curve
        ],
    }


def _sample_weight_summary(weights: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(weights)),
        "mean": float(np.mean(weights)),
        "max": float(np.max(weights)),
    }


def _oracle_ceiling(data: _Dataset, quality_floor: float, target_pass_rate: float) -> dict[str, float | bool]:
    both_fail_rate = float(np.mean((data.local_scores < quality_floor) & (data.cloud_scores < quality_floor)))
    max_pass_rate = 1.0 - both_fail_rate
    return {
        "max_achievable_pass_rate": max_pass_rate,
        "both_fail_rate": both_fail_rate,
        "target_pass_rate": target_pass_rate,
        "target_reachable": max_pass_rate >= target_pass_rate,
        "target_margin": max_pass_rate - target_pass_rate,
    }
