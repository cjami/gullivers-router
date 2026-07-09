"""Train and evaluate the routing model from cached judge scores and embeddings.

Reads the judged prompts and their embeddings, fits a classifier that predicts when the local
model is likely to miss the quality floor while cloud can rescue the answer, then chooses the
cheapest per-category threshold that satisfies the target pass rate on a calibration split and
reports it on a held-out test split.

The submission gate is a *portfolio* accuracy floor: only the blended pass rate must clear the
gate, so the target pass rate is the gate plus a safety margin, and thresholds are chosen per
category to spend cloud calls where local is weakest. A companion category model lets the runtime
recover each prompt's category from its embedding alone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from gullivers_router.inference.base import DEFAULT_INFERENCE_SEED
from gullivers_router.router.model import CategoryModel, RouterModel, save_bundle
from gullivers_router.training import evaluate, store
from gullivers_router.training.pipeline import Artifacts

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = DEFAULT_INFERENCE_SEED
DEFAULT_QUALITY_FLOOR = 4.0
DEFAULT_ACCURACY_GATE = 0.85
DEFAULT_TARGET_MARGIN = 0.025
_MIN_CATEGORY_CALIBRATION = 20
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


def _per_category_alpha(
    risk: np.ndarray,
    data: _Dataset,
    quality_floor: float,
    target_pass_rate: float,
    global_alpha: float,
) -> dict[str, float]:
    """Pick each category's cheapest threshold reaching the target, else fall back to global.

    Because every kept category clears the target, their weighted blend clears it too; categories
    with too few calibration rows to trust default to the global threshold.
    """
    category_array = np.array(data.categories)
    alphas: dict[str, float] = {}
    for category in sorted(set(data.categories)):
        mask = category_array == category
        if int(np.count_nonzero(mask)) < _MIN_CATEGORY_CALIBRATION:
            alphas[category] = global_alpha
            continue
        curve = evaluate.cost_pass_curve(risk[mask], data.local_scores[mask], data.cloud_scores[mask], quality_floor)
        point = evaluate.cloud_fraction_for_pass_rate(curve, target_pass_rate)
        alphas[category] = point.alpha if point is not None else global_alpha
    return alphas


def _runtime_categories(category_model: CategoryModel | None, data: _Dataset) -> list[str]:
    """Return the categories the deployed router would assign to these rows."""
    if category_model is None:
        return data.categories
    return [str(category) for category in category_model.predict(data.embeddings)]


def _fit_category_model(train: _Dataset) -> CategoryModel | None:
    """Fit the category head, or ``None`` when the data has fewer than two categories."""
    if len(set(train.categories)) < 2:  # noqa: PLR2004 - two classes required to classify
        return None
    return CategoryModel().fit(train.embeddings, np.array(train.categories))


def _deployed_routes(
    risk: np.ndarray,
    data: _Dataset,
    category_model: CategoryModel | None,
    alpha_by_category: dict[str, float],
    global_alpha: float,
) -> np.ndarray:
    """Route with the same predict-category-then-threshold policy the runtime uses."""
    if category_model is None:
        return (risk >= global_alpha).astype(int)
    predicted = category_model.predict(data.embeddings)
    thresholds = np.array([alpha_by_category.get(str(category), global_alpha) for category in predicted])
    return (risk >= thresholds).astype(int)


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


def _operating_metrics(model: RouterModel, val: _Dataset, routes: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    risk = model.predict_proba(val.embeddings)
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


def train_router(  # noqa: PLR0913 - each knob configures a distinct stage of the run
    out: Path,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
    quality_floor: float = DEFAULT_QUALITY_FLOOR,
    accuracy_gate: float = DEFAULT_ACCURACY_GATE,
    target_margin: float = DEFAULT_TARGET_MARGIN,
) -> dict:
    """Fit, select, and persist the router; return the metrics report.

    Thresholds target ``accuracy_gate + target_margin`` on the calibration split so the deployed
    blend clears the gate with headroom against calibration-to-test drift.
    """
    target_pass_rate = min(accuracy_gate + target_margin, 1.0)
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
    category_model = _fit_category_model(train)

    calibration_risk = model.predict_proba(calibration.embeddings)
    calibration_curve = evaluate.cost_pass_curve(
        calibration_risk,
        calibration.local_scores,
        calibration.cloud_scores,
        quality_floor,
    )
    calibration_point = _choose_alpha(calibration_curve, target_pass_rate)
    threshold_calibration = replace(
        calibration,
        categories=_runtime_categories(category_model, calibration),
    )
    alpha_by_category = _per_category_alpha(
        calibration_risk,
        threshold_calibration,
        quality_floor,
        target_pass_rate,
        calibration_point.alpha,
    )

    test_risk = model.predict_proba(test.embeddings)
    test_curve = evaluate.cost_pass_curve(test_risk, test.local_scores, test.cloud_scores, quality_floor)
    global_operating_point = evaluate.operating_point_at_alpha(
        test_risk,
        test.local_scores,
        test.cloud_scores,
        quality_floor,
        calibration_point.alpha,
    )
    deployed_routes = _deployed_routes(test_risk, test, category_model, alpha_by_category, calibration_point.alpha)
    deployed_metrics = evaluate.routed_metrics(deployed_routes, test.local_scores, test.cloud_scores, quality_floor)

    model.save(artifacts.router_model)
    save_bundle(
        artifacts.router_weights,
        risk=model,
        category=category_model,
        alpha_by_category=alpha_by_category,
        global_alpha=calibration_point.alpha,
    )

    metrics = _report(
        _ReportInputs(
            data=data,
            train=train,
            calibration=calibration,
            test=test,
            model=model,
            regularization=regularization,
            calibration_curve=calibration_curve,
            test_curve=test_curve,
            calibration_point=calibration_point,
            global_operating_point=global_operating_point,
            deployed_metrics=deployed_metrics,
            deployed_routes=deployed_routes,
            alpha_by_category=alpha_by_category,
            val_fraction=val_fraction,
            seed=seed,
            quality_floor=quality_floor,
            accuracy_gate=accuracy_gate,
            target_margin=target_margin,
            target_pass_rate=target_pass_rate,
        )
    )
    store.write_json(artifacts.router_metrics, metrics)
    print(
        f"wrote router -> {artifacts.router_weights} "
        f"(global alpha={calibration_point.alpha:.3f}, {len(alpha_by_category)} category thresholds, "
        f"test pass={deployed_metrics['pass_rate']:.3f} cloud={deployed_metrics['cloud_fraction']:.3f})"
    )
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


@dataclass(frozen=True, slots=True)
class _ReportInputs:
    data: _Dataset
    train: _Dataset
    calibration: _Dataset
    test: _Dataset
    model: RouterModel
    regularization: float
    calibration_curve: list[evaluate.OperatingPoint]
    test_curve: list[evaluate.OperatingPoint]
    calibration_point: evaluate.OperatingPoint
    global_operating_point: evaluate.OperatingPoint
    deployed_metrics: dict[str, float]
    deployed_routes: np.ndarray
    alpha_by_category: dict[str, float]
    val_fraction: float
    seed: int
    quality_floor: float
    accuracy_gate: float
    target_margin: float
    target_pass_rate: float


def _operating_point_record(point: evaluate.OperatingPoint) -> dict:
    return {
        "alpha": point.alpha,
        "cloud_fraction": point.cloud_fraction,
        "quality": point.quality,
        "pass_rate": point.pass_rate,
        "violation_rate": point.violation_rate,
        "rescue_rate": point.rescue_rate,
        "unnecessary_cloud_fraction": point.unnecessary_cloud_fraction,
    }


def _curve_records(curve: list[evaluate.OperatingPoint]) -> list[dict]:
    return [
        {
            "alpha": point.alpha,
            "cloud_fraction": point.cloud_fraction,
            "pass_rate": point.pass_rate,
            "violation_rate": point.violation_rate,
        }
        for point in curve
    ]


def _report(inputs: _ReportInputs) -> dict:
    data, train, calibration, test = inputs.data, inputs.train, inputs.calibration, inputs.test
    deployed = inputs.deployed_metrics
    calibration_gap = inputs.calibration_point.pass_rate - deployed["pass_rate"]
    return {
        "n_total": len(data),
        "n_train": len(train),
        "n_val": len(calibration) + len(test),
        "n_calibration": len(calibration),
        "n_test": len(test),
        "val_fraction": inputs.val_fraction,
        "calibration_fraction": len(calibration) / len(data),
        "test_fraction": len(test) / len(data),
        "seed": inputs.seed,
        "quality_floor": inputs.quality_floor,
        "accuracy_gate": inputs.accuracy_gate,
        "target_margin": inputs.target_margin,
        "target_pass_rate": inputs.target_pass_rate,
        "selected_regularization": inputs.regularization,
        "selected_alpha_source": "calibration",
        "threshold_category_source": "predicted",
        "sample_weight": _sample_weight_summary(train.sample_weights),
        "average_pass_rate": evaluate.average_pass_rate(inputs.test_curve),
        "calibration_average_pass_rate": evaluate.average_pass_rate(inputs.calibration_curve),
        "calibration_operating_point": _operating_point_record(inputs.calibration_point),
        "global_operating_point": _operating_point_record(inputs.global_operating_point),
        "per_category_alpha": inputs.alpha_by_category,
        "operating_point": {"policy": "per_category", **deployed},
        "calibration_to_test_gap": calibration_gap,
        "test_clears_gate": deployed["pass_rate"] >= inputs.accuracy_gate,
        "baselines": _baselines(test, inputs.quality_floor),
        "oracle_ceiling": _oracle_ceiling(test, inputs.quality_floor, inputs.target_pass_rate),
        **_operating_metrics(inputs.model, test, inputs.deployed_routes),
        "cost_pass_curve": _curve_records(inputs.test_curve),
        "calibration_cost_pass_curve": _curve_records(inputs.calibration_curve),
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
