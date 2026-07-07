"""Train and evaluate the routing model from cached judge scores and embeddings.

Reads the judged prompts and their embeddings, fits a classifier that predicts when the local
model is likely to miss the quality floor while cloud can rescue the answer, then chooses the
cheapest threshold that satisfies the target pass rate on a held-out split.
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
    categories: list[str]

    def __len__(self) -> int:
        return len(self.needs_cloud)


def _load_dataset(artifacts: Artifacts, quality_floor: float) -> _Dataset:
    """Join judge scores to embeddings by id, keeping only prompts present in both."""
    embeddings = store.read_map(artifacts.embeddings, value="embedding")
    rows = [row for row in store.read_records(artifacts.labels) if row["id"] in embeddings]
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
        categories=[row["category"] for row in rows],
    )


def _needs_cloud(local_scores: np.ndarray, cloud_scores: np.ndarray, quality_floor: float) -> np.ndarray:
    return ((local_scores < quality_floor) & (cloud_scores >= quality_floor)).astype(int)


def _best_regularization(x: np.ndarray, y: np.ndarray, seed: int) -> float:
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
            model = RouterModel(regularization=regularization).fit(x[train_idx], y[train_idx])
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
    return {
        "all_local_quality": all_local,
        "all_cloud_quality": all_cloud,
        "oracle_quality": oracle,
        "all_local_pass_rate": float(np.mean(data.local_scores >= quality_floor)),
        "all_cloud_pass_rate": float(np.mean(data.cloud_scores >= quality_floor)),
        "oracle_pass_rate": float(np.mean(np.maximum(data.local_scores, data.cloud_scores) >= quality_floor)),
        "oracle_cloud_fraction": float(np.mean(oracle_routes)),
        "both_fail_rate": float(np.mean((data.local_scores < quality_floor) & (data.cloud_scores < quality_floor))),
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
    from sklearn.model_selection import train_test_split

    artifacts = Artifacts(out)
    data = _load_dataset(artifacts, quality_floor)
    indices = np.arange(len(data))
    train_idx, val_idx = train_test_split(indices, test_size=val_fraction, random_state=seed, stratify=data.categories)
    train, val = _subset(data, train_idx), _subset(data, val_idx)

    regularization = _best_regularization(train.embeddings, train.needs_cloud, seed)
    model = RouterModel(regularization=regularization).fit(train.embeddings, train.needs_cloud)
    risk = model.predict_proba(val.embeddings)
    curve = evaluate.cost_pass_curve(risk, val.local_scores, val.cloud_scores, quality_floor)
    operating_point = _choose_alpha(curve, target_pass_rate)

    model.save(artifacts.router_model)
    model.to_numpy(artifacts.router_weights, alpha=operating_point.alpha)

    metrics = _report(
        data,
        train,
        val,
        model,
        regularization,
        curve,
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
        categories=[data.categories[i] for i in idx],
    )


def _report(  # noqa: PLR0913 - a report gathers every evaluation input
    data: _Dataset,
    train: _Dataset,
    val: _Dataset,
    model: RouterModel,
    regularization: float,
    curve: list[evaluate.OperatingPoint],
    operating_point: evaluate.OperatingPoint,
    val_fraction: float,
    seed: int,
    quality_floor: float,
    target_pass_rate: float,
) -> dict:
    return {
        "n_total": len(data),
        "n_train": len(train),
        "n_val": len(val),
        "val_fraction": val_fraction,
        "seed": seed,
        "quality_floor": quality_floor,
        "target_pass_rate": target_pass_rate,
        "selected_regularization": regularization,
        "average_pass_rate": evaluate.average_pass_rate(curve),
        "operating_point": {
            "alpha": operating_point.alpha,
            "cloud_fraction": operating_point.cloud_fraction,
            "quality": operating_point.quality,
            "pass_rate": operating_point.pass_rate,
            "violation_rate": operating_point.violation_rate,
            "rescue_rate": operating_point.rescue_rate,
            "unnecessary_cloud_fraction": operating_point.unnecessary_cloud_fraction,
        },
        "baselines": _baselines(val, quality_floor),
        **_operating_metrics(model, val, operating_point.alpha),
        "cost_pass_curve": [
            {
                "alpha": point.alpha,
                "cloud_fraction": point.cloud_fraction,
                "pass_rate": point.pass_rate,
                "violation_rate": point.violation_rate,
            }
            for point in curve
        ],
    }
