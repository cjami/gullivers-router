"""Train and evaluate the routing model from cached labels and embeddings.

Reads the labelled prompts and their embeddings, fits a hard-label and a margin-weighted logistic
router, and keeps whichever recovers more of the quality gap on a held-out, category-stratified
split. The winner is refit on all data and written three ways: the sklearn pipeline for later
inspection, a numpy ``.npz`` for the runtime, and a metrics report describing the chosen operating
point and how it was reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gullivers_router.router.model import RouterModel
from gullivers_router.training import evaluate, store
from gullivers_router.training.pipeline import Artifacts

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = 0
DEFAULT_TARGET_RECOVERY = 0.9
_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
_CV_SPLITS = 5


@dataclass(frozen=True, slots=True)
class _Dataset:
    embeddings: np.ndarray
    labels: np.ndarray
    local_scores: np.ndarray
    cloud_scores: np.ndarray
    categories: list[str]

    def __len__(self) -> int:
        return len(self.labels)


def _load_dataset(artifacts: Artifacts) -> _Dataset:
    """Join labels to embeddings by id, keeping only prompts present in both."""
    embeddings = store.read_map(artifacts.embeddings, value="embedding")
    rows = [row for row in store.read_records(artifacts.labels) if row["id"] in embeddings]
    if not rows:
        msg = "no rows with both a label and an embedding; run the label and embed stages first"
        raise ValueError(msg)
    return _Dataset(
        embeddings=np.array([embeddings[row["id"]] for row in rows], dtype=np.float64),
        labels=np.array([row["label"] for row in rows], dtype=int),
        local_scores=np.array([row["local_score"] for row in rows], dtype=np.float64),
        cloud_scores=np.array([row["cloud_score"] for row in rows], dtype=np.float64),
        categories=[row["category"] for row in rows],
    )


def _sample_weights(data: _Dataset, *, soft: bool) -> np.ndarray | None:
    """Confidence weights for the soft variant (larger score gap -> more trusted).

    A unit floor keeps every prompt in play, so a category with no score gap can never
    zero out a whole class and stall the fit.
    """
    if not soft:
        return None
    return 1.0 + np.abs(data.cloud_scores - data.local_scores)


def _best_c(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None, seed: int) -> float:
    """Pick the regularization strength by cross-validated AUC, defaulting when infeasible."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    splits = min(_CV_SPLITS, int(np.min(np.bincount(y))))
    if splits < 2 or len(np.unique(y)) < 2:  # noqa: PLR2004 - two classes / two folds minimum
        return 1.0
    folds = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    scored: dict[float, float] = {}
    for c in _C_GRID:
        aucs = []
        for train_idx, test_idx in folds.split(x, y):
            weight = None if sample_weight is None else sample_weight[train_idx]
            model = RouterModel(c=c).fit(x[train_idx], y[train_idx], weight)
            aucs.append(roc_auc_score(y[test_idx], model.predict_proba(x[test_idx])))
        scored[c] = float(np.mean(aucs))
    return max(scored, key=lambda candidate: scored[candidate])


@dataclass(frozen=True, slots=True)
class _Variant:
    name: str
    model: RouterModel
    c: float
    curve: list[evaluate.OperatingPoint]
    apgr: float


def _fit_variant(name: str, *, soft: bool, train: _Dataset, val: _Dataset, seed: int) -> _Variant:
    """Tune, fit on train, and score a variant's cost-quality curve on validation."""
    weight = _sample_weights(train, soft=soft)
    c = _best_c(train.embeddings, train.labels, weight, seed)
    model = RouterModel(c=c).fit(train.embeddings, train.labels, weight)
    curve = evaluate.cost_quality_curve(model.predict_proba(val.embeddings), val.local_scores, val.cloud_scores)
    return _Variant(name=name, model=model, c=c, curve=curve, apgr=evaluate.average_gap_recovered(curve))


def _choose_alpha(curve: list[evaluate.OperatingPoint], target_recovery: float) -> evaluate.OperatingPoint:
    """Return the cheapest point reaching the recovery target, else the one recovering the most."""
    reached = evaluate.cloud_fraction_for_recovery(curve, target_recovery)
    return reached if reached is not None else max(curve, key=lambda point: point.gap_recovered)


def _baselines(data: _Dataset) -> dict[str, float]:
    all_local = float(np.mean(data.local_scores))
    all_cloud = float(np.mean(data.cloud_scores))
    return {
        "all_local_quality": all_local,
        "all_cloud_quality": all_cloud,
        "random_quality": 0.5 * (all_local + all_cloud),
        "random_gap_recovered": 0.5,
    }


def _operating_metrics(model: RouterModel, val: _Dataset, alpha: float) -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score

    proba = model.predict_proba(val.embeddings)
    routes = (proba >= alpha).astype(int)
    return {
        "roc_auc": float(roc_auc_score(val.labels, proba)) if len(np.unique(val.labels)) > 1 else None,
        "accuracy": float(accuracy_score(val.labels, routes)),
        "per_category_cloud_rate": evaluate.per_category_cloud_rate(routes, val.categories),
    }


def train_router(
    out: Path,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
    target_recovery: float = DEFAULT_TARGET_RECOVERY,
) -> dict:
    """Fit, select, and persist the router; return the metrics report."""
    from sklearn.model_selection import train_test_split

    artifacts = Artifacts(out)
    data = _load_dataset(artifacts)
    indices = np.arange(len(data))
    train_idx, val_idx = train_test_split(indices, test_size=val_fraction, random_state=seed, stratify=data.categories)
    train, val = _subset(data, train_idx), _subset(data, val_idx)

    variants = [
        _fit_variant("hard", soft=False, train=train, val=val, seed=seed),
        _fit_variant("soft", soft=True, train=train, val=val, seed=seed),
    ]
    best = max(variants, key=lambda variant: variant.apgr)
    operating_point = _choose_alpha(best.curve, target_recovery)

    final = RouterModel(c=best.c).fit(data.embeddings, data.labels, _sample_weights(data, soft=best.name == "soft"))
    final.save(artifacts.router_model)
    final.to_numpy(artifacts.router_weights, alpha=operating_point.alpha)

    metrics = _report(data, train, val, variants, best, operating_point, val_fraction, seed, target_recovery)
    store.write_json(artifacts.router_metrics, metrics)
    print(f"wrote router -> {artifacts.router_weights} (variant={best.name}, alpha={operating_point.alpha:.3f})")
    return metrics


def _subset(data: _Dataset, idx: np.ndarray) -> _Dataset:
    return _Dataset(
        embeddings=data.embeddings[idx],
        labels=data.labels[idx],
        local_scores=data.local_scores[idx],
        cloud_scores=data.cloud_scores[idx],
        categories=[data.categories[i] for i in idx],
    )


def _report(  # noqa: PLR0913 - a report gathers every evaluation input
    data: _Dataset,
    train: _Dataset,
    val: _Dataset,
    variants: list[_Variant],
    best: _Variant,
    operating_point: evaluate.OperatingPoint,
    val_fraction: float,
    seed: int,
    target_recovery: float,
) -> dict:
    return {
        "n_total": len(data),
        "n_train": len(train),
        "n_val": len(val),
        "val_fraction": val_fraction,
        "seed": seed,
        "target_recovery": target_recovery,
        "selected_variant": best.name,
        "selected_c": best.c,
        "variant_apgr": {variant.name: variant.apgr for variant in variants},
        "operating_point": {
            "alpha": operating_point.alpha,
            "cloud_fraction": operating_point.cloud_fraction,
            "quality": operating_point.quality,
            "gap_recovered": operating_point.gap_recovered,
        },
        "baselines": _baselines(val),
        **_operating_metrics(best.model, val, operating_point.alpha),
        "cost_quality_curve": [
            {"alpha": point.alpha, "cloud_fraction": point.cloud_fraction, "gap_recovered": point.gap_recovered}
            for point in best.curve
        ],
    }
