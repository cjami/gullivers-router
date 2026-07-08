"""The routing model: a quality-floor risk classifier over query embeddings.

Given a prompt's embedding it predicts whether the local model is likely to fall below the
quality floor while the cloud model can rescue the answer. A per-category threshold on that
probability picks the route, so tuning the thresholds minimizes cloud calls subject to a quality
target. A companion :class:`CategoryModel` predicts each prompt's category from the same embedding,
so the runtime can pick the right threshold without the caller supplying a category.

Both models are trained with scikit-learn but reduce to linear weights over the L2-normalized
embedding, so :func:`save_bundle` exports them to a small ``.npz`` the runtime scores with numpy
alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_REGULARIZATION = 1.0
DEFAULT_ALPHA = 0.5
MAX_ITER = 1000


class RouterModel:
    """Logistic regression over L2-normalized embeddings, predicting ``P(needs_cloud)``."""

    def __init__(self, *, regularization: float = DEFAULT_REGULARIZATION) -> None:
        """Configure regularization strength."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer

        self._pipeline = make_pipeline(
            Normalizer(norm="l2"),
            LogisticRegression(C=regularization, class_weight="balanced", max_iter=MAX_ITER),
        )

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> RouterModel:
        """Fit on embeddings ``x`` and 0/1 ``needs_cloud`` targets."""
        self._pipeline.fit(x, y, logisticregression__sample_weight=sample_weight)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return ``P(needs_cloud)`` for each row of ``x``."""
        return self._pipeline.predict_proba(x)[:, 1]

    def route(self, x: np.ndarray, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
        """Return 1 (cloud) where predicted risk is at least ``alpha``."""
        return (self.predict_proba(x) >= alpha).astype(int)

    def save(self, path: Path) -> None:
        """Persist the fitted sklearn pipeline for reloading in training/eval."""
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: Path) -> RouterModel:
        """Reload a model saved by :meth:`save`."""
        import joblib

        model = cls.__new__(cls)
        model._pipeline = joblib.load(path)  # noqa: SLF001 - reconstructing an instance of this same class
        return model

    def coefficients(self) -> tuple[np.ndarray, np.float64]:
        """Return the weight vector and intercept over the normalized embedding."""
        classifier = self._pipeline[-1]
        return classifier.coef_[0].astype(np.float64), np.float64(classifier.intercept_[0])

    def to_numpy(self, path: Path, *, alpha: float = DEFAULT_ALPHA) -> None:
        """Export a single-threshold bundle (no category head) for numpy-only scoring."""
        save_bundle(path, risk=self, category=None, alpha_by_category={}, global_alpha=alpha)


class CategoryModel:
    """Multinomial logistic regression over L2-normalized embeddings, predicting the category."""

    def __init__(self, *, regularization: float = DEFAULT_REGULARIZATION) -> None:
        """Configure regularization strength."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer

        self._pipeline = make_pipeline(
            Normalizer(norm="l2"),
            LogisticRegression(C=regularization, class_weight="balanced", max_iter=MAX_ITER),
        )

    def fit(self, x: np.ndarray, categories: np.ndarray) -> CategoryModel:
        """Fit on embeddings ``x`` and string category targets."""
        self._pipeline.fit(x, categories)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the predicted category for each row of ``x``."""
        return self._pipeline.predict(x)

    def coefficients(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return per-class weights, biases, and class labels, expanding the binary case.

        For two classes sklearn keeps a single coefficient row; padding it to a zero row for the
        first class lets the runtime pick a category with a uniform ``argmax`` over all classes.
        """
        classifier = self._pipeline[-1]
        classes = [str(label) for label in classifier.classes_]
        coef = classifier.coef_.astype(np.float64)
        intercept = classifier.intercept_.astype(np.float64)
        if len(classes) == 2 and coef.shape[0] == 1:  # noqa: PLR2004 - binary logistic keeps one row
            coef = np.vstack([np.zeros_like(coef[0]), coef[0]])
            intercept = np.array([0.0, intercept[0]], dtype=np.float64)
        return coef, intercept, classes


def save_bundle(
    path: Path,
    *,
    risk: RouterModel,
    category: CategoryModel | None,
    alpha_by_category: dict[str, float],
    global_alpha: float,
) -> None:
    """Export the risk model, optional category head, and per-category thresholds to one ``.npz``.

    Embeddings are always scored L2-normalized, matching how both models were fitted.
    """
    weights, bias = risk.coefficients()
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "weights": weights,
        "bias": bias,
        "alpha": np.float64(global_alpha),
    }
    if category is not None:
        cat_weights, cat_bias, cat_classes = category.coefficients()
        arrays["cat_weights"] = cat_weights
        arrays["cat_bias"] = cat_bias
        arrays["cat_classes"] = np.array(cat_classes)
        arrays["cat_alpha"] = np.array(
            [alpha_by_category.get(name, global_alpha) for name in cat_classes], dtype=np.float64
        )
    np.savez(path, **arrays)


def load_numpy(path: Path) -> dict[str, np.ndarray]:
    """Load the exported ``.npz`` weights into a plain dict."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def probabilities(weights: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Score embeddings with exported weights, matching :meth:`RouterModel.predict_proba`."""
    logits = _features(weights, x) @ weights["weights"] + weights["bias"]
    return 1.0 / (1.0 + np.exp(-logits))


def predict_categories(weights: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray | None:
    """Predict each row's category, or ``None`` when the bundle carries no category head."""
    if "cat_classes" not in weights:
        return None
    scores = _features(weights, x) @ weights["cat_weights"].T + weights["cat_bias"]
    return weights["cat_classes"][np.argmax(scores, axis=1)]


def category_thresholds(weights: dict[str, np.ndarray], categories: np.ndarray) -> np.ndarray:
    """Map predicted categories to their thresholds, falling back to the global threshold."""
    global_alpha = float(weights["alpha"])
    alpha_by_category = dict(zip(weights["cat_classes"], weights["cat_alpha"], strict=True))
    return np.array([float(alpha_by_category.get(category, global_alpha)) for category in categories])


def _features(weights: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return _l2_normalize(x) if bool(weights.get("normalize", True)) else x


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)
