"""The routing model: a calibrated logistic classifier over query embeddings.

Given a prompt's embedding it predicts ``P(cloud)`` -- the probability the cloud model beats the
local one by a useful margin. A single threshold on that probability picks the route, so tuning
the threshold trades cloud cost against answer quality (see the training-time evaluation).

The model is trained with scikit-learn but reduces to a weight vector and intercept over the
L2-normalized embedding, so :meth:`RouterModel.to_numpy` exports it to a small ``.npz`` the runtime
can score with numpy alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_C = 1.0
DEFAULT_ALPHA = 0.5
MAX_ITER = 1000


class RouterModel:
    """Logistic regression over L2-normalized embeddings, predicting ``P(cloud)``."""

    def __init__(self, *, c: float = DEFAULT_C) -> None:
        """Configure regularization strength ``c`` (sklearn's inverse-L2 ``C``)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer

        self._pipeline = make_pipeline(
            Normalizer(norm="l2"),
            LogisticRegression(C=c, class_weight="balanced", max_iter=MAX_ITER),
        )

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> RouterModel:
        """Fit on embeddings ``x`` and 0/1 labels ``y``, optionally weighting samples."""
        self._pipeline.fit(x, y, logisticregression__sample_weight=sample_weight)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return ``P(cloud)`` for each row of ``x``."""
        return self._pipeline.predict_proba(x)[:, 1]

    def route(self, x: np.ndarray, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
        """Return 1 (cloud) where ``P(cloud) >= alpha``, else 0 (local)."""
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

    def to_numpy(self, path: Path, *, alpha: float = DEFAULT_ALPHA) -> None:
        """Export the weight vector, intercept, and threshold for numpy-only scoring."""
        classifier = self._pipeline[-1]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weights=classifier.coef_[0].astype(np.float64),
            bias=np.float64(classifier.intercept_[0]),
            alpha=np.float64(alpha),
            normalize=True,
        )


def load_numpy(path: Path) -> dict[str, np.ndarray]:
    """Load the exported ``.npz`` weights into a plain dict."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def probabilities(weights: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Score embeddings with exported weights, matching :meth:`RouterModel.predict_proba`."""
    features = _l2_normalize(x) if bool(weights["normalize"]) else x
    logits = features @ weights["weights"] + weights["bias"]
    return 1.0 / (1.0 + np.exp(-logits))


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)
