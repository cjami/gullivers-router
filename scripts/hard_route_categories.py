"""Override per-category routing thresholds in an exported router bundle.

A post-calibration hard-route: set a category's alpha to 0.0 so every prompt the
category head assigns to it routes to cloud (risk >= 0 is always true). Re-run after
any retrain, since retraining regenerates the calibrated thresholds.

Usage:
    uv run python scripts/hard_route_categories.py artifacts/training/router.npz \
        code_generation=0.0 code_debugging=0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def apply_overrides(bundle: Path, overrides: dict[str, float]) -> None:
    """Rewrite ``bundle`` in place with the given per-category alpha overrides."""
    data = dict(np.load(bundle, allow_pickle=False))
    if "cat_classes" not in data:
        msg = f"{bundle} has no category head; cannot set per-category alpha"
        raise SystemExit(msg)

    classes = [str(name) for name in data["cat_classes"]]
    unknown = sorted(set(overrides) - set(classes))
    if unknown:
        msg = f"unknown categories {unknown}; known: {classes}"
        raise SystemExit(msg)

    alpha = data["cat_alpha"].astype(np.float64).copy()
    for name, value in overrides.items():
        alpha[classes.index(name)] = value
    data["cat_alpha"] = alpha
    np.savez(bundle, **data)

    for name in overrides:
        print(f"  {name:<26} cat_alpha -> {alpha[classes.index(name)]:.4f}")


def _parse(pairs: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        overrides[name] = float(value)
    return overrides


def main(argv: list[str]) -> None:
    """Parse ``bundle path`` plus ``category=alpha`` pairs and apply the overrides."""
    if len(argv) < 2:  # noqa: PLR2004 - bundle path plus at least one override
        raise SystemExit(__doc__)
    bundle = Path(argv[0])
    print(f"overriding {bundle}")
    apply_overrides(bundle, _parse(argv[1:]))


if __name__ == "__main__":
    main(sys.argv[1:])
