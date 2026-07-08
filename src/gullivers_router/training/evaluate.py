"""Quality-floor evaluation for the router.

The router minimizes cloud calls while keeping routed answer quality above a chosen floor.
Sweeping the cloud-risk threshold traces the tradeoff between cost and pass rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """The router's behaviour at one decision threshold."""

    alpha: float
    cloud_fraction: float
    quality: float
    pass_rate: float
    violation_rate: float
    rescue_rate: float
    unnecessary_cloud_fraction: float


def routed_quality(routes: np.ndarray, local_scores: np.ndarray, cloud_scores: np.ndarray) -> float:
    """Mean judge score when each query takes its routed model's score."""
    chosen = np.where(routes == 1, cloud_scores, local_scores)
    return float(np.mean(chosen))


def pass_rate(routes: np.ndarray, local_scores: np.ndarray, cloud_scores: np.ndarray, quality_floor: float) -> float:
    """Fraction of routed answers meeting ``quality_floor``."""
    chosen = np.where(routes == 1, cloud_scores, local_scores)
    return float(np.mean(chosen >= quality_floor))


def cost_pass_curve(
    risk: np.ndarray,
    local_scores: np.ndarray,
    cloud_scores: np.ndarray,
    quality_floor: float,
) -> list[OperatingPoint]:
    """Trace quality-floor pass rate and cloud-call fraction across risk thresholds."""
    alphas = np.unique(
        np.concatenate(
            [
                [0.0, 1.0],
                risk,
                [np.nextafter(1.0, 2.0)],
            ]
        )
    )[::-1]
    return [operating_point_at_alpha(risk, local_scores, cloud_scores, quality_floor, float(alpha)) for alpha in alphas]


def operating_point_at_alpha(
    risk: np.ndarray,
    local_scores: np.ndarray,
    cloud_scores: np.ndarray,
    quality_floor: float,
    alpha: float,
) -> OperatingPoint:
    """Measure routed quality and cost at a fixed decision threshold."""
    routes = (risk >= alpha).astype(int)
    return OperatingPoint(alpha=float(alpha), **routed_metrics(routes, local_scores, cloud_scores, quality_floor))


def routed_metrics(
    routes: np.ndarray,
    local_scores: np.ndarray,
    cloud_scores: np.ndarray,
    quality_floor: float,
) -> dict[str, float]:
    """Quality, pass rate, and cost for an explicit per-row route vector.

    Unlike :func:`operating_point_at_alpha`, the routes need not come from a single threshold,
    so this also scores per-category or otherwise heterogeneous routing policies.
    """
    passes = pass_rate(routes, local_scores, cloud_scores, quality_floor)
    return {
        "cloud_fraction": float(np.mean(routes)),
        "quality": routed_quality(routes, local_scores, cloud_scores),
        "pass_rate": passes,
        "violation_rate": 1.0 - passes,
        "rescue_rate": _rescue_rate(routes, local_scores, cloud_scores, quality_floor),
        "unnecessary_cloud_fraction": _unnecessary_cloud_fraction(routes, local_scores, quality_floor),
    }


def _rescue_rate(
    routes: np.ndarray,
    local_scores: np.ndarray,
    cloud_scores: np.ndarray,
    quality_floor: float,
) -> float:
    """Fraction of all rows rescued by cloud when local would miss the floor."""
    rescued = (routes == 1) & (local_scores < quality_floor) & (cloud_scores >= quality_floor)
    return float(np.mean(rescued))


def _unnecessary_cloud_fraction(routes: np.ndarray, local_scores: np.ndarray, quality_floor: float) -> float:
    """Fraction of all rows sent to cloud even though local already met the floor."""
    unnecessary = (routes == 1) & (local_scores >= quality_floor)
    return float(np.mean(unnecessary))


def average_pass_rate(curve: list[OperatingPoint]) -> float:
    """Area under pass-rate vs cloud-fraction."""
    by_cost: dict[float, float] = {}
    for point in curve:
        by_cost[point.cloud_fraction] = max(by_cost.get(point.cloud_fraction, 0.0), point.pass_rate)
    costs = np.array(sorted(by_cost))
    pass_rates = np.array([by_cost[cost] for cost in costs])
    return float(np.trapezoid(pass_rates, costs))


def cloud_fraction_for_pass_rate(curve: list[OperatingPoint], target: float) -> OperatingPoint | None:
    """Return the cheapest operating point reaching the target pass rate, if any."""
    reaching = [point for point in curve if point.pass_rate >= target]
    if not reaching:
        return None
    return min(reaching, key=lambda point: point.cloud_fraction)


def per_category_cloud_rate(routes: np.ndarray, categories: list[str]) -> dict[str, float]:
    """Fraction routed to the cloud within each category."""
    rates: dict[str, float] = {}
    category_array = np.array(categories)
    for category in sorted(set(categories)):
        mask = category_array == category
        rates[category] = float(np.mean(routes[mask]))
    return rates
