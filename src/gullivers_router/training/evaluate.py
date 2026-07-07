"""Cost-quality evaluation for the router (outline §5, the alpha threshold).

Routing is not a plain accuracy problem: sending a query to the cloud buys quality at a cost, so
the model is judged by how much answer quality it recovers per cloud call. Sweeping the decision
threshold traces a cost-quality curve; its area (average gap recovered) summarizes the router, and
the curve locates the threshold that hits a target quality at the fewest cloud calls.
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
    gap_recovered: float


def routed_quality(routes: np.ndarray, local_scores: np.ndarray, cloud_scores: np.ndarray) -> float:
    """Mean judge score when each query takes its routed model's score."""
    chosen = np.where(routes == 1, cloud_scores, local_scores)
    return float(np.mean(chosen))


def gap_recovered(quality: float, all_local: float, all_cloud: float) -> float:
    """Fraction of the local->cloud quality gap that ``quality`` recovers."""
    gap = all_cloud - all_local
    if gap <= 0:
        return 0.0
    return (quality - all_local) / gap


def cost_quality_curve(
    proba: np.ndarray,
    local_scores: np.ndarray,
    cloud_scores: np.ndarray,
) -> list[OperatingPoint]:
    """Trace quality and cloud-call fraction across thresholds, sorted by descending alpha."""
    all_local = float(np.mean(local_scores))
    all_cloud = float(np.mean(cloud_scores))
    alphas = np.unique(np.concatenate([[0.0], proba, [np.nextafter(1.0, 2.0)]]))[::-1]
    points: list[OperatingPoint] = []
    for alpha in alphas:
        routes = (proba >= alpha).astype(int)
        quality = routed_quality(routes, local_scores, cloud_scores)
        points.append(
            OperatingPoint(
                alpha=float(alpha),
                cloud_fraction=float(np.mean(routes)),
                quality=quality,
                gap_recovered=gap_recovered(quality, all_local, all_cloud),
            )
        )
    return points


def average_gap_recovered(curve: list[OperatingPoint]) -> float:
    """Area under the gap-recovered vs cloud-fraction curve (RouteLLM's APGR)."""
    by_cost: dict[float, float] = {}
    for point in curve:
        by_cost[point.cloud_fraction] = max(by_cost.get(point.cloud_fraction, 0.0), point.gap_recovered)
    costs = np.array(sorted(by_cost))
    recovered = np.array([by_cost[cost] for cost in costs])
    return float(np.trapezoid(recovered, costs))


def cloud_fraction_for_recovery(curve: list[OperatingPoint], target: float) -> OperatingPoint | None:
    """Return the cheapest operating point recovering at least ``target`` of the gap, if any."""
    reaching = [point for point in curve if point.gap_recovered >= target]
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
