import numpy as np

from gullivers_router.training import evaluate


def test_gap_recovered_at_routing_extremes():
    local = np.array([2.0, 4.0, 6.0])
    cloud = np.array([5.0, 5.0, 8.0])
    all_local, all_cloud = float(local.mean()), float(cloud.mean())

    always_cloud = evaluate.routed_quality(np.ones(3, int), local, cloud)
    always_local = evaluate.routed_quality(np.zeros(3, int), local, cloud)

    assert evaluate.gap_recovered(always_cloud, all_local, all_cloud) == 1.0
    assert evaluate.gap_recovered(always_local, all_local, all_cloud) == 0.0


def test_gap_recovered_is_zero_when_no_gap():
    assert evaluate.gap_recovered(5.0, 5.0, 5.0) == 0.0


def test_cost_quality_curve_cloud_fraction_monotonic_in_alpha():
    proba = np.array([0.1, 0.4, 0.6, 0.9])
    local = np.array([1.0, 1.0, 1.0, 1.0])
    cloud = np.array([2.0, 2.0, 2.0, 2.0])

    curve = evaluate.cost_quality_curve(proba, local, cloud)

    alphas = [point.alpha for point in curve]
    fractions = [point.cloud_fraction for point in curve]
    assert alphas == sorted(alphas, reverse=True)
    assert fractions == sorted(fractions)


def test_average_gap_recovered_prefers_correct_ranking():
    local = np.ones(10)
    cloud = np.array([1.0] * 5 + [10.0] * 5)
    good = np.linspace(0, 1, 10)  # high proba where cloud helps
    bad = good[::-1]

    good_curve = evaluate.cost_quality_curve(good, local, cloud)
    bad_curve = evaluate.cost_quality_curve(bad, local, cloud)

    assert evaluate.average_gap_recovered(good_curve) > evaluate.average_gap_recovered(bad_curve)


def test_cloud_fraction_for_recovery_picks_cheapest():
    local = np.ones(10)
    cloud = np.array([1.0] * 5 + [10.0] * 5)
    proba = np.linspace(0, 1, 10)
    curve = evaluate.cost_quality_curve(proba, local, cloud)

    point = evaluate.cloud_fraction_for_recovery(curve, target=1.0)

    assert point is not None
    assert point.gap_recovered >= 1.0
    assert point.cloud_fraction == min(p.cloud_fraction for p in curve if p.gap_recovered >= 1.0)


def test_cloud_fraction_for_recovery_returns_none_when_unreachable():
    curve = evaluate.cost_quality_curve(np.array([0.2, 0.8]), np.array([1.0, 1.0]), np.array([1.0, 1.0]))

    assert evaluate.cloud_fraction_for_recovery(curve, target=0.5) is None


def test_per_category_cloud_rate():
    routes = np.array([1, 0, 1, 1])
    categories = ["math", "math", "code", "code"]

    rates = evaluate.per_category_cloud_rate(routes, categories)

    assert rates == {"code": 1.0, "math": 0.5}
