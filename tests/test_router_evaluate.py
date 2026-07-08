import numpy as np

from gullivers_router.training import evaluate


def test_pass_rate_at_routing_extremes():
    local = np.array([3.0, 4.0, 5.0])
    cloud = np.array([4.0, 4.0, 5.0])

    always_cloud = np.ones(3, int)
    always_local = np.zeros(3, int)

    assert evaluate.pass_rate(always_cloud, local, cloud, quality_floor=4.0) == 1.0
    assert evaluate.pass_rate(always_local, local, cloud, quality_floor=4.0) == 2 / 3


def test_cost_pass_curve_cloud_fraction_monotonic_in_alpha():
    proba = np.array([0.1, 0.4, 0.6, 0.9])
    local = np.array([3.0, 3.0, 3.0, 3.0])
    cloud = np.array([4.0, 4.0, 4.0, 4.0])

    curve = evaluate.cost_pass_curve(proba, local, cloud, quality_floor=4.0)

    alphas = [point.alpha for point in curve]
    fractions = [point.cloud_fraction for point in curve]
    assert alphas == sorted(alphas, reverse=True)
    assert fractions == sorted(fractions)


def test_average_pass_rate_prefers_correct_ranking():
    local = np.array([5.0] * 5 + [3.0] * 5)
    cloud = np.array([5.0] * 10)
    good = np.linspace(0, 1, 10)
    bad = good[::-1]

    good_curve = evaluate.cost_pass_curve(good, local, cloud, quality_floor=4.0)
    bad_curve = evaluate.cost_pass_curve(bad, local, cloud, quality_floor=4.0)

    assert evaluate.average_pass_rate(good_curve) > evaluate.average_pass_rate(bad_curve)


def test_cloud_fraction_for_pass_rate_picks_cheapest():
    local = np.array([5.0] * 5 + [3.0] * 5)
    cloud = np.array([5.0] * 10)
    proba = np.linspace(0, 1, 10)
    curve = evaluate.cost_pass_curve(proba, local, cloud, quality_floor=4.0)

    point = evaluate.cloud_fraction_for_pass_rate(curve, target=1.0)

    assert point is not None
    assert point.pass_rate >= 1.0
    assert point.cloud_fraction == min(p.cloud_fraction for p in curve if p.pass_rate >= 1.0)


def test_cloud_fraction_for_pass_rate_returns_none_when_unreachable():
    curve = evaluate.cost_pass_curve(
        np.array([0.2, 0.8]),
        np.array([3.0, 3.0]),
        np.array([3.0, 3.0]),
        quality_floor=4.0,
    )

    assert evaluate.cloud_fraction_for_pass_rate(curve, target=0.5) is None


def test_cost_pass_curve_reports_rescue_and_unnecessary_cloud():
    routes = np.array([0.9, 0.8, 0.1, 0.0])
    local = np.array([3.0, 5.0, 3.0, 5.0])
    cloud = np.array([5.0, 5.0, 3.0, 5.0])

    point = next(p for p in evaluate.cost_pass_curve(routes, local, cloud, quality_floor=4.0) if p.alpha == 0.8)

    assert point.cloud_fraction == 0.5
    assert point.rescue_rate == 0.25
    assert point.unnecessary_cloud_fraction == 0.25


def test_operating_point_at_alpha_uses_fixed_threshold():
    risk = np.array([0.9, 0.6, 0.4, 0.1])
    local = np.array([3.0, 5.0, 3.0, 5.0])
    cloud = np.array([5.0, 5.0, 5.0, 5.0])

    point = evaluate.operating_point_at_alpha(risk, local, cloud, quality_floor=4.0, alpha=0.5)

    assert point.alpha == 0.5
    assert point.cloud_fraction == 0.5
    assert point.pass_rate == 0.75


def test_per_category_cloud_rate():
    routes = np.array([1, 0, 1, 1])
    categories = ["math", "math", "code", "code"]

    rates = evaluate.per_category_cloud_rate(routes, categories)

    assert rates == {"code": 1.0, "math": 0.5}
