import numpy as np

from gullivers_router.router.model import (
    CategoryModel,
    RouterModel,
    category_thresholds,
    load_numpy,
    predict_categories,
    probabilities,
    save_bundle,
)


def _separable_data(seed=0):
    rng = np.random.default_rng(seed)
    local = rng.normal(loc=-1.0, size=(60, 4))
    cloud = rng.normal(loc=1.0, size=(60, 4))
    x = np.vstack([local, cloud])
    y = np.array([0] * 60 + [1] * 60)
    return x, y


def test_learns_separable_data():
    x, y = _separable_data()
    model = RouterModel().fit(x, y)

    assert (model.route(x) == y).mean() > 0.9


def test_save_load_roundtrip(tmp_path):
    x, y = _separable_data()
    model = RouterModel().fit(x, y)
    path = tmp_path / "router.joblib"
    model.save(path)

    reloaded = RouterModel.load(path)

    assert np.allclose(reloaded.predict_proba(x), model.predict_proba(x))


def test_numpy_export_matches_sklearn(tmp_path):
    x, y = _separable_data()
    model = RouterModel().fit(x, y)
    path = tmp_path / "router.npz"
    model.to_numpy(path, alpha=0.5)

    weights = load_numpy(path)

    assert np.allclose(probabilities(weights, x), model.predict_proba(x))
    assert float(weights["alpha"]) == 0.5


def test_fit_accepts_sample_weight():
    x, y = _separable_data()
    weight = np.abs(np.linspace(1.0, 3.0, len(y)))

    model = RouterModel().fit(x, y, weight)

    assert model.predict_proba(x).shape == (len(y),)


def _category_data(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[-2.0, 0.0], size=(40, 2))
    b = rng.normal(loc=[2.0, 0.0], size=(40, 2))
    x = np.vstack([a, b])
    y = np.array(["a"] * 40 + ["b"] * 40)
    return x, y


def test_bundle_scores_categories_and_thresholds(tmp_path):
    x, y = _category_data()
    risk = RouterModel().fit(x, (y == "b").astype(int))
    category = CategoryModel().fit(x, y)
    path = tmp_path / "bundle.npz"
    save_bundle(path, risk=risk, category=category, alpha_by_category={"a": 0.9, "b": 0.2}, global_alpha=0.5)

    weights = load_numpy(path)

    assert np.allclose(probabilities(weights, x), risk.predict_proba(x))
    predicted = predict_categories(weights, x)
    assert predicted is not None
    assert (predicted == category.predict(x)).all()
    thresholds = category_thresholds(weights, predicted)
    assert set(np.round(thresholds, 3).tolist()) <= {0.9, 0.2}


def test_predict_categories_is_none_without_category_head(tmp_path):
    x, y = _separable_data()
    model = RouterModel().fit(x, y)
    path = tmp_path / "router.npz"
    model.to_numpy(path, alpha=0.5)

    assert predict_categories(load_numpy(path), x) is None
