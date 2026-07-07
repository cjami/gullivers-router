import numpy as np

from gullivers_router.router.model import RouterModel, load_numpy, probabilities


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
