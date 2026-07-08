from gullivers_router.cli import build_parser, main
from gullivers_router.inference.base import DEFAULT_INFERENCE_SEED
from gullivers_router.router import DEFAULT_INPUT, DEFAULT_OUTPUT, DEFAULT_ROUTER_WEIGHTS
from gullivers_router.training import DEFAULT_CONCURRENCY


def test_main_returns_success():
    assert main([]) == 0


def test_parser_exposes_program_name():
    parser = build_parser()
    assert parser.prog == "gullivers-router"


def test_run_dispatches_to_router(monkeypatch):
    calls = []
    monkeypatch.setattr("gullivers_router.router.run", lambda **kwargs: calls.append(kwargs))
    assert (
        main(
            [
                "run",
                "--input",
                "examples/tasks.json",
                "--output",
                "artifacts/dev/results.json",
                "--router-weights",
                "artifacts/dev/router.npz",
                "--workers",
                "3",
                "--classify-only",
            ]
        )
        == 0
    )
    assert len(calls) == 1
    call = calls[0]
    assert str(call["input_path"]) == "examples\\tasks.json"
    assert str(call["output_path"]) == "artifacts\\dev\\results.json"
    assert str(call["router_weights"]) == "artifacts\\dev\\router.npz"
    assert call["workers"] == 3
    assert call["classify_only"] is True


def test_run_defaults_flow_through_to_router(monkeypatch):
    calls = []
    monkeypatch.setattr("gullivers_router.router.run", lambda **kwargs: calls.append(kwargs))

    assert main(["run"]) == 0

    [call] = calls
    assert call["input_path"] == DEFAULT_INPUT
    assert call["output_path"] == DEFAULT_OUTPUT
    assert call["router_weights"] == DEFAULT_ROUTER_WEIGHTS
    assert call["workers"] == DEFAULT_CONCURRENCY
    assert call["classify_only"] is False


def test_train_dispatches_to_training(monkeypatch):
    calls = []
    monkeypatch.setattr("gullivers_router.training.train", lambda **kwargs: calls.append(kwargs))
    assert main(["train", "--samples", "5", "--out", "artifacts/x", "--stages", "local,cloud", "--workers", "4"]) == 0
    assert calls == [{"samples_per_category": 5, "out": "artifacts/x", "stages": ("local", "cloud"), "workers": 4}]


def test_train_router_dispatches_quality_floor_options(monkeypatch):
    calls = []
    monkeypatch.setattr("gullivers_router.training.train_router", lambda *args, **kwargs: calls.append((args, kwargs)))

    assert (
        main(
            [
                "train-router",
                "--out",
                "artifacts/x",
                "--val-fraction",
                "0.25",
                "--seed",
                "7",
                "--quality-floor",
                "3.0",
                "--accuracy-gate",
                "0.80",
                "--margin",
                "0.05",
            ]
        )
        == 0
    )

    [(args, kwargs)] = calls
    assert str(args[0]) == "artifacts\\x"
    assert kwargs == {
        "val_fraction": 0.25,
        "seed": 7,
        "quality_floor": 3.0,
        "accuracy_gate": 0.80,
        "target_margin": 0.05,
    }


def test_train_router_default_seed_matches_global_seed(monkeypatch):
    calls = []
    monkeypatch.setattr("gullivers_router.training.train_router", lambda *args, **kwargs: calls.append((args, kwargs)))

    assert main(["train-router", "--out", "artifacts/x"]) == 0

    [(_, kwargs)] = calls
    assert kwargs["seed"] == DEFAULT_INFERENCE_SEED
