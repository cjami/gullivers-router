from gullivers_router.cli import build_parser, main


def test_main_returns_success():
    assert main([]) == 0


def test_parser_exposes_program_name():
    parser = build_parser()
    assert parser.prog == "gullivers-router"


def test_run_dispatches_to_router(monkeypatch):
    calls = []
    monkeypatch.setattr("gullivers_router.router.run", lambda: calls.append("run"))
    assert main(["run"]) == 0
    assert calls == ["run"]


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
                "4.0",
                "--target-pass-rate",
                "0.97",
            ]
        )
        == 0
    )

    [(args, kwargs)] = calls
    assert str(args[0]) == "artifacts\\x"
    assert kwargs == {"val_fraction": 0.25, "seed": 7, "quality_floor": 4.0, "target_pass_rate": 0.97}
