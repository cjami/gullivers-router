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
    monkeypatch.setattr("gullivers_router.training.train", lambda: calls.append("train"))
    assert main(["train"]) == 0
    assert calls == ["train"]
