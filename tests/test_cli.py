from gullivers_router.cli import build_parser, main


def test_main_returns_success():
    assert main([]) == 0


def test_parser_exposes_program_name():
    parser = build_parser()
    assert parser.prog == "gullivers-router"
