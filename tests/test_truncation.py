from gullivers_router.inference.truncation import truncate_head_tail


def test_short_input_passes_through():
    tokens = list(range(10))
    assert truncate_head_tail(tokens, limit=100) == tokens


def test_exact_limit_passes_through():
    tokens = list(range(2048))
    assert truncate_head_tail(tokens) == tokens


def test_long_input_keeps_head_and_tail_in_order():
    tokens = list(range(5000))
    result = truncate_head_tail(tokens, limit=2048, head=1024, tail=1024)
    assert len(result) == 2048
    assert result[:1024] == list(range(1024))
    assert result[1024:] == list(range(5000 - 1024, 5000))
