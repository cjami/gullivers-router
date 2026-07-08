from gullivers_router.model_selection import select_model


def test_prefers_earliest_family_in_preference_order():
    allowed = ["gemma-4-31b-it", "minimax-m3", "kimi-k2p7-code"]
    assert select_model(allowed) == "minimax-m3"


def test_matches_family_as_case_insensitive_substring():
    allowed = ["accounts/fireworks/models/MiniMax-M3"]
    assert select_model(allowed) == "accounts/fireworks/models/MiniMax-M3"


def test_falls_through_to_next_family_when_preferred_absent():
    allowed = ["kimi-k2p7-code", "gemma-4-31b-it", "gemma-4-26b-a4b-it"]
    assert select_model(allowed) == "gemma-4-31b-it"


def test_first_allowed_wins_within_the_same_family():
    allowed = ["gemma-4-26b-a4b-it", "gemma-4-31b-it"]
    assert select_model(allowed) == "gemma-4-26b-a4b-it"


def test_falls_back_to_first_allowed_when_no_family_matches():
    allowed = ["mystery-a", "mystery-b"]
    assert select_model(allowed) == "mystery-a"


def test_honours_a_custom_preference_order():
    allowed = ["gemma-4-31b-it", "kimi-k2p7-code"]
    assert select_model(allowed, ["kimi", "gemma"]) == "kimi-k2p7-code"
