from gullivers_router.router.ner import (
    Entity,
    answer_named_entities,
    extract_date_entities,
    parse_model_entities,
    source_text,
)


class FakeNamedEntityModel:
    def __init__(self, response: str):
        self.response = response

    def extract(self, text: str) -> str:
        return self.response


def test_source_text_extracts_final_quoted_passage():
    prompt = "Extract entities from: 'Jordan met O'Reilly Media in Boston.'"

    assert source_text(prompt) == "Jordan met O'Reilly Media in Boston."


def test_answer_merges_model_entities_and_dates_in_source_order():
    prompt = (
        "Extract all named entities from: 'On March 15 2023, Sundar Pichai announced that Google "
        "would open a lab in Zurich with ETH Zurich.'"
    )
    model = FakeNamedEntityModel('{"PER":["Sundar Pichai"],"ORG":["Google","ETH Zurich"],"LOC":["Zurich"],"MISC":[]}')

    assert answer_named_entities(prompt, model) == (
        "March 15 2023: DATE\nSundar Pichai: PERSON\nGoogle: ORGANIZATION\nZurich: LOCATION\nETH Zurich: ORGANIZATION"
    )


def test_extract_date_entities_supports_relative_iso_and_written_dates():
    source = "The review began last March, resumes on 2026-08-17, and ends next Tuesday."

    assert [entity.text for entity in extract_date_entities(source)] == [
        "last March",
        "2026-08-17",
        "next Tuesday",
    ]


def test_parse_model_entities_ignores_unknown_types_and_non_source_values():
    source = "Ada Lovelace worked with the Royal Society in London."
    raw = (
        'Result: {"PER":["Ada Lovelace","Grace Hopper"],'
        '"ORG":["Royal Society"],"LOC":["London"],"MISC":["mathematics"]}'
    )

    assert parse_model_entities(raw, source) == [
        Entity("Ada Lovelace", "PERSON", 0),
        Entity("Royal Society", "ORGANIZATION", 29),
        Entity("London", "LOCATION", 46),
    ]
