"""Named-entity extraction, date recovery, and source-order formatting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from dateparser.search import search_dates

if TYPE_CHECKING:
    from gullivers_router.inference.base import NamedEntityModel

type EntityType = Literal["PERSON", "ORGANIZATION", "LOCATION", "DATE"]

_TYPE_MAP: dict[str, EntityType] = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
}
_QUOTED_SOURCE = re.compile(r":\s*(['\"])(.+)\1\s*$", re.DOTALL)
_LABELED_SOURCE = re.compile(r"(?:from|text)\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)
_RELATIVE_PREFIX = re.compile(r"\b(?:last|next|this)\s+$", re.IGNORECASE)
_LEADING_DATE_PREPOSITION = re.compile(r"^(?:on|at)\s+", re.IGNORECASE)
_TRAILING_DATE_CONNECTOR = re.compile(r"(?:\s*,)?\s+(?:and|or)\s*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Entity:
    """One exact source span and its normalized entity type."""

    text: str
    entity_type: EntityType
    start: int


def answer_named_entities(prompt: str, model: NamedEntityModel) -> str:
    """Extract entities and dates, returning one source-ordered entity per line."""
    source = source_text(prompt)
    model_entities = parse_model_entities(model.extract(source), source)
    entities = merge_entities(model_entities, extract_date_entities(source))
    return "\n".join(f"{entity.text}: {entity.entity_type}" for entity in entities)


def source_text(prompt: str) -> str:
    """Separate a quoted or labeled source passage from its task instruction."""
    quoted = _QUOTED_SOURCE.search(prompt)
    if quoted is not None:
        return quoted.group(2).strip()
    labeled = list(_LABELED_SOURCE.finditer(prompt))
    if labeled:
        return labeled[-1].group(1).strip().strip("'\"")
    return prompt.strip()


def parse_model_entities(raw: str, source: str) -> list[Entity]:
    """Parse Minibase JSON and retain only exact spans with supported types."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        return []
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []

    entities: list[Entity] = []
    for raw_type, values in payload.items():
        entity_type = _TYPE_MAP.get(str(raw_type).upper())
        if entity_type is None or not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            text = value.strip()
            match = re.search(re.escape(text), source, re.IGNORECASE)
            if match is not None:
                entities.append(
                    Entity(
                        text=source[match.start() : match.end()],
                        entity_type=entity_type,
                        start=match.start(),
                    )
                )
    return entities


def extract_date_entities(source: str) -> list[Entity]:
    """Find natural-language date spans and preserve their exact source spelling."""
    matches = search_dates(source, languages=["en"], settings={"PREFER_DATES_FROM": "past"}) or []
    entities: list[Entity] = []
    cursor = 0
    for raw_text, _parsed in matches:
        start = source.casefold().find(raw_text.casefold(), cursor)
        if start < 0:
            start = source.casefold().find(raw_text.casefold())
        if start < 0:
            continue
        end = start + len(raw_text)

        relative = _RELATIVE_PREFIX.search(source[:start])
        if relative is not None:
            start = relative.start()

        span = source[start:end]
        preposition = _LEADING_DATE_PREPOSITION.match(span)
        if preposition is not None:
            start += preposition.end()
            span = source[start:end]

        connector = _TRAILING_DATE_CONNECTOR.search(span)
        if connector is not None:
            end = start + connector.start()
            span = source[start:end]

        span = span.strip(" ,.;:()[]{}'\"")
        if not span:
            continue
        normalized_start = source.find(span, start, end)
        if normalized_start < 0:
            continue
        entities.append(Entity(text=span, entity_type="DATE", start=normalized_start))
        cursor = end
    return entities


def merge_entities(*groups: list[Entity]) -> list[Entity]:
    """Deduplicate entities and return them in first-source-occurrence order."""
    unique: dict[tuple[str, EntityType], Entity] = {}
    for entity in (item for group in groups for item in group):
        key = (entity.text.casefold(), entity.entity_type)
        existing = unique.get(key)
        if existing is None or entity.start < existing.start:
            unique[key] = entity
    return sorted(unique.values(), key=lambda entity: (entity.start, -len(entity.text), entity.entity_type))
