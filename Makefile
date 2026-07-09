.PHONY: setup test lint format score-practice

PRACTICE_TASKS ?= examples/practice_tasks.json
PRACTICE_RESULTS ?= outputs/results.json
PRACTICE_ANSWER_SET ?= examples/practice_answer_set.json
PRACTICE_SCORE ?= outputs/practice_score.json

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ty check

format:
	uv run ruff check --fix .
	uv run ruff format .

score-practice:
	uv run gullivers-router score-practice \
		--tasks $(PRACTICE_TASKS) \
		--results $(PRACTICE_RESULTS) \
		--answer-set $(PRACTICE_ANSWER_SET) \
		--output $(PRACTICE_SCORE)
