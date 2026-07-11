.PHONY: setup test lint format practice

PRACTICE_TASKS ?= examples/practice_tasks.json
PRACTICE_RESULTS ?= outputs/results.json
PRACTICE_ANSWER_SET ?= examples/practice_answer_set.json
PRACTICE_SCORE ?= outputs/practice_score.json
PRACTICE_ROUTER_WEIGHTS ?= artifacts/training/router.npz

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

practice:
	uv run gullivers-router run \
		--input $(PRACTICE_TASKS) \
		--output $(PRACTICE_RESULTS) \
		--router-weights $(PRACTICE_ROUTER_WEIGHTS)
	uv run gullivers-router score-practice \
		--tasks $(PRACTICE_TASKS) \
		--results $(PRACTICE_RESULTS) \
		--answer-set $(PRACTICE_ANSWER_SET) \
		--output $(PRACTICE_SCORE)
