# Gulliver's Router

A low-cost routing system that splits traffic between a free local model and a
heavy-duty cloud AI based on query difficulty.

## Setup

```sh
uv sync
```

## Development

| Command       | Description                                      |
| ------------- | ------------------------------------------------ |
| `make setup`  | Install dependencies (`uv sync`).                |
| `make test`   | Run the test suite (`pytest`).                   |
| `make lint`   | Lint (`ruff check`) and type-check (`ty check`). |
| `make format` | Auto-fix and format with `ruff`.                 |

Run the CLI with `uv run gullivers-router` or `uv run python -m gullivers_router`.
