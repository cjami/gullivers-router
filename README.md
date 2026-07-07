# Gulliver's Router

A low-cost routing system that splits traffic between a free local model and a
heavy-duty cloud AI based on query difficulty.

## Setup

```sh
make setup   # uv sync
cp .env.example .env   # then fill in HF_TOKEN and FIREWORKS_API_KEY
```

`make setup` installs `llama-cpp-python` from a prebuilt **Vulkan** wheel (pinned to
`https://abetlen.github.io/llama-cpp-python/whl/vulkan` in `pyproject.toml`). If no wheel
matches your platform, build from source instead:

```sh
CMAKE_ARGS="-DGGML_VULKAN=on" uv pip install --no-binary llama-cpp-python llama-cpp-python
```

Models are downloaded automatically from HuggingFace on first use, using the repo id and
filename configured per role in `.env` (see `.env.example`). Each role — local generation,
embeddings, runtime cloud, and the training judge — can point at any supported provider
(`llama`, `openai`, or `fireworks`).

## Usage

| Command                        | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| `uv run gullivers-router run`  | Run the batch router (local vs cloud).            |
| `uv run gullivers-router train`| Train the matrix-factorization router (offline).  |

The runtime defaults to the hackathon file contract:

```sh
uv run gullivers-router run \
  --input /input/tasks.json \
  --output /output/results.json \
  --router-weights artifacts/training/router.npz
```

For local routing diagnostics without model completions:

```sh
uv run gullivers-router run \
  --input examples/tasks.json \
  --output artifacts/routes.json \
  --router-weights artifacts/training/router.npz \
  --classify-only
```

## Development

| Command       | Description                                      |
| ------------- | ------------------------------------------------ |
| `make setup`  | Install dependencies (`uv sync`).                |
| `make test`   | Run the test suite (`pytest`).                   |
| `make lint`   | Lint (`ruff check`) and type-check (`ty check`). |
| `make format` | Auto-fix and format with `ruff`.                 |

Run the CLI with `uv run gullivers-router` or `uv run python -m gullivers_router`.
