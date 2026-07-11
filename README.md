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
filename configured per role in `.env` (see `.env.example`). Qwen3 Embedding 0.6B uses
last-token pooling to produce 1024-dimensional vectors. Each role — local generation,
embeddings, runtime cloud, and the training judge — can point at any supported provider
(`llama`, `openai`, or `fireworks`).

Local development defaults use the Vulkan llama.cpp wheel and GPU offload. The submission
Docker image overrides llama.cpp to CPU mode with `LOCAL_N_GPU_LAYERS=0`,
`LOCAL_N_THREADS=2`, and `LOCAL_N_CTX=2048`.

## Usage

| Command                        | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| `uv run gullivers-router run`          | Run the batch router (local vs cloud).                    |
| `uv run gullivers-router score-practice` | Score practice results against the answer set with the judge. |
| `uv run gullivers-router train`        | Build the router training dataset (offline).              |
| `uv run gullivers-router train-router` | Train the routing model from judge scores and embeddings. |

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

To let borderline local routes verify the local answer before accepting it:

```sh
uv run gullivers-router run \
  --input examples/practice_tasks.json \
  --output outputs/results.json \
  --router-weights artifacts/training/router.npz \
  --local-cascade
```

## Docker submission smoke test

The Dockerfile builds a CPU-only `linux/amd64` image and includes the local Gemma E2B text
GGUF, the Qwen3 0.6B summarisation GGUF, the Minibase NER-Standard GGUF, the Qwen3 embedding GGUF used by the router, and
`artifacts/training/router.npz`.

```sh
docker buildx build --platform linux/amd64 --load -t gullivers-router:local .
mkdir -p outputs
uv run python -c "import numpy as np; np.savez('outputs/router-all-local.npz', weights=np.zeros(1024, dtype=np.float64), bias=np.float64(0), alpha=np.float64(2), normalize=True)"
docker run --rm --memory=4g --cpus=2 \
  -e CLOUD_PROVIDER=llama \
  -e CLOUD_REPO_ID=google/gemma-4-E2B-it-qat-q4_0-gguf \
  -e CLOUD_FILENAME=gemma-4-E2B_q4_0-it.gguf \
  -e CLOUD_N_CTX=2048 \
  -e CLOUD_N_GPU_LAYERS=0 \
  -e CLOUD_N_THREADS=2 \
  -e CLOUD_MODEL_ROOT=/app/models \
  -v "$PWD/examples/practice_tasks.json:/input/tasks.json:ro" \
  -v "$PWD/outputs:/output" \
  --entrypoint gullivers-router \
  gullivers-router:local run \
  --input /input/tasks.json \
  --output /output/results.json \
  --router-weights /output/router-all-local.npz
```

This all-local smoke test writes `outputs/results.json` without requiring Fireworks credentials.

## Local practice scoring

Use the included practice pack to score routing changes before submitting:

```sh
make practice
```

This generates answers with the router, then grades them with the configured `judge` role.

The report includes per-task LLM grades and a percentage score. Missing answers are counted as
incorrect without calling the judge.

## Publishing to GHCR

`.github/workflows/publish-image.yml` builds the CPU image and pushes it to
`ghcr.io/<owner>/gullivers-router`. It downloads the required GGUFs from HuggingFace (cached
between runs), so the runner does not need them committed.

Cut a release by pushing a tag:

```sh
git tag v1.0
git push origin v1.0
```

Each run publishes immutable `:v1.0` and `:sha-<commit>` tags plus a moving `:latest`. Pin the
evaluation program to an immutable tag so a redeploy never serves stale bytes. To force a clean
rebuild (ignoring the Docker layer cache), run the workflow manually from the Actions tab with
`no_cache` enabled.

One-time setup:

- Add an `HF_TOKEN` repository secret whose HuggingFace account has accepted the gated Gemma
  license; otherwise the model download returns 403.
- After the first publish, make the package public: repo → Packages → `gullivers-router` →
  Package settings → Change visibility → Public. It stays public for later pushes and lets the
  evaluation program pull anonymously even while the repo is private.

## Development

| Command       | Description                                      |
| ------------- | ------------------------------------------------ |
| `make setup`  | Install dependencies (`uv sync`).                |
| `make test`   | Run the test suite (`pytest`).                   |
| `make lint`   | Lint (`ruff check`) and type-check (`ty check`). |
| `make format` | Auto-fix and format with `ruff`.                 |

Run the CLI with `uv run gullivers-router` or `uv run python -m gullivers_router`.
