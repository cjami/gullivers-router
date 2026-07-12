![Gulliver's Router cover](gullivers_router_cover.png)

# Gulliver's Router

Gulliver's Router reduces cloud usage by handling suitable prompts locally. It sends work to
the cloud when the expected improvement in answer quality justifies the extra cost.

Under the hood, it uses small classifiers trained on judged local and cloud responses. They flag
prompts where the local model is likely to fall short, identify the task category, and let the
runtime use specialist paths for jobs like named entity extraction and arithmetic. The result is
a router that stays practical for resource-constrained environments.

This was built for the AMD Developer Hackathon: ACT II.

## How it works

```text
prompt
  |
  v
Qwen3 embedding (0.6B)
  |
  +--> category classifier --> task-specific threshold and execution lane
  |
  +--> risk classifier -----> probability that local quality will fall short
                                  |
                 +----------------+----------------+
                 |                |                |
             local model      specialist       cloud model
               Gemma         NER / arithmetic     MiniMax
```

Each prompt is embedded once. Two small logistic models run over that embedding: one predicts
the task category and the other estimates whether cloud can rescue a weak local answer. The
router uses a separately calibrated risk threshold for each category, because the same score
does not carry the same meaning for sentiment, code and multi-step reasoning.

Thresholds are selected to use as few cloud calls as possible while meeting a target pass
rate. The trained scikit-learn models are exported to a 76 KB NumPy bundle, so production
routing is just a handful of NumPy operations with no scikit-learn dependency.

## Execution lanes

The runtime uses the fastest suitable path after classification:

| Work | Path |
| --- | --- |
| General local tasks | Gemma 4 E2B Q4 |
| Summarisation | Gemma 4 E2B Q4 |
| Named entities | Minibase NER-Standard plus date parsing |
| Simple arithmetic | Safe deterministic evaluator |
| Code generation and debugging | Cloud |
| Other high-risk prompts | Cloud |

Gemma handles general tasks and summaries while NER runs on a second local lane. Cloud
requests run concurrently in the background. Both local lanes start with one CPU thread;
when the NER lane finishes, Gemma takes both threads. Models are released between
phases to keep memory use within the submission limit.

### Deadline-aware scheduling

Local work is ordered using the prompt length and its risk relative to the learned category
threshold. Long prompts with plenty of threshold headroom run first because they offer strong
cloud-token savings. Borderline prompts stay near the back of the queue, where they are cheaper
to hand off if local inference is slower than expected.

The batch has an eight-minute local deadline, measured from startup. Gemma generation stops at
the deadline and the current task, together with any queued local work, moves to the concurrent
cloud pool. Completed local and NER answers are retained, and results are still written in the
original input order. The deadline can be changed with `--local-deadline-seconds` or disabled
with a value of `0`.

## Training the router

We sampled 8,000 real user questions from a
[cleaned, categorised mirror](https://huggingface.co/datasets/OpenLeecher/lmsys_chat_1m_clean)
of [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m). The sampler streams
1,000 questions for each of eight categories: factual knowledge, mathematical reasoning,
sentiment classification, summarisation, named-entity recognition, code debugging, logical
reasoning and code generation.

Gemma 4 E2B and MiniMax M3 answered every question. GLM-5.2 then judged each pair, with the
model names hidden and answer order balanced, rating their quality and choosing the stronger
response. A question is labelled as needing cloud when Gemma falls below the quality floor
and MiniMax clears it.

The answered and judged training tasks are archived on Hugging Face at
[cjami/gullivers-router-training-data](https://huggingface.co/datasets/cjami/gullivers-router-training-data).

We use 6,400 rows for training, 800 for threshold calibration and 800 for held-out testing.
The pipeline is resumable at every stage and exports both `router.npz` and
`router_metrics.json`, keeping the model and its operating-point history together.

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and a Fireworks API key.

```sh
make setup
cp .env.example .env
```

Add `HF_TOKEN` and `FIREWORKS_API_KEY` to `.env`. Models download from Hugging Face on first
use. Defaults match the CPU-only submission environment; local GPU acceleration can be enabled
through the role-specific environment settings.

## Running it

Run a batch using the hackathon file contract:

```sh
uv run gullivers-router run \
  --input /input/tasks.json \
  --output /output/results.json \
  --router-weights artifacts/training/router.npz
```

Run and judge the included practice set with:

```sh
make practice
```

Useful commands:

| Command | Purpose |
| --- | --- |
| `uv run gullivers-router run` | Route a batch and generate answers. |
| `uv run gullivers-router run --classify-only` | Inspect routes without generating answers. |
| `uv run gullivers-router score-practice` | Grade practice answers as pass or fail. |
| `uv run gullivers-router train` | Build the resumable training dataset. |
| `uv run gullivers-router train-router` | Fit, calibrate and export the router. |

## Docker

The Dockerfile builds a self-contained CPU-only `linux/amd64` image with all three quantised
GGUF models and the calibrated router bundle.

```sh
docker buildx build --platform linux/amd64 --load -t gullivers-router:local .
docker run --rm --memory=4g --cpus=2 \
  -e FIREWORKS_API_KEY \
  -v "$PWD/examples/practice_tasks.json:/input/tasks.json:ro" \
  -v "$PWD/outputs:/output" \
  gullivers-router:local
```

## Development

```sh
make test
make lint
make format
```
