FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# Install the package in a single pass, pulling llama-cpp-python as a prebuilt CPU wheel
# from the abetlen index. No source compilation, so the image needs no build toolchain.
RUN python -m pip install --upgrade pip \
    && python -m pip install --prefix=/install \
       --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
       --only-binary=llama-cpp-python \
       .

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /install /usr/local
COPY models/google/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf \
    /app/models/google/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf
COPY models/Minibase/NER-Standard/model.gguf \
    /app/models/Minibase/NER-Standard/model.gguf
COPY models/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf \
    /app/models/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf
COPY artifacts/training/router.npz /app/artifacts/training/router.npz

ENTRYPOINT ["gullivers-router", "run", "--input", "/input/tasks.json", "--output", "/output/results.json", "--router-weights", "/app/artifacts/training/router.npz"]
