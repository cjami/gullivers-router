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

ENV PYTHONUNBUFFERED=1 \
    LOCAL_PROVIDER=llama \
    LOCAL_REPO_ID=google/gemma-4-E2B-it-qat-q4_0-gguf \
    LOCAL_FILENAME=gemma-4-E2B_q4_0-it.gguf \
    LOCAL_N_CTX=2048 \
    LOCAL_N_GPU_LAYERS=0 \
    LOCAL_FLASH_ATTN=false \
    LOCAL_N_THREADS=2 \
    LOCAL_MODEL_ROOT=/app/models \
    EMBEDDING_PROVIDER=llama \
    EMBEDDING_REPO_ID=ggml-org/embeddinggemma-300M-GGUF \
    EMBEDDING_FILENAME=*Q8_0.gguf \
    EMBEDDING_N_CTX=2048 \
    EMBEDDING_N_GPU_LAYERS=0 \
    EMBEDDING_N_THREADS=2 \
    EMBEDDING_MODEL_ROOT=/app/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /install /usr/local
COPY models/google/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf \
    /app/models/google/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf
COPY models/ggml-org/embeddinggemma-300M-GGUF/ \
    /app/models/ggml-org/embeddinggemma-300M-GGUF/
COPY artifacts/training/router.npz /app/artifacts/training/router.npz

ENTRYPOINT ["gullivers-router", "run", "--input", "/input/tasks.json", "--output", "/output/results.json", "--router-weights", "/app/artifacts/training/router.npz"]
