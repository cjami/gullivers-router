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
    LOCAL_ENABLE_THINKING=false \
    LOCAL_MAX_TOKENS=1024 \
    LOCAL_N_THREADS=2 \
    LOCAL_MODEL_ROOT=/app/models \
    SPECIALIST_PROVIDER=llama \
    SPECIALIST_REPO_ID=unsloth/Qwen3-0.6B-GGUF \
    SPECIALIST_FILENAME=Qwen3-0.6B-Q4_0.gguf \
    SPECIALIST_N_CTX=2048 \
    SPECIALIST_N_GPU_LAYERS=0 \
    SPECIALIST_FLASH_ATTN=false \
    SPECIALIST_ENABLE_THINKING=false \
    SPECIALIST_TEMPERATURE=0.0 \
    SPECIALIST_TOP_P=1.0 \
    SPECIALIST_TOP_K=1 \
    SPECIALIST_MAX_TOKENS=512 \
    SPECIALIST_N_THREADS=2 \
    SPECIALIST_MODEL_ROOT=/app/models \
    EMBEDDING_PROVIDER=llama \
    EMBEDDING_REPO_ID=Qwen/Qwen3-Embedding-0.6B-GGUF \
    EMBEDDING_FILENAME=Qwen3-Embedding-0.6B-Q8_0.gguf \
    EMBEDDING_N_CTX=2048 \
    EMBEDDING_N_GPU_LAYERS=0 \
    EMBEDDING_N_THREADS=2 \
    EMBEDDING_POOLING_TYPE=last \
    EMBEDDING_MODEL_ROOT=/app/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /install /usr/local
COPY models/google/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf \
    /app/models/google/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf
COPY models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf \
    /app/models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf
COPY models/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf \
    /app/models/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf
COPY artifacts/training/router.npz /app/artifacts/training/router.npz

ENTRYPOINT ["gullivers-router", "run", "--input", "/input/tasks.json", "--output", "/output/results.json", "--router-weights", "/app/artifacts/training/router.npz"]
