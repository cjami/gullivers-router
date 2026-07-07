# Gulliver's Router

## Project Description

A low-cost routing system that splits traffic between a free local model and a heavy-duty cloud AI based on query difficulty.

## Project Structure

```
src/gullivers_router/
  config.py            Per-role model config (Settings) loaded from .env.
  inference/           Provider-agnostic clients shared by runtime and training.
    base.py            ChatModel / EmbeddingModel / BatchChatModel protocols.
    truncation.py      Head-and-tail token truncation for embeddings.
    llama_cpp.py       Local GGUF backends (auto-downloaded from HuggingFace).
    openai_compat.py   Chat over any OpenAI-compatible endpoint (e.g. Fireworks).
    fireworks_batch.py Batch chat via the Fireworks Batch API (training only).
    factory.py         Build the backend for a role's configured provider.
  router/              Runtime path: single-call routing (skeleton).
  training/            Offline path: dataset load -> batch generate -> combine (skeleton).
    dataset.py         Prompt sourcing/shaping from OpenLeecher/lmsys_chat_1m_clean.
    combine.py         Align local + cloud batch results into judge-ready pairs.
```

Roles (`local`, `embedding`, `cloud`, `judge`) are each independently configurable to any
provider via `.env`. The runtime router uses single calls; training uses the batch path.

## Development Workflow

- Always use modern Python practices for Python 3.12+.
- Use TDD where appropriate to keep a considered design and protect key behaviours.
- Do not test content, configurations or anything that is likely to change by design.
- Tidy-up and refactor after changes - make sure to follow SOLID principles.
- Run `make lint` and `make test` after changes.
- Use `uv run` for Python commands.
- Do not commit changes unless instructed.

## Comments

- Keep all comments concise, clear, and suitable for inclusion in final production.
- Only use comments when the intent cannot be explained through thoughtful naming or code structure.
