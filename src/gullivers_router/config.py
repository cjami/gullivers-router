"""Environment-driven configuration for each model role."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from gullivers_router.inference.base import Provider
from gullivers_router.model_selection import select_model

if TYPE_CHECKING:
    from collections.abc import Mapping

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
QWEN_ROUTING_PREFIX = ""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """One role's binding to a provider, model, and source."""

    provider: Provider
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    repo_id: str | None = None
    filename: str | None = None
    timeout_seconds: float | None = None
    n_ctx: int | None = None
    n_gpu_layers: int | None = None
    flash_attn: bool | None = None
    enable_thinking: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    n_threads: int | None = None
    model_root: Path | None = None
    pooling_type: str | None = None
    input_prefix: str | None = None


@dataclass(frozen=True, slots=True)
class _RoleDefaults:
    provider: Provider
    model: str | None = None
    base_url: str | None = None
    repo_id: str | None = None
    filename: str | None = None
    n_ctx: int | None = None
    n_gpu_layers: int | None = None
    flash_attn: bool | None = None
    enable_thinking: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    n_threads: int | None = None
    model_root: Path | None = None
    pooling_type: str | None = None
    input_prefix: str | None = None


_ROLE_DEFAULTS: dict[str, _RoleDefaults] = {
    "LOCAL": _RoleDefaults(
        provider=Provider.LLAMA,
        repo_id="google/gemma-4-E2B-it-qat-q4_0-gguf",
        filename="gemma-4-E2B_q4_0-it.gguf",
        n_ctx=2048,
        n_gpu_layers=-1,
        flash_attn=True,
        enable_thinking=False,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        model_root=Path("models"),
    ),
    "EMBEDDING": _RoleDefaults(
        provider=Provider.LLAMA,
        repo_id="Qwen/Qwen3-Embedding-0.6B-GGUF",
        filename="Qwen3-Embedding-0.6B-Q8_0.gguf",
        n_ctx=2048,
        pooling_type="last",
        input_prefix=QWEN_ROUTING_PREFIX,
        model_root=Path("models"),
    ),
    "CLOUD": _RoleDefaults(
        provider=Provider.FIREWORKS,
        model="accounts/fireworks/models/minimax-m3",
        base_url=FIREWORKS_BASE_URL,
    ),
    "JUDGE": _RoleDefaults(
        provider=Provider.FIREWORKS,
        model="accounts/fireworks/models/glm-5p2",
        base_url=FIREWORKS_BASE_URL,
    ),
}


def _role_config(env: Mapping[str, str], role: str) -> ModelConfig:
    defaults = _ROLE_DEFAULTS[role]

    def value(key: str, default: str | None) -> str | None:
        return env.get(f"{role}_{key}", default)

    provider_name = value("PROVIDER", defaults.provider.value)
    provider = Provider(provider_name)
    base_url = value("BASE_URL", defaults.base_url)
    if provider == Provider.FIREWORKS and env.get("FIREWORKS_BASE_URL"):
        base_url = env["FIREWORKS_BASE_URL"]
    model = _resolve_model(env, role, provider, value("MODEL", defaults.model))
    return ModelConfig(
        provider=provider,
        model=model,
        api_key=env.get(f"{role}_API_KEY") or env.get("FIREWORKS_API_KEY"),
        base_url=base_url,
        repo_id=value("REPO_ID", defaults.repo_id),
        filename=value("FILENAME", defaults.filename),
        timeout_seconds=_optional_float(value("TIMEOUT_SECONDS", None)),
        n_ctx=_optional_int(value("N_CTX", _string(defaults.n_ctx))),
        n_gpu_layers=_optional_int(value("N_GPU_LAYERS", _string(defaults.n_gpu_layers))),
        flash_attn=_optional_bool(value("FLASH_ATTN", _string(defaults.flash_attn))),
        enable_thinking=_optional_bool(value("ENABLE_THINKING", _string(defaults.enable_thinking))),
        temperature=_optional_float(value("TEMPERATURE", _string(defaults.temperature))),
        top_p=_optional_float(value("TOP_P", _string(defaults.top_p))),
        top_k=_optional_int(value("TOP_K", _string(defaults.top_k))),
        max_tokens=_optional_int(value("MAX_TOKENS", _string(defaults.max_tokens))),
        n_threads=_optional_int(value("N_THREADS", _string(defaults.n_threads))),
        model_root=_optional_path(value("MODEL_ROOT", _string(defaults.model_root))),
        pooling_type=value("POOLING_TYPE", defaults.pooling_type),
        input_prefix=value("INPUT_PREFIX", defaults.input_prefix),
    )


def _resolve_model(env: Mapping[str, str], role: str, provider: Provider, configured: str | None) -> str | None:
    """Override the cloud model from ``ALLOWED_MODELS`` when the harness pins the allowlist."""
    if role != "CLOUD" or provider != Provider.FIREWORKS:
        return configured
    allowed = _allowed_models(env)
    return select_model(allowed) if allowed else configured


def _allowed_models(env: Mapping[str, str]) -> list[str]:
    raw = env.get("ALLOWED_MODELS")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _optional_float(raw: str | None) -> float | None:
    return float(raw) if raw else None


def _optional_int(raw: str | None) -> int | None:
    return int(raw) if raw else None


def _optional_bool(raw: str | None) -> bool | None:
    if raw is None or raw == "":
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    msg = f"invalid boolean value: {raw}"
    raise ValueError(msg)


def _optional_path(raw: str | None) -> Path | None:
    return Path(raw) if raw else None


def _string(value: object | None) -> str | None:
    return str(value) if value is not None else None


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved model bindings for every role."""

    hf_token: str | None
    local: ModelConfig
    embedding: ModelConfig
    cloud: ModelConfig
    judge: ModelConfig

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Load settings, reading ``.env`` when no explicit mapping is given."""
        if env is None:
            load_dotenv()
            env = os.environ
        return cls(
            hf_token=env.get("HF_TOKEN"),
            local=_role_config(env, "LOCAL"),
            embedding=_role_config(env, "EMBEDDING"),
            cloud=_role_config(env, "CLOUD"),
            judge=_role_config(env, "JUDGE"),
        )
