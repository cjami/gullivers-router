"""Environment-driven configuration for each model role."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from gullivers_router.inference.base import Provider

if TYPE_CHECKING:
    from collections.abc import Mapping

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """One role's binding to a provider, model, and source."""

    provider: Provider
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    repo_id: str | None = None
    filename: str | None = None


@dataclass(frozen=True, slots=True)
class _RoleDefaults:
    provider: Provider
    model: str | None = None
    base_url: str | None = None
    repo_id: str | None = None
    filename: str | None = None


_ROLE_DEFAULTS: dict[str, _RoleDefaults] = {
    "LOCAL": _RoleDefaults(
        provider=Provider.LLAMA,
        repo_id="google/gemma-4-31B-it-qat-q4_0-gguf",
        filename="*q4_0-it.gguf",
    ),
    "EMBEDDING": _RoleDefaults(
        provider=Provider.LLAMA,
        repo_id="ggml-org/embeddinggemma-300M-GGUF",
        filename="*Q8_0.gguf",
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
    return ModelConfig(
        provider=provider,
        model=value("MODEL", defaults.model),
        api_key=env.get(f"{role}_API_KEY") or env.get("FIREWORKS_API_KEY"),
        base_url=base_url,
        repo_id=value("REPO_ID", defaults.repo_id),
        filename=value("FILENAME", defaults.filename),
    )


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
