from pathlib import Path

import pytest

from gullivers_router.config import Settings
from gullivers_router.inference.base import Provider
from gullivers_router.inference.factory import build_chat_model


def test_from_env_populates_every_role():
    settings = Settings.from_env({})
    assert settings.local.provider == Provider.LLAMA
    assert settings.embedding.provider == Provider.LLAMA
    assert settings.cloud.provider == Provider.FIREWORKS
    assert settings.judge.provider == Provider.FIREWORKS


def test_role_provider_can_be_overridden():
    env = {"CLOUD_PROVIDER": "openai", "CLOUD_MODEL": "custom", "CLOUD_API_KEY": "k"}
    settings = Settings.from_env(env)
    assert settings.cloud.provider == Provider.OPENAI
    assert settings.cloud.model == "custom"
    assert settings.cloud.api_key == "k"


def test_shared_fireworks_key_is_reused_across_roles():
    settings = Settings.from_env({"FIREWORKS_API_KEY": "shared"})
    assert settings.cloud.api_key == "shared"
    assert settings.judge.api_key == "shared"


def test_fireworks_base_url_overrides_role_and_default_urls():
    settings = Settings.from_env(
        {
            "FIREWORKS_BASE_URL": "https://proxy.example/v1",
            "CLOUD_BASE_URL": "https://role.example/v1",
            "JUDGE_BASE_URL": "https://judge.example/v1",
        }
    )

    assert settings.cloud.base_url == "https://proxy.example/v1"
    assert settings.judge.base_url == "https://proxy.example/v1"


def test_fireworks_base_url_does_not_override_non_fireworks_roles():
    settings = Settings.from_env(
        {
            "FIREWORKS_BASE_URL": "https://proxy.example/v1",
            "CLOUD_PROVIDER": "openai",
            "CLOUD_BASE_URL": "https://openai.example/v1",
            "CLOUD_MODEL": "custom",
            "CLOUD_API_KEY": "key",
        }
    )

    assert settings.cloud.base_url == "https://openai.example/v1"


def test_allowed_models_selects_cloud_model_by_preference():
    settings = Settings.from_env({"ALLOWED_MODELS": "gemma-4-31b-it,minimax-m3,kimi-k2p7-code"})

    assert settings.cloud.model == "minimax-m3"


def test_allowed_models_overrides_configured_cloud_model():
    settings = Settings.from_env({"ALLOWED_MODELS": "minimax-m3", "CLOUD_MODEL": "configured-model"})

    assert settings.cloud.model == "minimax-m3"


def test_allowed_models_falls_back_to_first_when_no_family_matches():
    settings = Settings.from_env({"ALLOWED_MODELS": "mystery-a,mystery-b"})

    assert settings.cloud.model == "mystery-a"


def test_allowed_models_does_not_override_non_fireworks_cloud():
    settings = Settings.from_env(
        {"ALLOWED_MODELS": "minimax-m3", "CLOUD_PROVIDER": "openai", "CLOUD_MODEL": "custom", "CLOUD_API_KEY": "k"}
    )

    assert settings.cloud.model == "custom"


def test_allowed_models_does_not_change_the_training_judge():
    settings = Settings.from_env({"ALLOWED_MODELS": "minimax-m3"})

    assert settings.judge.model == "accounts/fireworks/models/glm-5p2"


def test_role_timeout_seconds_can_be_configured():
    settings = Settings.from_env({"CLOUD_TIMEOUT_SECONDS": "45.5"})

    assert settings.cloud.timeout_seconds == 45.5


def test_llama_runtime_options_can_be_configured():
    settings = Settings.from_env(
        {
            "LOCAL_N_CTX": "2048",
            "LOCAL_N_GPU_LAYERS": "0",
            "LOCAL_FLASH_ATTN": "false",
            "LOCAL_ENABLE_THINKING": "false",
            "LOCAL_TEMPERATURE": "0.7",
            "LOCAL_TOP_P": "0.8",
            "LOCAL_TOP_K": "32",
            "LOCAL_MAX_TOKENS": "1024",
            "LOCAL_N_THREADS": "2",
            "LOCAL_MODEL_ROOT": "/app/models",
        }
    )

    assert settings.local.n_ctx == 2048
    assert settings.local.n_gpu_layers == 0
    assert settings.local.flash_attn is False
    assert settings.local.enable_thinking is False
    assert settings.local.temperature == 0.7
    assert settings.local.top_p == 0.8
    assert settings.local.top_k == 32
    assert settings.local.max_tokens == 1024
    assert settings.local.n_threads == 2
    assert settings.local.model_root == Path("/app/models")


def test_llama_runtime_defaults_keep_training_profile():
    settings = Settings.from_env({})

    assert settings.local.n_ctx == 2048
    assert settings.local.n_gpu_layers == -1
    assert settings.local.flash_attn is True
    assert settings.local.enable_thinking is False
    assert settings.local.temperature == 1.0
    assert settings.local.top_p == 0.95
    assert settings.local.top_k == 64
    assert settings.local.max_tokens is None
    assert settings.local.model_root == Path("models")


def test_judge_disables_reasoning_but_cloud_leaves_it_unset():
    settings = Settings.from_env({})

    assert settings.judge.enable_thinking is False
    assert settings.cloud.enable_thinking is None


def test_invalid_bool_runtime_option_raises():
    with pytest.raises(ValueError, match="invalid boolean"):
        Settings.from_env({"LOCAL_FLASH_ATTN": "maybe"})


def test_missing_key_raises_only_when_role_is_built():
    settings = Settings.from_env({"CLOUD_PROVIDER": "openai", "CLOUD_MODEL": "custom"})
    with pytest.raises(ValueError, match="API key"):
        build_chat_model(settings.cloud)
