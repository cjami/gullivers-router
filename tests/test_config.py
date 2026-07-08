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


def test_allowed_models_is_ignored_for_now():
    settings = Settings.from_env({"ALLOWED_MODELS": "allowed-a,allowed-b", "CLOUD_MODEL": "configured-model"})

    assert settings.cloud.model == "configured-model"


def test_role_timeout_seconds_can_be_configured():
    settings = Settings.from_env({"CLOUD_TIMEOUT_SECONDS": "45.5"})

    assert settings.cloud.timeout_seconds == 45.5


def test_missing_key_raises_only_when_role_is_built():
    settings = Settings.from_env({"CLOUD_PROVIDER": "openai", "CLOUD_MODEL": "custom"})
    with pytest.raises(ValueError, match="API key"):
        build_chat_model(settings.cloud)
