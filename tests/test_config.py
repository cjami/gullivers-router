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


def test_missing_key_raises_only_when_role_is_built():
    settings = Settings.from_env({"CLOUD_PROVIDER": "openai", "CLOUD_MODEL": "custom"})
    with pytest.raises(ValueError, match="API key"):
        build_chat_model(settings.cloud)
