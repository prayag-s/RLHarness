"""Enterprise provider targets (G13 residual): Azure OpenAI and self-hosted OpenAI-compatible.

Procurement at a large customer rarely accepts "send your data to api.deepseek.com". Azure OpenAI is
the usual answer, and an internal vLLM/TGI endpoint is the other. Both speak a variant of the OpenAI
schema that breaks a naive client — different auth header, deployment in the path, API version as a
query parameter — so the request shapes are asserted here rather than trusted.
"""

from __future__ import annotations

import json

import pytest

from rl_ide.llm.client import (
    LLMConfig,
    OpenAICompatibleLLMClient,
    AzureOpenAILLMClient,
    create_llm_client,
    validate_llm_connection,
)


class _Response:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _fake_httpx(monkeypatch, captured: dict, reply: str = "pong"):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, headers=None, json=None):  # noqa: A002 - httpx signature
            captured["url"] = url
            captured["headers"] = headers or {}
            captured["body"] = json or {}
            return _Response(
                {
                    "choices": [{"message": {"content": reply}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                }
            )

    monkeypatch.setattr("rl_ide.llm.client.httpx.Client", _Client)


def test_azure_uses_the_deployment_in_the_path_and_an_api_key_header(monkeypatch):
    captured: dict = {}
    _fake_httpx(monkeypatch, captured)
    config = LLMConfig(
        provider="azure_openai",
        base_url="https://acme.openai.azure.com",
        model_name="gpt-4o-prod",
        api_key="azure-secret",
    )

    assert AzureOpenAILLMClient(config).generate("hi", system_prompt="be brief") == "pong"

    assert captured["url"] == (
        "https://acme.openai.azure.com/openai/deployments/gpt-4o-prod/chat/completions"
        "?api-version=2024-10-21"
    )
    assert captured["headers"]["api-key"] == "azure-secret"
    assert "Authorization" not in captured["headers"], "Azure uses api-key, not a bearer token"
    assert captured["body"]["messages"][0] == {"role": "system", "content": "be brief"}


def test_azure_accepts_a_full_deployment_url(monkeypatch):
    captured: dict = {}
    _fake_httpx(monkeypatch, captured)
    config = LLMConfig(
        provider="azure_openai",
        base_url="https://acme.openai.azure.com/openai/deployments/gpt-4o-prod",
        model_name="gpt-4o-prod",
        api_key="k",
    )
    AzureOpenAILLMClient(config).generate("hi")
    assert captured["url"].startswith(
        "https://acme.openai.azure.com/openai/deployments/gpt-4o-prod/chat/completions"
    )


def test_azure_without_an_endpoint_explains_what_is_missing():
    with pytest.raises(ValueError) as excinfo:
        AzureOpenAILLMClient(LLMConfig(provider="azure_openai", api_key="k", model_name="d")).generate("hi")
    assert "endpoint" in str(excinfo.value).lower()


def test_azure_without_a_deployment_names_the_field():
    with pytest.raises(ValueError) as excinfo:
        AzureOpenAILLMClient(
            LLMConfig(provider="azure_openai", api_key="k", base_url="https://acme.openai.azure.com")
        ).generate("hi")
    assert "deployment" in str(excinfo.value).lower()


def test_self_hosted_works_without_a_key_and_sends_one_when_configured(monkeypatch):
    captured: dict = {}
    _fake_httpx(monkeypatch, captured)
    config = LLMConfig(
        provider="openai_compatible", base_url="http://vllm.internal:8000/v1", model_name="llama-3-70b"
    )

    assert OpenAICompatibleLLMClient(config).generate("hi") == "pong"
    assert captured["url"] == "http://vllm.internal:8000/v1/chat/completions"
    assert "Authorization" not in captured["headers"]
    assert captured["body"]["model"] == "llama-3-70b"

    keyed = LLMConfig(
        provider="openai_compatible",
        base_url="http://vllm.internal:8000/v1",
        model_name="llama-3-70b",
        api_key="internal-token",
    )
    OpenAICompatibleLLMClient(keyed).generate("hi")
    assert captured["headers"]["Authorization"] == "Bearer internal-token"


def test_self_hosted_without_a_base_url_says_so():
    with pytest.raises(ValueError) as excinfo:
        OpenAICompatibleLLMClient(LLMConfig(provider="openai_compatible", model_name="m")).generate("hi")
    assert "base_url" in str(excinfo.value)


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("azure", "AzureOpenAILLMClient"),
        ("azure-openai", "AzureOpenAILLMClient"),
        ("vllm", "OpenAICompatibleLLMClient"),
        ("self_hosted", "OpenAICompatibleLLMClient"),
        ("openai-compatible", "OpenAICompatibleLLMClient"),
    ],
)
def test_provider_aliases_resolve(alias, expected):
    config = LLMConfig(provider=alias, base_url="http://x/v1", model_name="m", api_key="k")
    assert type(create_llm_client(config)).__name__ == expected


def test_validation_names_the_missing_setting(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    azure = validate_llm_connection(LLMConfig(provider="azure_openai"))
    assert azure["success"] is False
    assert "AZURE_OPENAI_API_KEY" in azure["error"]

    compat = validate_llm_connection(LLMConfig(provider="openai_compatible"))
    assert compat["success"] is False
    assert "base URL" in compat["error"]


def test_config_fills_defaults_from_the_environment(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    config = LLMConfig(provider="azure_openai")
    assert config.base_url == "https://env.openai.azure.com"
    assert config.model_name == "env-deployment"
    assert config.api_key == "env-key"

    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://local:8000/v1")
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "local-model")
    compat = LLMConfig(provider="openai_compatible")
    assert compat.base_url == "http://local:8000/v1"
    assert compat.model_name == "local-model"
