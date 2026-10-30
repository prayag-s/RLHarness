"""Unit tests verifying mandatory LLM verification and studio access gating."""

import os
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from rl_ide.api.server import app
from rl_ide.llm.client import (
    LLMConfig,
    _llm_state,
    get_llm_status,
    is_llm_ready,
    is_llm_verified,
    get_active_llm_client,
    verify_and_set_llm_config,
    validate_llm_connection,
)


@pytest.fixture(autouse=True)
def reset_llm_state():
    """Resets global LLM state before and after each test."""
    _llm_state["config"] = None
    _llm_state["client"] = None
    _llm_state["is_verified"] = False
    _llm_state["last_error"] = None
    _llm_state["latency_ms"] = None
    yield
    _llm_state["config"] = None
    _llm_state["client"] = None
    _llm_state["is_verified"] = False
    _llm_state["last_error"] = None
    _llm_state["latency_ms"] = None


def test_unverified_llm_status_and_gating():
    """Unverified state blocks only the model-backed actions, and says why.

    Exploration (upload, listing, reading an arena) is deliberately not gated: a prospective user
    must be able to evaluate the studio before buying a key (G13).
    """
    client = TestClient(app)

    # 1. Check status endpoint returns unverified
    status_resp = client.get("/api/v4/settings/llm")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["is_verified"] is False
    assert status_data["has_api_key"] is False

    # 2. Simulate non-test environment to test mandatory gating
    with patch("rl_ide.llm.client.is_test_mode", return_value=False), \
         patch("rl_ide.api.server.is_llm_ready", return_value=False):
        
        chat_resp = client.post(
            "/api/v4/architect/chat",
            json={"user_prompt": "Build an agent"},
        )
        assert chat_resp.status_code == 403
        # The refusal must be machine-readable, not only prose: the UI switches on the reason.
        assert chat_resp.headers.get("X-RL-Reason") == "llm-key-required"
        assert "needs a model" in chat_resp.json()["detail"]

        start_resp = client.post(
            "/api/v4/sessions/start",
            json={
                "blueprint": {
                    "task_title": "Test",
                    "domain_summary": "Test",
                    "data_source_type": "pure_simulation",
                    "observation_features": ["f1"],
                    "action_labels": ["a1"],
                    "arena_visual_description": "grid",
                    "interactive_mode_description": "play",
                    "success_metric_name": "rate",
                    "target_metric_threshold": 80.0,
                    "max_auto_tune_rounds": 2,
                    "initial_reward_components": [],
                }
            },
        )
        assert start_resp.status_code == 403

        nudge_resp = client.post(
            "/api/v4/sessions/dummy_sess/nudge",
            json={"feedback": "tune higher"},
        )
        assert nudge_resp.status_code == 403


def test_invalid_key_fails_verification():
    """Verify that an invalid API key fails verification and leaves studio gated."""
    client = TestClient(app)

    # Attempt to configure deepseek with bad credentials
    with patch("rl_ide.llm.client.DeepSeekLLMClient.generate", side_effect=Exception("401 Unauthorized: Invalid API Key")):
        resp = client.post(
            "/api/v4/settings/llm",
            json={
                "provider": "deepseek",
                "api_key": "sk-invalid-key-9999",
                "model_name": "deepseek-chat",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["is_verified"] is False
        assert "401" in data["error"] or "Unauthorized" in data["error"]
        assert _llm_state["is_verified"] is False


def test_valid_key_verifies_and_activates_globally():
    """Verify that a successful connection activates the LLM and unblocks studio."""
    client = TestClient(app)

    with patch("rl_ide.llm.client.DeepSeekLLMClient.generate", return_value="pong"):
        resp = client.post(
            "/api/v4/settings/llm",
            json={
                "provider": "deepseek",
                "api_key": "sk-valid-key-12345678",
                "model_name": "deepseek-chat",
                "base_url": "https://api.deepseek.com/v1",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["is_verified"] is True
        assert data["provider"] == "deepseek"
        assert _llm_state["is_verified"] is True

        # Check status endpoint reflects verified state and masked key
        status_resp = client.get("/api/v4/settings/llm")
        assert status_resp.status_code == 200
        s_data = status_resp.json()
        assert s_data["is_verified"] is True
        assert s_data["provider"] == "deepseek"
        assert s_data["has_api_key"] is True
        assert s_data["masked_key"].startswith("sk-v")
        assert s_data["masked_key"].endswith("5678")


def test_fallback_rejected_in_production_mode():
    """Verify that Offline Fallback cannot be activated as verified provider in production."""
    with patch("rl_ide.llm.client.is_test_mode", return_value=False):
        config = LLMConfig(provider="fallback")
        success, msg, res = verify_and_set_llm_config(config, save_to_disk=False)
        assert success is False
        assert res["is_verified"] is False
        assert "Offline Fallback cannot be used" in res["error"]


def test_offline_test_mode_never_reaches_a_provider(monkeypatch):
    """A regression run must stay free and hermetic even when a real key is present."""
    from rl_ide.llm.client import FallbackCodeSynthesizer, init_llm_state

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-a-real-looking-key-12345678")
    provider_calls: list[str] = []
    monkeypatch.setattr(
        "rl_ide.llm.client.DeepSeekLLMClient.generate",
        lambda self, *args, **kwargs: provider_calls.append("call") or "pong",
    )

    init_llm_state()

    assert provider_calls == [], "startup must not probe a paid provider during tests"
    assert _llm_state["is_verified"] is False
    assert is_llm_ready() is True  # offline studio access stays unlocked
    assert isinstance(get_active_llm_client(), FallbackCodeSynthesizer)


def test_live_llm_opt_in_restores_provider_verification(monkeypatch, tmp_path):
    """RL_IDE_LIVE_LLM=1 is the only way back to the real provider inside pytest."""
    from rl_ide.llm.client import init_llm_state

    monkeypatch.setenv("RL_IDE_LIVE_LLM", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-live-key-12345678")
    monkeypatch.setattr("rl_ide.llm.client.DeepSeekLLMClient.generate", lambda self, *a, **k: "pong")
    monkeypatch.setattr(
        "rl_ide.llm.client.get_config_persistence_path", lambda: tmp_path / "absent.json"
    )

    init_llm_state()

    assert _llm_state["is_verified"] is True
    assert _llm_state["config"].provider == "deepseek"
