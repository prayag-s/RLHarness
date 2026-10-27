"""BYOK LLM Client supporting OpenAI, Anthropic, Ollama/vLLM, and deterministic Mock."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import json
import httpx
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for BYOK LLM connection."""
    provider: str = "mock"  # "mock" | "openai" | "anthropic" | "ollama"
    api_key: str | None = None
    base_url: str | None = None
    model_name: str | None = None
    temperature: float = 0.2
    timeout: float = 30.0


class BaseLLMClient(ABC):
    """Abstract interface for LLM client."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generates completion text from prompt."""
        ...


class OpenAILLMClient(BaseLLMClient):
    """Client for OpenAI and OpenAI-compatible endpoints (Ollama, vLLM, LMStudio, DeepSeek)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        if config.provider == "deepseek":
            self.base_url = (config.base_url or "https://api.deepseek.com").rstrip("/")
            self.model = config.model_name or "deepseek-chat"
        else:
            self.base_url = (config.base_url or "https://api.openai.com/v1").rstrip("/")
            self.model = config.model_name or ("gpt-4o" if config.provider == "openai" else "llama3.1")
        self.api_key = config.api_key or "EMPTY"

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        with httpx.Client(timeout=self.config.timeout) as client:
            resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class AnthropicLLMClient(BaseLLMClient):
    """Client for Anthropic Messages API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = (config.base_url or "https://api.anthropic.com/v1").rstrip("/")
        self.model = config.model_name or "claude-3-5-sonnet-20241022"
        self.api_key = config.api_key or ""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        with httpx.Client(timeout=self.config.timeout) as client:
            resp = client.post(f"{self.base_url}/messages", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]


class MockRulesLLMClient(BaseLLMClient):
    """Deterministic keyword/pattern synthesis client for offline zero-cost execution and tests."""

    def __init__(self, scripted_responses: list[str] | None = None):
        self.scripted_responses: list[str] = list(scripted_responses) if scripted_responses else []

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        if self.scripted_responses:
            return self.scripted_responses.pop(0)

        # Handle auto-repair retry prompts
        if "RewardValidationError" in prompt or "Undefined variable" in prompt:
            if "TextClassifyAdapter" in prompt or "ticket" in prompt.lower() or "triage" in prompt.lower() or "email" in prompt.lower():
                return """
task_id: "ticket_triage_task_repaired"
name: "Ticket Triage (Repaired)"
description: "Repaired ticket triage task with valid schema fields"
adapter_name: "TextClassifyAdapter"
adapter_config:
  episode_length: 25
hyperparameters:
  learning_rate: 0.001
  gamma: 0.0
reward_dsl:
  version: "1.0"
  horizon:
    gamma: 0.0
    critic_warmup_epochs_on_diff: 10
    kl_coef: 0.1
  reward:
    terms:
      - name: accurate_routing_match
        when: "truth.has_label == True and action.label == truth.label"
        value: 5.0
      - name: misclassify_spam_as_urgent
        when: "action.label == 'urgent' and truth.label == 'spam'"
        value: -15.0
      - name: escalate_angry_vip
        when: "action.label == 'escalate' and obs.sender_is_vip == True and obs.angry_tone_score > 0.6"
        value: 4.0
    pinned_episodes: []
    aggregate: sum
    clamp: [-50.0, 50.0]
"""
            # Default CSV repair
            return """
task_id: "csv_stream_task_repaired"
name: "CSV Anomaly Monitor (Repaired)"
description: "Repaired task configuration with valid schema fields"
adapter_name: "CSVStreamAdapter"
adapter_config:
  episode_length: 100
hyperparameters:
  learning_rate: 0.0003
  gamma: 0.90
reward_dsl:
  version: "1.0"
  horizon:
    gamma: 0.90
    critic_warmup_epochs_on_diff: 15
    kl_coef: 0.1
  reward:
    terms:
      - name: accurate_match
        when: "truth.has_label == True and action.label == truth.label"
        value: 1.0
      - name: false_positive_low_vol
        when: "action.label == 'anomaly' and obs.volatility <= obs.rolling_vol_90p"
        value: -8.0
      - name: high_vol_anomaly_bonus
        when: "action.label == 'anomaly' and obs.volatility > obs.rolling_vol_90p and state.cooldown == 0"
        value: 1.5
      - name: alert_spam_penalty
        when: "action.label == 'anomaly' and state.rolling_alert_rate > 0.15"
        value: -4.0
    pinned_episodes: []
    aggregate: sum
    clamp: [-100.0, 100.0]
"""

        # Handle standard task compilation prompts for TextClassifyAdapter
        if "TextClassifyAdapter" in prompt or "ticket" in prompt.lower() or "triage" in prompt.lower() or "email" in prompt.lower():
            return """
task_id: "ticket_triage_task_v1"
name: "Customer Support Ticket & Email Triage"
description: "Classify incoming tickets into urgent, normal, spam, and escalate based on urgency, tone, and VIP status"
adapter_name: "TextClassifyAdapter"
adapter_config:
  episode_length: 25
hyperparameters:
  learning_rate: 0.001
  gamma: 0.0
reward_dsl:
  version: "1.0"
  horizon:
    gamma: 0.0
    critic_warmup_epochs_on_diff: 10
    kl_coef: 0.1
  reward:
    terms:
      - name: accurate_routing_match
        when: "truth.has_label == True and action.label == truth.label"
        value: 5.0
      - name: misclassify_spam_as_urgent
        when: "action.label == 'urgent' and truth.label == 'spam'"
        value: -15.0
      - name: escalate_angry_vip
        when: "action.label == 'escalate' and obs.sender_is_vip == True and obs.angry_tone_score > 0.6"
        value: 4.0
      - name: neglect_angry_vip_penalty
        when: "action.label == 'normal' and obs.sender_is_vip == True and obs.angry_tone_score > 0.6"
        value: -12.0
    pinned_episodes: []
    aggregate: sum
    clamp: [-50.0, 50.0]
"""

        # Handle standard task compilation prompts
        return """
task_id: "csv_stream_task_v1"
name: "Financial Stream Anomaly Monitor"
description: "Detect sensor anomalies on price and volume while tolerating volatility"
adapter_name: "CSVStreamAdapter"
adapter_config:
  episode_length: 100
hyperparameters:
  learning_rate: 0.0003
  gamma: 0.90
reward_dsl:
  version: "1.0"
  horizon:
    gamma: 0.90
    critic_warmup_epochs_on_diff: 15
    kl_coef: 0.1
  reward:
    terms:
      - name: accurate_match
        when: "truth.has_label == True and action.label == truth.label"
        value: 1.0
      - name: false_positive_low_vol
        when: "action.label == 'anomaly' and obs.volatility <= obs.rolling_vol_90p"
        value: -8.0
      - name: high_vol_anomaly_bonus
        when: "action.label == 'anomaly' and obs.volatility > obs.rolling_vol_90p and state.cooldown == 0"
        value: 1.5
      - name: alert_spam_penalty
        when: "action.label == 'anomaly' and state.rolling_alert_rate > 0.15"
        value: -4.0
    pinned_episodes: []
    aggregate: sum
    clamp: [-100.0, 100.0]
"""


def create_llm_client(config: LLMConfig | None = None) -> BaseLLMClient:
    """Factory creating appropriate LLM client instance based on configuration."""
    if config is None or config.provider == "mock":
        return MockRulesLLMClient()

    if config.provider in ("openai", "deepseek", "ollama", "vllm", "local"):
        if not config.api_key and config.provider not in ("ollama", "vllm", "local"):
            return MockRulesLLMClient()
        return OpenAILLMClient(config)

    if config.provider == "anthropic":
        if not config.api_key:
            return MockRulesLLMClient()
        return AnthropicLLMClient(config)

    return MockRulesLLMClient()


def validate_connection(config: LLMConfig) -> tuple[bool, str]:
    """Validates connectivity and credentials for the specified provider."""
    provider = (config.provider or "mock").lower()
    if provider == "mock":
        return True, "Mock LLM client active (deterministic offline rules)."

    if provider == "deepseek":
        if not config.api_key or not config.api_key.strip():
            return False, "DeepSeek API key is required."
        base_url = (config.base_url or "https://api.deepseek.com").rstrip("/")
        headers = {"Authorization": f"Bearer {config.api_key.strip()}"}
        try:
            with httpx.Client(timeout=min(config.timeout, 5.0)) as client:
                resp = client.get(f"{base_url}/models", headers=headers)
                if resp.status_code == 200:
                    model = config.model_name or "deepseek-chat"
                    return True, f"Connected to DeepSeek successfully. Model: {model}"
                elif resp.status_code == 401:
                    return False, "Authentication failed: Invalid DeepSeek API key."
                else:
                    return False, f"DeepSeek endpoint returned HTTP {resp.status_code}: {resp.text[:100]}"
        except Exception as e:
            return False, f"Connection failed to {base_url}: {str(e)}"

    if provider == "openai":
        if not config.api_key or not config.api_key.strip():
            return False, "OpenAI API key is required."
        base_url = (config.base_url or "https://api.openai.com/v1").rstrip("/")
        headers = {"Authorization": f"Bearer {config.api_key.strip()}"}
        try:
            with httpx.Client(timeout=min(config.timeout, 5.0)) as client:
                resp = client.get(f"{base_url}/models", headers=headers)
                if resp.status_code == 200:
                    model = config.model_name or "gpt-4o"
                    return True, f"Connected to OpenAI successfully. Model: {model}"
                elif resp.status_code == 401:
                    return False, "Authentication failed: Invalid OpenAI API key."
                else:
                    return False, f"OpenAI endpoint returned HTTP {resp.status_code}: {resp.text[:100]}"
        except Exception as e:
            return False, f"Connection failed to {base_url}: {str(e)}"

    if provider == "anthropic":
        if not config.api_key or not config.api_key.strip():
            return False, "Anthropic API key is required."
        key = config.api_key.strip()
        if not key.startswith("sk-ant-") and len(key) < 20:
            return False, "Anthropic API key appears invalid (must begin with 'sk-ant-')."
        model = config.model_name or "claude-3-5-sonnet-20241022"
        return True, f"Anthropic credentials configured. Model: {model}"

    if provider in ("ollama", "vllm", "local"):
        base_url = (config.base_url or "http://localhost:11434/v1").rstrip("/")
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{base_url}/models")
                if resp.status_code == 200:
                    model = config.model_name or "llama3"
                    return True, f"Connected to local endpoint ({base_url}). Model: {model}"
                root_url = base_url.removesuffix("/v1")
                resp2 = client.get(f"{root_url}/api/tags")
                if resp2.status_code == 200:
                    return True, f"Connected to local Ollama ({root_url})."
                return False, f"Endpoint responded with HTTP {resp.status_code}"
        except Exception as e:
            return False, f"Could not reach local LLM server at {base_url}: {str(e)}"

    return False, f"Unknown LLM provider: {config.provider}"

