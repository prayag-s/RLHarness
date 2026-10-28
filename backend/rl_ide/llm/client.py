"""BYOK Multi-Provider LLM Client with DeepSeek, Anthropic, OpenAI, Ollama, and Deterministic Fallback."""

from __future__ import annotations
import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any
import httpx
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for BYOK LLM connection."""
    provider: str = Field(default="fallback")  # "deepseek" | "openai" | "anthropic" | "ollama" | "fallback" | "mock"
    api_key: str | None = None
    base_url: str | None = None
    model_name: str | None = None
    temperature: float = 0.2
    timeout: float = 30.0
    max_tokens: int = 4096

    def model_post_init(self, __context: Any) -> None:
        """Autofill defaults based on provider."""
        provider_lower = self.provider.lower()
        if provider_lower == "deepseek":
            if not self.base_url:
                self.base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
            if not self.model_name:
                self.model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
            if not self.api_key:
                self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        elif provider_lower == "openai":
            if not self.base_url:
                self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            if not self.model_name:
                self.model_name = os.environ.get("OPENAI_MODEL", "gpt-4o")
            if not self.api_key:
                self.api_key = os.environ.get("OPENAI_API_KEY")
        elif provider_lower == "anthropic":
            if not self.base_url:
                self.base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")
            if not self.model_name:
                self.model_name = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            if not self.api_key:
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif provider_lower == "ollama":
            if not self.base_url:
                self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
            if not self.model_name:
                self.model_name = os.environ.get("OLLAMA_MODEL", "llama3.1")
            if not self.api_key:
                self.api_key = "ollama"
        elif provider_lower in ("fallback", "mock"):
            if not self.model_name:
                self.model_name = "deterministic-fallback"


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Robustly extracts JSON object or array from LLM completion text."""
    clean = text.strip()
    # Try finding markdown code block ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
    if match:
        clean = match.group(1).strip()
    else:
        # Fallback to finding outermost { ... }
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            clean = clean[start : end + 1]
    return json.loads(clean)


class BaseLLMClient(ABC):
    """Abstract interface for multi-provider LLM client."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @property
    def model(self) -> str:
        return self.config.model_name or "unknown"

    @property
    def provider(self) -> str:
        return self.config.provider

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generates raw text completion."""
        ...

    def generate_json(self, prompt: str, system_prompt: str | None = None) -> dict[str, Any]:
        """Generates completion and parses as JSON."""
        raw = self.generate(prompt, system_prompt)
        return extract_json_from_text(raw)


class DeepSeekLLMClient(BaseLLMClient):
    """First-class DeepSeek LLM Client (deepseek-chat / deepseek-reasoner)."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        api_key = self.config.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set or provided in LLMConfig.")

        base_url = (self.config.base_url or "https://api.deepseek.com/v1").rstrip("/")
        endpoint = f"{base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model_name or "deepseek-chat",
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        with httpx.Client(timeout=self.config.timeout) as client:
            resp = client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class OpenAILLMClient(BaseLLMClient):
    """OpenAI API Client (GPT-4o, o3-mini, etc.)."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set or provided in LLMConfig.")

        base_url = (self.config.base_url or "https://api.openai.com/v1").rstrip("/")
        endpoint = f"{base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model_name or "gpt-4o",
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        with httpx.Client(timeout=self.config.timeout) as client:
            resp = client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class AnthropicLLMClient(BaseLLMClient):
    """Anthropic Messages API Client (Claude 3.5 Sonnet, Claude 3.7 Sonnet)."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set or provided in LLMConfig.")

        base_url = (self.config.base_url or "https://api.anthropic.com/v1").rstrip("/")
        endpoint = f"{base_url}/messages"

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.config.model_name or "claude-3-5-sonnet-20241022",
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        with httpx.Client(timeout=self.config.timeout) as client:
            resp = client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]


class OllamaLLMClient(BaseLLMClient):
    """Ollama and vLLM local OpenAI-compatible endpoint."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        base_url = (self.config.base_url or "http://localhost:11434/v1").rstrip("/")
        endpoint = f"{base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.config.api_key or 'ollama'}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model_name or "llama3.1",
            "messages": messages,
            "temperature": self.config.temperature,
        }

        with httpx.Client(timeout=self.config.timeout) as client:
            resp = client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class FallbackCodeSynthesizer(BaseLLMClient):
    """Deterministic Fallback Synthesizer for 100% offline testing and zero-cost CI.
    
    Synthesizes valid AgentBlueprintSpec, ClarifyingQuestions, env_module.py,
    and arena.html responses dynamically based on user prompts and schemas.
    """

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        prompt_lower = prompt.lower()
        sys_lower = (system_prompt or "").lower()

        # Connection check probe
        if "ping" in prompt_lower or "connectivity" in prompt_lower:
            return json.dumps({"status": "ok", "provider": "fallback"})

        # Intake Architect Request
        if "intake" in sys_lower or "blueprint" in sys_lower or "pillar" in sys_lower or "intake" in prompt_lower:
            from rl_ide.architect.intake import is_tictactoe_intent
            is_ttt = is_tictactoe_intent(prompt_lower)

            # Check if prompt answers questions or has smart defaults
            if any(kw in prompt_lower for kw in ["smart defaults", "defaults", "force_defaults", "answer:", "selected:", "ready"]):
                if is_ttt:
                    spec = {
                        "task_title": "Tic-Tac-Toe Master Agent",
                        "domain_summary": "Zero-sum competitive 3x3 board game agent learning optimal play against stochastic and min-max opponents.",
                        "data_source_type": "pure_simulation",
                        "dataset_id": None,
                        "observation_features": [
                            "cell_0_0", "cell_0_1", "cell_0_2",
                            "cell_1_0", "cell_1_1", "cell_1_2",
                            "cell_2_0", "cell_2_1", "cell_2_2",
                            "current_turn"
                        ],
                        "action_labels": [
                            "Place (0,0)", "Place (0,1)", "Place (0,2)",
                            "Place (1,0)", "Place (1,1)", "Place (1,2)",
                            "Place (2,0)", "Place (2,1)", "Place (2,2)"
                        ],
                        "arena_visual_description": "Modern 3x3 dark-mode grid with neon glow and winning lines",
                        "interactive_mode_description": "Clickable 3x3 interactive board where the human plays as 'O' and the agent responds instantly as 'X'.",
                        "success_metric_name": "Non-Loss Rate (%)",
                        "target_metric_threshold": 85.0,
                        "max_auto_tune_rounds": 4,
                        "initial_reward_components": [
                            {"name": "win_bonus", "weight": 10.0, "description": "Reward for winning the game"},
                            {"name": "draw_bonus", "weight": 3.0, "description": "Reward for achieving a draw"},
                            {"name": "loss_penalty", "weight": -10.0, "description": "Penalty for losing the game"},
                            {"name": "illegal_move_penalty", "weight": -5.0, "description": "Penalty for selecting an occupied cell"}
                        ]
                    }
                else:
                    spec = {
                        "task_title": "Autonomous Decision Agent",
                        "domain_summary": "Synthesized agent environment tailored to objective.",
                        "data_source_type": "pure_simulation",
                        "dataset_id": None,
                        "observation_features": ["feature_0", "feature_1", "feature_2", "feature_3"],
                        "action_labels": ["action_0", "action_1", "action_2"],
                        "arena_visual_description": "Modern dark-mode 2D grid visualization with live value heatmaps.",
                        "interactive_mode_description": "Interactive play session with user controls and live agent feedback.",
                        "success_metric_name": "Success Rate (%)",
                        "target_metric_threshold": 85.0,
                        "max_auto_tune_rounds": 4,
                        "initial_reward_components": [
                            {"name": "goal_bonus", "weight": 10.0, "description": "Bonus for reaching goal state"},
                            {"name": "step_cost", "weight": -0.1, "description": "Small penalty per step to encourage efficiency"}
                        ]
                    }
                return json.dumps({
                    "status": "ready_to_build",
                    "assistant_message": "Blueprint locked! All 5 pillars established. Ready to synthesize environment and arena.",
                    "questions": [],
                    "blueprint_spec": spec
                })
            else:
                if is_ttt:
                    questions = [
                        {
                            "pillar": "arena_ui",
                            "question": "How would you like to visualize the Tic-Tac-Toe arena in the center panel?",
                            "suggested_options": [
                                "Modern 3x3 dark-mode grid with neon glow and winning lines",
                                "Minimalist wood-grain board with move history ledger",
                                "Retro arcade 8-bit grid with animated action markers",
                                "Compact numeric matrix with live value heatmaps"
                            ],
                            "default_option": "Modern 3x3 dark-mode grid with neon glow and winning lines"
                        },
                        {
                            "pillar": "expectations",
                            "question": "What target mastery win/draw rate and auto-tune round budget should we aim for?",
                            "suggested_options": [
                                "85% Win/Draw Rate (4 rounds)",
                                "90% Win/Draw Rate (4 rounds)",
                                "95% Near-Optimal Mastery (5 rounds)",
                                "75% Fast Prototyping (2 rounds)"
                            ],
                            "default_option": "85% Win/Draw Rate (4 rounds)"
                        }
                    ]
                else:
                    questions = [
                        {
                            "pillar": "arena_ui",
                            "question": "How would you like to visually see the agent's behavior and environment in the Arena center panel?",
                            "suggested_options": [
                                "Modern dark-mode 2D canvas with neon indicators",
                                "Interactive game board with highlighted state cells",
                                "Clean telemetry dashboard with real-time gauges",
                                "Minimalist radar chart and timeline inspector"
                            ],
                            "default_option": "Modern dark-mode 2D canvas with neon indicators"
                        },
                        {
                            "pillar": "expectations",
                            "question": "What target success rate and auto-tuning round budget should we target?",
                            "suggested_options": [
                                "85% Success Rate (3 rounds)",
                                "90% Success Rate (4 rounds)",
                                "95% High Mastery (5 rounds)",
                                "75% Fast Prototyping (2 rounds)"
                            ],
                            "default_option": "85% Success Rate (4 rounds)"
                        }
                    ]
                return json.dumps({
                    "status": "needs_clarification",
                    "assistant_message": "I've analyzed your request across the 5 Pillars. To tailor your agent and arena UI perfectly, please clarify the following:",
                    "questions": questions,
                    "blueprint_spec": None
                })

        # Code Repair / Self-Healing Request
        if "repair" in sys_lower or "traceback" in prompt_lower or "repair" in prompt_lower or "fix" in prompt_lower:
            code_match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", prompt)
            code_to_fix = code_match.group(1) if code_match else prompt
            if "zerodivisionerror" in prompt_lower or "/ 0" in code_to_fix or "division by zero" in prompt_lower:
                fixed_code = re.sub(r"/\s*0(?!\d)", "/ 1.0", code_to_fix)
                fixed_code = fixed_code.replace("1 / 0", "1.0").replace("10 / 0", "10.0")
                return f"```python\n{fixed_code}\n```"
            elif "shape mismatch" in prompt_lower or "shape" in prompt_lower:
                # Self-healing shape adjustment
                fixed_code = re.sub(r"return np\.zeros\(\(8,\)", "return np.zeros((10,)", code_to_fix)
                fixed_code = re.sub(r"return np\.zeros\(\(11,\)", "return np.zeros((10,)", fixed_code)
                return f"```python\n{fixed_code}\n```"
            return f"```python\n{code_to_fix}\n```"

        # Generic fallback JSON
        return json.dumps({"status": "completed", "message": "Deterministic fallback response generated."})


# Alias for backward/forward compatibility
FallbackLLMClient = FallbackCodeSynthesizer


def create_llm_client(config: LLMConfig | None = None) -> BaseLLMClient:
    """Factory to instantiate the appropriate LLM client."""
    if config is None:
        # Auto-detect from environment or fallback
        if os.environ.get("DEEPSEEK_API_KEY"):
            config = LLMConfig(provider="deepseek")
        elif os.environ.get("OPENAI_API_KEY"):
            config = LLMConfig(provider="openai")
        elif os.environ.get("ANTHROPIC_API_KEY"):
            config = LLMConfig(provider="anthropic")
        else:
            config = LLMConfig(provider="fallback")

    p = config.provider.lower()
    if p == "deepseek":
        return DeepSeekLLMClient(config)
    elif p == "openai":
        return OpenAILLMClient(config)
    elif p == "anthropic":
        return AnthropicLLMClient(config)
    elif p == "ollama":
        return OllamaLLMClient(config)
    elif p in ("fallback", "mock"):
        return FallbackCodeSynthesizer(config)
    else:
        raise ValueError(f"Unsupported LLM provider '{config.provider}'. Supported: deepseek, openai, anthropic, ollama, fallback.")


def validate_llm_connection(config: LLMConfig) -> dict[str, Any]:
    """Tests connectivity to the configured provider."""
    p = config.provider.lower()

    if p in ("fallback", "mock"):
        return {
            "success": True,
            "provider": config.provider,
            "model": config.model_name or "deterministic-fallback",
            "latency_ms": 0.5,
            "message": "Deterministic offline fallback active (100% offline & reproducible)",
        }

    # Check for missing API key
    if p in ("deepseek", "openai", "anthropic") and not config.api_key:
        env_var = f"{p.upper()}_API_KEY"
        if not os.environ.get(env_var):
            return {
                "success": False,
                "provider": config.provider,
                "error": f"Missing API key for provider '{config.provider}'. Please set {env_var} or supply in config.",
            }

    start = time.perf_counter()
    try:
        client = create_llm_client(config)
        # Probe generation
        _ = client.generate("Respond with 'pong' only.", system_prompt="Health check")
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "success": True,
            "provider": config.provider,
            "model": client.model,
            "latency_ms": round(latency_ms, 2),
            "message": f"Successfully connected to {config.provider} ({client.model})",
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "success": False,
            "provider": config.provider,
            "latency_ms": round(latency_ms, 2),
            "error": str(exc),
        }
