"""Natural Language to TaskConfig Compiler with automatic AST validation and self-repair."""

from __future__ import annotations
import re
from typing import Any
import yaml
from pydantic import ValidationError

from rl_ide.contracts.adapter import DomainAdapter
from rl_ide.adapters.stream_adapter import CSVStreamAdapter
from rl_ide.adapters.text_classify_adapter import TextClassifyAdapter
from rl_ide.adapters.auto_data_adapter import AutoDataAdapter
from rl_ide.compiler.models import TaskConfig
from rl_ide.compiler.llm_client import BaseLLMClient, MockRulesLLMClient, create_llm_client
from rl_ide.reward.evaluator import RewardValidator, RewardValidationError

ADAPTER_REGISTRY: dict[str, type[DomainAdapter]] = {
    "CSVStreamAdapter": CSVStreamAdapter,
    "TextClassifyAdapter": TextClassifyAdapter,
    "AutoDataAdapter": AutoDataAdapter,
}


def _strip_markdown_code_fences(text: str) -> str:
    """Removes ```yaml and ``` wrapper fences if present."""
    text = text.strip()
    match = re.search(r"```(?:yaml)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # If starts with ``` and ends with ```
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return text


class NLToConfigCompiler:
    """Compiles natural language specifications into validated TaskConfig instances."""

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        adapter_registry: dict[str, type[DomainAdapter]] | None = None,
        max_retries: int = 2,
    ):
        self.llm_client = llm_client if llm_client is not None else MockRulesLLMClient()
        self.adapter_registry = dict(adapter_registry or ADAPTER_REGISTRY)
        self.max_retries = max_retries

    def compile(
        self,
        user_prompt: str,
        adapter_name: str,
        adapter: DomainAdapter | None = None,
        task_id: str | None = None,
    ) -> TaskConfig:
        """Translates user English prose into a validated TaskConfig.
        
        Inspects adapter schema, invokes LLM, runs RewardValidator,
        and automatically retries with feedback on validation errors.
        """
        # Resolve adapter
        if adapter is None:
            adapter_cls = self.adapter_registry.get(adapter_name)
            if adapter_cls is None:
                raise ValueError(
                    f"Unknown adapter '{adapter_name}'. Registered: {list(self.adapter_registry.keys())}"
                )
            if adapter_name == "AutoDataAdapter":
                import pandas as pd
                sample_df = pd.DataFrame({
                    "amount": [100.0, 5500.0, 20.0, 12000.0],
                    "risk_score": [0.1, 0.9, 0.2, 0.85],
                    "has_po": [True, False, True, False],
                    "department": ["sales", "engineering", "sales", "ops"],
                    "target": ["pass", "flag", "pass", "flag"],
                })
                adapter = adapter_cls(sample_df)
            else:
                adapter = adapter_cls()

        schema = adapter.event_schema()
        obs_spec = adapter.observations()
        action_spec = adapter.actions()

        system_prompt = (
            "You are an expert RL Task Compiler. You convert natural language descriptions of tasks "
            "into a structured TaskConfig YAML. You MUST ONLY use variables present in the provided event_schema. "
            "Never invent undefined fields. Output ONLY valid YAML without markdown commentary."
        )

        base_prompt = (
            f"User Task Request: {user_prompt}\n\n"
            f"Target Adapter: {adapter_name}\n"
            f"Horizon Type: {obs_spec.horizon_type} (default_gamma: {obs_spec.default_gamma})\n"
            f"Action Labels: {action_spec.labels}\n"
            f"Legal Event Schema Variables:\n"
            + "\n".join(f"  - {k}: {v}" for k, v in schema.items())
            + "\n\n"
            "Emit a valid TaskConfig YAML with fields:\n"
            "- task_id: string\n"
            "- name: string\n"
            "- description: string\n"
            "- adapter_name: string\n"
            "- adapter_config: dict\n"
            "- hyperparameters: dict (learning_rate, gamma, etc.)\n"
            "- reward_dsl: version, horizon, reward.terms (name, when, value), aggregate, clamp\n"
        )

        prompt = base_prompt
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            raw_response = self.llm_client.generate(prompt=prompt, system_prompt=system_prompt)
            clean_yaml = _strip_markdown_code_fences(raw_response)

            try:
                task_config = TaskConfig.from_yaml(clean_yaml)

                # Ensure adapter_name matches
                task_config.adapter_name = adapter_name
                if task_id:
                    task_config.task_id = task_id

                # Static validation against schema
                RewardValidator.validate(task_config.reward_dsl, schema=schema)
                return task_config

            except (RewardValidationError, ValidationError, yaml.YAMLError, ValueError) as err:
                last_error = err
                if attempt < self.max_retries:
                    # Construct repair prompt feeding the error back to the LLM
                    prompt = (
                        f"{base_prompt}\n\n"
                        f"ATTENTION: Your previous proposal failed validation with error:\n"
                        f"{str(err)}\n\n"
                        "Please repair the TaskConfig YAML. Ensure EVERY variable in 'when' and 'value' "
                        f"exists exactly in the Legal Event Schema ({list(schema.keys())}).\n"
                        "Output repaired YAML only:"
                    )
                else:
                    break

        if isinstance(last_error, RewardValidationError):
            raise last_error
        raise RewardValidationError(f"Failed to compile task config after {self.max_retries} retries: {last_error}")
