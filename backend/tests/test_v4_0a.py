"""Phase 4A Verification Test Suite.

Verifies:
1. Underspecified prompts return status="needs_clarification" with clickable suggestion pills across 5 Pillars.
2. Resolving clarifying questions (or "Use Smart Defaults") produces a valid AgentBlueprintSpec.
3. Ingested datasets (CSV/PDF) produce data-aware AgentBlueprintSpec referencing dataset columns.
4. Multi-provider LLM client routing (DeepSeek, OpenAI, Anthropic, Ollama, Fallback) and connection validation.
"""

import io
import json
import os
import pandas as pd
import pytest

from rl_ide.architect import (
    IntakeArchitectAgent,
    ClarifyingQuestion,
    AgentBlueprintSpec,
    IntakeResponse,
    PILLAR_ARENA_UI,
    PILLAR_EXPECTATIONS,
    PILLAR_OBJECTIVE,
    PILLAR_DATA,
)
from rl_ide.ingest import load_bytes, IngestedDataset
from rl_ide.llm import (
    LLMConfig,
    create_llm_client,
    validate_llm_connection,
    DeepSeekLLMClient,
    OpenAILLMClient,
    AnthropicLLMClient,
    OllamaLLMClient,
    FallbackCodeSynthesizer,
)


# -----------------------------------------------------------------------------
# Test 1: Underspecified Prompt Clarification
# -----------------------------------------------------------------------------

def test_underspecified_prompt_returns_clarifying_questions():
    """An underspecified prompt must return status='needs_clarification' with pills."""
    architect = IntakeArchitectAgent()
    prompt = "Train an agent to play tic-tac-toe"
    response = architect.interview(user_prompt=prompt)

    assert isinstance(response, IntakeResponse)
    assert response.status == "needs_clarification"
    assert response.blueprint_spec is None
    assert len(response.questions) >= 2

    # Verify Arena UI and Expectations pillars are probed
    pillars = [q.pillar for q in response.questions]
    assert PILLAR_ARENA_UI in pillars
    assert PILLAR_EXPECTATIONS in pillars

    # Verify each question has clickable suggestion pills and a default option
    for q in response.questions:
        assert isinstance(q, ClarifyingQuestion)
        assert len(q.question) > 10
        assert len(q.suggested_options) >= 3, f"Pillar {q.pillar} should offer >= 3 options"
        assert q.default_option in q.suggested_options, f"Default option '{q.default_option}' must be in suggested options"


# -----------------------------------------------------------------------------
# Test 2: Answering Clarifying Questions & Smart Defaults
# -----------------------------------------------------------------------------

def test_answering_questions_produces_valid_blueprint():
    """Submitting answers to the clarifying questions produces a valid AgentBlueprintSpec."""
    architect = IntakeArchitectAgent()
    prompt = "Train an agent to play tic-tac-toe"

    answers = {
        "arena_ui": "Minimalist wood-grain board with move history ledger",
        "expectations": "95% Near-Optimal Mastery (5 rounds)",
    }

    response = architect.interview(user_prompt=prompt, user_answers=answers)

    assert response.status == "ready_to_build"
    assert len(response.questions) == 0
    assert response.blueprint_spec is not None

    blueprint = response.blueprint_spec
    assert isinstance(blueprint, AgentBlueprintSpec)
    assert "Tic-Tac-Toe" in blueprint.task_title
    assert blueprint.data_source_type == "pure_simulation"
    assert blueprint.dataset_id is None
    assert blueprint.arena_visual_description == "Minimalist wood-grain board with move history ledger"
    assert blueprint.target_metric_threshold == 95.0
    assert blueprint.max_auto_tune_rounds == 5
    assert len(blueprint.observation_features) == 10  # 9 cells + turn indicator
    assert len(blueprint.action_labels) == 9          # 9 board cells
    assert len(blueprint.initial_reward_components) >= 3

    # Ensure blueprint is strictly serializable to JSON
    spec_json = blueprint.model_dump_json()
    spec_data = json.loads(spec_json)
    assert spec_data["target_metric_threshold"] == 95.0


def test_smart_defaults_produces_valid_blueprint():
    """Clicking 'Use Smart Defaults & Start' locks the blueprint immediately."""
    architect = IntakeArchitectAgent()
    prompt = "Train an agent to play tic-tac-toe"

    # Express button with force_smart_defaults=True
    response = architect.interview(user_prompt=prompt, force_smart_defaults=True)

    assert response.status == "ready_to_build"
    assert response.blueprint_spec is not None
    assert response.blueprint_spec.target_metric_threshold == 85.0
    assert response.blueprint_spec.max_auto_tune_rounds == 4
    assert len(response.blueprint_spec.action_labels) == 9

    # Natural-language express trigger
    nl_response = architect.interview(user_prompt="Train an agent to play tic-tac-toe. Use smart defaults and build now.")
    assert nl_response.status == "ready_to_build"
    assert nl_response.blueprint_spec is not None


# -----------------------------------------------------------------------------
# Test 3: Data-Aware Ingestion Integration (CSV and Schema Awareness)
# -----------------------------------------------------------------------------

def test_ingested_csv_produces_data_aware_blueprint():
    """Uploading a custom CSV produces a blueprint referencing the ingested columns."""
    architect = IntakeArchitectAgent()

    # Create synthetic customer churn dataset
    csv_content = b"""customer_id,monthly_charges,tenure_months,support_calls,contract_type,churned
cust_001,75.50,12,3,month-to-month,1
cust_002,29.99,36,0,two-year,0
cust_003,110.20,5,6,month-to-month,1
cust_004,45.00,48,1,one-year,0
"""
    dataset = load_bytes(filename="telecom_churn.csv", content=csv_content)
    assert isinstance(dataset, IngestedDataset)
    assert dataset.format == "csv"
    assert len(dataset.df) == 4

    # Underspecified prompt with dataset should probe data-specific preferences
    resp_clarify = architect.interview(
        user_prompt="Train an agent to detect customer churn from this CSV",
        dataset=dataset,
    )
    assert resp_clarify.status == "needs_clarification"
    assert len(resp_clarify.questions) >= 2

    # Resolving with smart defaults
    resp_built = architect.interview(
        user_prompt="Train an agent to detect customer churn from this CSV",
        dataset=dataset,
        force_smart_defaults=True,
    )
    assert resp_built.status == "ready_to_build"
    blueprint = resp_built.blueprint_spec
    assert blueprint is not None
    assert blueprint.data_source_type == "uploaded_dataset"
    assert blueprint.dataset_id == dataset.dataset_id

    # Observation features must reference the actual CSV columns
    for col in ["customer_id", "monthly_charges", "tenure_months", "support_calls"]:
        assert col in blueprint.observation_features

    assert len(blueprint.action_labels) >= 2
    assert "Accuracy" in blueprint.success_metric_name or "Rate" in blueprint.success_metric_name
    assert blueprint.target_metric_threshold >= 70.0


def test_ingested_document_produces_data_aware_blueprint():
    """Uploading a document/text policy produces a valid data-aware blueprint."""
    architect = IntakeArchitectAgent()

    doc_content = b"""# Financial Audit Policy
Rule 1: If transaction amount exceeds $10,000, mark as high risk.
Rule 2: International wire transfers require manual manager review.
Rule 3: Daily transactions under $100 are automatically cleared.
"""
    dataset = load_bytes(filename="compliance_rules.md", content=doc_content, role="knowledge_rules")
    assert isinstance(dataset, IngestedDataset)
    assert len(dataset.extracted_rules) >= 1

    resp = architect.interview(
        user_prompt="Train an agent to audit transactions following these compliance rules",
        dataset=dataset,
        force_smart_defaults=True,
    )
    assert resp.status == "ready_to_build"
    blueprint = resp.blueprint_spec
    assert blueprint is not None
    assert blueprint.data_source_type == "uploaded_dataset"
    assert blueprint.dataset_id == dataset.dataset_id
    assert len(blueprint.observation_features) >= 1


# -----------------------------------------------------------------------------
# Test 4: Multi-Provider LLM Client Routing & Validation
# -----------------------------------------------------------------------------

def test_deepseek_llm_client_routing():
    """Verify DeepSeek provider configuration and routing."""
    config = LLMConfig(provider="deepseek", api_key="sk-test-deepseek-key-1234")
    client = create_llm_client(config)

    assert isinstance(client, DeepSeekLLMClient)
    assert client.provider == "deepseek"
    assert client.model == "deepseek-chat"
    assert client.config.base_url == "https://api.deepseek.com/v1"
    assert client.config.api_key == "sk-test-deepseek-key-1234"


def test_openai_llm_client_routing():
    """Verify OpenAI provider configuration and routing."""
    config = LLMConfig(provider="openai", api_key="sk-test-openai-key")
    client = create_llm_client(config)

    assert isinstance(client, OpenAILLMClient)
    assert client.provider == "openai"
    assert client.model == "gpt-4o"
    assert client.config.base_url == "https://api.openai.com/v1"


def test_anthropic_llm_client_routing():
    """Verify Anthropic provider configuration and routing."""
    config = LLMConfig(provider="anthropic", api_key="sk-ant-test-key")
    client = create_llm_client(config)

    assert isinstance(client, AnthropicLLMClient)
    assert client.provider == "anthropic"
    assert client.model == "claude-3-5-sonnet-20241022"
    assert client.config.base_url == "https://api.anthropic.com/v1"


def test_ollama_llm_client_routing():
    """Verify Ollama provider configuration and routing."""
    config = LLMConfig(provider="ollama")
    client = create_llm_client(config)

    assert isinstance(client, OllamaLLMClient)
    assert client.provider == "ollama"
    assert client.model == "llama3.1"
    assert client.config.base_url == "http://localhost:11434/v1"


def test_fallback_synthesizer_and_connection_validation():
    """Verify FallbackCodeSynthesizer executes offline and validates cleanly."""
    config = LLMConfig(provider="fallback")
    client = create_llm_client(config)

    assert isinstance(client, FallbackCodeSynthesizer)
    assert client.provider == "fallback"
    assert client.model == "deterministic-fallback"

    # Fallback client generate probe
    reply = client.generate("test connectivity probe")
    assert isinstance(reply, str)
    data = json.loads(reply)
    assert data["status"] in ("ok", "completed")

    # validate_llm_connection on fallback must be 100% green
    val = validate_llm_connection(config)
    assert val["success"] is True
    assert val["provider"] == "fallback"
    assert "deterministic" in val["message"].lower()

    # validate_llm_connection for unconfigured cloud provider should fail gracefully
    unconfigured = LLMConfig(provider="deepseek", api_key=None)
    # Ensure env var is empty for this check
    orig_key = os.environ.pop("DEEPSEEK_API_KEY", None)
    try:
        val_fail = validate_llm_connection(unconfigured)
        assert val_fail["success"] is False
        assert "Missing API key" in val_fail["error"]
    finally:
        if orig_key:
            os.environ["DEEPSEEK_API_KEY"] = orig_key


def test_models_json_roundtrip():
    """Verify Pydantic models serialization round-trip across all structures."""
    q = ClarifyingQuestion(
        pillar="arena_ui",
        question="Which style?",
        suggested_options=["Option A", "Option B", "Option C"],
        default_option="Option A",
    )
    q_dict = q.model_dump()
    q_restored = ClarifyingQuestion.model_validate(q_dict)
    assert q_restored.pillar == "arena_ui"
    assert q_restored.suggested_options == ["Option A", "Option B", "Option C"]

    spec = AgentBlueprintSpec(
        task_title="Test Agent",
        domain_summary="Test Summary",
        data_source_type="pure_simulation",
        observation_features=["x", "y"],
        action_labels=["left", "right"],
        arena_visual_description="2D grid",
        interactive_mode_description="interactive clicks",
        success_metric_name="Score",
        target_metric_threshold=90.0,
        max_auto_tune_rounds=3,
        initial_reward_components=[{"name": "bonus", "weight": 1.0}],
    )
    spec_dict = spec.model_dump()
    spec_restored = AgentBlueprintSpec.model_validate(spec_dict)
    assert spec_restored.task_title == "Test Agent"
    assert spec_restored.target_metric_threshold == 90.0

    resp = IntakeResponse(
        status="ready_to_build",
        assistant_message="Ready",
        questions=[],
        blueprint_spec=spec,
    )
    resp_dict = resp.model_dump()
    resp_restored = IntakeResponse.model_validate(resp_dict)
    assert resp_restored.status == "ready_to_build"
    assert resp_restored.blueprint_spec is not None
    assert resp_restored.blueprint_spec.task_title == "Test Agent"


def test_tictactoe_fuzzy_intent_and_typo_handling():
    """Verify that colloquial typo variants and abbreviations correctly route to Tic-Tac-Toe."""
    architect = IntakeArchitectAgent()

    typo_prompts = [
        "Train an unbeatable tic toe agent. I want to see in train in",
        "Train an unbeatable tic tax tow agent. I want to watch it tr",
        "train a ttt agent",
        "create an unbeatable 3x3 board game agent",
        "agent that plays noughts and crosses",
        "build an agent for x and o",
        "tick tack toe bot",
    ]

    for p in typo_prompts:
        # First turn: should ask Tic-Tac-Toe specific clarifying questions
        resp = architect.interview(user_prompt=p)
        assert resp.status == "needs_clarification", f"Prompt failed clarification: {p}"
        assert any("tic-tac-toe" in q.question.lower() or "3x3" in q.suggested_options[0].lower() for q in resp.questions), (
            f"Expected Tic-Tac-Toe clarifying question for prompt: {p}"
        )

        # Express build / smart defaults should lock the Tic-Tac-Toe blueprint
        resp_defaults = architect.interview(user_prompt=p, force_smart_defaults=True)
        assert resp_defaults.status == "ready_to_build"
        assert resp_defaults.blueprint_spec is not None
        bp = resp_defaults.blueprint_spec
        assert "Tic-Tac-Toe" in bp.task_title
        assert len(bp.observation_features) == 10
        assert len(bp.action_labels) == 9
        assert "3x3" in bp.interactive_mode_description or "3x3" in bp.arena_visual_description
