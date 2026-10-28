"""Conversational Intake Architect Agent for RL-Native Agent IDE v4.0."""

from __future__ import annotations
import json
import re
from typing import Any
from rl_ide.architect.models import (
    ClarifyingQuestion,
    AgentBlueprintSpec,
    IntakeResponse,
    PILLAR_OBJECTIVE,
    PILLAR_DATA,
    PILLAR_ARENA_UI,
    PILLAR_EXPECTATIONS,
    PILLAR_INTERACTION,
)
from rl_ide.llm.client import BaseLLMClient, create_llm_client, FallbackCodeSynthesizer
from rl_ide.ingest.loader import IngestedDataset


ARCHITECT_SYSTEM_PROMPT = """You are the Lead AI Agent Architect for an autonomous Reinforcement Learning IDE.
Your job is to interview the user briefly to lock down a complete `AgentBlueprintSpec` before our code synthesizer builds their custom RL environment, visual Arena UI, and autonomous training loop.

Check if the user's prompt (plus any attached dataset schema) clearly specifies the 5 Pillars:
1. Objective & Actions: What the agent controls and what good behavior looks like.
2. Data / Environment Source: Uploaded data/SQL/PDFs vs. a synthesized game/simulation.
3. Visual Arena Design: How they want to visually watch the agent on screen (e.g., 2D grid, canvas physics, card inspector, dashboard).
4. Target Expectations: The target success metric threshold (0-100%) and how many autonomous self-tuning rounds to run.
5. Live Interaction Mode: How the human can play against or test the agent interactively.

RULES:
- If critical details (especially Visual Arena preference or Target Expectations) are missing on the first turn, set `status="needs_clarification"` and ask at most 2 to 3 concise, high-impact questions.
- Every question MUST include 3-4 clickable `suggested_options` and a sensible `default_option`.
- If the user answers your questions OR says "use defaults / build now", immediately set `status="ready_to_build"` and output the complete `AgentBlueprintSpec` in strict JSON.
"""


def is_tictactoe_intent(text: str) -> bool:
    """Fuzzy and regex domain intent detector for Tic-Tac-Toe variants and typos."""
    if not text:
        return False
    t = text.lower().strip()
    direct_kws = [
        "tic-tac-toe", "tictactoe", "tic tac toe", "tic-tac", "tictac",
        "tic tac", "tic toe", "tic-toe", "tictoc", "tic toc",
        "tick-tack-toe", "tick tack toe", "tick tack", "tic tack toe",
        "noughts and crosses", "noughts & crosses", "noughts", "crosses",
        "three in a row", "3 in a row", "ttt agent", "ttt game", "ttt board",
    ]
    if any(kw in t for kw in direct_kws):
        return True

    if re.search(r"\bttt\b", t):
        return True

    # Match typo / phonetic variants: "tic"/"tik"/"tek"/"tick" + "tac"/"tax"/"tak"/"tok"/"toe"/"tow"/"to"/"tack"
    if re.search(r"\b(tic|tik|tek|tick)\b", t) and re.search(r"\b(tac|tax|tak|tok|toe|tow|to|tack)\b", t):
        return True

    # Match X and O variants: "x and o", "x's and o's", "xs and os", "x & o", "x/o"
    if re.search(r"\b(x['s]?\s*(?:and|&)\s*o['s]?)\b", t) or re.search(r"\bx\s*/\s*o\b", t):
        return True

    # Match 3x3 board/game/agent specifications
    if re.search(r"\b(3x3|3\s*by\s*3)\b", t) and any(w in t for w in ["game", "board", "grid", "agent", "play", "win", "player"]):
        return True

    return False


def is_tictactoe_spec(blueprint: AgentBlueprintSpec) -> bool:
    """Checks if an AgentBlueprintSpec represents a Tic-Tac-Toe domain."""
    if not blueprint:
        return False
    if is_tictactoe_intent(blueprint.task_title) or is_tictactoe_intent(blueprint.domain_summary):
        return True
    if len(blueprint.action_labels) == 9 and (
        len(blueprint.observation_features) in (9, 10)
        or any("place" in a.lower() or "cell" in a.lower() for a in blueprint.action_labels)
        or "3x3" in blueprint.arena_visual_description.lower()
    ):
        return True
    return False


class IntakeArchitectAgent:
    """Interviews the user to establish the 5 Pillars and locks the AgentBlueprintSpec."""

    def __init__(self, llm_client: BaseLLMClient | None = None):
        self.client = llm_client or create_llm_client()

    def interview(
        self,
        user_prompt: str,
        dataset: IngestedDataset | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        user_answers: dict[str, str] | None = None,
        force_smart_defaults: bool = False,
    ) -> IntakeResponse:
        """Evaluates user intent and returns either clarifying questions or locked blueprint."""
        prompt_clean = user_prompt.strip()
        prompt_lower = prompt_clean.lower()
        answers = user_answers or {}

        # 1. Check if user explicitly asked for defaults or instant build
        wants_defaults = (
            force_smart_defaults
            or any(kw in prompt_lower for kw in [
                "use smart defaults", "use defaults", "default", "build now",
                "start now", "start training", "force_defaults"
            ])
        )

        # If using live non-fallback LLM, attempt LLM call first
        if not isinstance(self.client, FallbackCodeSynthesizer) and not wants_defaults:
            try:
                llm_response = self._call_llm_intake(prompt_clean, dataset, conversation_history, answers)
                if llm_response:
                    return llm_response
            except Exception:
                # Gracefully fall back to deterministic intake engine
                pass

        # 2. Deterministic Intake Engine (used for Fallback, offline testing, and smart defaults)
        return self._deterministic_intake(
            prompt_clean,
            dataset,
            answers,
            wants_defaults=wants_defaults,
            conversation_history=conversation_history,
        )

    def _call_llm_intake(
        self,
        prompt: str,
        dataset: IngestedDataset | None,
        conversation_history: list[dict[str, str]] | None,
        answers: dict[str, str],
    ) -> IntakeResponse | None:
        """Queries the configured LLM for intake evaluation."""
        context_parts = [f"User Prompt: {prompt}"]
        if dataset:
            cols = list(dataset.df.columns)[:20] if hasattr(dataset, "df") and dataset.df is not None else []
            context_parts.append(f"Attached Dataset ID: {dataset.dataset_id}, Name: {dataset.name}, Columns: {cols}")
        if answers:
            context_parts.append(f"User Answers to Previous Questions: {json.dumps(answers)}")
        if conversation_history:
            history_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in conversation_history[-4:])
            context_parts.append(f"Recent Conversation:\n{history_text}")

        instruction = (
            "Analyze the above against the 5 Pillars. Respond in valid JSON adhering to IntakeResponse schema "
            "with keys: status, assistant_message, questions, blueprint_spec."
        )
        full_query = "\n\n".join(context_parts) + "\n\n" + instruction

        raw_json = self.client.generate_json(full_query, system_prompt=ARCHITECT_SYSTEM_PROMPT)
        return IntakeResponse.model_validate(raw_json)

    def _deterministic_intake(
        self,
        prompt: str,
        dataset: IngestedDataset | None,
        answers: dict[str, str],
        wants_defaults: bool,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> IntakeResponse:
        """Deterministic, highly reliable intake resolver across the 5 Pillars."""
        prompt_lower = prompt.lower()
        history_text = " ".join(m.get("content", "") for m in (conversation_history or []))
        answers_text = " ".join(str(v) for v in answers.values())
        combined_text = f"{prompt} {history_text} {answers_text}".strip()

        # Detect domain with fuzzy typo & abbreviation support
        is_tictactoe = is_tictactoe_intent(combined_text)
        is_grid = any(kw in prompt_lower for kw in ["grid", "maze", "catcher", "navigation"]) and not is_tictactoe
        is_data = (dataset is not None or any(kw in prompt_lower for kw in ["csv", "dataset", "table", "churn", "fraud", "classify", "predict"])) and not is_tictactoe

        # Check pillar coverage
        has_arena_ui = (
            "arena_ui" in answers
            or (
                any(kw in prompt_lower for kw in ["visualize", "arena", "dark-mode", "grid", "canvas", "board", "dashboard", "radar", "inspector"])
                and not is_tictactoe  # user simply mentioning tic-tac-toe doesn't mean they specified arena styling
            )
        )
        # If user explicitly specified styling in prompt, count it
        if any(kw in prompt_lower for kw in ["neon", "wood", "retro", "pixel", "table view", "dark mode", "modern 3x3"]):
            has_arena_ui = True

        has_expectations = (
            "expectations" in answers
            or re.search(r"\b\d{1,3}%\b", prompt) is not None
            or any(kw in prompt_lower for kw in ["win rate", "accuracy", "threshold", "rounds", "metric"])
        )

        # Check if clarification is needed
        needs_clarification = (not wants_defaults) and (not has_arena_ui or not has_expectations) and len(answers) == 0

        if needs_clarification:
            questions = []
            if not has_arena_ui:
                if is_tictactoe:
                    q = ClarifyingQuestion(
                        pillar=PILLAR_ARENA_UI,
                        question="How would you like to visualize the Tic-Tac-Toe arena in the center panel?",
                        suggested_options=[
                            "Modern 3x3 dark-mode grid with neon glow and winning lines",
                            "Minimalist wood-grain board with move history ledger",
                            "Retro arcade 8-bit grid with animated action markers",
                            "Compact numeric matrix with live value heatmaps",
                        ],
                        default_option="Modern 3x3 dark-mode grid with neon glow and winning lines",
                        question_id=PILLAR_ARENA_UI,
                    )
                elif is_data:
                    q = ClarifyingQuestion(
                        pillar=PILLAR_ARENA_UI,
                        question="How would you like to visualize predictions and records in the Arena center panel?",
                        suggested_options=[
                            "Interactive record inspector card with confidence gauge",
                            "Real-time streaming ledger with live audit heatmaps",
                            "Multi-metric telemetry dashboard with ROC/confusion sparklines",
                            "Compact tabular feed with color-coded classification tags",
                        ],
                        default_option="Interactive record inspector card with confidence gauge",
                        question_id=PILLAR_ARENA_UI,
                    )
                else:
                    q = ClarifyingQuestion(
                        pillar=PILLAR_ARENA_UI,
                        question="How would you like to visually see the agent's behavior in the Arena center panel?",
                        suggested_options=[
                            "Modern dark-mode 2D canvas with neon indicators",
                            "Interactive simulation field with trajectory traces",
                            "Clean telemetry dashboard with real-time gauges",
                            "Minimalist radar chart and timeline inspector",
                        ],
                        default_option="Modern dark-mode 2D canvas with neon indicators",
                        question_id=PILLAR_ARENA_UI,
                    )
                questions.append(q)

            if not has_expectations:
                if is_tictactoe:
                    q = ClarifyingQuestion(
                        pillar=PILLAR_EXPECTATIONS,
                        question="What target mastery win/draw rate and auto-tune round budget should we aim for?",
                        suggested_options=[
                            "85% Win/Draw Rate (4 rounds)",
                            "90% Win/Draw Rate (4 rounds)",
                            "95% Near-Optimal Mastery (5 rounds)",
                            "75% Fast Prototyping (2 rounds)",
                        ],
                        default_option="85% Win/Draw Rate (4 rounds)",
                        question_id=PILLAR_EXPECTATIONS,
                    )
                elif is_data:
                    q = ClarifyingQuestion(
                        pillar=PILLAR_EXPECTATIONS,
                        question="What target accuracy / success threshold and auto-tune rounds budget should we target?",
                        suggested_options=[
                            "90% Target Accuracy (4 rounds)",
                            "95% High Precision (5 rounds)",
                            "85% Balanced Classification (3 rounds)",
                            "75% Quick Baseline (2 rounds)",
                        ],
                        default_option="90% Target Accuracy (4 rounds)",
                        question_id=PILLAR_EXPECTATIONS,
                    )
                else:
                    q = ClarifyingQuestion(
                        pillar=PILLAR_EXPECTATIONS,
                        question="What target success rate and auto-tuning round budget should we target?",
                        suggested_options=[
                            "85% Success Rate (4 rounds)",
                            "90% Success Rate (4 rounds)",
                            "95% High Mastery (5 rounds)",
                            "75% Fast Prototyping (2 rounds)",
                        ],
                        default_option="85% Success Rate (4 rounds)",
                        question_id=PILLAR_EXPECTATIONS,
                    )
                questions.append(q)

            return IntakeResponse(
                status="needs_clarification",
                assistant_message=(
                    "I've analyzed your objective across the 5 Pillars. To tailor your custom RL environment, "
                    "visual Arena UI, and autonomous tuning loop, please clarify the following preferences:"
                ),
                questions=questions,
                blueprint_spec=None,
            )

        # 3. Build Locked Blueprint
        # Parse arena_ui
        arena_ui_choice = answers.get("arena_ui") or answers.get(PILLAR_ARENA_UI)
        if not arena_ui_choice:
            if is_tictactoe:
                arena_ui_choice = "Modern 3x3 dark-mode grid with neon glow and winning lines"
            elif is_data:
                arena_ui_choice = "Interactive record inspector card with confidence gauge"
            else:
                arena_ui_choice = "Modern dark-mode 2D canvas with neon indicators"

        # Parse expectations & threshold
        expectations_choice = answers.get("expectations") or answers.get(PILLAR_EXPECTATIONS) or ""
        threshold = 85.0
        max_rounds = 4

        # Extract percentage threshold
        match_pct = re.search(r"(\d+(?:\.\d+)?)%", expectations_choice or prompt)
        if match_pct:
            threshold = float(match_pct.group(1))

        # Extract rounds
        match_rnd = re.search(r"(\d+)\s*rounds?", expectations_choice or prompt, re.IGNORECASE)
        if match_rnd:
            max_rounds = int(match_rnd.group(1))

        # Build domain-specific blueprint
        if is_tictactoe:
            task_title = "Tic-Tac-Toe Master Agent"
            domain_summary = "Zero-sum competitive 3x3 board game agent learning optimal play against stochastic and min-max opponents."
            data_source_type = "pure_simulation"
            dataset_id = None
            observation_features = [
                "cell_0_0", "cell_0_1", "cell_0_2",
                "cell_1_0", "cell_1_1", "cell_1_2",
                "cell_2_0", "cell_2_1", "cell_2_2",
                "current_turn"
            ]
            action_labels = [
                "Place (0,0)", "Place (0,1)", "Place (0,2)",
                "Place (1,0)", "Place (1,1)", "Place (1,2)",
                "Place (2,0)", "Place (2,1)", "Place (2,2)"
            ]
            interactive_mode = "Clickable 3x3 interactive board where the human plays as 'O' and the agent responds instantly as 'X'."
            success_metric = "Non-Loss Rate (%)"
            reward_components = [
                {"name": "win_bonus", "weight": 10.0, "description": "Reward for winning the game"},
                {"name": "draw_bonus", "weight": 3.0, "description": "Reward for achieving a draw"},
                {"name": "loss_penalty", "weight": -10.0, "description": "Penalty for losing the game"},
                {"name": "illegal_move_penalty", "weight": -5.0, "description": "Penalty for selecting an occupied cell"},
            ]
        elif is_data and dataset is not None:
            # Data-aware blueprint referencing dataset columns
            raw_cols = list(dataset.df.columns) if hasattr(dataset, "df") and dataset.df is not None else []
            # Sanitize observation features
            observation_features = [str(c).strip() for c in raw_cols if str(c).strip()]
            if not observation_features:
                observation_features = ["feature_0", "feature_1", "feature_2", "feature_3"]

            task_title = f"{dataset.name} Decision Agent"
            domain_summary = f"Autonomous sequential decision and classification agent operating over ingested dataset '{dataset.name}'."
            data_source_type = "uploaded_dataset"
            dataset_id = dataset.dataset_id

            # Determine sensible action labels
            action_labels = ["Accept / Class 0", "Review / Class 1", "Escalate / Class 2"]
            if len(observation_features) > 2:
                # If there's a binary-like target column, extract or use clean binary actions
                action_labels = ["Approve / Normal", "Flag / High-Risk"]

            interactive_mode = "Inspect any custom row from the dataset or adjust feature sliders to test agent predictions live."
            success_metric = "Classification Accuracy (%)"
            reward_components = [
                {"name": "correct_decision", "weight": 1.0, "description": "Reward for correct classification / action"},
                {"name": "incorrect_decision", "weight": -1.0, "description": "Penalty for false positive or false negative"},
                {"name": "high_confidence_bonus", "weight": 0.2, "description": "Bonus for well-calibrated decisions"},
            ]
        else:
            task_title = "Autonomous Task Synthesizer"
            domain_summary = f"Custom RL agent synthesized for: {prompt[:60]}"
            data_source_type = "pure_simulation"
            dataset_id = None
            observation_features = ["state_pos_x", "state_pos_y", "velocity_x", "velocity_y", "target_distance"]
            action_labels = ["Move Left", "Move Right", "Move Up", "Move Down", "Hold Position"]
            interactive_mode = "Click or drag target points in the arena to interactively guide and test the agent."
            success_metric = "Target Achievement Rate (%)"
            reward_components = [
                {"name": "goal_bonus", "weight": 10.0, "description": "Reward for reaching target state"},
                {"name": "distance_delta", "weight": 1.0, "description": "Progress reward for closing distance to target"},
                {"name": "step_cost", "weight": -0.05, "description": "Small efficiency step penalty"},
            ]

        spec = AgentBlueprintSpec(
            task_title=task_title,
            domain_summary=domain_summary,
            data_source_type=data_source_type,
            dataset_id=dataset_id,
            observation_features=observation_features,
            action_labels=action_labels,
            arena_visual_description=arena_ui_choice,
            interactive_mode_description=interactive_mode,
            success_metric_name=success_metric,
            target_metric_threshold=threshold,
            max_auto_tune_rounds=max_rounds,
            initial_reward_components=reward_components,
        )

        return IntakeResponse(
            status="ready_to_build",
            assistant_message=(
                f"Blueprint locked for '{task_title}'! All 5 pillars are established. "
                "Ready to proceed with JIT environment synthesis and arena generation."
            ),
            questions=[],
            blueprint_spec=spec,
        )
