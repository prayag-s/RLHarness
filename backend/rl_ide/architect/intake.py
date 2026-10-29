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
from rl_ide.llm.client import BaseLLMClient, create_llm_client, FallbackCodeSynthesizer, get_active_llm_client
from rl_ide.llm.prompts import INTAKE_ARCHITECT_SYSTEM_PROMPT, ARCHITECT_SYSTEM_PROMPT
from rl_ide.ingest.loader import IngestedDataset


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


def extract_action_labels_from_prompt(text: str) -> list[str] | None:
    """Extracts explicit quoted action labels from user prompt."""
    if not text:
        return None
    m = re.search(r"(?:actions|classify into|into)\s*:?\s*([\'\"][^\.\n]+)", text, re.IGNORECASE)
    if m:
        acts = re.findall(r"[\'\"]([^\'\"]+)[\'\"]", m.group(1))
        if len(acts) >= 2:
            return [a.strip() for a in acts if a.strip()]
    return None


def extract_observation_features_from_prompt(text: str) -> list[str] | None:
    """Extracts explicit observation features from phrases like 'observing a, b, c. Actions'."""
    if not text:
        return None
    m = re.search(r"observing\s+([^.]+?)\.\s*Actions", text, re.IGNORECASE)
    if m:
        parts = re.split(r",\s*|\s+and\s+", m.group(1).strip())
        feats = [p.strip() for p in parts if p.strip()]
        if len(feats) >= 2:
            return feats
    return None


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


def is_blackjack_intent(text: str) -> bool:
    """Fuzzy and regex domain intent detector for Blackjack and card game variants."""
    if not text:
        return False
    t = text.lower().strip()
    direct_kws = [
        "blackjack", "black-jack", "black jack", "blacjack", "blakjack", "blackjak",
        "blacckjack", "twenty-one", "twenty one", "21 card", "game of 21", "card game",
        "casino card", "dealer card", "hit or stand", "hit and stand", "split or double",
        "double down", "card counting", "card count", "playing cards", "card agent",
        "blackjack table", "card table", "casino table"
    ]
    if any(kw in t for kw in direct_kws):
        return True
    if re.search(r"\bblackjack\b", t) or re.search(r"\bblack\s*jack\b", t):
        return True
    if re.search(r"\b(card|cards|dealer|shoe|deck|hand)\b", t) and any(w in t for w in ["21", "hit", "stand", "bust", "double", "table"]):
        return True
    return False


def is_blackjack_spec(blueprint: AgentBlueprintSpec) -> bool:
    """Checks if an AgentBlueprintSpec represents a Blackjack / card game domain."""
    if not blueprint:
        return False
    if is_blackjack_intent(blueprint.task_title) or is_blackjack_intent(blueprint.domain_summary):
        return True
    action_str = " ".join(blueprint.action_labels or []).lower()
    if "hit" in action_str and "stand" in action_str:
        return True
    obs_str = " ".join(blueprint.observation_features or []).lower()
    if any(w in obs_str for w in ["player_total", "dealer_upcard", "usable_ace", "hand_total"]):
        return True
    if "blackjack" in (blueprint.arena_visual_description or "").lower():
        return True
    return False


class IntakeArchitectAgent:
    """Interviews the user to establish the 5 Pillars and locks the AgentBlueprintSpec."""

    def __init__(self, llm_client: BaseLLMClient | None = None):
        self._llm_client = llm_client

    @property
    def client(self) -> BaseLLMClient:
        if self._llm_client:
            return self._llm_client
        return get_active_llm_client()

    def interview(
        self,
        user_prompt: str,
        dataset: IngestedDataset | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        user_answers: dict[str, str] | None = None,
        force_smart_defaults: bool = False,
        training_params: dict[str, Any] | None = None,
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

        # If using live non-fallback LLM, use live LLM directly
        if not isinstance(self.client, FallbackCodeSynthesizer):
            try:
                llm_response = self._call_llm_intake(
                    prompt_clean,
                    dataset,
                    conversation_history,
                    answers,
                    wants_defaults=wants_defaults,
                    training_params=training_params,
                )
                if llm_response:
                    return llm_response
            except Exception:
                # Gracefully fall back to deterministic intake engine
                pass

        # 2. Deterministic Intake Engine (used for offline Fallback in test mode)
        return self._deterministic_intake(
            prompt_clean,
            dataset,
            answers,
            wants_defaults=wants_defaults,
            conversation_history=conversation_history,
            training_params=training_params,
        )

    def _call_llm_intake(
        self,
        prompt: str,
        dataset: IngestedDataset | None,
        conversation_history: list[dict[str, str]] | None,
        answers: dict[str, str],
        wants_defaults: bool = False,
        training_params: dict[str, Any] | None = None,
    ) -> IntakeResponse | None:
        """Queries the configured LLM for intake evaluation."""
        context_parts = [f"User Prompt: {prompt}"]
        if dataset:
            cols = list(dataset.df.columns)[:20] if hasattr(dataset, "df") and dataset.df is not None else []
            context_parts.append(f"Attached Dataset ID: {dataset.dataset_id}, Name: {dataset.name}, Columns: {cols}")
        if answers:
            context_parts.append(f"User Answers to Previous Questions: {json.dumps(answers)}")
        if training_params:
            context_parts.append(f"User Explicit Training Parameters: {json.dumps(training_params)}")
        if conversation_history:
            history_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in conversation_history[-4:])
            context_parts.append(f"Recent Conversation:\n{history_text}")

        if wants_defaults or answers:
            instruction = (
                "The user has either provided clarifying answers or requested smart defaults. "
                "Synthesize the complete AgentBlueprintSpec tailored directly to this domain and problem. "
                "Respond in valid JSON adhering to IntakeResponse schema with status: 'ready_to_build', "
                "assistant_message, questions: [], and blueprint_spec populated with all 5 pillars: "
                "task_title, domain_summary, data_source_type, observation_features, action_labels, "
                "arena_visual_description, interactive_mode_description, success_metric_name, "
                "target_metric_threshold, max_auto_tune_rounds, episodes_per_round, initial_reward_components."
            )
        else:
            instruction = (
                "Analyze the user's agent objective against the 5 Pillars (Observation, Action, Reward, "
                "Arena UI, Expectations, Interaction). If critical decisions require clarification, return status: "
                "'needs_clarification' with clarifying questions using unique IDs 'q_objective', 'q_arena_ui', 'q_interaction'. "
                "If intent is already clear, return status: 'ready_to_build' with blueprint_spec."
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
        training_params: dict[str, Any] | None = None,
    ) -> IntakeResponse:
        """Deterministic, highly reliable intake resolver across the 5 Pillars."""
        prompt_lower = prompt.lower()
        history_text = " ".join(m.get("content", "") for m in (conversation_history or []))
        answers_text = " ".join(str(v) for v in answers.values())
        combined_text = f"{prompt} {history_text} {answers_text}".strip()

        # Detect domain with fuzzy typo & abbreviation support
        is_tictactoe = is_tictactoe_intent(combined_text)
        is_blackjack = is_blackjack_intent(combined_text) and not is_tictactoe
        is_grid = any(kw in prompt_lower for kw in ["grid", "maze", "catcher", "navigation"]) and not is_tictactoe and not is_blackjack
        is_data = (dataset is not None or any(kw in prompt_lower for kw in ["csv", "dataset", "table", "churn", "fraud", "classify", "predict"])) and not is_tictactoe and not is_blackjack

        # Check pillar coverage
        has_arena_ui = (
            "arena_ui" in answers or "q_arena_ui" in answers
            or (
                any(kw in prompt_lower for kw in ["visualize", "arena", "dark-mode", "grid", "canvas", "board", "dashboard", "radar", "inspector", "felt", "blackjack table", "card table"])
                and not is_tictactoe
            )
        )
        if any(kw in prompt_lower for kw in ["neon", "wood", "retro", "pixel", "table view", "dark mode", "modern 3x3"]):
            has_arena_ui = True

        has_expectations = (
            "expectations" in answers or "q_expectations" in answers
            or re.search(r"\b\d{1,3}%\b", prompt) is not None
            or any(kw in prompt_lower for kw in ["win rate", "accuracy", "threshold", "rounds", "metric"])
        )

        has_objective = "q_objective" in answers or "objective" in answers
        has_interaction = "q_interaction" in answers or "interaction" in answers

        # Check if clarification is needed: prompt is underspecified and no answers given yet
        needs_clarification = (not wants_defaults) and len(answers) == 0 and (not has_arena_ui or not has_expectations or not has_objective or not has_interaction)

        if needs_clarification:
            questions = []

            # 1. Strategy & Rules (q_objective)
            if is_tictactoe:
                q_obj = ClarifyingQuestion(
                    id="q_objective",
                    question_id="q_objective",
                    pillar=PILLAR_OBJECTIVE,
                    question="What core strategic behavior and reward priorities should the agent master?",
                    suggested_options=[
                        "Unbeatable minimax play: Win whenever possible and force a draw otherwise",
                        "Aggressive center-and-corner control with high win pressure",
                        "Defensive counter-puncher: Prioritize immediate threat blocking",
                        "Balanced self-play exploration with anti-loss penalty",
                    ],
                    default_option="Unbeatable minimax play: Win whenever possible and force a draw otherwise",
                )
            elif is_blackjack:
                q_obj = ClarifyingQuestion(
                    id="q_objective",
                    question_id="q_objective",
                    pillar=PILLAR_OBJECTIVE,
                    question="What blackjack strategy and dealer rule profile should the agent learn?",
                    suggested_options=[
                        "Casino Basic Strategy: Mathematically optimal hit/stand/double decisions",
                        "Aggressive Payout Maximizer: High-variance doubling and split play",
                        "Conservative Anti-Bust Strategy: Stand early to force dealer busts",
                        "Dynamic Card Counter: Risk-adjusted thresholds against shoe penetration",
                    ],
                    default_option="Casino Basic Strategy: Mathematically optimal hit/stand/double decisions",
                )
            elif is_data:
                q_obj = ClarifyingQuestion(
                    id="q_objective",
                    question_id="q_objective",
                    pillar=PILLAR_OBJECTIVE,
                    question="What classification strategy and error penalty model should the policy prioritize?",
                    suggested_options=[
                        "Balanced precision and recall with equal false positive/negative penalties",
                        "High precision: Heavily penalize false alarms",
                        "High recall: Prioritize catching all positive target occurrences",
                        "Cost-sensitive decision making with tiered penalty matrix",
                    ],
                    default_option="Balanced precision and recall with equal false positive/negative penalties",
                )
            else:
                q_obj = ClarifyingQuestion(
                    id="q_objective",
                    question_id="q_objective",
                    pillar=PILLAR_OBJECTIVE,
                    question="What primary domain goal and reward incentives should the agent optimize?",
                    suggested_options=[
                        "Rapid goal completion with minimum steps and distance penalties",
                        "Energy-efficient conservative navigation avoiding danger zones",
                        "Dynamic adaptability under environmental perturbations",
                        "Exploration-first milestone discovery",
                    ],
                    default_option="Rapid goal completion with minimum steps and distance penalties",
                )
            questions.append(q_obj)

            # 2. Arena Visualization (q_arena_ui)
            if is_tictactoe:
                q_ui = ClarifyingQuestion(
                    id="q_arena_ui",
                    question_id="q_arena_ui",
                    pillar=PILLAR_ARENA_UI,
                    question="How would you like to visualize the Tic-Tac-Toe arena in the center panel?",
                    suggested_options=[
                        "Modern 3x3 dark-mode grid with neon glow and winning lines",
                        "Minimalist wood-grain board with move history ledger",
                        "Retro arcade 8-bit grid with animated action markers",
                        "Compact numeric matrix with live value heatmaps",
                    ],
                    default_option="Modern 3x3 dark-mode grid with neon glow and winning lines",
                )
            elif is_blackjack:
                q_ui = ClarifyingQuestion(
                    id="q_arena_ui",
                    question_id="q_arena_ui",
                    pillar=PILLAR_ARENA_UI,
                    question="How would you like to style the Blackjack Casino Arena in the center stage?",
                    suggested_options=[
                        "Classic Emerald Felt: Casino oval card table with dealer/player cards & chip bankroll",
                        "High-Roller Neon Casino: Glowing neon felt with live odds and probability bars",
                        "Slate Card Room: Clean modern table with crisp SVG cards and payout telemetry",
                        "Retro Casino Arcade: 8-bit pixel art blackjack table with animated chip stacks",
                    ],
                    default_option="Classic Emerald Felt: Casino oval card table with dealer/player cards & chip bankroll",
                )
            elif is_data:
                q_ui = ClarifyingQuestion(
                    id="q_arena_ui",
                    question_id="q_arena_ui",
                    pillar=PILLAR_ARENA_UI,
                    question="How would you like to visualize predictions and records in the Arena center panel?",
                    suggested_options=[
                        "Interactive record inspector card with confidence gauge",
                        "Real-time streaming ledger with live audit heatmaps",
                        "Multi-metric telemetry dashboard with ROC/confusion sparklines",
                        "Compact tabular feed with color-coded classification tags",
                    ],
                    default_option="Interactive record inspector card with confidence gauge",
                )
            else:
                q_ui = ClarifyingQuestion(
                    id="q_arena_ui",
                    question_id="q_arena_ui",
                    pillar=PILLAR_ARENA_UI,
                    question="How would you like to visually see the agent's behavior in the Arena center panel?",
                    suggested_options=[
                        "Modern dark-mode 2D canvas with neon indicators",
                        "Interactive simulation field with trajectory traces",
                        "Clean telemetry dashboard with real-time gauges",
                        "Minimalist radar chart and timeline inspector",
                    ],
                    default_option="Modern dark-mode 2D canvas with neon indicators",
                )
            questions.append(q_ui)

            # 3. Interactive Mode (q_interaction)
            if is_tictactoe:
                q_int = ClarifyingQuestion(
                    id="q_interaction",
                    question_id="q_interaction",
                    pillar=PILLAR_INTERACTION,
                    question="How will you interact with and test the agent live in the Arena?",
                    suggested_options=[
                        "Click-to-place human vs agent interactive board with reset button",
                        "Step-by-step turn inspection with move undo and probability radar",
                        "Automated self-play spectator mode with speed slider",
                        "Simulated opponent difficulty toggle (Random, Tactical, Master)",
                    ],
                    default_option="Click-to-place human vs agent interactive board with reset button",
                )
            elif is_blackjack:
                q_int = ClarifyingQuestion(
                    id="q_interaction",
                    question_id="q_interaction",
                    pillar=PILLAR_INTERACTION,
                    question="How do you want to interactively play against or test the Blackjack agent?",
                    suggested_options=[
                        "Live Player Mode: Deal cards and play Hit/Stand/Double decisions as a human",
                        "Agent Advisor Mode: Inspect real-time action probabilities and recommended moves",
                        "Continuous Auto-Play: Watch the policy play successive hands with speed controls",
                        "Dealer Rule Simulator: Toggle dealer stands on soft 17 vs hits soft 17",
                    ],
                    default_option="Live Player Mode: Deal cards and play Hit/Stand/Double decisions as a human",
                )
            elif is_data:
                q_int = ClarifyingQuestion(
                    id="q_interaction",
                    question_id="q_interaction",
                    pillar=PILLAR_INTERACTION,
                    question="How will you test individual data records and scenarios interactively?",
                    suggested_options=[
                        "Interactive feature sliders with live probability prediction readout",
                        "Row-by-row dataset inspector with manual override button",
                        "Batch record test upload with confusion matrix recalculation",
                        "Live threshold sensitivity tuning bar",
                    ],
                    default_option="Interactive feature sliders with live probability prediction readout",
                )
            else:
                q_int = ClarifyingQuestion(
                    id="q_interaction",
                    question_id="q_interaction",
                    pillar=PILLAR_INTERACTION,
                    question="How will you interactively test and guide the agent in the Arena?",
                    suggested_options=[
                        "Direct point-and-click target guidance with obstacle placement",
                        "Manual keyboard/button control override for agent comparison",
                        "State injection inspector with live policy value readout",
                        "Continuous physics sandbox with gravity/friction sliders",
                    ],
                    default_option="Direct point-and-click target guidance with obstacle placement",
                )
            questions.append(q_int)

            # 4. Target Expectations (q_expectations)
            if is_tictactoe:
                q_exp = ClarifyingQuestion(
                    id="q_expectations",
                    question_id="q_expectations",
                    pillar=PILLAR_EXPECTATIONS,
                    question="What target mastery win/draw rate and auto-tune round budget should we aim for?",
                    suggested_options=[
                        "90% Win/Draw Rate (5 rounds)",
                        "95% Near-Optimal Mastery (8 rounds)",
                        "85% Win/Draw Rate (4 rounds)",
                        "75% Fast Prototyping (2 rounds)",
                    ],
                    default_option="90% Win/Draw Rate (5 rounds)",
                )
            elif is_blackjack:
                q_exp = ClarifyingQuestion(
                    id="q_expectations",
                    question_id="q_expectations",
                    pillar=PILLAR_EXPECTATIONS,
                    question="What target win/non-bust success rate and auto-tuning round budget should we target?",
                    suggested_options=[
                        "50% Win / Non-Bust Rate (5 rounds)",
                        "55% Advantage Play (8 rounds)",
                        "45% Baseline Mastery (3 rounds)",
                        "40% Rapid Prototype (2 rounds)",
                    ],
                    default_option="50% Win / Non-Bust Rate (5 rounds)",
                )
            elif is_data:
                q_exp = ClarifyingQuestion(
                    id="q_expectations",
                    question_id="q_expectations",
                    pillar=PILLAR_EXPECTATIONS,
                    question="What target accuracy / success threshold and auto-tune rounds budget should we target?",
                    suggested_options=[
                        "90% Target Accuracy (5 rounds)",
                        "95% High Precision (8 rounds)",
                        "85% Balanced Classification (4 rounds)",
                        "75% Quick Baseline (2 rounds)",
                    ],
                    default_option="90% Target Accuracy (5 rounds)",
                )
            else:
                q_exp = ClarifyingQuestion(
                    id="q_expectations",
                    question_id="q_expectations",
                    pillar=PILLAR_EXPECTATIONS,
                    question="What target success rate and auto-tuning round budget should we target?",
                    suggested_options=[
                        "90% Target Mastery (5 rounds)",
                        "95% High Mastery (8 rounds)",
                        "85% Target Success (4 rounds)",
                        "75% Fast Prototyping (2 rounds)",
                    ],
                    default_option="90% Target Mastery (5 rounds)",
                )
            questions.append(q_exp)

            return IntakeResponse(
                status="needs_clarification",
                assistant_message=(
                    "I've analyzed your objective across the 5 Pillars. To tailor your custom RL environment, "
                    "visual Arena UI, and autonomous tuning loop, please clarify the following preferences:"
                ),
                questions=questions,
                blueprint_spec=None,
                recommended_training_params={
                    "max_auto_tune_rounds": 5,
                    "episodes_per_round": 200,
                    "target_metric_threshold": 90.0,
                },
            )

        # 3. Build Locked Blueprint
        # Parse arena_ui
        arena_ui_choice = answers.get("q_arena_ui") or answers.get("arena_ui") or answers.get(PILLAR_ARENA_UI)
        if not arena_ui_choice:
            if is_tictactoe:
                arena_ui_choice = "Modern 3x3 dark-mode grid with neon glow and winning lines"
            elif is_data:
                arena_ui_choice = "Interactive record inspector card with confidence gauge"
            else:
                arena_ui_choice = "Modern dark-mode 2D canvas with neon indicators"

        # Parse expectations & threshold
        expectations_choice = answers.get("q_expectations") or answers.get("expectations") or answers.get(PILLAR_EXPECTATIONS) or ""
        threshold = 85.0 if wants_defaults else 90.0
        max_rounds = 4 if wants_defaults else 5
        episodes_per_round = 200

        # Check explicit training parameters from arguments or answers
        effective_params = dict(training_params or {})
        for k in ["max_auto_tune_rounds", "episodes_per_round", "target_metric_threshold"]:
            if k in answers:
                effective_params[k] = answers[k]

        if "target_metric_threshold" in effective_params:
            try:
                threshold = float(effective_params["target_metric_threshold"])
            except (ValueError, TypeError):
                pass
        else:
            match_pct = re.search(r"(\d+(?:\.\d+)?)%", expectations_choice or prompt)
            if match_pct:
                threshold = float(match_pct.group(1))

        if "max_auto_tune_rounds" in effective_params:
            try:
                max_rounds = int(effective_params["max_auto_tune_rounds"])
            except (ValueError, TypeError):
                pass
        else:
            match_rnd = re.search(r"(\d+)\s*rounds?", expectations_choice or prompt, re.IGNORECASE)
            if match_rnd:
                max_rounds = int(match_rnd.group(1))

        if "episodes_per_round" in effective_params:
            try:
                episodes_per_round = int(effective_params["episodes_per_round"])
            except (ValueError, TypeError):
                pass
        else:
            match_eps = re.search(r"(\d+)\s*episodes?", prompt, re.IGNORECASE)
            if match_eps:
                episodes_per_round = int(match_eps.group(1))

        # Parse interactive mode choice
        interactive_choice = answers.get("q_interaction") or answers.get("interaction") or answers.get(PILLAR_INTERACTION)

        # Parse strategy/objective choice
        objective_choice = answers.get("q_objective") or answers.get("objective") or answers.get(PILLAR_OBJECTIVE)

        # Build domain-specific blueprint
        prompt_actions = extract_action_labels_from_prompt(prompt)
        prompt_obs = extract_observation_features_from_prompt(prompt)
        is_robotics_kw = any(kw in prompt_lower for kw in ["rover", "robot", "amr", "warehouse", "obstacle", "dock", "turn", "cartpole", "pendulum"])
        is_energy = ("solar" in prompt_lower or "grid_price" in prompt_lower or "tariff" in prompt_lower or ("battery" in prompt_lower and not is_robotics_kw))

        if is_tictactoe:
            task_title = "Tic-Tac-Toe Master Agent"
            domain_summary = (
                f"Zero-sum competitive 3x3 board game agent. Strategy: {objective_choice}"
                if objective_choice
                else "Zero-sum competitive 3x3 board game agent learning optimal play against stochastic and min-max opponents."
            )
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
            interactive_mode = interactive_choice or "Clickable 3x3 interactive board where the human plays as 'O' and the agent responds instantly as 'X'."
            success_metric = "Non-Loss Rate (%)"
            reward_components = [
                {"name": "win_bonus", "weight": 10.0, "description": "Reward for winning the game"},
                {"name": "draw_bonus", "weight": 3.0, "description": "Reward for achieving a draw"},
                {"name": "loss_penalty", "weight": -10.0, "description": "Penalty for losing the game"},
                {"name": "illegal_move_penalty", "weight": -5.0, "description": "Penalty for selecting an occupied cell"},
            ]
        elif is_blackjack:
            task_title = "Blackjack 21 Card Agent"
            domain_summary = (
                f"Classic Blackjack (Twenty-One) casino card game. Strategy: {objective_choice}"
                if objective_choice
                else "Classic Blackjack (Twenty-One) casino card game learning optimal hit, stand, and double-down decisions against a dealer."
            )
            data_source_type = "pure_simulation"
            dataset_id = None
            observation_features = [
                "player_total", "dealer_upcard_value", "has_usable_ace", "can_double_down"
            ]
            action_labels = [
                "Hit (Draw Card)", "Stand (Hold Hand)", "Double Down (2x Bet)"
            ]
            interactive_mode = interactive_choice or "Interactive casino card table: play live hands as a human with Hit/Stand/Double controls, or inspect agent decisions."
            success_metric = "Win / Non-Bust Rate (%)"
            threshold = 50.0 if threshold == 90.0 else threshold
            reward_components = [
                {"name": "win_payout", "weight": 1.0, "description": "Standard 1:1 payout for beating dealer hand"},
                {"name": "blackjack_bonus", "weight": 1.5, "description": "3:2 bonus payout for natural 21 Blackjack"},
                {"name": "loss_penalty", "weight": -1.0, "description": "Loss penalty when dealer total beats player"},
                {"name": "bust_penalty", "weight": -1.5, "description": "Penalty for exceeding 21 (busting)"},
                {"name": "push_reward", "weight": 0.0, "description": "Zero return for tied hand (push)"},
            ]
        elif is_energy:
            task_title = "Solar & Commercial Battery Arbitrage Controller"
            domain_summary = (
                f"Smart edge controller observing solar generation, building load, battery SOC, and dynamic grid prices. Strategy: {objective_choice}"
                if objective_choice
                else "Smart edge controller optimizing solar and commercial battery charge/discharge and grid arbitrage."
            )
            data_source_type = "pure_simulation"
            dataset_id = None
            observation_features = prompt_obs or ["solar_kw", "building_load_kw", "battery_soc_pct", "grid_price_kwh"]
            action_labels = prompt_actions or ["Charge Battery", "Discharge to Building", "Sell Back to Grid", "Idle Hold"]
            interactive_mode = interactive_choice or "Interactive energy flow simulator with animated power arrows and live battery SOC inspection."
            success_metric = "Tariff Arbitrage Efficiency (%)"
            reward_components = [
                {"name": "arbitrage_savings", "weight": 1.0, "description": "Net dollar savings on building energy bill"},
                {"name": "peak_discharge_bonus", "weight": 0.5, "description": "Reward for discharging during high tariff hours"},
                {"name": "deep_discharge_penalty", "weight": -5.0, "description": "Penalty for dropping battery SOC below 15%"},
            ]
        elif is_data:
            if dataset is not None:
                raw_cols = list(dataset.df.columns) if hasattr(dataset, "df") and dataset.df is not None else []
                observation_features = [str(c).strip() for c in raw_cols if str(c).strip()]
                if not observation_features:
                    observation_features = ["feature_0", "feature_1", "feature_2", "feature_3"]

                if any(kw in prompt_lower for kw in ["prior-auth", "clinical", "hipaa", "triage"]):
                    task_title = "Prior-Authorization Clinical Triage Agent"
                    success_metric = "Clinical Triage Accuracy (%)"
                elif any(kw in prompt_lower for kw in ["wire", "ach", "fraud interceptor"]):
                    task_title = "Real-Time Wire & ACH Fraud Interceptor"
                    success_metric = "Fraud Interception Accuracy (%)"
                elif any(kw in prompt_lower for kw in ["k8s", "kubernetes", "auto-scaler"]):
                    task_title = "Kubernetes Cluster Auto-Scaler & SLA Protector"
                    success_metric = "SLA Compliance Rate (%)"
                elif any(kw in prompt_lower for kw in ["flash-sale", "flash sale", "dynamic pricing"]):
                    task_title = "Flash-Sale Dynamic Pricing Agent"
                    success_metric = "Revenue Optimization Rate (%)"
                elif any(kw in prompt_lower for kw in ["nda", "contract", "clause risk"]):
                    task_title = "Contract Clause Risk Auditor"
                    success_metric = "Clause Classification Accuracy (%)"
                elif any(kw in prompt_lower for kw in ["travel", "expense", "t&e", "receipt"]):
                    task_title = "Travel & Expense (T&E) Policy Auditor"
                    success_metric = "Audit Compliance Rate (%)"
                elif any(kw in prompt_lower for kw in ["retention", "churn", "csm", "retained mrr"]):
                    task_title = "VIP Customer Churn & Retention Optimizer"
                    success_metric = "Net Retention Rate (%)"
                else:
                    task_title = f"{dataset.name} Decision Agent"
                    success_metric = "Classification Accuracy (%)"

                domain_summary = (
                    f"Autonomous sequential decision and classification agent operating over ingested dataset '{dataset.name}'. Strategy: {objective_choice}"
                    if objective_choice
                    else f"Autonomous sequential decision and classification agent operating over ingested dataset '{dataset.name}'."
                )
                data_source_type = "uploaded_dataset"
                dataset_id = dataset.dataset_id
            else:
                observation_features = prompt_obs or ["transaction_amount", "user_velocity_score", "country_risk_index", "device_trust_score"]
                task_title = "Fraud & Risk Classification Agent"
                domain_summary = f"Autonomous tabular decision and fraud risk classification agent. Strategy: {objective_choice or 'High-precision risk flagging'}"
                data_source_type = "uploaded_dataset"
                dataset_id = None
                success_metric = "Classification Accuracy (%)"

            if prompt_actions:
                action_labels = prompt_actions
            else:
                action_labels = ["Approve / Normal", "Flag / High-Risk"]
                if len(observation_features) > 4:
                    action_labels = ["Accept / Class 0", "Review / Class 1", "Escalate / Class 2"]

            interactive_mode = interactive_choice or "Inspect any custom row from the dataset or adjust feature sliders to test agent predictions live."
            reward_components = [
                {"name": "correct_decision", "weight": 1.0, "description": "Reward for correct classification / action"},
                {"name": "incorrect_decision", "weight": -1.0, "description": "Penalty for false positive or false negative"},
                {"name": "high_confidence_bonus", "weight": 0.2, "description": "Bonus for well-calibrated decisions"},
            ]
        else:
            if any(kw in prompt_lower for kw in ["cartpole", "pole", "pendulum", "balance", "inverted pendulum"]):
                task_title = "CartPole Physics Balancing Agent"
                domain_summary = f"Continuous control inverted pendulum physics agent learning upright stabilization against gravity. Strategy: {objective_choice or 'Pole balance stabilization'}"
                data_source_type = "pure_simulation"
                dataset_id = None
                observation_features = ["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"]
                action_labels = prompt_actions or ["Push Cart Left", "Push Cart Right"]
                interactive_mode = interactive_choice or "Interactive physics simulation: apply directional push disturbances to test recovery."
                success_metric = "Balance Stability Rate (%)"
                reward_components = [
                    {"name": "upright_bonus", "weight": 1.0, "description": "Reward for maintaining pole angle within threshold"},
                    {"name": "centering_bonus", "weight": 0.2, "description": "Bonus for keeping cart near track center"},
                    {"name": "fall_penalty", "weight": -10.0, "description": "Penalty for pole falling past failure threshold"},
                ]
            else:
                is_warehouse = any(kw in prompt_lower for kw in ["warehouse", "amr", "loading dock"])
                task_title = "2D AMR Warehouse Navigation Rover" if is_warehouse else ("2D Maze Navigation Rover Agent" if any(kw in prompt_lower for kw in ["rover", "maze", "navigation"]) else "Autonomous Task Synthesizer")
                domain_summary = f"Custom RL agent synthesized for: {prompt[:60]}. Priority: {objective_choice or 'Balanced efficiency'}"
                data_source_type = "pure_simulation"
                dataset_id = None
                observation_features = prompt_obs or ["state_pos_x", "state_pos_y", "velocity_x", "velocity_y", "target_distance"]
                action_labels = prompt_actions or ["Move Left", "Move Right", "Move Up", "Move Down", "Hold Position"]
                interactive_mode = interactive_choice or "Click or drag target points in the arena to interactively guide and test the agent."
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
            episodes_per_round=episodes_per_round,
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
            recommended_training_params={
                "max_auto_tune_rounds": max_rounds,
                "episodes_per_round": episodes_per_round,
                "target_metric_threshold": threshold,
            },
        )
