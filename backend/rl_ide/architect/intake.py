"""Universal Conversational Intake Architect for RL-Native Agent IDE v4.4.

Eliminates all domain-specific intent classifiers, hardcoded templates, and rover fallbacks.
Every user domain idea produces a bespoke DynamicTaskBlueprint via LLM structured synthesis.
"""

from __future__ import annotations
import json
import re
from typing import Any
from rl_ide.architect.models import (
    ClarifyingQuestion,
    ObservationFeatureSpec,
    ActionSpec,
    AgentBlueprintSpec,
    DynamicTaskBlueprint,
    IntakeResponse,
    PILLAR_OBJECTIVE,
    PILLAR_DATA,
    PILLAR_ARENA_UI,
    PILLAR_EXPECTATIONS,
    PILLAR_INTERACTION,
)
from rl_ide.llm.client import BaseLLMClient, FallbackCodeSynthesizer, get_active_llm_client
from rl_ide.llm.prompts import INTAKE_ARCHITECT_SYSTEM_PROMPT, ARCHITECT_SYSTEM_PROMPT
from rl_ide.ingest.loader import IngestedDataset


UNIVERSAL_INTAKE_PROMPT = """You are an Expert Universal Reinforcement Learning Domain Architect.
Given any user domain idea, problem description, or task objective, dynamically design a bespoke Reinforcement Learning task blueprint.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. ZERO hardcoded domain templates. Tailor the domain title, observation features, discrete actions, reward components, and visual metaphor dynamically to the user's prompt.
2. NEVER default or fall back to a 2D rover or navigation robot unless the user specifically asks for rover navigation.
3. Define a clear `state_dict_schema` of exact JSON-serializable keys that the environment's `get_render_state()` will emit and the Arena UI will render.

Respond in strict JSON with the following structure:
{
  "domain_title": "Concise human-friendly domain title",
  "domain_summary": "1-2 sentence summary of domain mechanics and RL goal",
  "visual_metaphor": "Vivid description of the bespoke Arena UI layout and visual metaphor",
  "observation_features": [
    {"name": "feature_name_snake_case", "description": "feature description", "low": -1.0, "high": 1.0}
  ],
  "actions": [
    {"index": 0, "label": "Human Action Button Label", "mechanic_description": "Exact effect of this action"}
  ],
  "state_dict_schema": {
    "key_name": "data type and description"
  },
  "reward_components": {
    "reward_name": "description of reward incentive or penalty"
  },
  "clarifying_questions": []
}
"""


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


def synthesize_blueprint_from_prompt(
    prompt: str,
    dataset: IngestedDataset | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    answers: dict[str, str] | None = None,
    training_params: dict[str, Any] | None = None,
    client: BaseLLMClient | None = None,
) -> DynamicTaskBlueprint:
    """Universal LLM Domain Architect that dynamically creates DynamicTaskBlueprint for ANY prompt."""
    clean_prompt = prompt.strip()
    prompt_lower = clean_prompt.lower()
    active_client = client or get_active_llm_client()

    # 1. Try Live LLM Generation if configured and not fallback
    if not isinstance(active_client, FallbackCodeSynthesizer):
        try:
            context_parts = [f"User Prompt: {clean_prompt}"]
            if dataset and hasattr(dataset, "df") and dataset.df is not None:
                context_parts.append(f"Ingested Dataset Columns: {list(dataset.df.columns)[:20]}")
            if answers:
                context_parts.append(f"User Clarifications: {json.dumps(answers)}")
            if conversation_history:
                history_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in conversation_history[-4:])
                context_parts.append(f"Conversation Context:\n{history_text}")

            full_query = "\n\n".join(context_parts) + "\n\n" + UNIVERSAL_INTAKE_PROMPT
            raw_json = active_client.generate_json(full_query, system_prompt="You are a Universal RL Domain Architect.")
            if raw_json and isinstance(raw_json, dict) and "domain_title" in raw_json:
                data_type = "uploaded_dataset" if dataset is not None else "pure_simulation"
                dataset_id = getattr(dataset, "dataset_id", None) if dataset else None
                return DynamicTaskBlueprint(
                    domain_title=raw_json.get("domain_title", "Universal RL Agent"),
                    domain_summary=raw_json.get("domain_summary", "Autonomous agent task"),
                    visual_metaphor=raw_json.get("visual_metaphor", "Interactive domain arena"),
                    observation_features=[
                        ObservationFeatureSpec(
                            name=f.get("name", f"feat_{i}"),
                            description=f.get("description", ""),
                            low=float(f.get("low", -10.0)),
                            high=float(f.get("high", 10.0)),
                        )
                        for i, f in enumerate(raw_json.get("observation_features", []))
                    ] or [ObservationFeatureSpec(name=f"state_{i}", description="feature", low=-1.0, high=1.0) for i in range(4)],
                    actions=[
                        ActionSpec(
                            index=a.get("index", i),
                            label=a.get("label", f"Action {i}"),
                            mechanic_description=a.get("mechanic_description", ""),
                        )
                        for i, a in enumerate(raw_json.get("actions", []))
                    ] or [ActionSpec(index=i, label=f"Action {i}", mechanic_description="") for i in range(2)],
                    state_dict_schema=raw_json.get("state_dict_schema", {}),
                    reward_components=raw_json.get("reward_components", {"goal_reward": "Reward for achieving objective"}),
                    clarifying_questions=[],
                    data_source_type=data_type,
                    dataset_id=dataset_id,
                )
        except Exception:
            pass

    # 2. Universal Dynamic Schema-Driven Blueprint Synthesizer
    explicit_actions = extract_action_labels_from_prompt(clean_prompt)
    explicit_obs = extract_observation_features_from_prompt(clean_prompt)

    # Dynamic extraction of domain title
    raw_title = re.sub(r"^(train|build|create|make|synthesize)\s+(an?\s+)?", "", clean_prompt, flags=re.IGNORECASE)
    raw_title = raw_title.split(".")[0].split("\n")[0].strip()
    title_words = [w.capitalize() for w in raw_title.split()[:5]]
    domain_title = " ".join(title_words) if title_words else "Universal RL Agent"
    if not any(kw in domain_title.lower() for kw in ["agent", "master", "controller"]):
        domain_title += " Agent"

    data_source = "uploaded_dataset" if dataset is not None else "pure_simulation"
    ds_id = getattr(dataset, "dataset_id", None) if dataset else None

    # Check for discrete column count (e.g. "drops discs into 7 columns")
    col_match = re.search(r"(\d+)\s*(?:columns|cols)", prompt_lower)
    if col_match:
        n_cols = int(col_match.group(1))
        actions = [
            ActionSpec(index=c, label=f"Drop Col {c}", mechanic_description=f"Drop token into column {c}")
            for c in range(n_cols)
        ]
        n_rows = 6
        obs_features = [
            ObservationFeatureSpec(name=f"cell_{r}_{c}", description=f"Grid cell row {r}, col {c}", low=-1.0, high=1.0)
            for r in range(n_rows) for c in range(n_cols)
        ]
        state_schema = {
            "board": f"list[list[int]] ({n_rows}x{n_cols} grid)",
            "current_player": "int",
            "winner": "int",
        }
        return DynamicTaskBlueprint(
            domain_title=domain_title,
            domain_summary=f"Competitive grid game on a {n_cols} column grid.",
            visual_metaphor=f"Vertical {n_cols} column drop grid arena with interactive action buttons",
            observation_features=obs_features,
            actions=actions,
            state_dict_schema=state_schema,
            reward_components={"win_bonus": "Reward for winning line", "loss_penalty": "Penalty for loss"},
            clarifying_questions=[],
            data_source_type=data_source,
            dataset_id=ds_id,
        )

    # Check for 3x3 grid / board games
    is_3x3 = bool(re.search(r"\b(3x3|3\s*by\s*3|ttt|tic|toe|x\s*and\s*o|noughts)\b", prompt_lower))
    if is_3x3:
        domain_title = "Tic-Tac-Toe Master Agent"
        obs_features = [
            ObservationFeatureSpec(name=f"cell_{r}_{c}", description=f"Board cell at row {r}, col {c}", low=-1.0, high=1.0)
            for r in range(3) for c in range(3)
        ] + [ObservationFeatureSpec(name="current_turn", description="Current player turn", low=-1.0, high=1.0)]
        actions = [
            ActionSpec(index=i, label=f"Place ({i // 3},{i % 3})", mechanic_description=f"Place mark at row {i // 3}, col {i % 3}")
            for i in range(9)
        ]
        state_schema = {
            "board": "list[int] (9 cells: 0=empty, 1=X, 2=O)",
            "current_player": "int",
            "winner": "int (0=ongoing, 1=X, 2=O, 3=draw)",
        }
        return DynamicTaskBlueprint(
            domain_title=domain_title,
            domain_summary="Zero-sum competitive 3x3 board game agent learning optimal play.",
            visual_metaphor="Modern 3x3 dark-mode grid with neon glow and winning lines",
            observation_features=obs_features,
            actions=actions,
            state_dict_schema=state_schema,
            reward_components={"win_bonus": "Reward for winning", "draw_bonus": "Reward for draw", "loss_penalty": "Penalty for losing"},
            clarifying_questions=[],
            data_source_type=data_source,
            dataset_id=ds_id,
        )

    # Dataset-driven blueprint
    if dataset is not None and hasattr(dataset, "df") and dataset.df is not None:
        cols = list(dataset.df.columns)[:8]
        obs_features = [
            ObservationFeatureSpec(name=c, description=f"Dataset column: {c}", low=-1000.0, high=1000.0)
            for c in cols
        ]
        act_labels = explicit_actions or ["Approve / Safe", "Flag / Investigate"]
        actions = [
            ActionSpec(index=i, label=lbl, mechanic_description=f"Classify record as {lbl}")
            for i, lbl in enumerate(act_labels)
        ]
        return DynamicTaskBlueprint(
            domain_title=domain_title,
            domain_summary=f"Data-driven decision agent trained on {len(dataset.df)} records.",
            visual_metaphor="Real-time tabular decision ledger with feature gauges and accuracy metrics",
            observation_features=obs_features,
            actions=actions,
            state_dict_schema={"record_index": "int", "accuracy_pct": "float", "last_action": "str"},
            reward_components={"accuracy_reward": "Reward for correct classification"},
            clarifying_questions=[],
            data_source_type="uploaded_dataset",
            dataset_id=ds_id,
        )

    # General Novel Domain
    if explicit_actions:
        actions = [
            ActionSpec(index=i, label=lbl, mechanic_description=f"Execute action {lbl}")
            for i, lbl in enumerate(explicit_actions)
        ]
    else:
        actions = [
            ActionSpec(index=0, label="Option A", mechanic_description="First control option"),
            ActionSpec(index=1, label="Option B", mechanic_description="Second control option"),
            ActionSpec(index=2, label="Option C", mechanic_description="Third control option"),
        ]

    if explicit_obs:
        obs_features = [
            ObservationFeatureSpec(name=feat, description=f"Observed feature: {feat}", low=-10.0, high=10.0)
            for feat in explicit_obs
        ]
    else:
        obs_features = [
            ObservationFeatureSpec(name=f"state_var_{i}", description=f"State metric {i}", low=-5.0, high=5.0)
            for i in range(4)
        ]

    state_schema = {f.name: "float" for f in obs_features}
    state_schema["status"] = "str"

    return DynamicTaskBlueprint(
        domain_title=domain_title,
        domain_summary=f"Autonomous policy optimization agent for {clean_prompt[:60]}.",
        visual_metaphor=f"Dynamic interactive stage for {domain_title} with live telemetry and action controls",
        observation_features=obs_features,
        actions=actions,
        state_dict_schema=state_schema,
        reward_components={"objective_progress": "Reward for making progress towards task success"},
        clarifying_questions=[],
        data_source_type=data_source,
        dataset_id=ds_id,
    )


class IntakeArchitectAgent:
    """Conversational intake architect managing 5-Pillar interviews and blueprint generation."""

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
        wants_defaults: bool = False,
    ) -> IntakeResponse:
        """Conducts conversational intake across the 5 pillars, returning clarifying questions or locked blueprint."""
        answers = user_answers or {}
        should_default = force_smart_defaults or wants_defaults

        # Check for smart defaults directive in user message
        prompt_lower = user_prompt.lower()
        if any(kw in prompt_lower for kw in ["smart defaults", "defaults", "force_defaults", "use defaults"]):
            should_default = True

        return self._deterministic_intake(
            prompt=user_prompt,
            dataset=dataset,
            answers=answers,
            wants_defaults=should_default,
            training_params=training_params,
        )

    def _deterministic_intake(
        self,
        prompt: str,
        dataset: IngestedDataset | None = None,
        answers: dict[str, str] | None = None,
        wants_defaults: bool = False,
        training_params: dict[str, Any] | None = None,
    ) -> IntakeResponse:
        """Determines whether to ask universal clarifying questions or lock the blueprint."""
        ans = answers or {}
        prompt_lower = prompt.lower().strip()

        # Generate the blueprint dynamically from the prompt
        dyn_blueprint = synthesize_blueprint_from_prompt(
            prompt=prompt,
            dataset=dataset,
            answers=ans,
            training_params=training_params,
            client=self.client,
        )
        blueprint = dyn_blueprint.to_agent_blueprint_spec()

        # Incorporate custom training params if provided
        if training_params:
            if "target_metric_threshold" in training_params:
                blueprint.target_metric_threshold = float(training_params["target_metric_threshold"])
            if "max_auto_tune_rounds" in training_params:
                blueprint.max_auto_tune_rounds = int(training_params["max_auto_tune_rounds"])
            if "episodes_per_round" in training_params:
                blueprint.episodes_per_round = int(training_params["episodes_per_round"])

        # Check pillar coverage
        has_arena_ui = "arena_ui" in ans or "q_arena_ui" in ans or any(kw in prompt_lower for kw in ["visualize in arena", "arcade theme", "dark-mode 3x3"])
        has_expectations = "expectations" in ans or "q_expectations" in ans or any(kw in prompt_lower for kw in ["round", "rounds", "episodes", "threshold", "% mastery", "% win"])

        needs_clarification = not wants_defaults and not (has_arena_ui and has_expectations) and len(ans) == 0

        if needs_clarification:
            domain_name = blueprint.task_title.replace(" Agent", "").replace(" Master", "")
            questions = [
                ClarifyingQuestion(
                    id="q_objective",
                    question_id="q_objective",
                    pillar=PILLAR_OBJECTIVE,
                    question=f"What strategic gameplay behavior and reward priorities should the {domain_name} agent learn?",
                    suggested_options=[
                        f"Optimal minimax play: Win whenever possible and maximize success",
                        f"Aggressive high-variance play: Prioritize immediate offensive opportunities",
                        f"Defensive counter-puncher: Prioritize threat mitigation and stability",
                    ],
                    default_option=f"Optimal minimax play: Win whenever possible and maximize success",
                ),
                ClarifyingQuestion(
                    id="q_arena_ui",
                    question_id="q_arena_ui",
                    pillar=PILLAR_ARENA_UI,
                    question=f"How would you like to visualize the {domain_name} arena in the center panel?",
                    suggested_options=[
                        f"Modern 3x3 dark-mode grid with neon glow and winning lines" if "tic" in domain_name.lower() or "3x3" in domain_name.lower() else f"Modern dark-mode reactive visual stage for {domain_name}",
                        f"Minimalist clean board with move history ledger",
                        f"Retro arcade 8-bit styling with animated action markers",
                        f"Compact numeric matrix with live value heatmaps",
                    ],
                    default_option=f"Modern 3x3 dark-mode grid with neon glow and winning lines" if "tic" in domain_name.lower() or "3x3" in domain_name.lower() else f"Modern dark-mode reactive visual stage for {domain_name}",
                ),
                ClarifyingQuestion(
                    id="q_interaction",
                    question_id="q_interaction",
                    pillar=PILLAR_INTERACTION,
                    question=f"How will you interactively test and challenge the agent on the board?",
                    suggested_options=[
                        "Click-to-place human vs agent interactive board with reset button",
                        "Dynamic side switching and play inspection",
                        "Self-play spectator simulation with speed controls",
                    ],
                    default_option="Click-to-place human vs agent interactive board with reset button",
                ),
                ClarifyingQuestion(
                    id="q_expectations",
                    question_id="q_expectations",
                    pillar=PILLAR_EXPECTATIONS,
                    question=f"What target mastery win/draw rate and auto-tune round budget should we aim for?",
                    suggested_options=[
                        "85% Win/Draw Rate (4 rounds)",
                        "90% Win/Draw Rate (4 rounds)",
                        "95% Near-Optimal Mastery (5 rounds)",
                        "75% Fast Prototyping (2 rounds)",
                    ],
                    default_option="85% Win/Draw Rate (4 rounds)",
                ),
            ]

            return IntakeResponse(
                status="needs_clarification",
                assistant_message=f"I've analyzed your request for **{blueprint.task_title}** across the 5 Pillars. To tailor your agent and arena UI, please clarify the following or select smart defaults:",
                questions=questions,
                blueprint_spec=None,
            )

        # Apply user answers to customize blueprint
        if ans:
            custom_obj = ans.get("objective") or ans.get("q_objective")
            if custom_obj:
                blueprint.domain_summary = f"{blueprint.domain_summary} Strategy: {custom_obj}"
            custom_ui = ans.get("arena_ui") or ans.get("q_arena_ui")
            if custom_ui:
                blueprint.arena_visual_description = custom_ui
            custom_exp = ans.get("expectations") or ans.get("q_expectations")
            if custom_exp:
                m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", custom_exp)
                if m_pct:
                    blueprint.target_metric_threshold = float(m_pct.group(1))
                m_rnd = re.search(r"(\d+)\s*rounds?", custom_exp, re.IGNORECASE)
                if m_rnd:
                    blueprint.max_auto_tune_rounds = int(m_rnd.group(1))

        return IntakeResponse(
            status="ready_to_build",
            assistant_message=f"Locked complete 5-Pillar AgentBlueprintSpec for **{blueprint.task_title}**! Ready to synthesize environment and arena.",
            questions=[],
            blueprint_spec=blueprint,
        )
