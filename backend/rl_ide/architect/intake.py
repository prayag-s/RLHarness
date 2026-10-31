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
from rl_ide.security.constraints import extract_hard_constraints


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


_LIST_SPLIT = re.compile(r"\s*(?:,|;|\s+or\s+|\s+and\s+)\s*", re.IGNORECASE)


def _clean_items(raw: str) -> list[str]:
    """Splits an enumeration into usable label candidates."""
    text = raw.strip().strip(".:;")
    items = [
        re.sub(r"^(?:or|and|nor)\s+", "", i.strip(" .:;\"'"), flags=re.IGNORECASE)
        for i in _LIST_SPLIT.split(text)
        if i.strip(" .:;\"'")
    ]
    cleaned: list[str] = []
    for item in items:
        if not item or len(item) > 60:
            continue
        if item.lower().startswith(("using ", "use ", "and use", "since ", "because ")):
            break
        if item not in cleaned:
            cleaned.append(item)
    return cleaned


def _looks_like_labels(items: list[str]) -> bool:
    """Rejects prose that happens to contain commas."""
    if not 2 <= len(items) <= 12:
        return False
    capitalised = sum(1 for i in items if i[:1].isupper() or i[:1].isdigit())
    return capitalised >= max(2, int(0.6 * len(items)))


def extract_action_labels_from_prompt(text: str) -> list[str] | None:
    """Extracts explicitly enumerated action labels from the user prompt.

    Handles quoted lists (``actions: 'A', 'B'``) and plain enumerations introduced by a cue
    (``route each ticket to the correct queue: A, B, or C``) without any domain vocabulary.
    """
    if not text:
        return None

    quoted = re.search(r"(?:actions?|options?|queues?|into)\s*:?\s*([\'\"][^\.\n]+)", text, re.IGNORECASE)
    if quoted:
        acts = [a.strip() for a in re.findall(r"[\'\"]([^\'\"]+)[\'\"]", quoted.group(1)) if a.strip()]
        if len(acts) >= 2:
            return acts

    cue_patterns = [
        r"(?:actions?|options?|queues?|categories|classes|decision|label)\s*(?:are|is)?\s*:\s*([^\n.;]+)",
        r"(?:classify|route|sort|assign|label|categoris|categoriz|dispatch)\b[^:.;\n]{0,60}?\binto\s*:?\s*([^\n.;]+)",
        r"(?:select|choose|decide)\b[^:.;\n]{0,40}?(?:between|among|from)\s*:?\s*([^\n.;]+)",
    ]
    for pattern in cue_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        items = _clean_items(match.group(1))
        if _looks_like_labels(items):
            return items

    colon = re.search(r":\s*([^\n.;]+)", text)
    if colon:
        items = _clean_items(colon.group(1))
        if len(items) >= 3 and _looks_like_labels(items):
            return items

    trailing = re.search(r"\bto\s+([A-Z0-9][^.;\n]{10,220})", text)
    if trailing:
        items = _clean_items(trailing.group(1))
        if len(items) >= 3 and _looks_like_labels(items):
            return items
    return None


def enforce_explicit_actions(
    blueprint: DynamicTaskBlueprint,
    explicit_actions: list[str] | None,
) -> DynamicTaskBlueprint:
    """The user's enumerated actions are a contract: never silently collapse them.

    A generic 2-action blueprint for a prompt that names four queues silently degrades the task
    (and the reported metric) without any error, so the declared list wins.
    """
    if not explicit_actions or len(explicit_actions) < 2:
        return blueprint

    def _norm(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()

    current = [_norm(a.label) for a in blueprint.actions]
    wanted = [_norm(label) for label in explicit_actions]
    if current == wanted:
        return blueprint

    blueprint.actions = [
        ActionSpec(index=i, label=label, mechanic_description=f"Execute action {label}")
        for i, label in enumerate(explicit_actions)
    ]
    return blueprint


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


def _synthesize_blueprint_from_prompt_impl(
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
    raw_title = re.sub(
        r"^(train|build|create|make|synthesize)\s+(an?\s+)?(agent\s+to\s+)?(play\s+|for\s+)?",
        "",
        clean_prompt,
        flags=re.IGNORECASE,
    )
    raw_title = raw_title.split(".")[0].split("\n")[0].strip()
    def _title_word(word: str) -> str:
        return "-".join(part[:1].upper() + part[1:] if part else "" for part in word.split("-"))

    title_words = [_title_word(w) for w in raw_title.split()[:5]]
    domain_title = " ".join(title_words) if title_words else "Universal RL Agent"
    if not any(kw in domain_title.lower() for kw in ["agent", "master", "controller"]):
        domain_title += " Agent"

    data_source = "uploaded_dataset" if dataset is not None else "pure_simulation"
    ds_id = getattr(dataset, "dataset_id", None) if dataset else None

    # Generic structure inference (no domain nouns/templates).
    # 1) Explicit grid dimensions: "3x3", "6 x 7", "4 by 4"
    grid_match = re.search(r"\b(\d+)\s*(?:x|×|by)\s*(\d+)\b", prompt_lower)
    if grid_match:
        rows, cols = int(grid_match.group(1)), int(grid_match.group(2))
        n_cells = rows * cols
        actions = [
            ActionSpec(
                index=i,
                label=f"Cell {i // cols},{i % cols}",
                mechanic_description=f"Select grid cell at row {i // cols}, column {i % cols}",
            )
            for i in range(n_cells)
        ]
        obs_features = [
            ObservationFeatureSpec(
                name=f"cell_{i // cols}_{i % cols}",
                description=f"Grid cell row {i // cols}, column {i % cols}",
                low=-1.0,
                high=1.0,
            )
            for i in range(n_cells)
        ] + [ObservationFeatureSpec(name="current_turn", description="Whose turn it is", low=-1.0, high=1.0)]
        state_schema = {
            "board": f"list[{n_cells}] cell values ({rows}x{cols} grid, row-major)",
            "current_player": "int",
            "winner": "int (0=ongoing)",
        }
        return DynamicTaskBlueprint(
            domain_title=domain_title,
            domain_summary=f"Turn-based {rows}x{cols} grid task.",
            visual_metaphor=f"{rows}x{cols} interactive grid arena with per-cell selection",
            observation_features=obs_features,
            actions=actions,
            state_dict_schema=state_schema,
            reward_components={
                "goal_bonus": "Reward for achieving the objective",
                "invalid_action_penalty": "Penalty for an illegal selection",
                "draw_bonus": "Reward for a neutral/terminal draw",
            },
            clarifying_questions=[],
            data_source_type=data_source,
            dataset_id=ds_id,
        )

    # 2) Explicit discrete choice count: "7 columns", "4 options", "3 classes", "5 lanes"
    count_match = re.search(
        r"\b(\d+)\s*(columns?|cols?|lanes?|options?|choices?|actions?|classes?|categories?|bins?|levels?|floors?|stops?|servers?|pumps?|queues?|tracks?)\b",
        prompt_lower,
    )
    if count_match and 2 <= int(count_match.group(1)) <= 50:
        n_choices = int(count_match.group(1))
        noun = count_match.group(2).rstrip("s")
        actions = [
            ActionSpec(
                index=i,
                label=f"{noun.capitalize()} {i + 1}",
                mechanic_description=f"Choose {noun} {i + 1}",
            )
            for i in range(n_choices)
        ]
        obs_features = [
            ObservationFeatureSpec(name=f"state_{i + 1}", description=f"Observed state for {noun} {i + 1}", low=-10.0, high=10.0)
            for i in range(n_choices)
        ]
        state_schema = {
            f"state_{i + 1}": "float" for i in range(n_choices)
        }
        state_schema["selected_index"] = "int"
        return DynamicTaskBlueprint(
            domain_title=domain_title,
            domain_summary=f"Discrete decision task over {n_choices} {noun}s.",
            visual_metaphor=f"Decision console with {n_choices} {noun} controls and live telemetry",
            observation_features=obs_features,
            actions=actions,
            state_dict_schema=state_schema,
            reward_components={"objective_progress": "Reward for moving towards the objective"},
            clarifying_questions=[],
            data_source_type=data_source,
            dataset_id=ds_id,
        )

    # Dataset-driven blueprint
    if dataset is not None and hasattr(dataset, "df") and dataset.df is not None:
        from rl_ide.ingest.profiler import infer_label_column

        act_labels = explicit_actions or ["Approve / Safe", "Flag / Investigate"]
        actions = [
            ActionSpec(index=i, label=lbl, mechanic_description=f"Classify record as {lbl}")
            for i, lbl in enumerate(act_labels)
        ]
        label_column = infer_label_column(dataset.df, len(actions))
        # Never expose the target column as an observation (label leakage).
        cols = [c for c in dataset.df.columns if c != label_column][:8]
        obs_features = [
            ObservationFeatureSpec(name=c, description=f"Dataset column: {c}", low=-1000.0, high=1000.0)
            for c in cols
        ]
        if label_column:
            summary = f"Data-driven decision agent trained on {len(dataset.df)} records."
            metaphor = "Real-time tabular decision ledger with feature gauges and accuracy metrics"
            state_schema = {"record_index": "int", "accuracy_pct": "float", "last_action": "int"}
            data_source = "uploaded_dataset"
        else:
            # Telemetry-only dataset: it informs observations, but the task is a control problem.
            summary = f"Decision agent operating on live telemetry from {len(dataset.df)} records."
            metaphor = "Live telemetry dashboard with action controls and metric history"
            state_schema = {c: "float" for c in cols}
            state_schema["selected_index"] = "int"
            data_source = "hybrid_doc_and_sim"
        return DynamicTaskBlueprint(
            domain_title=domain_title,
            domain_summary=summary,
            visual_metaphor=metaphor,
            observation_features=obs_features,
            actions=actions,
            state_dict_schema=state_schema,
            reward_components={"accuracy_reward": "Reward for correct classification"},
            clarifying_questions=[],
            data_source_type=data_source,
            dataset_id=ds_id,
            label_column=label_column,
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


# Words that mean "this case unfolds over several decisions" rather than "classify this row".
_SEQUENTIAL_CUES = (
    "journey", "playbook", "lifecycle", "funnel", "over time", "over the quarter", "over a quarter",
    "multi-step", "multistep", "multi step", "step by step", "sequence of", "sequential",
    "onboarding flow", "nurture", "cadence", "follow-up plan", "retention play", "renewal play",
    "multi-touch", "multitouch", "multi touch", "claim journey", "escalation path", "over several",
)
# Words that mean the opposite: a single decision we should not turn into a journey.
_ONE_SHOT_CUES = ("classify", "triage each", "route each", "categorise each", "categorize each", "score each")
_SEQUENTIAL_STAGE_RE = re.compile(
    r"(\d{1,2})\s*[-\u2013]?\s*(?:step|stage|phase|touch|cycle)s?", re.IGNORECASE
)


def infer_task_structuring(prompt: str) -> tuple[str, int]:
    """Decides whether a request describes a journey or a single decision.

    A renewal play, a claim journey or a multi-touch cadence is one case that unfolds over several
    decisions with an outcome at the end; treating it as N independent guesses is a modelling error
    the user should not have to correct afterwards (G10). Explicit "one-shot" wording wins over a
    stray mention of "steps".
    """
    text = (prompt or "").lower()
    if any(cue in text for cue in _ONE_SHOT_CUES):
        return "one_shot", 1
    matched = [cue for cue in _SEQUENTIAL_CUES if cue in text]
    if not matched:
        return "one_shot", 1
    stages = _SEQUENTIAL_STAGE_RE.search(text)
    if stages:
        length = max(1, min(int(stages.group(1)), 12))
    else:
        # Five stages is the shortest length at which "state across steps" matters.
        length = 5
    return "sequential", length


def apply_task_structuring(blueprint: AgentBlueprintSpec, prompt: str) -> AgentBlueprintSpec:
    """Sets `task_structuring`/`episode_length` from the request unless it was already chosen."""
    structuring, length = infer_task_structuring(prompt)
    if structuring == "one_shot":
        return blueprint
    blueprint.task_structuring = structuring
    blueprint.episode_length = length
    if "journey" not in (blueprint.success_metric_name or "").lower():
        blueprint.success_metric_name = "Journey Completion Rate (%)"
    return blueprint


def apply_answers_to_blueprint(
    blueprint: AgentBlueprintSpec,
    answers: dict[str, Any] | None,
) -> AgentBlueprintSpec:
    """Applies optional intake answers (by question id or pillar) to a blueprint in place."""
    ans = answers or {}

    def _pick(*keys: str) -> str | None:
        for key in keys:
            val = ans.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None

    custom_obj = _pick("objective", "q_objective")
    if custom_obj:
        blueprint.domain_summary = f"{blueprint.domain_summary} Objective: {custom_obj}"
    custom_ui = _pick("arena_ui", "q_arena_ui")
    if custom_ui:
        blueprint.arena_visual_description = custom_ui
        blueprint.visual_metaphor = custom_ui
    custom_exp = _pick("expectations", "q_expectations")
    if custom_exp:
        m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", custom_exp)
        if m_pct:
            blueprint.target_metric_threshold = float(m_pct.group(1))
        m_rnd = re.search(r"(\d+)\s*rounds?", custom_exp, re.IGNORECASE)
        if m_rnd:
            blueprint.max_auto_tune_rounds = int(m_rnd.group(1))

    return blueprint


def synthesize_blueprint_from_prompt(
    prompt: str,
    dataset: IngestedDataset | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    answers: dict[str, str] | None = None,
    training_params: dict[str, Any] | None = None,
    client: BaseLLMClient | None = None,
) -> DynamicTaskBlueprint:
    """Generates the blueprint, then attaches any explicit hard constraints from the objective.

    The core synthesis is unchanged; this wrapper makes "must never X when feature >= N" a
    first-class, executable rule instead of prose the reward function may or may not learn.
    """
    blueprint = _synthesize_blueprint_from_prompt_impl(
        prompt=prompt,
        dataset=dataset,
        conversation_history=conversation_history,
        answers=answers,
        training_params=training_params,
        client=client,
    )
    try:
        feature_names = (
            [str(c) for c in dataset.df.columns]
            if dataset is not None and hasattr(dataset, "df")
            else []
        )
        labels = [str(a.label) for a in blueprint.actions]
        blueprint.hard_constraints = [
            c.model_dump()
            for c in extract_hard_constraints(prompt, labels, feature_names)
        ]
    except Exception:
        # Constraint extraction must never break synthesis; an empty list is the safe default.
        blueprint.hard_constraints = []
    return blueprint


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
            conversation_history=conversation_history,
        )

    def _deterministic_intake(
        self,
        prompt: str,
        dataset: IngestedDataset | None = None,
        answers: dict[str, str] | None = None,
        wants_defaults: bool = False,
        training_params: dict[str, Any] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> IntakeResponse:
        """Determines whether to ask universal clarifying questions or lock the blueprint."""
        ans = answers or {}
        prompt_lower = prompt.lower().strip()

        # Generate the blueprint dynamically from the prompt
        dyn_blueprint = synthesize_blueprint_from_prompt(
            prompt=prompt,
            dataset=dataset,
            conversation_history=conversation_history,
            answers=ans,
            training_params=training_params,
            client=self.client,
        )
        dyn_blueprint = enforce_explicit_actions(
            dyn_blueprint, extract_action_labels_from_prompt(prompt)
        )
        blueprint = dyn_blueprint.to_agent_blueprint_spec()
        # A journey-shaped request must not be modelled as N independent guesses (G10).
        blueprint = apply_task_structuring(blueprint, prompt)

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
            is_data_task = blueprint.data_source_type == "uploaded_dataset"
            metaphor = blueprint.visual_metaphor or f"a live view of {domain_name}"
            questions = [
                ClarifyingQuestion(
                    id="q_objective",
                    question_id="q_objective",
                    pillar=PILLAR_OBJECTIVE,
                    question=f"What outcome and trade-offs should the {domain_name} agent optimize?",
                    suggested_options=[
                        "Maximize the primary success metric while avoiding costly mistakes",
                        "Maximize speed/efficiency even if risk is slightly higher",
                        "Prioritize safety and stability over peak performance",
                    ],
                    default_option="Maximize the primary success metric while avoiding costly mistakes",
                ),
                ClarifyingQuestion(
                    id="q_arena_ui",
                    question_id="q_arena_ui",
                    pillar=PILLAR_ARENA_UI,
                    question=f"How should the {domain_name} arena look in the center panel?",
                    suggested_options=[
                        f"Render it as described: {metaphor}",
                        "Minimalist dark-mode layout focused on the main metric",
                        "Detailed telemetry dashboard with history and per-action probabilities",
                        "Retro arcade styling with animated state changes",
                    ],
                    default_option=f"Render it as described: {metaphor}",
                ),
                ClarifyingQuestion(
                    id="q_interaction",
                    question_id="q_interaction",
                    pillar=PILLAR_INTERACTION,
                    question=(
                        "How do you want to inspect or override individual records?"
                        if is_data_task
                        else f"How do you want to test the {domain_name} agent interactively?"
                    ),
                    suggested_options=(
                        [
                            "Step through records and override the agent's decision",
                            "Review misclassified records with the agent's probabilities",
                            "Run a batch audit and export the confusion summary",
                        ]
                        if is_data_task
                        else [
                            "Interactive controls to drive the environment turn by turn",
                            "Watch live training episodes with replay",
                            "Both: play manually and watch training",
                        ]
                    ),
                    default_option=(
                        "Step through records and override the agent's decision"
                        if is_data_task
                        else "Interactive controls to drive the environment turn by turn"
                    ),
                ),
                ClarifyingQuestion(
                    id="q_expectations",
                    question_id="q_expectations",
                    pillar=PILLAR_EXPECTATIONS,
                    question="What success threshold and auto-tune round budget should we aim for?",
                    suggested_options=[
                        f"85% {blueprint.success_metric_name} (4 rounds)",
                        f"90% {blueprint.success_metric_name} (4 rounds)",
                        f"95% {blueprint.success_metric_name} (5 rounds)",
                        "75% Fast prototyping (2 rounds)",
                    ],
                    default_option=f"85% {blueprint.success_metric_name} (4 rounds)",
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
            blueprint = apply_answers_to_blueprint(blueprint, ans)

        return IntakeResponse(
            status="ready_to_build",
            assistant_message=f"Locked complete 5-Pillar AgentBlueprintSpec for **{blueprint.task_title}**! Ready to synthesize environment and arena.",
            questions=[],
            blueprint_spec=blueprint,
        )
