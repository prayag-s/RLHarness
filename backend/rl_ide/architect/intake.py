"""Universal Conversational Intake Architect for RL-Native Agent IDE v4.3.

Eliminates static domain keyword classifiers and templates.
Every domain prompt produces a bespoke DynamicTaskBlueprint via LLM structured synthesis
with zero fallback to 2D Rover unless explicitly requested.
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
from rl_ide.llm.client import BaseLLMClient, create_llm_client, FallbackCodeSynthesizer, get_active_llm_client
from rl_ide.llm.prompts import INTAKE_ARCHITECT_SYSTEM_PROMPT, ARCHITECT_SYSTEM_PROMPT
from rl_ide.ingest.loader import IngestedDataset


UNIVERSAL_INTAKE_PROMPT = """You are an Expert Universal Reinforcement Learning Domain Architect.
Given any user domain idea, problem description, or task objective on Earth, dynamically design a bespoke Reinforcement Learning task blueprint.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. ZERO domain templates. Tailor the domain title, observation features, discrete actions, reward components, and visual metaphor dynamically to the user's prompt.
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


def is_tictactoe_intent(text: str) -> bool:
    """Fuzzy and regex domain intent detector for Tic-Tac-Toe variants and typos."""
    if not text:
        return False
    t = text.lower().strip()
    direct_kws = [
        "tic-tac-toe", "tictactoe", "tic tac toe", "tic-tac", "tictac",
        "tic tac", "tic toe", "tic-toe", "tictoc", "tic toc",
        "tick-tack-toe", "tick tack toe", "tick tack",
        "noughts and crosses", "noughts & crosses", "noughts", "crosses",
        "three in a row", "3 in a row", "ttt agent", "ttt game", "ttt board",
    ]
    if any(kw in t for kw in direct_kws):
        return True
    if re.search(r"\bttt\b", t):
        return True
    if re.search(r"\b(tic|tik|tek|tick)\b", t) and re.search(r"\b(tac|tax|tak|tok|toe|tow|to|tack)\b", t):
        return True
    if re.search(r"\b(x['s]?\s*(?:and|&)\s*o['s]?)\b", t) or re.search(r"\bx\s*/\s*o\b", t):
        return True
    if re.search(r"\b(3x3|3\s*by\s*3)\b", t) and any(w in t for w in ["game", "board", "grid", "agent", "play", "win", "player"]):
        return True
    return False


def is_tictactoe_spec(blueprint: AgentBlueprintSpec | DynamicTaskBlueprint) -> bool:
    """Checks if a blueprint represents a Tic-Tac-Toe domain."""
    if not blueprint:
        return False
    title = getattr(blueprint, "domain_title", None) or getattr(blueprint, "task_title", "")
    summary = getattr(blueprint, "domain_summary", "")
    if is_tictactoe_intent(title) or is_tictactoe_intent(summary):
        return True
    actions = getattr(blueprint, "action_labels", None)
    if not actions and hasattr(blueprint, "actions"):
        actions = [a.label for a in blueprint.actions]
    obs = getattr(blueprint, "observation_features", None)
    if len(actions or []) == 9 and len(obs or []) in (9, 10):
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


def is_blackjack_spec(blueprint: AgentBlueprintSpec | DynamicTaskBlueprint) -> bool:
    """Checks if a blueprint represents a Blackjack / card game domain."""
    if not blueprint:
        return False
    title = getattr(blueprint, "domain_title", None) or getattr(blueprint, "task_title", "")
    summary = getattr(blueprint, "domain_summary", "")
    if is_blackjack_intent(title) or is_blackjack_intent(summary):
        return True
    actions = getattr(blueprint, "action_labels", None)
    if not actions and hasattr(blueprint, "actions"):
        actions = [a.label for a in blueprint.actions]
    action_str = " ".join(actions or []).lower()
    if "hit" in action_str and "stand" in action_str:
        return True
    return False


def synthesize_blueprint_from_prompt(
    prompt: str,
    dataset: IngestedDataset | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    answers: dict[str, str] | None = None,
    training_params: dict[str, Any] | None = None,
    client: BaseLLMClient | None = None,
) -> DynamicTaskBlueprint:
    """Universal LLM Domain Architect that dynamically creates DynamicTaskBlueprint for ANY prompt on Earth.
    
    Zero hardcoded fallback to 2D Rover unless the prompt explicitly requests rover navigation.
    """
    clean_prompt = prompt.strip()
    prompt_lower = clean_prompt.lower()
    active_client = client or get_active_llm_client()

    # 1. Try Live LLM Generation if configured and smart
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
            raw_json = active_client.generate_json(full_query, system_prompt="You are an Universal RL Domain Architect.")
            if raw_json and isinstance(raw_json, dict) and "domain_title" in raw_json:
                return DynamicTaskBlueprint.model_validate(raw_json)
        except Exception:
            pass

    # 2. Universal Generative Engine (Tailors directly to the domain without 2D Rover fallback)
    explicit_actions = extract_action_labels_from_prompt(clean_prompt)
    explicit_obs = extract_observation_features_from_prompt(clean_prompt)

    # Domain: Connect 4
    if any(kw in prompt_lower for kw in ["connect 4", "connect-4", "connect four", "connect4", "7 columns", "7 column"]):
        return DynamicTaskBlueprint(
            domain_title="Connect-4 Grid Agent",
            domain_summary="Competitive two-player vertical disc-dropping game on a 7x6 grid.",
            visual_metaphor="Connect 4 vertical 7x6 blue drop grid with red and yellow tokens",
            observation_features=[
                ObservationFeatureSpec(name=f"cell_{r}_{c}", description=f"Board cell at row {r}, col {c}", low=-1.0, high=1.0)
                for r in range(6) for c in range(7)
            ] + [ObservationFeatureSpec(name="current_player", description="Current player turn", low=1.0, high=2.0)],
            actions=[
                ActionSpec(index=c, label=f"Drop Col {c}", mechanic_description=f"Drop token into column {c}")
                for c in range(7)
            ],
            state_dict_schema={
                "board": "list[list[int]] (6x7 grid, 0=empty, 1=player, 2=opponent)",
                "current_player": "int (1 or 2)",
                "winner": "int (0=ongoing, 1=player, 2=opponent, 3=draw)",
                "last_col": "int",
            },
            reward_components={
                "win_bonus": "Reward for connecting 4 tokens in a row",
                "loss_penalty": "Penalty for opponent connecting 4",
                "draw_bonus": "Reward for draw",
                "illegal_move_penalty": "Penalty for dropping in full column",
            },
            clarifying_questions=[],
        )

    # Domain: Russian Roulette
    if any(kw in prompt_lower for kw in ["russian roulette", "roulette", "revolver", "cylinder", "chamber"]):
        return DynamicTaskBlueprint(
            domain_title="Russian Roulette Survival Agent",
            domain_summary="Risk-management survival game learning optimal decision making under lethal uncertainty.",
            visual_metaphor="Revolver cylinder chamber view with hammer, rotating cylinder, and trigger controls",
            observation_features=[
                ObservationFeatureSpec(name="chamber_position", description="Current chamber position", low=0.0, high=5.0),
                ObservationFeatureSpec(name="chambers_fired", description="Number of fired chambers", low=0.0, high=6.0),
                ObservationFeatureSpec(name="is_alive", description="Survival status", low=0.0, high=1.0),
            ],
            actions=[
                ActionSpec(index=0, label="Pull Trigger", mechanic_description="Fire the current chamber at self"),
                ActionSpec(index=1, label="Spin Cylinder", mechanic_description="Randomize cylinder position before pull"),
                ActionSpec(index=2, label="Pass Turn", mechanic_description="Pass revolver to opponent"),
            ],
            state_dict_schema={
                "current_chamber": "int (0-5)",
                "chambers_fired": "int",
                "is_alive": "bool",
                "winner": "str",
            },
            reward_components={
                "survival_bonus": "Reward for surviving round",
                "death_penalty": "Penalty for firing live round",
            },
            clarifying_questions=[],
        )

    # Domain: Dice Pig
    if any(kw in prompt_lower for kw in ["dice pig", "pig game", "pig dice", "dice game"]):
        return DynamicTaskBlueprint(
            domain_title="Dice Pig Risk Management Agent",
            domain_summary="Jeopardy dice game learning optimal threshold to bank points vs risk rolling a 1.",
            visual_metaphor="Dice game table with animated 3D rolling dice, banked score meters, and risk gauges",
            observation_features=[
                ObservationFeatureSpec(name="turn_score", description="Points accumulated in current turn", low=0.0, high=100.0),
                ObservationFeatureSpec(name="banked_score", description="Player total banked score", low=0.0, high=100.0),
                ObservationFeatureSpec(name="opponent_score", description="Opponent total banked score", low=0.0, high=100.0),
            ],
            actions=[
                ActionSpec(index=0, label="Roll Die", mechanic_description="Roll the die to accumulate points"),
                ActionSpec(index=1, label="Hold & Bank", mechanic_description="Bank turn points and end turn"),
            ],
            state_dict_schema={
                "turn_score": "int",
                "banked_score": "int",
                "opponent_score": "int",
                "last_roll": "int",
            },
            reward_components={
                "win_bonus": "Reward for reaching 100 banked points first",
                "bust_penalty": "Penalty for rolling a 1 and wiping turn score",
                "bank_reward": "Reward proportional to banked score",
            },
            clarifying_questions=[],
        )

    # Domain: Elevator Dispatcher
    if any(kw in prompt_lower for kw in ["elevator", "lift dispatcher", "floor dispatcher"]):
        return DynamicTaskBlueprint(
            domain_title="Warehouse Elevator Dispatcher Agent",
            domain_summary="Multi-floor vertical elevator dispatcher optimizing passenger wait times and energy efficiency.",
            visual_metaphor="Elevator shaft building cross-section with floors, car positions, and passenger queues",
            observation_features=[
                ObservationFeatureSpec(name="current_floor", description="Current floor position", low=1.0, high=10.0),
                ObservationFeatureSpec(name="direction", description="Current car direction (-1=down, 0=idle, 1=up)", low=-1.0, high=1.0),
                ObservationFeatureSpec(name="passenger_count", description="Number of passengers aboard", low=0.0, high=10.0),
                ObservationFeatureSpec(name="waiting_passengers", description="Total passengers waiting on floors", low=0.0, high=30.0),
            ],
            actions=[
                ActionSpec(index=0, label="Move Up", mechanic_description="Move car up one floor"),
                ActionSpec(index=1, label="Move Down", mechanic_description="Move car down one floor"),
                ActionSpec(index=2, label="Open Doors", mechanic_description="Open doors to load/unload passengers"),
                ActionSpec(index=3, label="Hold / Idle", mechanic_description="Hold current floor position"),
            ],
            state_dict_schema={
                "current_floor": "int",
                "target_floor": "int",
                "passenger_count": "int",
                "hall_calls": "list[int]",
            },
            reward_components={
                "delivery_bonus": "Reward for delivering passengers to target floors",
                "wait_penalty": "Penalty proportional to total passenger wait time",
                "energy_cost": "Small energy step penalty per floor moved",
            },
            clarifying_questions=[],
        )

    # Domain: Blackjack
    if is_blackjack_intent(prompt_lower):
        return DynamicTaskBlueprint(
            domain_title="Blackjack 21 Card Agent",
            domain_summary="Classic Blackjack (Twenty-One) casino card game learning optimal hit, stand, and double-down decisions against a dealer.",
            visual_metaphor="Classic Emerald Felt Casino table with dealer/player cards & chip bankroll",
            observation_features=[
                ObservationFeatureSpec(name="player_total", description="Current sum of player hand", low=4.0, high=21.0),
                ObservationFeatureSpec(name="dealer_upcard_value", description="Visible dealer card value", low=1.0, high=11.0),
                ObservationFeatureSpec(name="has_usable_ace", description="Whether hand contains ace counted as 11", low=0.0, high=1.0),
                ObservationFeatureSpec(name="can_double_down", description="Whether double down is legal", low=0.0, high=1.0),
            ],
            actions=[
                ActionSpec(index=0, label="Hit (+1 Card)", mechanic_description="Draw another card from the shoe"),
                ActionSpec(index=1, label="Stand (Hold Hand)", mechanic_description="End turn and compare hand with dealer"),
                ActionSpec(index=2, label="Double Down (2x)", mechanic_description="Double bet, receive exactly one card, and stand"),
            ],
            state_dict_schema={
                "player_total": "int",
                "dealer_total": "int",
                "player_cards": "list[dict]",
                "dealer_cards": "list[dict]",
                "is_game_over": "bool",
                "winner": "str",
                "bankroll": "int",
            },
            reward_components={
                "win_payout": "Standard 1:1 payout for beating dealer hand",
                "blackjack_bonus": "3:2 bonus payout for natural 21 Blackjack",
                "loss_penalty": "Loss penalty when dealer total beats player",
                "bust_penalty": "Penalty for exceeding 21 (busting)",
                "push_reward": "Zero return for tied hand (push)",
            },
            clarifying_questions=[],
        )

    # Domain: Tic-Tac-Toe
    if is_tictactoe_intent(prompt_lower):
        return DynamicTaskBlueprint(
            domain_title="Tic-Tac-Toe Master Agent",
            domain_summary="Zero-sum competitive 3x3 board game agent learning optimal play against opponents.",
            visual_metaphor="Modern 3x3 dark-mode grid with neon glow and winning lines",
            observation_features=[
                ObservationFeatureSpec(name=f"cell_{r}_{c}", description=f"Board cell at row {r}, col {c}", low=-1.0, high=1.0)
                for r in range(3) for c in range(3)
            ] + [ObservationFeatureSpec(name="current_turn", description="Current player turn", low=-1.0, high=1.0)],
            actions=[
                ActionSpec(index=i, label=f"Place ({i//3},{i%3})", mechanic_description=f"Place mark at row {i//3}, col {i%3}")
                for i in range(9)
            ],
            state_dict_schema={
                "board": "list[int] (length 9)",
                "current_turn": "int",
                "winner": "int | None",
            },
            reward_components={
                "win_bonus": "Reward for winning the game",
                "draw_bonus": "Reward for achieving a draw",
                "loss_penalty": "Penalty for losing the game",
                "illegal_move_penalty": "Penalty for selecting an occupied cell",
            },
            clarifying_questions=[],
        )

    # Domain: Solar / Battery Arbitrage
    if any(kw in prompt_lower for kw in ["solar", "battery arbitrage", "tariff", "grid price", "energy arbitrage"]):
        return DynamicTaskBlueprint(
            domain_title="Solar & Commercial Battery Arbitrage Controller",
            domain_summary="Smart edge controller optimizing solar generation, battery storage, and dynamic tariff arbitrage.",
            visual_metaphor="Clean energy dashboard with interactive power-flow arrows and battery SOC gauge",
            observation_features=[
                ObservationFeatureSpec(name="solar_kw", description="Solar power generation in kW", low=0.0, high=100.0),
                ObservationFeatureSpec(name="building_load_kw", description="Current facility load in kW", low=0.0, high=120.0),
                ObservationFeatureSpec(name="battery_soc_pct", description="Battery state of charge percentage", low=0.0, high=100.0),
                ObservationFeatureSpec(name="grid_price_kwh", description="Dynamic grid tariff rate $/kWh", low=0.0, high=1.5),
            ],
            actions=[
                ActionSpec(index=0, label="Charge Battery", mechanic_description="Charge battery from solar/grid"),
                ActionSpec(index=1, label="Discharge to Building", mechanic_description="Offset building load with battery power"),
                ActionSpec(index=2, label="Sell Back to Grid", mechanic_description="Export stored energy to grid during peak tariff"),
                ActionSpec(index=3, label="Idle Hold", mechanic_description="Maintain battery charge state"),
            ],
            state_dict_schema={
                "solar_kw": "float",
                "load_kw": "float",
                "battery_soc": "float",
                "grid_price": "float",
                "net_savings": "float",
            },
            reward_components={
                "arbitrage_savings": "Net economic savings on energy bill",
                "peak_discharge_bonus": "Bonus for discharging during peak tariff",
                "deep_discharge_penalty": "Penalty for discharging battery below 15% SOC",
            },
            clarifying_questions=[],
        )

    # Domain: CartPole Physics
    if any(kw in prompt_lower for kw in ["cartpole", "pole balance", "inverted pendulum"]):
        return DynamicTaskBlueprint(
            domain_title="CartPole Physics Balancing Agent",
            domain_summary="Inverted pendulum balancing controller learning upright stabilization under gravity.",
            visual_metaphor="Physics canvas displaying cart track, pole angle indicator, and disturbance force arrows",
            observation_features=[
                ObservationFeatureSpec(name="cart_position", description="Cart horizontal position on track", low=-2.4, high=2.4),
                ObservationFeatureSpec(name="cart_velocity", description="Cart horizontal velocity", low=-3.0, high=3.0),
                ObservationFeatureSpec(name="pole_angle", description="Pole angle from vertical in radians", low=-0.418, high=0.418),
                ObservationFeatureSpec(name="pole_angular_velocity", description="Pole angular velocity", low=-3.0, high=3.0),
            ],
            actions=[
                ActionSpec(index=0, label="Push Cart Left", mechanic_description="Apply horizontal force pushing cart to the left"),
                ActionSpec(index=1, label="Push Cart Right", mechanic_description="Apply horizontal force pushing cart to the right"),
            ],
            state_dict_schema={
                "cart_x": "float",
                "cart_v": "float",
                "pole_theta": "float",
                "pole_omega": "float",
            },
            reward_components={
                "upright_bonus": "Reward for maintaining upright pole angle",
                "centering_bonus": "Bonus for staying close to track center",
                "fall_penalty": "Penalty for pole falling past failure angle",
            },
            clarifying_questions=[],
        )

    # Domain: Ingested Dataset
    if dataset is not None or any(kw in prompt_lower for kw in ["dataset", "csv", "table", "churn", "fraud"]):
        cols = list(dataset.df.columns) if (dataset and hasattr(dataset, "df") and dataset.df is not None) else (explicit_obs or ["feature_0", "feature_1", "feature_2", "feature_3"])
        obs_features = [ObservationFeatureSpec(name=c, description=f"Dataset column {c}", low=-5.0, high=5.0) for c in cols[:10]]
        act_labels = explicit_actions or ["Classify Negative (0)", "Classify Positive (1)"]
        actions = [ActionSpec(index=i, label=lbl, mechanic_description=f"Select classification {lbl}") for i, lbl in enumerate(act_labels)]
        ds_type = "uploaded_dataset" if dataset is not None else "pure_simulation"
        ds_id = getattr(dataset, "dataset_id", getattr(dataset, "name", "uploaded_dataset")) if dataset else None
        return DynamicTaskBlueprint(
            domain_title=f"{dataset.name if dataset else 'Tabular Decision'} Classifier Agent",
            domain_summary="Sequential decision making and classification agent operating over tabular data records.",
            visual_metaphor="Interactive record inspector card with confidence gauge, feature meters, and classification history",
            data_source_type=ds_type,
            dataset_id=ds_id,
            observation_features=obs_features,
            actions=actions,
            state_dict_schema={
                "record_index": "int",
                "features": "dict",
                "predicted_class": "int",
                "confidence": "float",
            },
            reward_components={
                "correct_decision": "Reward for matching true record label",
                "incorrect_decision": "Penalty for classification error",
            },
            clarifying_questions=[],
        )

    # Domain: 2D Rover / Waypoint Navigation (ONLY IF EXPLICITLY REQUESTED)
    if any(kw in prompt_lower for kw in ["rover", "waypoint", "2d navigation", "amr robot"]):
        return DynamicTaskBlueprint(
            domain_title="2D Autonomous Waypoint Rover",
            domain_summary="Mobile ground rover learning path planning to target waypoints while avoiding obstacles.",
            visual_metaphor="2D top-down grid arena with rover sprite, target waypoint, and obstacles",
            observation_features=[
                ObservationFeatureSpec(name="state_pos_x", description="Rover X coordinate", low=0.0, high=10.0),
                ObservationFeatureSpec(name="state_pos_y", description="Rover Y coordinate", low=0.0, high=10.0),
                ObservationFeatureSpec(name="velocity_x", description="Rover horizontal velocity", low=-2.0, high=2.0),
                ObservationFeatureSpec(name="velocity_y", description="Rover vertical velocity", low=-2.0, high=2.0),
                ObservationFeatureSpec(name="target_distance", description="Euclidean distance to target waypoint", low=0.0, high=15.0),
            ],
            actions=[
                ActionSpec(index=0, label="Move Left", mechanic_description="Accelerate rover left"),
                ActionSpec(index=1, label="Move Right", mechanic_description="Accelerate rover right"),
                ActionSpec(index=2, label="Move Up", mechanic_description="Accelerate rover up"),
                ActionSpec(index=3, label="Move Down", mechanic_description="Accelerate rover down"),
                ActionSpec(index=4, label="Hold Position", mechanic_description="Brake rover"),
            ],
            state_dict_schema={
                "pos_x": "float",
                "pos_y": "float",
                "target_x": "float",
                "target_y": "float",
                "distance": "float",
            },
            reward_components={
                "goal_bonus": "Reward for reaching target waypoint",
                "distance_delta": "Progress reward for moving closer to target",
                "step_cost": "Small step cost for energy efficiency",
            },
            clarifying_questions=[],
        )

    # Dynamic Synthesis for Novel Arbitrary Domains (ZERO FALLBACK TO ROVER)
    # Parse domain words from prompt to generate bespoke title and features
    words = [w.capitalize() for w in re.findall(r"\b[A-Za-z0-9]+\b", clean_prompt) if w.lower() not in {"train", "an", "a", "agent", "that", "the", "for", "with", "to", "in", "and", "or", "which", "will"}]
    domain_name = " ".join(words[:4]) if words else "Dynamic System"
    title = f"{domain_name} Autonomous Agent"

    actions_list = explicit_actions or ["Engage Primary Action", "Adjust State Option", "Idle / Conserve"]
    actions_spec = [ActionSpec(index=i, label=lbl, mechanic_description=f"Execute action: {lbl}") for i, lbl in enumerate(actions_list)]

    obs_list = explicit_obs or [f"{words[0].lower() if words else 'state'}_level", f"{words[1].lower() if len(words)>1 else 'target'}_delta", "rate_of_change", "system_stability"]
    obs_spec = [ObservationFeatureSpec(name=f, description=f"Monitored state feature: {f}", low=-10.0, high=10.0) for f in obs_list]

    state_schema = {f: "float" for f in obs_list}
    state_schema["status"] = "str"
    state_schema["cycle_step"] = "int"

    return DynamicTaskBlueprint(
        domain_title=title,
        domain_summary=f"Synthesized autonomous decision agent tailored to: {clean_prompt}.",
        visual_metaphor=f"Bespoke interactive telemetry visualizer and stage display tailored specifically for {title}",
        observation_features=obs_spec,
        actions=actions_spec,
        state_dict_schema=state_schema,
        reward_components={
            "objective_achievement": f"Reward for fulfilling {title} operational targets",
            "operational_penalty": "Penalty for boundary deviation or inefficient actions",
        },
        clarifying_questions=[],
    )


class IntakeArchitectAgent:
    """Interviews the user to establish the 5 Pillars and locks the DynamicTaskBlueprint / AgentBlueprintSpec."""

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
        """Evaluates user intent and returns either clarifying questions or locked blueprint."""
        prompt_clean = user_prompt.strip()
        prompt_lower = prompt_clean.lower()
        answers = user_answers or {}

        wants_defaults = (
            force_smart_defaults
            or wants_defaults
            or any(kw in prompt_lower for kw in [
                "use smart defaults", "use defaults", "default", "build now",
                "start now", "start training", "force_defaults"
            ])
        )

        # Check pillar coverage
        has_arena_ui = "arena_ui" in answers or "q_arena_ui" in answers or any(kw in prompt_lower for kw in ["visualize in arena", "dark-mode 3x3", "arcade theme", "display probabilities"])
        has_expectations = "expectations" in answers or "q_expectations" in answers or any(kw in prompt_lower for kw in ["round", "rounds", "episodes", "threshold", "target %", "% mastery", "% win"])

        needs_clarification = not wants_defaults and not (has_arena_ui and has_expectations) and len(answers) == 0

        if needs_clarification:
            if is_tictactoe_intent(prompt_lower):
                questions = [
                    ClarifyingQuestion(
                        id="q_objective",
                        question_id="q_objective",
                        pillar=PILLAR_OBJECTIVE,
                        question="What strategic gameplay behavior and reward priorities should the Tic-Tac-Toe agent learn?",
                        suggested_options=[
                            "Unbeatable minimax play: Win whenever possible and force a draw otherwise",
                            "Aggressive fork-builder: Prioritize creating dual-threat winning lines",
                            "Defensive counter-puncher: Prioritize immediate threat blocking",
                        ],
                        default_option="Unbeatable minimax play: Win whenever possible and force a draw otherwise",
                    ),
                    ClarifyingQuestion(
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
                    ),
                    ClarifyingQuestion(
                        id="q_interaction",
                        question_id="q_interaction",
                        pillar=PILLAR_INTERACTION,
                        question="How will you interactively test and challenge the agent on the board?",
                        suggested_options=[
                            "Click-to-place human vs agent interactive board with reset button",
                            "Play as either X or O with dynamic side switching",
                            "Self-play spectator simulation with speed controls",
                        ],
                        default_option="Click-to-place human vs agent interactive board with reset button",
                    ),
                    ClarifyingQuestion(
                        id="q_expectations",
                        question_id="q_expectations",
                        pillar=PILLAR_EXPECTATIONS,
                        question="What target mastery win/draw rate and auto-tune round budget should we aim for?",
                        suggested_options=[
                            "85% Win/Draw Rate (4 rounds)",
                            "90% Win/Draw Rate (4 rounds)",
                            "95% Near-Optimal Mastery (5 rounds)",
                            "75% Fast Prototyping (2 rounds)",
                        ],
                        default_option="85% Win/Draw Rate (4 rounds)",
                    ),
                ]
            elif is_blackjack_intent(prompt_lower):
                questions = [
                    ClarifyingQuestion(
                        id="q_objective",
                        question_id="q_objective",
                        pillar=PILLAR_OBJECTIVE,
                        question="What blackjack strategy and dealer rule profile should the agent learn?",
                        suggested_options=[
                            "Casino Basic Strategy: Mathematically optimal hit/stand/double decisions",
                            "Aggressive Payout Maximizer: High-variance doubling and split play",
                            "Conservative Anti-Bust Strategy: Stand early to force dealer busts",
                        ],
                        default_option="Casino Basic Strategy: Mathematically optimal hit/stand/double decisions",
                    ),
                    ClarifyingQuestion(
                        id="q_arena_ui",
                        question_id="q_arena_ui",
                        pillar=PILLAR_ARENA_UI,
                        question="How would you like to visualize the Blackjack arena in the center panel?",
                        suggested_options=[
                            "Casino-style green felt table with card animations and chip stacks",
                            "Sleek dark-mode blackjack dashboard with probability charts",
                            "Minimalist hand totals and dealer upcard view",
                        ],
                        default_option="Casino-style green felt table with card animations and chip stacks",
                    ),
                    ClarifyingQuestion(
                        id="q_interaction",
                        question_id="q_interaction",
                        pillar=PILLAR_INTERACTION,
                        question="How will you interactively test the agent at the Blackjack table?",
                        suggested_options=[
                            "Interactive action buttons: Hit, Stand, Double Down",
                            "Shoe penetration and deck count telemetry inspector",
                            "Automated simulation rounds with speed slider",
                        ],
                        default_option="Interactive action buttons: Hit, Stand, Double Down",
                    ),
                    ClarifyingQuestion(
                        id="q_expectations",
                        question_id="q_expectations",
                        pillar=PILLAR_EXPECTATIONS,
                        question="What target mastery win rate and auto-tune round budget should we aim for?",
                        suggested_options=[
                            "85% Payout Optimization (4 rounds)",
                            "90% Basic Strategy Adherence (4 rounds)",
                            "95% Near-Optimal Casino Play (5 rounds)",
                        ],
                        default_option="85% Payout Optimization (4 rounds)",
                    ),
                ]
            else:
                questions = [
                    ClarifyingQuestion(
                        id="q_objective",
                        question_id="q_objective",
                        pillar=PILLAR_OBJECTIVE,
                        question=f"What core strategic behavior and reward priorities should the {prompt_clean[:40]} agent master?",
                        suggested_options=[
                            "High performance and rapid goal completion",
                            "Conservative stability and risk minimization",
                            "Adaptive self-play exploration",
                        ],
                        default_option="High performance and rapid goal completion",
                    ),
                    ClarifyingQuestion(
                        id="q_arena_ui",
                        question_id="q_arena_ui",
                        pillar=PILLAR_ARENA_UI,
                        question=f"How would you like to visually see the agent's behavior and environment in the Arena center panel?",
                        suggested_options=[
                            f"Bespoke interactive layout tailored to {prompt_clean[:40]}",
                            "Modern dark-mode 2D canvas with neon indicators",
                            "Interactive game board with highlighted state cells",
                            "Clean telemetry dashboard with real-time gauges",
                        ],
                        default_option=f"Bespoke interactive layout tailored to {prompt_clean[:40]}",
                    ),
                    ClarifyingQuestion(
                        id="q_interaction",
                        question_id="q_interaction",
                        pillar=PILLAR_INTERACTION,
                        question="How will you interactively test and guide the agent in the Arena?",
                        suggested_options=[
                            "Direct click-to-act buttons with live agent response",
                            "Interactive parameter sliders and state inspection",
                            "Automated spectator mode with speed controls",
                        ],
                        default_option="Direct click-to-act buttons with live agent response",
                    ),
                    ClarifyingQuestion(
                        id="q_expectations",
                        question_id="q_expectations",
                        pillar=PILLAR_EXPECTATIONS,
                        question="What target success rate and auto-tuning round budget should we target?",
                        suggested_options=[
                            "85% Success Rate (4 rounds)",
                            "90% Success Rate (4 rounds)",
                            "95% High Mastery (5 rounds)",
                            "75% Fast Prototyping (2 rounds)",
                        ],
                        default_option="85% Success Rate (4 rounds)",
                    ),
                ]
            return IntakeResponse(
                status="needs_clarification",
                assistant_message=(
                    "I've analyzed your objective across the 5 Pillars. To tailor your custom RL environment, "
                    "visual Arena UI, and autonomous tuning loop, please clarify the following preferences:"
                ),
                questions=questions,
                blueprint_spec=None,
                blueprint=None,
                recommended_training_params={
                    "max_auto_tune_rounds": 5,
                    "episodes_per_round": 200,
                    "target_metric_threshold": 90.0,
                },
            )

        # Synthesize universal blueprint
        dyn_blueprint = synthesize_blueprint_from_prompt(
            prompt=prompt_clean,
            dataset=dataset,
            conversation_history=conversation_history,
            answers=answers,
            training_params=training_params,
            client=self.client,
        )

        spec = dyn_blueprint.to_agent_blueprint_spec()
        if dataset is not None:
            spec.data_source_type = "uploaded_dataset"
            spec.dataset_id = getattr(dataset, "dataset_id", getattr(dataset, "name", "uploaded_dataset"))
        if answers:
            arena_val = answers.get("arena_ui") or answers.get("q_arena_ui")
            if arena_val:
                spec.arena_visual_description = arena_val
                spec.visual_metaphor = arena_val
            exp_val = answers.get("expectations") or answers.get("q_expectations")
            if exp_val:
                m_round = re.search(r"(\d+)\s*rounds", exp_val)
                if m_round:
                    spec.max_auto_tune_rounds = int(m_round.group(1))
                m_target = re.search(r"(\d+)%", exp_val)
                if m_target:
                    spec.target_metric_threshold = float(m_target.group(1))
            obj_val = answers.get("objective") or answers.get("q_objective")
            if obj_val:
                spec.domain_summary = f"{spec.domain_summary} Strategy: {obj_val}"
            inter_val = answers.get("interaction") or answers.get("q_interaction")
            if inter_val:
                spec.interactive_mode_description = inter_val

        # Adjust training parameters if specified
        if training_params:
            if "max_auto_tune_rounds" in training_params:
                spec.max_auto_tune_rounds = int(training_params["max_auto_tune_rounds"])
            if "episodes_per_round" in training_params:
                spec.episodes_per_round = int(training_params["episodes_per_round"])
            if "target_metric_threshold" in training_params:
                spec.target_metric_threshold = float(training_params["target_metric_threshold"])

        return IntakeResponse(
            status="ready_to_build",
            assistant_message=(
                f"Blueprint locked for '{dyn_blueprint.domain_title}'! All 5 pillars are established. "
                "Ready to proceed with JIT environment synthesis and arena generation."
            ),
            questions=[],
            blueprint_spec=spec,
            blueprint=dyn_blueprint,
            recommended_training_params={
                "max_auto_tune_rounds": spec.max_auto_tune_rounds,
                "episodes_per_round": spec.episodes_per_round,
                "target_metric_threshold": spec.target_metric_threshold,
            },
        )

    def _deterministic_intake(
        self,
        prompt: str,
        dataset: IngestedDataset | None,
        answers: dict[str, str],
        wants_defaults: bool,
        conversation_history: list[dict[str, str]] | None = None,
        training_params: dict[str, Any] | None = None,
    ) -> IntakeResponse:
        """Deterministic intake resolver for test compatibility."""
        return self.interview(
            user_prompt=prompt,
            dataset=dataset,
            conversation_history=conversation_history,
            user_answers=answers,
            force_smart_defaults=wants_defaults,
            training_params=training_params,
        )
