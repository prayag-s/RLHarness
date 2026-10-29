"""Continuous Conversational Studio Omni-Agent for RL-Native Agent IDE v4.2.

Provides a single unified conversational router (StudioOmniAgent) eliminating the rigid
"Intake -> Locked Blueprint -> Training Done" state machine. Users can converse at any stage
(drafting, arena preview, training, paused, completed) to:
- Instantly preview and restyle mock-animated Arena UI (PREVIEW_OR_UPDATE_ARENA)
- Modify agent strategy, behavior, and reward formulations (UPDATE_AGENT_CHARACTERISTICS)
- Modify environment mechanics and action spaces (UPDATE_ENV_MECHANICS)
- Lock arena and run or continue incremental training (RUN_OR_CONTINUE_TRAINING)
- Ask diagnostic questions and discuss telemetry (CLARIFY_OR_DISCUSS)
"""

from __future__ import annotations
import ast
from enum import Enum
import json
import logging
from pathlib import Path
import re
from typing import Any, Callable
from pydantic import BaseModel, Field

from rl_ide.architect.models import (
    AgentBlueprintSpec,
    ClarifyingQuestion,
    ExecutionTarget,
    ChatActionPlan,
)
from rl_ide.architect.intake import IntakeArchitectAgent, is_tictactoe_intent, is_tictactoe_spec
from rl_ide.synthesizer.arena_builder import JITArenaSynthesizer
from rl_ide.synthesizer.env_builder import JITEnvironmentSynthesizer, load_synthesized_env_class, instantiate_env
from rl_ide.synthesizer.verifier import SandboxSmokeTester
from rl_ide.state.conversation_store import (
    ConversationStore,
    get_conversation_store,
    utc_now_iso,
    generate_title_from_prompt,
)
from rl_ide.autopilot.loop_controller import (
    AutonomousLoopController,
    SessionStateStore,
    SessionState,
    parse_nudge_meta_command,
)
from rl_ide.autopilot.auto_tuner import AutoTunerAgent
from rl_ide.llm.client import BaseLLMClient, get_active_llm_client, FallbackCodeSynthesizer, UnconfiguredLLMClient


class StudioActionType(str, Enum):
    CLARIFY_OR_DISCUSS = "clarify_or_discuss"       # Ask/answer questions, explain telemetry, or refine spec
    PREVIEW_OR_UPDATE_ARENA = "update_arena"        # Generate or restyle arena.html with live mock animation
    UPDATE_AGENT_CHARACTERISTICS = "update_agent"   # Modify compute_reward() / demonstrations / personality
    UPDATE_ENV_MECHANICS = "update_env_mechanics"   # Modify state/action space or rules in env_module.py
    RUN_OR_CONTINUE_TRAINING = "run_training"       # Start or continue training for N rounds / M episodes


class StudioChatRequest(BaseModel):
    user_message: str
    dataset_id: str | None = None
    selected_options: dict[str, Any] = Field(default_factory=dict)
    training_params: dict[str, Any] | None = None
    force_action: StudioActionType | None = None
    run_sync: bool = False


class StudioChatResponse(BaseModel):
    conversation_id: str
    status: str
    assistant_message: str
    message: str = ""
    action_types: list[StudioActionType] = Field(default_factory=list)
    blueprint_spec: AgentBlueprintSpec | None = None
    blueprint: AgentBlueprintSpec | None = None
    arena_html: str | None = None
    env_code: str | None = None
    arena_locked: bool = False
    questions: list[ClarifyingQuestion] = Field(default_factory=list)
    clarifying_questions: list[ClarifyingQuestion] = Field(default_factory=list)
    quick_chips: list[str] = Field(default_factory=list)
    training_params: dict[str, Any] = Field(default_factory=dict)
    current_round: int = 1
    best_metric: float = 0.0
    action_plan: ChatActionPlan | None = None

    def model_post_init(self, __context: Any) -> None:
        if not self.message and self.assistant_message:
            self.message = self.assistant_message
        if not self.assistant_message and self.message:
            self.assistant_message = self.message
        if not self.blueprint and self.blueprint_spec:
            self.blueprint = self.blueprint_spec
        if not self.blueprint_spec and self.blueprint:
            self.blueprint_spec = self.blueprint
        if not self.clarifying_questions and self.questions:
            self.clarifying_questions = self.questions
        if not self.questions and self.clarifying_questions:
            self.questions = self.clarifying_questions


class StudioOmniAgent:
    """Unified Conversational Agent & Studio Router (Codex-style continuous loop)."""

    def __init__(
        self,
        conv_store: ConversationStore | None = None,
        session_store: SessionStateStore | None = None,
        loop_controller: AutonomousLoopController | None = None,
        llm_client: BaseLLMClient | None = None,
    ):
        self.conv_store = conv_store or get_conversation_store()
        self.session_store = session_store or SessionStateStore()
        self.loop_controller = loop_controller or AutonomousLoopController(self.session_store)
        self._llm_client = llm_client
        self.arena_synthesizer = JITArenaSynthesizer(llm_client)
        self.env_synthesizer = JITEnvironmentSynthesizer(llm_client)
        self.smoke_tester = SandboxSmokeTester()

    @property
    def client(self) -> BaseLLMClient:
        if self._llm_client:
            return self._llm_client
        return get_active_llm_client()

    def classify_intents(
        self,
        user_message: str,
        conversation: dict[str, Any],
        force_action: StudioActionType | None = None,
    ) -> list[StudioActionType]:
        """Classifies user turn into one or more composable Studio Intent Actions."""
        if force_action:
            return [force_action]

        text = user_message.lower().strip()
        actions: list[StudioActionType] = []

        has_arena = bool(conversation.get("arena_html"))
        has_blueprint = bool(conversation.get("blueprint_spec"))
        arena_locked = bool(conversation.get("arena_locked", False))

        # 1. First Turn check
        if not has_arena or not has_blueprint:
            return [StudioActionType.PREVIEW_OR_UPDATE_ARENA]

        # 2. Arena Look / Restyling / UI / Interaction intent keywords
        arena_kws = [
            "arena", "area", "arnea", "stage", "look", "theme", "color", "cell", "cells", "canvas",
            "scoreboard", "victory badge", "badge", "badges", "banner", "gold", "arcade", "hud", "radar",
            "display", "visual", "larger", "restyle", "redesign", "bigger", "retro",
            "meter", "arrow", "arrows", "highlight", "highlights", "hipaa", "power-flow", "flow",
            "ui", "chart", "gauge", "burn-rate", "play", "player", "play as", "play both",
            "x and o", "x or o", "switch", "toggle", "first turn", "who goes first", "side", "sides",
            "human", "symbol", "grid", "board", "controls", "button", "buttons", "interactive",
            "interface", "layout", "click", "clickable"
        ]
        has_arena_intent = any(kw in text for kw in arena_kws) and not any(
            kw in text for kw in ["lock arena", "lock & start", "lock and start", "confirm and start", "start training"]
        )

        # 3. Training / Continuing Training / Lock Arena keywords
        train_kws = [
            "lock arena", "lock & start", "lock and start", "confirm and start",
            "looks good, start training", "train more", "continue training", "train for",
            "more rounds", "episodes per round", "train now", "start training", "start now",
            "tune more", "run training"
        ]
        has_train_intent = any(kw in text for kw in train_kws) or (
            ("round" in text or "episode" in text) and any(w in text for w in ["train", "run", "more"])
        )

        # 4. Agent behavior / rewards keywords
        agent_kws = [
            "passive", "aggressive", "behavior", "personality", "stalling", "strategy",
            "prioritize", "attack", "bonus", "penalize", "reward", "mistakes", "defense",
            "heuristic", "bonus for winning", "never", "always", "route", "forbid", "disallow",
            "reduce", "false positive", "false positives", "prevent", "avoid", "strictly",
            "prefer", "require", "cardiac", "payroll", "clearance"
        ]
        has_agent_intent = any(kw in text for kw in agent_kws)

        # 5. Environment mechanics / rules keywords
        env_kws = [
            "mechanic", "mechanics", "rules", "rule", "4x4", "5x5", "board to", "grid size", "board size",
            "add action", "hold action", "remove action", "lives", "rules of the game",
            "action space", "observation space", "turbulence", "wind", "friction", "gravity",
            "obstacle", "obstacles", "physics", "disturbance", "environment rules"
        ]
        has_env_intent = any(kw in text for kw in env_kws)

        # 6. Diagnostic questions
        diag_kws = ["why", "how", "what", "explain", "diagnose", "tell me", "loss in round", "suboptimal", "reason"]
        has_diag_intent = any(kw in text for kw in diag_kws) or ("?" in text and not has_train_intent)
        is_pure_question = has_diag_intent and (
            text.startswith(("why", "what", "how", "explain", "tell me", "can you", "is there", "why did"))
            or ("?" in text and not any(text.startswith(cmd) for cmd in ["change", "make", "update", "set", "train", "add", "penalize", "prioritize", "chnge"]))
        )

        # Assemble composable actions
        if is_pure_question and not has_train_intent:
            actions.append(StudioActionType.CLARIFY_OR_DISCUSS)
            return actions

        if has_agent_intent:
            actions.append(StudioActionType.UPDATE_AGENT_CHARACTERISTICS)
        if has_env_intent:
            actions.append(StudioActionType.UPDATE_ENV_MECHANICS)
        if has_arena_intent:
            actions.append(StudioActionType.PREVIEW_OR_UPDATE_ARENA)
        if has_train_intent:
            actions.append(StudioActionType.RUN_OR_CONTINUE_TRAINING)

        # LLM intent classification fallback if keywords were ambiguous
        if not actions and not is_pure_question and not isinstance(self.client, (FallbackCodeSynthesizer, UnconfiguredLLMClient)):
            try:
                llm_actions = self._classify_with_llm(user_message)
                if llm_actions:
                    return llm_actions
            except Exception:
                pass

        if not actions and has_diag_intent:
            actions.append(StudioActionType.CLARIFY_OR_DISCUSS)

        if not actions:
            actions.append(StudioActionType.CLARIFY_OR_DISCUSS)

        return actions

    def _classify_with_llm(self, user_message: str) -> list[StudioActionType]:
        """Classifies intent using active LLM when static keywords do not trigger."""
        prompt = (
            f"Classify this user request for an autonomous RL studio into one of these actions:\n"
            f"User message: \"{user_message}\"\n\n"
            f"Valid actions:\n"
            f"- PREVIEW_OR_UPDATE_ARENA: changing arena layout, UI, visual theme, canvas, board, player controls, or interactive play\n"
            f"- UPDATE_AGENT_CHARACTERISTICS: changing reward terms, penalties, aggression, or strategy\n"
            f"- UPDATE_ENV_MECHANICS: changing environment physics, rules, grid size, or action space\n"
            f"- RUN_OR_CONTINUE_TRAINING: training for more rounds, episodes, or locking arena\n"
            f"- CLARIFY_OR_DISCUSS: asking an informational question or discussing\n\n"
            f"Respond with JSON format: {{\"action\": \"PREVIEW_OR_UPDATE_ARENA\"}}"
        )
        raw = self.client.generate_json(prompt, system_prompt="You are an intent classification agent.")
        action_name = raw.get("action")
        for act in StudioActionType:
            if act.value == action_name or act.name == action_name:
                return [act]
        return []

    def _plan_chat_action(
        self,
        user_message: str,
        conversation: dict[str, Any] | None = None,
        current_env_code: str = "",
    ) -> ChatActionPlan:
        """Inspects live code and user prompt to formulate a code-aware multi-layer ChatActionPlan."""
        conv = conversation or {}
        text = user_message.lower().strip()

        # 1. Honesty Guardrail: Check for unsupported external capabilities
        unsupported_keywords = [
            "discord", "voice", "audio", "microphone", "sound", "stream 3d", "3d webgl",
            "webgl animation", "webgl", "shader", "shaders", "multiplayer", "networking",
            "webhook", "external api", "satellite", "quantum", "vr headset", "augmented reality",
            "slack bot", "twitch"
        ]
        if any(kw in text for kw in unsupported_keywords):
            matched = [kw for kw in unsupported_keywords if kw in text]
            reason = f"Integration with {', '.join(matched)} is outside the supported RL arena runtime capabilities."
            return ChatActionPlan(
                supported=False,
                unsupported_reason=reason,
                scope_classification="unsupported",
                targets=[ExecutionTarget.UNSUPPORTED_CAPABILITY],
                requires_retrain=False,
                surgical_plan="Zero code mutations applied.",
                user_facing_summary=f"I do not know how to do that yet: {reason}",
            )

        # 2. Context summary of current environment methods
        env_methods_summary = "Methods: reset(), step(), interactive_step(), compute_reward()"
        if current_env_code:
            try:
                tree = ast.parse(current_env_code)
                funcs = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                if funcs:
                    env_methods_summary = f"Detected methods in env_module.py: {', '.join(funcs)}"
            except Exception:
                pass

        # 3. LLM-based planning if smart client is configured
        if not isinstance(self.client, (FallbackCodeSynthesizer, UnconfiguredLLMClient)):
            try:
                prompt = (
                    f"You are the Code-Aware Agentic Chat Planner for an Autonomous RL Studio.\n"
                    f"User message: \"{user_message}\"\n\n"
                    f"Environment Context:\n{env_methods_summary}\n\n"
                    f"Supported Execution Targets:\n"
                    f"- \"inference_strategy\": Modifies interactive_step() / policy wrapper (Arena/Eval action selection, e.g. opening randomization, top-k filtering, fallback heuristics).\n"
                    f"- \"reward_shaping\": Modifies compute_reward() / RewardDSL (Training incentives, bonuses, penalties).\n"
                    f"- \"env_sim_mechanics\": Modifies reset() / step() / opponent rules or physics.\n"
                    f"- \"training_hyperparams\": Modifies entropy coef, burst rounds, or exploration.\n"
                    f"- \"unsupported_capability\": If user asks for capabilities outside the runtime (e.g. Discord, WebGL shaders, multiplayer, voice).\n\n"
                    f"Strict System Rules:\n"
                    f"1. If user asks to change how the agent selects moves in the Arena (e.g. randomizing its first/opening move among good options), route to \"inference_strategy\" so interactive_step() is patched directly—do NOT route solely to \"reward_shaping\".\n"
                    f"2. If user requests both a training reward/mechanic and an arena inference change, include both targets and set scope_classification=\"both\", requires_retrain=true.\n"
                    f"3. If user requests an unsupported capability, set supported=false and targets=[\"unsupported_capability\"].\n\n"
                    f"Respond with JSON adhering to:\n"
                    f"{{\n"
                    f"  \"supported\": true,\n"
                    f"  \"unsupported_reason\": null,\n"
                    f"  \"scope_classification\": \"arena_only\" | \"training_only\" | \"both\" | \"unsupported\",\n"
                    f"  \"targets\": [\"inference_strategy\"],\n"
                    f"  \"requires_retrain\": false,\n"
                    f"  \"surgical_plan\": \"...\",\n"
                    f"  \"user_facing_summary\": \"...\"\n"
                    f"}}"
                )
                res = self.client.generate_json(prompt, system_prompt="You are a code-aware agentic planner.")
                if res and isinstance(res, dict) and "supported" in res and "targets" in res:
                    return ChatActionPlan.model_validate(res)
            except Exception as ex:
                logging.warning(f"LLM ChatActionPlan error: {ex}")

        # 4. High-precision deterministic fallback planner
        targets: list[ExecutionTarget] = []
        requires_retrain = False

        # Check for inference strategy (Arena move selection / opening randomization)
        has_inference = (
            any(kw in text for kw in ["randomize", "random", "stochastic", "sample"]) and
            any(kw in text for kw in ["first move", "first", "opening", "openings", "turn 1", "arena", "option", "options"])
        ) or "interactive_step" in text or "inference" in text

        # Check for reward shaping (Reward terms / training incentives)
        has_reward = (
            any(kw in text for kw in ["reward", "penalize", "bonus", "penalty", "incentive", "compute_reward"]) or
            ("corner" in text and any(w in text for w in ["training", "train", "reward", "openings"]))
        )

        # Check for environment mechanics
        has_env_sim = (
            "mechanic" in text or "physics" in text or "grid size" in text or "board size" in text or
            ("self-play" in text and "training" in text) or "both self-play" in text or "rules" in text
        )

        # Check for training hyperparams
        has_training_params = any(kw in text for kw in ["hyperparameter", "entropy", "learning rate", "burst length", "batch size"])

        if has_inference:
            targets.append(ExecutionTarget.INFERENCE_STRATEGY)
        if has_reward:
            targets.append(ExecutionTarget.REWARD_SHAPING)
            requires_retrain = True
        if has_env_sim:
            targets.append(ExecutionTarget.ENV_SIM_MECHANICS)
            requires_retrain = True
        if has_training_params:
            targets.append(ExecutionTarget.TRAINING_HYPERPARAMS)
            requires_retrain = True

        if not targets:
            targets.append(ExecutionTarget.INFERENCE_STRATEGY)
            scope = "arena_only"
        elif ExecutionTarget.INFERENCE_STRATEGY in targets and (ExecutionTarget.REWARD_SHAPING in targets or ExecutionTarget.ENV_SIM_MECHANICS in targets):
            scope = "both"
            requires_retrain = True
        elif ExecutionTarget.INFERENCE_STRATEGY in targets:
            scope = "arena_only"
            requires_retrain = False
        else:
            scope = "training_only"
            requires_retrain = True

        surgical_steps = []
        if ExecutionTarget.INFERENCE_STRATEGY in targets:
            surgical_steps.append("Patch interactive_step() in env_module.py for turn-aware action selection.")
        if ExecutionTarget.REWARD_SHAPING in targets:
            surgical_steps.append("Rewrite compute_reward() in env_module.py and recalibrate Critic head.")
        if ExecutionTarget.ENV_SIM_MECHANICS in targets:
            surgical_steps.append("Update reset() and step() simulation rules in env_module.py.")
        if ExecutionTarget.TRAINING_HYPERPARAMS in targets:
            surgical_steps.append("Adjust training hyperparameters in session configuration.")

        return ChatActionPlan(
            supported=True,
            unsupported_reason=None,
            scope_classification=scope,
            targets=targets,
            requires_retrain=requires_retrain,
            surgical_plan=" ".join(surgical_steps),
            user_facing_summary=f"Formulated execution plan for scope '{scope}' targeting {', '.join([t.value for t in targets])}.",
        )

    def plan_chat_action(
        self,
        user_message: str,
        conversation: dict[str, Any] | None = None,
        current_env_code: str = "",
    ) -> ChatActionPlan:
        """Public alias for _plan_chat_action."""
        return self._plan_chat_action(user_message, conversation, current_env_code)

    def _patch_interactive_inference(
        self,
        current_code: str,
        user_request: str,
        plan: ChatActionPlan | None = None,
    ) -> str:
        """Surgically patches interactive_step() in env_module.py to modify action selection / inference strategy."""
        code = current_code
        req_lower = user_request.lower()

        # Check if opening move randomization is requested
        is_opening_randomization = any(kw in req_lower for kw in ["randomize", "random", "stochastic"]) and any(
            kw in req_lower for kw in ["first", "opening", "openings", "start", "turn 1", "option", "options"]
        )

        if is_opening_randomization or (plan and ExecutionTarget.INFERENCE_STRATEGY in plan.targets):
            code = code.replace("if probs else", "if probs is not None else")
            turn_aware_replacement = (
                "{indent}action_idx, probs = policy_predict_fn({args})\n"
                "{indent}is_first_move = (np.count_nonzero(getattr(self, 'board', [])) == 0) or (getattr(self, 'step_count', 0) == 0)\n"
                "{indent}if is_first_move:\n"
                "{indent}    board_arr = getattr(self, 'board', None)\n"
                "{indent}    legal_indices = [i for i, v in enumerate(board_arr) if v == 0] if (board_arr is not None and len(board_arr) > 0) else list(range(9))\n"
                "{indent}    if probs is not None and len(probs) > 0:\n"
                "{indent}        candidate_probs = np.array([probs[i] for i in legal_indices if i < len(probs)], dtype=np.float64)\n"
                "{indent}        max_p = np.max(candidate_probs) if len(candidate_probs) > 0 else 0.0\n"
                "{indent}        good_moves = [legal_indices[i] for i, p in enumerate(candidate_probs) if p >= 0.7 * max_p]\n"
                "{indent}    else:\n"
                "{indent}        good_moves = []\n"
                "{indent}    if len(good_moves) <= 1:\n"
                "{indent}        strategic_openings = [i for i in [0, 2, 4, 6, 8] if i in legal_indices]\n"
                "{indent}        if len(strategic_openings) > 1:\n"
                "{indent}            good_moves = strategic_openings\n"
                "{indent}    if good_moves:\n"
                "{indent}        action_idx = int(np.random.choice(good_moves))\n"
                "{indent}    else:\n"
                "{indent}        action_idx = int(action_idx)\n"
                "{indent}else:\n"
                "{indent}    action_idx = int(action_idx)"
            )

            pattern = re.compile(
                r"([ \t]*)(action_idx,\s*probs\s*=\s*policy_predict_fn\((.*?)\)\s*\n[ \t]*action_idx\s*=\s*int\(action_idx\))"
            )

            def _replace_match(m):
                indent = m.group(1)
                args = m.group(3)
                return turn_aware_replacement.format(indent=indent, args=args)

            if pattern.search(code):
                patched = pattern.sub(_replace_match, code)
                try:
                    ast.parse(patched)
                    code = patched
                except Exception as ex:
                    logging.warning(f"AST parse validation failed after patching interactive_step: {ex}")

        return code

    def process_message(
        self,
        conversation_id: str,
        user_message: str,
        dataset: Any | None = None,
        dataset_id: str | None = None,
        selected_options: dict[str, Any] | None = None,
        training_params: dict[str, Any] | None = None,
        force_action: StudioActionType | None = None,
        broadcast_callback: Callable[[dict[str, Any]], None] | None = None,
        run_sync: bool = False,
    ) -> StudioChatResponse:
        """Single unified entrypoint for continuous conversational studio interaction."""
        conv = self.conv_store.get_conversation(conversation_id)
        if not conv:
            conv = self.conv_store.create_conversation(
                conversation_id=conversation_id,
                title=generate_title_from_prompt(user_message),
                training_params=training_params,
                dataset_id=dataset_id,
            )

        ws_dir = self.conv_store.workspaces_dir / conversation_id
        ws_dir.mkdir(parents=True, exist_ok=True)

        effective_dataset_id = dataset_id or conv.get("dataset_id")
        active_dataset = dataset
        if active_dataset is None and effective_dataset_id:
            try:
                from rl_ide.api.server import ingested_datasets_store
                active_dataset = ingested_datasets_store.get(effective_dataset_id)
            except Exception:
                pass

        selected_opts = selected_options or {}
        t_params = dict(conv.get("training_params", {}) or {})
        if training_params:
            t_params.update(training_params)

        # Detect control directives (e.g. "for 100%", "train for 5 rounds", "200 episodes") directly from user message
        meta_cmd = parse_nudge_meta_command(user_message)
        if meta_cmd:
            if meta_cmd.get("rounds") is not None:
                t_params["max_auto_tune_rounds"] = meta_cmd["rounds"]
            if meta_cmd.get("episodes") is not None:
                t_params["episodes_per_round"] = meta_cmd["episodes"]
            if meta_cmd.get("target") is not None:
                t_params["target_metric_threshold"] = meta_cmd["target"]

        intents = self.classify_intents(user_message, conv, force_action=force_action)

        # Log user message
        history = list(conv.get("chat_history", []))
        history.append({
            "role": "user",
            "content": user_message,
            "timestamp": utc_now_iso(),
            "selected_options": selected_opts if selected_opts else None,
        })

        blueprint_dict = conv.get("blueprint_spec")
        blueprint: AgentBlueprintSpec | None = None
        if blueprint_dict:
            try:
                blueprint = AgentBlueprintSpec.model_validate(blueprint_dict)
                if t_params:
                    if "target_metric_threshold" in t_params:
                        blueprint.target_metric_threshold = float(t_params["target_metric_threshold"])
                    if "max_auto_tune_rounds" in t_params:
                        blueprint.max_auto_tune_rounds = int(t_params["max_auto_tune_rounds"])
                    if "episodes_per_round" in t_params:
                        blueprint.episodes_per_round = int(t_params["episodes_per_round"])
            except Exception:
                pass

        current_arena_html = conv.get("arena_html") or ""
        current_env_code = conv.get("env_code") or ""
        env_file = ws_dir / "env_module.py"
        if not current_env_code and env_file.exists():
            current_env_code = env_file.read_text(encoding="utf-8")

        arena_locked = bool(conv.get("arena_locked", False))
        status = conv.get("status", "clarifying")
        assistant_reply = ""
        questions: list[ClarifyingQuestion] = []
        quick_chips: list[str] = []

        # 0. Agentic Chat Planning
        action_plan = self._plan_chat_action(
            user_message=user_message,
            conversation=conv,
            current_env_code=current_env_code,
        )

        # Honesty Guardrail: Zero mutations, instant honest refusal for unsupported requests
        if not action_plan.supported or ExecutionTarget.UNSUPPORTED_CAPABILITY in action_plan.targets:
            status = "unsupported"
            assistant_reply = f"I do not know how to do that yet: {action_plan.unsupported_reason}"
            history.append({
                "role": "assistant",
                "content": assistant_reply,
                "arena_locked": arena_locked,
                "timestamp": utc_now_iso(),
            })
            self.conv_store.save_or_update(
                conversation_id=conversation_id,
                status=status,
                chat_history=history,
            )
            return StudioChatResponse(
                conversation_id=conversation_id,
                status=status,
                assistant_message=assistant_reply,
                action_types=[StudioActionType.CLARIFY_OR_DISCUSS],
                blueprint_spec=blueprint,
                arena_html=current_arena_html,
                env_code=current_env_code,
                arena_locked=arena_locked,
                questions=[],
                quick_chips=quick_chips,
                training_params=t_params,
                current_round=conv.get("current_round", 1),
                best_metric=conv.get("best_metric", 0.0),
                action_plan=action_plan,
            )

        # ---------------------------------------------------------------------
        # CASE 1: FIRST TURN — Immediate Mock-Animated Arena Preview
        # ---------------------------------------------------------------------
        if not current_arena_html or not blueprint:
            intake_agent = IntakeArchitectAgent(self._llm_client)

            intake_resp = intake_agent.interview(
                user_prompt=user_message,
                dataset=active_dataset,
                user_answers=selected_opts,
                force_smart_defaults=False,
                training_params=t_params,
            )

            # Build draft blueprint if not fully locked yet
            if intake_resp.blueprint_spec:
                blueprint = intake_resp.blueprint_spec
            else:
                # Fast deterministic blueprint so arena preview can generate instantly without duplicate LLM latency!
                fallback_resp = intake_agent._deterministic_intake(
                    prompt=user_message,
                    dataset=active_dataset,
                    answers=selected_opts,
                    wants_defaults=True,
                    training_params=t_params,
                )
                blueprint = fallback_resp.blueprint_spec

            if blueprint and t_params:
                if "target_metric_threshold" in t_params:
                    blueprint.target_metric_threshold = float(t_params["target_metric_threshold"])
                if "max_auto_tune_rounds" in t_params:
                    blueprint.max_auto_tune_rounds = int(t_params["max_auto_tune_rounds"])
                if "episodes_per_round" in t_params:
                    blueprint.episodes_per_round = int(t_params["episodes_per_round"])

            if intake_resp.questions:
                questions = intake_resp.questions
            elif intake_resp.clarifying_questions:
                questions = intake_resp.clarifying_questions

            # Generate initial Mock-Animated arena.html
            dummy_sample_state = {"board": [0] * 9, "game_status": "Mock Arena Preview"}
            if active_dataset is not None:
                rec_preview = {}
                if hasattr(active_dataset, "df") and active_dataset.df is not None and len(active_dataset.df) > 0:
                    rec_preview = {k: str(v) for k, v in active_dataset.df.iloc[0].to_dict().items()}
                dummy_sample_state = {
                    "record_index": 0,
                    "total_records": len(active_dataset.df) if hasattr(active_dataset, "df") and active_dataset.df is not None else 100,
                    "accuracy_pct": 100.0,
                    "current_record_preview": rec_preview,
                }
            arena_html_str, _ = self.arena_synthesizer.synthesize(
                blueprint=blueprint,
                sample_arena_state=dummy_sample_state,
                session_id=conversation_id,
                workspace_dir=ws_dir,
            )
            current_arena_html = arena_html_str

            status = "arena_preview"
            arena_locked = False

            assistant_reply = (
                f"✨ Generated live mock-animated Arena preview for **{blueprint.task_title}**! "
                "You can see it animating with realistic mock moves and telemetry in the center stage. "
                "Ask me in chat to adjust the look & feel, or click **[ 🔒 Lock Arena & Start Training ]** when ready."
            )

            quick_chips = [
                "🎨 Neon Cyberpunk Theme",
                "📊 Add Detailed Telemetry HUD",
                "🔍 Make Board / Canvas Larger",
                "🔒 Lock Arena & Start Training",
            ]

            history.append({
                "role": "assistant",
                "content": assistant_reply,
                "blueprint": blueprint.model_dump(),
                "arena_locked": False,
                "timestamp": utc_now_iso(),
            })

            self.conv_store.save_or_update(
                conversation_id=conversation_id,
                status=status,
                title=blueprint.task_title or conv.get("title", "New Agent"),
                chat_history=history,
                blueprint_spec=blueprint.model_dump(),
                arena_html=current_arena_html,
                arena_locked=False,
                training_params=t_params,
                dataset_id=effective_dataset_id,
            )

            if broadcast_callback:
                broadcast_callback({"event": "arena_updated", "arena_html": current_arena_html})

            return StudioChatResponse(
                conversation_id=conversation_id,
                status=status,
                assistant_message=assistant_reply,
                action_types=[StudioActionType.PREVIEW_OR_UPDATE_ARENA],
                blueprint_spec=blueprint,
                arena_html=current_arena_html,
                env_code=current_env_code,
                arena_locked=False,
                questions=questions,
                quick_chips=quick_chips,
                training_params=t_params,
                current_round=conv.get("current_round", 1),
                best_metric=conv.get("best_metric", 0.0),
                action_plan=action_plan,
            )

        # ---------------------------------------------------------------------
        # CASE 2: MID / POST-TRAINING CONVERSATIONAL ACTIONS (Multi-Target Chaining)
        # ---------------------------------------------------------------------

        # Multi-Target Layer 1: ENV_SIM_MECHANICS (Gym mechanics / rules)
        if ExecutionTarget.ENV_SIM_MECHANICS in action_plan.targets or StudioActionType.UPDATE_ENV_MECHANICS in intents:
            env_file = ws_dir / "env_module.py"
            if blueprint:
                blueprint.domain_summary = f"{blueprint.domain_summary}. Rules: {user_message}"
            code_str, _, env_cls = self.env_synthesizer.synthesize(
                blueprint=blueprint,
                workspace_dir=ws_dir,
            )
            verified, healed_code, _ = self.smoke_tester.verify_and_heal(
                code_str=code_str,
                blueprint=blueprint,
                workspace_dir=ws_dir,
            )
            if verified:
                current_env_code = healed_code
                env_file.write_text(healed_code, encoding="utf-8")
                try:
                    updated_env_cls = load_synthesized_env_class(env_file)
                    active_env = instantiate_env(updated_env_cls)
                    self.session_store.set_env(conversation_id, active_env)
                except Exception:
                    pass
                assistant_reply += "🛠️ Environment mechanics updated and verified via sandbox smoke tester.\n"

        # Multi-Target Layer 2: REWARD_SHAPING (compute_reward / RewardDSL)
        if ExecutionTarget.REWARD_SHAPING in action_plan.targets or StudioActionType.UPDATE_AGENT_CHARACTERISTICS in intents:
            env_file = ws_dir / "env_module.py"
            if not current_env_code and env_file.exists():
                current_env_code = env_file.read_text(encoding="utf-8")

            # Update compute_reward in env code
            updated_code = self._rewrite_reward_function(
                current_code=current_env_code,
                user_feedback=user_message,
                blueprint=blueprint,
            )

            verified, healed_code, _ = self.smoke_tester.verify_and_heal(
                code_str=updated_code,
                blueprint=blueprint,
                workspace_dir=ws_dir,
            )
            if verified:
                current_env_code = healed_code
                env_file.write_text(healed_code, encoding="utf-8")

                # Hot-swap env class in session store
                try:
                    updated_env_cls = load_synthesized_env_class(env_file)
                    active_env = instantiate_env(updated_env_cls)
                    self.session_store.set_env(conversation_id, active_env)
                    # Calibrate Critic on existing policy weights
                    self.loop_controller.calibrate_critic(conversation_id, env=active_env, epochs=10)
                except Exception as ex:
                    print(f"Warning: Critic calibration or env reload: {ex}")

                assistant_reply += (
                    "🧠 Agent characteristics & reward formulation updated! "
                    "Calibrated Critic value head on existing policy weights. Progress is preserved.\n"
                )

        # Multi-Target Layer 3: INFERENCE_STRATEGY (interactive_step / live action selection)
        if ExecutionTarget.INFERENCE_STRATEGY in action_plan.targets:
            env_file = ws_dir / "env_module.py"
            if not current_env_code and env_file.exists():
                current_env_code = env_file.read_text(encoding="utf-8")

            patched_code = self._patch_interactive_inference(
                current_code=current_env_code,
                user_request=user_message,
                plan=action_plan,
            )

            verified, healed_code, _ = self.smoke_tester.verify_and_heal(
                code_str=patched_code,
                blueprint=blueprint,
                workspace_dir=ws_dir,
            )
            if verified:
                current_env_code = healed_code
                env_file.write_text(healed_code, encoding="utf-8")

                try:
                    updated_env_cls = load_synthesized_env_class(env_file)
                    active_env = instantiate_env(updated_env_cls)
                    self.session_store.set_env(conversation_id, active_env)
                except Exception as ex:
                    print(f"Warning: Env reload after inference patch: {ex}")

                assistant_reply += "🎯 Arena inference strategy updated! Patched interactive_step() for turn-aware action selection.\n"

        # Multi-Target Layer 4: Visual Arena & Controls
        has_visual_intent = any(kw in user_message.lower() for kw in [
            "look", "theme", "color", "canvas", "neon", "cyberpunk", "restyle", "dark", "gold",
            "scoreboard", "badge", "display", "larger", "bigger", "retro", "hud", "layout",
            "x and o", "x or o", "play as", "switch", "both", "who goes first", "player side", "symbol"
        ])
        if (StudioActionType.PREVIEW_OR_UPDATE_ARENA in intents and has_visual_intent) or (has_visual_intent and not assistant_reply):
            new_arena_html = self.arena_synthesizer.update_arena(
                blueprint=blueprint,
                current_arena_html=current_arena_html,
                user_visual_request=user_message,
                sample_arena_state=None,
                workspace_dir=ws_dir,
            )
            current_arena_html = new_arena_html

            is_play_feature = any(kw in user_message.lower() for kw in ["x and o", "x or o", "play as", "switch", "both", "who goes first", "player side", "symbol"])
            if is_play_feature and blueprint and is_tictactoe_spec(blueprint):
                env_file = ws_dir / "env_module.py"
                if env_file.exists():
                    env_code_txt = env_file.read_text(encoding="utf-8")
                    if "human_symbol" not in env_code_txt:
                        code_str, _, env_cls = self.env_synthesizer.synthesize(blueprint=blueprint, workspace_dir=ws_dir)
                        verified, healed_code, _ = self.smoke_tester.verify_and_heal(
                            code_str=code_str,
                            blueprint=blueprint,
                            workspace_dir=ws_dir,
                        )
                        if verified:
                            current_env_code = healed_code
                            env_file.write_text(healed_code, encoding="utf-8")
                            try:
                                updated_env_cls = load_synthesized_env_class(env_file)
                                active_env = instantiate_env(updated_env_cls)
                                self.session_store.set_env(conversation_id, active_env)
                            except Exception:
                                pass

            if is_play_feature:
                assistant_reply += "🎮 Arena updated! Added interactive controls to play as either **X** (First Move) or **O** (Agent First), with live side switching. Neural network weights remain 100% intact!\n"
            else:
                assistant_reply += "🎨 Visual Arena updated and hot-reloaded live. Neural network weights remain 100% intact!\n"

            if broadcast_callback:
                broadcast_callback({"event": "arena_updated", "arena_html": current_arena_html})

        # Multi-Target Layer 5: TRAINING_HYPERPARAMS / RUN_OR_CONTINUE_TRAINING
        wants_training = (
            StudioActionType.RUN_OR_CONTINUE_TRAINING in intents or
            ExecutionTarget.TRAINING_HYPERPARAMS in action_plan.targets or
            (action_plan.requires_retrain and any(kw in user_message.lower() for kw in ["train", "start", "run", "more", "burst"]))
        )
        if wants_training:
            arena_locked = True
            meta_cmd = parse_nudge_meta_command(user_message)
            if meta_cmd:
                if meta_cmd.get("rounds") is not None:
                    t_params["max_auto_tune_rounds"] = meta_cmd["rounds"]
                if meta_cmd.get("episodes") is not None:
                    t_params["episodes_per_round"] = meta_cmd["episodes"]
                if meta_cmd.get("target") is not None:
                    t_params["target_metric_threshold"] = meta_cmd["target"]

            if blueprint:
                if "target_metric_threshold" in t_params:
                    blueprint.target_metric_threshold = float(t_params["target_metric_threshold"])
                if "max_auto_tune_rounds" in t_params:
                    blueprint.max_auto_tune_rounds = int(t_params["max_auto_tune_rounds"])
                if "episodes_per_round" in t_params:
                    blueprint.episodes_per_round = int(t_params["episodes_per_round"])

            # Ensure environment module exists
            env_file = ws_dir / "env_module.py"
            if not env_file.exists() or not current_env_code:
                code_str, _, _ = self.env_synthesizer.synthesize(blueprint=blueprint, dataset=active_dataset, workspace_dir=ws_dir)
                verified, healed_code, _ = self.smoke_tester.verify_and_heal(
                    code_str=code_str,
                    blueprint=blueprint,
                    dataset=active_dataset,
                    workspace_dir=ws_dir,
                )
                current_env_code = healed_code
                env_file.write_text(healed_code, encoding="utf-8")

            env_cls = load_synthesized_env_class(env_file)
            eps_per_burst = int(t_params.get("episodes_per_round", 50))
            add_rounds = int(t_params.get("max_auto_tune_rounds", 2))

            if run_sync:
                trained_session = self.loop_controller.run_loop(
                    session_id=conversation_id,
                    blueprint=blueprint,
                    env_cls=env_cls,
                    dataset=active_dataset,
                    on_event_callback=broadcast_callback,
                    max_rounds=add_rounds,
                    episodes_per_burst=eps_per_burst,
                    continue_training=True,
                )
                status = trained_session.status
                assistant_reply += (
                    f"🔄 Training completed through Round {trained_session.current_round}! "
                    f"Metric achieved: {trained_session.current_metric}%. Ready for interactive play.\n"
                )
            else:
                status = "training"
                assistant_reply += (
                    f"🚀 Arena locked. Continuous training initiated for {add_rounds} rounds "
                    f"({eps_per_burst} episodes/round).\n"
                )

        # Intent E: CLARIFY_OR_DISCUSS (if no code updates were run)
        if not assistant_reply:
            assistant_reply = self._generate_diagnostic_explanation(conv, user_message)

        # Compute dynamic quick chips based on current state
        if not arena_locked:
            quick_chips = [
                "🎨 Neon Cyberpunk Theme",
                "📊 Add Detailed Telemetry HUD",
                "🔍 Make Board / Canvas Larger",
                "🔒 Lock Arena & Start Training",
            ]
        else:
            quick_chips = [
                "🔄 Train 5 More Rounds (500 eps/round)",
                "🧠 Make Agent More Aggressive",
                "🛡️ Penalize Mistakes More Heavily",
                "🎨 Redesign Arena Look",
            ]

        # Persist conversation state
        history.append({
            "role": "assistant",
            "content": assistant_reply.strip(),
            "arena_locked": arena_locked,
            "timestamp": utc_now_iso(),
        })

        session = self.session_store.get_session(conversation_id)
        current_r = session.current_round if session else conv.get("current_round", 1)
        best_m = session.current_metric if session else conv.get("best_metric", 0.0)
        r_hist = [r.model_dump() for r in session.rounds_history] if session else conv.get("rounds_history", [])
        eps_list = session.episodes if session else conv.get("episodes", [])
        model_path = session.latest_model_zip_path if session else conv.get("latest_model_zip_path")

        self.conv_store.save_or_update(
            conversation_id=conversation_id,
            status=status,
            chat_history=history,
            blueprint_spec=blueprint.model_dump() if blueprint else None,
            training_params=t_params,
            dataset_id=effective_dataset_id,
            arena_html=current_arena_html,
            env_code=current_env_code,
            arena_locked=arena_locked,
            current_round=current_r,
            best_metric=best_m,
            rounds_history=r_hist,
            episodes=eps_list,
            latest_model_zip_path=model_path,
        )

        effective_actions = list(intents)
        target_map = {
            ExecutionTarget.INFERENCE_STRATEGY: StudioActionType.PREVIEW_OR_UPDATE_ARENA,
            ExecutionTarget.REWARD_SHAPING: StudioActionType.UPDATE_AGENT_CHARACTERISTICS,
            ExecutionTarget.ENV_SIM_MECHANICS: StudioActionType.UPDATE_ENV_MECHANICS,
            ExecutionTarget.TRAINING_HYPERPARAMS: StudioActionType.RUN_OR_CONTINUE_TRAINING,
        }
        for t in action_plan.targets:
            act = target_map.get(t)
            if act and act not in effective_actions:
                effective_actions.append(act)

        return StudioChatResponse(
            conversation_id=conversation_id,
            status=status,
            assistant_message=assistant_reply.strip(),
            action_types=effective_actions,
            blueprint_spec=blueprint,
            arena_html=current_arena_html,
            env_code=current_env_code,
            arena_locked=arena_locked,
            questions=questions,
            quick_chips=quick_chips,
            training_params=t_params,
            current_round=current_r,
            best_metric=best_m,
            action_plan=action_plan,
        )

    def _rewrite_reward_function(
        self,
        current_code: str,
        user_feedback: str,
        blueprint: AgentBlueprintSpec,
    ) -> str:
        """Modifies compute_reward() according to natural language behavior directives."""
        if not isinstance(self.client, FallbackCodeSynthesizer):
            try:
                auto_tuner = AutoTunerAgent(self.client)
                prompt = (
                    f"The user wants to update the agent's behavior and reward formulation:\n"
                    f"Directives: \"{user_feedback}\"\n\n"
                    f"Current Environment Code:\n{current_code}\n\n"
                    f"Rewrite `compute_reward(self, state, action, next_state, info)` to reflect these directives. "
                    f"Preserve all method signatures and clamp reward terms to [-10.0, 10.0]. Return full updated Python module inside ```python ... ```."
                )
                res = self.client.generate(prompt)
                match = re.search(r"```python\s*([\s\S]*?)```", res)
                if match and "def compute_reward" in match.group(1):
                    return match.group(1).strip()
            except Exception:
                pass

        # Deterministic reward rewrite based on directive keywords
        code = current_code
        fb = user_feedback.lower()

        # Check for specific reward tweaks
        additions = []
        if "stalling" in fb or "quick" in fb or "faster" in fb:
            additions.append('        # Quick win bonus & stalling penalty\n        steps = info.get("steps", 1)\n        reward_breakdown["quick_win_bonus"] = 2.0 / max(1, steps)\n        reward_breakdown["stalling_penalty"] = -0.05 * float(steps)\n')
        if "aggressive" in fb or "fork" in fb:
            additions.append('        # Tactical aggression & fork reward\n        reward_breakdown["tactical_aggression"] = 1.5\n')
        if "center" in fb:
            additions.append('        # Center control priority\n        reward_breakdown["center_control"] = 2.0\n')
        if "corner" in fb:
            additions.append('        # Corner opening strategic bonus\n        if action in [0, 2, 6, 8]:\n            reward_breakdown["corner_opening_bonus"] = 2.0\n')
        if "mistake" in fb or "penalize" in fb:
            additions.append('        # Mistake avoidance penalty\n        reward_breakdown["mistake_penalty"] = -2.5\n')
        if "cardiac" in fb:
            additions.append('        # Cardiac symptom safety: never deny cardiac cases, route to urgent review\n        if info.get("has_cardiac_symptoms", 0) and action == 3:\n            reward_breakdown["cardiac_denial_penalty"] = -10.0\n        elif info.get("has_cardiac_symptoms", 0) and action == 2:\n            reward_breakdown["cardiac_urgent_route_bonus"] = 5.0\n')
        if "verified" in fb or "payroll" in fb or "false positive" in fb:
            additions.append('        # Reduce false positives on verified corporate payroll accounts\n        if info.get("is_verified_corporate", 0) and action != 0:\n            reward_breakdown["false_freeze_penalty"] = -5.0\n')
        if "clearance" in fb or "7 days" in fb or "forbid" in fb:
            additions.append('        # Forbid clearance when > 7 days remaining\n        if info.get("day_of_sale", 0) <= 7 and action == 3:\n            reward_breakdown["premature_clearance_penalty"] = -8.0\n')
        if "liability" in fb or "100,000" in fb or "100000" in fb:
            additions.append('        # Strict liability cap verification over $100,000\n        if info.get("max_dollar_value", 0) >= 100000 and action != 1:\n            reward_breakdown["unlimited_liability_penalty"] = -6.0\n')
        if "obstacle" in fb or "safety" in fb:
            additions.append('        # Increased obstacle safety clearance reward\n        reward_breakdown["obstacle_safety_clearance"] = 2.0\n')
        if "login_drop" in fb or "discount" in fb or "vip" in fb:
            additions.append('        # Penalize offering discount when login_drop_pct < 20%\n        if info.get("login_drop_pct", 0) < 20.0 and action == 2:\n            reward_breakdown["premature_discount_penalty"] = -5.0\n')

        if additions and "def compute_reward" in code:
            injection = "".join(additions)
            # Inject right before the return statement of compute_reward
            ret_match = re.search(r"(def compute_reward[\s\S]*?)(return\s+.*?,.*?breakdown.*?\n)", code)
            if ret_match:
                prefix = ret_match.group(1)
                ret_stmt = ret_match.group(2)
                # Ensure terms are summed into total
                sum_code = "        total_reward = float(sum(reward_breakdown.values()))\n        "
                code = code.replace(ret_match.group(0), prefix + injection + sum_code + ret_stmt)
            else:
                # Add comment and breakdown keys
                code = code.replace("def compute_reward", "# Behavior directive: " + user_feedback[:40] + "\n    def compute_reward")

        return code

    def _generate_diagnostic_explanation(self, conv: dict[str, Any], query: str) -> str:
        """Generates conversational diagnosis or answers questions using LLM or smart telemetry."""
        best_metric = conv.get("best_metric", 0.0)
        curr_round = conv.get("current_round", 1)
        r_hist = conv.get("rounds_history", [])
        title = conv.get("title", "Agent")

        if not isinstance(self.client, (FallbackCodeSynthesizer, UnconfiguredLLMClient)):
            try:
                prompt = (
                    f"You are the AI Studio Assistant for an Autonomous Reinforcement Learning Platform.\n"
                    f"Current Project: '{title}'.\n"
                    f"Current Round: {curr_round}, Best Success Metric: {best_metric}%.\n"
                    f"Rounds History: {r_hist}\n\n"
                    f"The user says: \"{query}\"\n\n"
                    f"Provide a helpful, precise, conversational response to the user. "
                    f"If the user is asking to modify the arena, agent, or environment, explain clearly what actions you can take."
                )
                res = self.client.generate(prompt, timeout=15.0)
                if res and res.strip():
                    return res.strip()
            except Exception:
                pass

        rounds_summary = ""
        if r_hist:
            rounds_summary = ", ".join([f"Round {r.get('round_num')}: {r.get('metric')}%" for r in r_hist])

        return (
            f"💬 **Diagnostic Analysis for {title}:**\n"
            f"- Current Success Rate: **{best_metric}%** across **{curr_round}** trained rounds.\n"
            f"- History: {rounds_summary or 'Initial round in progress'}.\n"
            "The agent is continuously refining its policy against tactical opponent heuristics. "
            "To guide its learning, you can ask me to: tweak reward weights, make it more aggressive, "
            "or train for additional rounds!"
        )
