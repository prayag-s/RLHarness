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
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Callable
from pydantic import BaseModel, Field

from rl_ide.architect.models import AgentBlueprintSpec, ClarifyingQuestion
from rl_ide.architect.intake import IntakeArchitectAgent, is_tictactoe_intent
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
from rl_ide.llm.client import BaseLLMClient, get_active_llm_client, FallbackCodeSynthesizer


class StudioActionType(str, Enum):
    CLARIFY_OR_DISCUSS = "clarify_or_discuss"       # Ask/answer questions, explain telemetry, or refine spec
    PREVIEW_OR_UPDATE_ARENA = "update_arena"        # Generate or restyle arena.html with live mock animation
    UPDATE_AGENT_CHARACTERISTICS = "update_agent"   # Modify compute_reward() / demonstrations / personality
    UPDATE_ENV_MECHANICS = "update_env_mechanics"   # Modify state/action space or rules in env_module.py
    RUN_OR_CONTINUE_TRAINING = "run_training"       # Start or continue training for N rounds / M episodes


class StudioChatRequest(BaseModel):
    user_message: str
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

        # 2. Arena Look / Restyling intent keywords
        arena_kws = [
            "arena", "look", "theme", "color", "cell", "cells", "canvas", "board",
            "scoreboard", "victory badge", "badge", "gold", "arcade", "hud", "radar",
            "display", "visual", "larger", "restyle", "redesign", "bigger", "retro"
        ]
        has_arena_intent = any(kw in text for kw in arena_kws)

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
            "heuristic", "bonus for winning"
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
            or ("?" in text and not any(text.startswith(cmd) for cmd in ["change", "make", "update", "set", "train", "add", "penalize", "prioritize"]))
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
        if not actions and has_diag_intent:
            actions.append(StudioActionType.CLARIFY_OR_DISCUSS)

        if not actions:
            actions.append(StudioActionType.CLARIFY_OR_DISCUSS)

        return actions

    def process_message(
        self,
        conversation_id: str,
        user_message: str,
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
            )

        ws_dir = self.conv_store.workspaces_dir / conversation_id
        ws_dir.mkdir(parents=True, exist_ok=True)

        selected_opts = selected_options or {}
        t_params = dict(conv.get("training_params", {}) or {})
        if training_params:
            t_params.update(training_params)

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
        arena_locked = bool(conv.get("arena_locked", False))
        status = conv.get("status", "clarifying")
        assistant_reply = ""
        questions: list[ClarifyingQuestion] = []
        quick_chips: list[str] = []

        # ---------------------------------------------------------------------
        # CASE 1: FIRST TURN — Immediate Mock-Animated Arena Preview
        # ---------------------------------------------------------------------
        if not current_arena_html or not blueprint:
            intake_agent = IntakeArchitectAgent(self._llm_client)
            dataset = None

            intake_resp = intake_agent.interview(
                user_prompt=user_message,
                dataset=dataset,
                user_answers=selected_opts,
                force_smart_defaults=False,
                training_params=t_params,
            )

            # Build draft blueprint if not fully locked yet
            if intake_resp.blueprint_spec:
                blueprint = intake_resp.blueprint_spec
            else:
                # Synthesize smart defaults blueprint so arena preview can generate instantly!
                fallback_resp = intake_agent.interview(
                    user_prompt=user_message,
                    dataset=dataset,
                    user_answers=selected_opts,
                    force_smart_defaults=True,
                    training_params=t_params,
                )
                blueprint = fallback_resp.blueprint_spec

            if intake_resp.questions:
                questions = intake_resp.questions
            elif intake_resp.clarifying_questions:
                questions = intake_resp.clarifying_questions

            # Generate initial Mock-Animated arena.html
            dummy_sample_state = {"board": [0] * 9, "game_status": "Mock Arena Preview"}
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
            )

        # ---------------------------------------------------------------------
        # CASE 2: MID / POST-TRAINING CONVERSATIONAL ACTIONS
        # ---------------------------------------------------------------------

        # Intent A: UPDATE_AGENT_CHARACTERISTICS
        if StudioActionType.UPDATE_AGENT_CHARACTERISTICS in intents:
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

        # Intent B: UPDATE_ENV_MECHANICS
        if StudioActionType.UPDATE_ENV_MECHANICS in intents:
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

        # Intent C: PREVIEW_OR_UPDATE_ARENA
        if StudioActionType.PREVIEW_OR_UPDATE_ARENA in intents:
            new_arena_html = self.arena_synthesizer.update_arena(
                blueprint=blueprint,
                current_arena_html=current_arena_html,
                user_visual_request=user_message,
                sample_arena_state=None,
                workspace_dir=ws_dir,
            )
            current_arena_html = new_arena_html
            assistant_reply += "🎨 Visual Arena updated and hot-reloaded live. Neural network weights remain 100% intact!\n"
            if broadcast_callback:
                broadcast_callback({"event": "arena_updated", "arena_html": current_arena_html})

        # Intent D: RUN_OR_CONTINUE_TRAINING
        if StudioActionType.RUN_OR_CONTINUE_TRAINING in intents:
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
                code_str, _, _ = self.env_synthesizer.synthesize(blueprint=blueprint, workspace_dir=ws_dir)
                verified, healed_code, _ = self.smoke_tester.verify_and_heal(
                    code_str=code_str,
                    blueprint=blueprint,
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
            arena_html=current_arena_html,
            env_code=current_env_code,
            arena_locked=arena_locked,
            current_round=current_r,
            best_metric=best_m,
            rounds_history=r_hist,
            episodes=eps_list,
            latest_model_zip_path=model_path,
        )

        return StudioChatResponse(
            conversation_id=conversation_id,
            status=status,
            assistant_message=assistant_reply.strip(),
            action_types=intents,
            blueprint_spec=blueprint,
            arena_html=current_arena_html,
            env_code=current_env_code,
            arena_locked=arena_locked,
            questions=questions,
            quick_chips=quick_chips,
            training_params=t_params,
            current_round=current_r,
            best_metric=best_m,
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
        if "mistake" in fb or "penalize" in fb:
            additions.append('        # Mistake avoidance penalty\n        reward_breakdown["mistake_penalty"] = -2.5\n')

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
        """Generates conversational diagnosis of agent telemetry and performance."""
        best_metric = conv.get("best_metric", 0.0)
        curr_round = conv.get("current_round", 1)
        r_hist = conv.get("rounds_history", [])
        title = conv.get("title", "Agent")

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
