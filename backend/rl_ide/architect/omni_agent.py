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
from rl_ide.architect.intake import (
    IntakeArchitectAgent,
    apply_answers_to_blueprint,
    synthesize_blueprint_from_prompt,
)
from rl_ide.synthesizer.arena_builder import JITArenaSynthesizer
from rl_ide.synthesizer.env_builder import JITEnvironmentSynthesizer, load_synthesized_env_class, instantiate_env
from rl_ide.synthesizer.verifier import SandboxSmokeTester
from rl_ide.state.records_store import save_records
from rl_ide.state.job_store import JobCancelled
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
    TrainingInProgressError,
    parse_nudge_meta_command,
    record_gate_decision,
)
from rl_ide.autopilot.auto_tuner import AutoTunerAgent
from rl_ide.autopilot.eval_corpus import EvalCorpusStore, greedy_predictor
from rl_ide.autopilot.verifier_gate import VerifierGate, VerifierSnapshot, offline_metric
from rl_ide.llm.client import BaseLLMClient, get_active_llm_client, FallbackCodeSynthesizer, UnconfiguredLLMClient
from rl_ide.llm.prompts import with_english_rule


class StudioActionType(str, Enum):
    CLARIFY_OR_DISCUSS = "clarify_or_discuss"       # Ask/answer questions, explain telemetry, or refine spec
    PREVIEW_OR_UPDATE_ARENA = "update_arena"        # Generate or restyle arena.html with live mock animation
    UPDATE_AGENT_CHARACTERISTICS = "update_agent"   # Modify compute_reward() / demonstrations / personality
    UPDATE_ENV_MECHANICS = "update_env_mechanics"   # Modify state/action space or rules in env_module.py
    UPDATE_INFERENCE = "update_inference"           # Patch action selection inside interactive_step()
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
    # "bespoke" (LLM canvas), "fallback" (schema-driven preview because the canvas failed) or
    # "preview" (shown while the bespoke canvas is still generating). Lets the UI label a
    # generic preview instead of presenting it as the final artefact.
    arena_source: str | None = None
    questions: list[ClarifyingQuestion] = Field(default_factory=list)
    clarifying_questions: list[ClarifyingQuestion] = Field(default_factory=list)
    quick_chips: list[str] = Field(default_factory=list)
    training_params: dict[str, Any] = Field(default_factory=dict)
    current_round: int = 1
    best_metric: float = 0.0
    action_plan: ChatActionPlan | None = None
    # Journey Coach findings for this turn. A dict avoids importing the coach package from the
    # agent module; the API fills it from CoachReport.model_dump().
    guidance: dict[str, Any] | None = None

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
        plan: ChatActionPlan | None = None,
    ) -> list[StudioActionType]:
        """Maps an agentic ChatActionPlan to composable Studio intent actions."""
        if force_action:
            return [force_action]

        has_arena = bool(conversation.get("arena_html"))
        has_blueprint = bool(conversation.get("blueprint_spec"))

        # 1. First Turn check
        if not has_arena or not has_blueprint:
            return [StudioActionType.PREVIEW_OR_UPDATE_ARENA]

        if plan is None:
            current_env_code = conversation.get("env_code", "")
            plan = self._plan_chat_action(user_message, conversation=conversation, current_env_code=current_env_code)

        actions: list[StudioActionType] = []
        for target in plan.targets:
            if target == ExecutionTarget.ARENA_VISUAL_UI:
                if StudioActionType.PREVIEW_OR_UPDATE_ARENA not in actions:
                    actions.append(StudioActionType.PREVIEW_OR_UPDATE_ARENA)
            elif target == ExecutionTarget.INFERENCE_STRATEGY:
                if StudioActionType.UPDATE_INFERENCE not in actions:
                    actions.append(StudioActionType.UPDATE_INFERENCE)
            elif target == ExecutionTarget.REWARD_SHAPING:
                if StudioActionType.UPDATE_AGENT_CHARACTERISTICS not in actions:
                    actions.append(StudioActionType.UPDATE_AGENT_CHARACTERISTICS)
            elif target == ExecutionTarget.ENV_SIM_MECHANICS:
                if StudioActionType.UPDATE_ENV_MECHANICS not in actions:
                    actions.append(StudioActionType.UPDATE_ENV_MECHANICS)
            elif target == ExecutionTarget.TRAINING_HYPERPARAMS:
                if StudioActionType.RUN_OR_CONTINUE_TRAINING not in actions:
                    actions.append(StudioActionType.RUN_OR_CONTINUE_TRAINING)
            elif target == ExecutionTarget.REANCHOR_DOMAIN:
                if StudioActionType.PREVIEW_OR_UPDATE_ARENA not in actions:
                    actions.append(StudioActionType.PREVIEW_OR_UPDATE_ARENA)
            elif target == ExecutionTarget.UNSUPPORTED_CAPABILITY:
                if StudioActionType.CLARIFY_OR_DISCUSS not in actions:
                    actions.append(StudioActionType.CLARIFY_OR_DISCUSS)

        if not actions:
            actions.append(StudioActionType.CLARIFY_OR_DISCUSS)

        return actions

    # Domain-free capability patterns (word boundaries prevent substrings like "invoice" -> "voice").
    _UNSUPPORTED_PATTERNS: list[tuple[str, str]] = [
        (r"\bdiscord\b", "Discord integration"),
        (r"\bvoice channels?\b", "live voice channels"),
        (r"\bwebhooks?\b", "outbound webhooks"),
        (r"\bmultiplayer\s+(matchmaking|networking)\b", "multiplayer matchmaking/networking"),
        (r"\bwebgl\b|\bshaders?\b", "WebGL/shader rendering"),
        (r"\bfix protocol\b|\bbloomberg\b", "external FIX/Bloomberg terminals"),
        (r"\bexternal api\b", "external API integrations"),
        (r"\b(3d|virtual reality|vr)\s+(rendering|audio|headset)\b", "3D/VR rendering"),
        (r"\baudio\s+(streaming|synthesis)\b", "audio streaming/synthesis"),
    ]

    # Statements that ask for a change (as opposed to a question about the current state).
    _DIRECTIVE_RE = re.compile(
        r"\b(change|set|update|make|modify|patch|rewrite|tune|increase|decrease|adjust|add|remove|"
        r"randomi[sz]e|stochastic|sample|train|start|run|lock|switch|pivot|use|replace|toggle|"
        r"penalize|penalty|reward|prioritize|show|display|render|restyle|continue|resume)\b"
    )
    _VISUAL_RE = re.compile(
        r"\b(visual|theme|style|colou?r|layout|size|larger|bigger|scoreboard|hud|display|render|"
        r"badge|meter|dashboard|chart|animate|animated|animation|arrows?|look and feel|panel)\b"
    )
    _EXPLICIT_TRAINING_RE = re.compile(
        r"(\bstart\s+training\b|\bcontinue\s+training\b|\bresume\s+training\b|"
        r"\block\b[^.]*\barena\b|\btrain\s+(for|now|again)\b|\btrain\s+\d+\s+(more\s+)?rounds?\b|"
        r"\b(run|do)\s+\d+\s+(more\s+)?rounds?\b|\bmore\s+rounds?\b|\bstart\s+training\b)"
    )

    def _plan_chat_action(
        self,
        user_message: str,
        conversation: dict[str, Any] | None = None,
        current_env_code: str = "",
    ) -> ChatActionPlan:
        """Formulates a multi-layer ChatActionPlan from the prompt and the live environment.

        The deterministic router is a *domain-free intent grammar*: it reasons about verbs and
        visual/style nouns, never about particular games or industries. A configured LLM performs
        the same classification with richer context and falls back to this grammar.
        """
        conv = conversation or {}
        text = user_message.lower().strip()

        # 1. Honesty guardrail: capabilities outside the runtime.
        for pattern, label in self._UNSUPPORTED_PATTERNS:
            if re.search(pattern, text):
                reason = f"Integration with {label} is outside the supported RL arena runtime capabilities."
                return ChatActionPlan(
                    supported=False,
                    unsupported_reason=reason,
                    scope_classification="unsupported",
                    targets=[ExecutionTarget.UNSUPPORTED_CAPABILITY],
                    requires_retrain=False,
                    surgical_instructions="Zero code mutations applied.",
                    surgical_plan="Zero code mutations applied.",
                    user_facing_summary=f"I do not know how to do that yet: {reason}",
                )

        # 2. Pure discussion: an explanation request that contains no change directive.
        first_word = text.split(" ", 1)[0] if text else ""
        looks_like_question = (
            text.endswith("?")
            or first_word in {"why", "what", "how", "when", "where", "which", "who", "explain", "describe", "tell"}
            or re.search(r"\b(explain|describe|tell me|walk me through|why is|why did|what are|what is|how does|how do)\b", text)
        )
        if looks_like_question:
            tail = text.split("?", 1)[1] if "?" in text else ""
            actionable_tail = re.search(
                r"\b(change|set|update|make|modify|add|remove|randomi[sz]e|train|start|lock|switch|replace|toggle|rewrite|tune|penalize)\b",
                tail,
            )
            conjunction = re.search(
                r"\b(and|then|also|but|instead)\b[^.]{0,60}\b(change|set|update|make|modify|add|remove|"
                r"randomi[sz]e|train|start|lock|switch|replace|toggle|rewrite|tune|penalize)\b",
                text,
            )
            actionable = bool(actionable_tail or conjunction)
        else:
            actionable = True

        if looks_like_question and not actionable:
            return ChatActionPlan(
                supported=True,
                unsupported_reason=None,
                scope_classification="discussion",
                targets=[],
                requires_retrain=False,
                surgical_instructions="Zero code mutations applied. Provide conversational explanation.",
                surgical_plan="Zero code mutations applied. Provide conversational explanation.",
                user_facing_summary="Explaining agent behavior and current environment dynamics.",
            )

        # 3. Optional LLM planning with a compact structural summary of the live code.
        env_methods_summary = "Methods: reset(), step(), interactive_step(), compute_reward()"
        if current_env_code:
            try:
                tree = ast.parse(current_env_code)
                funcs = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                detail = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name in ("interactive_step", "compute_reward"):
                        segment = ast.get_source_segment(current_env_code, node) or ""
                        if segment:
                            detail.append(segment[:2000])
                env_methods_summary = f"Detected methods in env_module.py: {', '.join(funcs)}"
                if detail:
                    env_methods_summary += "\n\nRelevant method bodies:\n" + "\n\n".join(detail)
            except Exception:
                pass

        if not isinstance(self.client, (FallbackCodeSynthesizer, UnconfiguredLLMClient)):
            try:
                prompt = (
                    f"You are the Code-Aware Agentic Chat Planner for an Autonomous RL Studio.\n"
                    f"User message: \"{user_message}\"\n\n"
                    f"Environment Context:\n{env_methods_summary}\n\n"
                    f"Blueprint: {json.dumps(conv.get('blueprint_spec') or {}, default=str)[:2000]}\n\n"
                    f"Supported Execution Targets:\n"
                    f"- \"reanchor_domain\": the user pivoted to a different task/domain.\n"
                    f"- \"inference_strategy\": change how actions are selected during interactive play (randomization, top-k, masking).\n"
                    f"- \"reward_shaping\": change compute_reward() / training incentives.\n"
                    f"- \"env_sim_mechanics\": change reset()/step() rules, physics or opponents.\n"
                    f"- \"arena_visual_ui\": restyle or rebuild the visual arena.\n"
                    f"- \"training_hyperparams\": run/continue training or change its budget.\n"
                    f"- \"unsupported_capability\": outside the runtime.\n\n"
                    f"Rules:\n"
                    f"1. Arena behaviour changes route to inference_strategy, never reward_shaping alone.\n"
                    f"2. Compound requests may include multiple targets; set requires_retrain when reward/env/training change.\n"
                    f"3. Pure questions route to scope_classification=\"discussion\" with targets=[].\n"
                    f"4. Unbuilt capabilities route to supported=false with targets=[\"unsupported_capability\"].\n\n"
                    f"Respond with JSON: {{\"supported\": true, \"unsupported_reason\": null, "
                    f"\"scope_classification\": \"arena_only\", \"targets\": [\"inference_strategy\"], "
                    f"\"requires_retrain\": false, \"surgical_instructions\": \"...\", "
                    f"\"user_facing_summary\": \"...\"}}"
                )
                res = self.client.generate_json(
                    with_english_rule(prompt),
                    system_prompt=with_english_rule("You are a code-aware agentic planner."),
                )
                if res and isinstance(res, dict) and "supported" in res and "targets" in res:
                    return ChatActionPlan.model_validate(res)
            except Exception as ex:
                logging.warning(f"LLM ChatActionPlan error: {ex}")

        targets: list[ExecutionTarget] = []

        if (
            re.search(r"\b(switch|pivot|move)\s+(to|into)\b", text)
            or re.search(r"\b(change|switch)\s+(the\s+)?(domain|task|game|problem)\b", text)
            or re.search(r"\b(train|build|create|make)\b[^.]{0,60}\binstead\b", text)
            or (re.search(r"\btrain\s+(a|an)\b[^.]*\bagent\b", text) and bool(conv.get("env_code")))
        ):
            targets.append(ExecutionTarget.REANCHOR_DOMAIN)
        if re.search(
            r"\b(randomi[sz]e|stochastic|top-?k|temperature|sampling|opening|first move|turn 1|action selection|inference)\b",
            text,
        ):
            targets.append(ExecutionTarget.INFERENCE_STRATEGY)
        if re.search(r"\b(reward|penal(ize|ise|ty|izing|ising)|bonus|incentive)\b", text):
            targets.append(ExecutionTarget.REWARD_SHAPING)
        if re.search(r"\b(mechanics?|physics|dynamics|opponents?|transition|simulation)\b", text):
            targets.append(ExecutionTarget.ENV_SIM_MECHANICS)
        if self._VISUAL_RE.search(text):
            targets.append(ExecutionTarget.ARENA_VISUAL_UI)
        if self._EXPLICIT_TRAINING_RE.search(text):
            targets.append(ExecutionTarget.TRAINING_HYPERPARAMS)

        # Ordered, de-duplicated target list (execution order: reanchor, env, reward, inference, ui, training).
        order = [
            ExecutionTarget.REANCHOR_DOMAIN,
            ExecutionTarget.ENV_SIM_MECHANICS,
            ExecutionTarget.REWARD_SHAPING,
            ExecutionTarget.INFERENCE_STRATEGY,
            ExecutionTarget.ARENA_VISUAL_UI,
            ExecutionTarget.TRAINING_HYPERPARAMS,
        ]
        targets = [t for t in order if t in targets]

        has_train_impact = any(
            t in targets for t in (ExecutionTarget.REWARD_SHAPING, ExecutionTarget.ENV_SIM_MECHANICS, ExecutionTarget.TRAINING_HYPERPARAMS)
        )
        has_arena_impact = any(
            t in targets for t in (ExecutionTarget.INFERENCE_STRATEGY, ExecutionTarget.ARENA_VISUAL_UI)
        )

        if not targets:
            scope = "discussion"
        elif has_train_impact and has_arena_impact:
            scope = "both"
        elif has_train_impact:
            scope = "training_only"
        else:
            scope = "arena_only"

        requires_retrain = any(
            t in targets for t in (ExecutionTarget.REWARD_SHAPING, ExecutionTarget.ENV_SIM_MECHANICS, ExecutionTarget.TRAINING_HYPERPARAMS)
        )

        steps = {
            ExecutionTarget.REANCHOR_DOMAIN: "Re-anchor the task blueprint and re-synthesize environment + arena.",
            ExecutionTarget.INFERENCE_STRATEGY: "Patch interactive_step() to adjust action selection.",
            ExecutionTarget.ARENA_VISUAL_UI: "Restyle the arena UI.",
            ExecutionTarget.REWARD_SHAPING: "Update compute_reward() using a safe function-level transformation.",
            ExecutionTarget.ENV_SIM_MECHANICS: "Update reset()/step() rules.",
            ExecutionTarget.TRAINING_HYPERPARAMS: "Adjust the training budget and continue training.",
        }
        plan_desc = " ".join(steps[t] for t in targets) or "No execution needed."

        return ChatActionPlan(
            supported=True,
            unsupported_reason=None,
            scope_classification=scope,
            targets=targets,
            requires_retrain=requires_retrain,
            surgical_instructions=plan_desc,
            surgical_plan=plan_desc,
            user_facing_summary=f"Formulated execution plan for scope '{scope}' targeting {', '.join(t.value for t in targets) or 'discussion'}.",
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
    ) -> tuple[str, bool]:
        """Patches interactive_step() action selection in an action-space-aware way.

        Returns ``(code, applied)``. ``applied`` is False when the expected inference call
        pattern is absent, so callers never report success for a no-op.
        """
        code = current_code
        if not code or "def interactive_step" not in code:
            return code, False

        replacement = (
            "{indent}action_idx, probs = policy_predict_fn({args})\n"
            "{indent}_n_actions = int(getattr(self, 'n_actions', len(probs) if probs is not None else 1))\n"
            "{indent}_board = getattr(self, 'board', None)\n"
            "{indent}try:\n"
            "{indent}    _legal = [int(a) for a in (self.get_valid_actions() or [])] if hasattr(self, 'get_valid_actions') else []\n"
            "{indent}except Exception:\n"
            "{indent}    _legal = []\n"
            "{indent}if not _legal and _board is not None:\n"
            "{indent}    _legal = [i for i, v in enumerate(_board) if v == 0]\n"
            "{indent}_legal = [a for a in _legal if 0 <= a < _n_actions] or list(range(_n_actions))\n"
            "{indent}_is_first = (int(getattr(self, 'step_count', 0)) == 0) or (_board is not None and np.count_nonzero(_board) == 0)\n"
            "{indent}if _is_first and len(_legal) > 1:\n"
            "{indent}    if probs is not None and len(probs) == _n_actions:\n"
            "{indent}        _p = np.asarray(probs, dtype=np.float64)\n"
            "{indent}        _ranked = sorted(_legal, key=lambda a: float(_p[a]), reverse=True)\n"
            "{indent}        _best = float(_p[_ranked[0]]) if _ranked else 0.0\n"
            "{indent}        _good = [a for a in _ranked if float(_p[a]) >= 0.7 * _best]\n"
            "{indent}        if len(_good) < 2:\n"
            "{indent}            _good = _ranked[:min(3, len(_ranked))]\n"
            "{indent}    else:\n"
            "{indent}        _good = _legal[:min(3, len(_legal))]\n"
            "{indent}    action_idx = int(np.random.choice(_good))\n"
            "{indent}else:\n"
            "{indent}    action_idx = int(action_idx)\n"
            "{indent}self.last_move = int(action_idx)"
        )

        pattern = re.compile(
            r"([ \t]*)(action_idx,\s*probs\s*=\s*policy_predict_fn\((.*?)\)\s*\n[ \t]*action_idx\s*=\s*int\(action_idx\))"
        )

        def _replace_match(m):
            indent = m.group(1)
            args = m.group(3)
            return replacement.format(indent=indent, args=args)

        if not pattern.search(code):
            return code, False

        patched = pattern.sub(_replace_match, code)
        if "import numpy as np" not in patched:
            patched = "import numpy as np\n" + patched
        try:
            ast.parse(patched)
        except SyntaxError as ex:
            logging.warning(f"AST parse validation failed after patching interactive_step: {ex}")
            return code, False

        return patched, True

    def _neutral_sample_state(self, blueprint: AgentBlueprintSpec) -> dict[str, Any]:
        """Declared-contract sample state used only when no environment could be instantiated."""
        sample: dict[str, Any] = {}
        for key, type_desc in (blueprint.state_dict_schema or {}).items():
            text = str(type_desc).lower()
            if "list" in text:
                sample[key] = []
            elif "int" in text:
                sample[key] = 0
            elif "float" in text:
                sample[key] = 0.0
            else:
                sample[key] = ""
        return sample

    def _render_sample_state(self, env_cls: type | None, dataset: Any) -> dict[str, Any]:
        """Captures the environment's real render state for arena synthesis."""
        if env_cls is None:
            return {}
        records = dataset.df.to_dict(orient="records") if (dataset is not None and hasattr(dataset, "df")) else None
        env = instantiate_env(env_cls, dataset_records=records)
        _, info = env.reset()
        return dict(info.get("arena_state") or {})

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
        progress_callback: Callable[[str, str | None], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        owner_id: str | None = None,
    ) -> StudioChatResponse:
        """Single unified entrypoint for continuous conversational studio interaction.

        ``progress_callback(stage, detail)`` reports the turn's stage so a long synthesis can be
        polled instead of guessed at; ``should_cancel()`` is checked at the same boundaries and
        raises :class:`JobCancelled`, so nothing is interrupted mid-write.
        """

        def report(stage: str, detail: str | None = None) -> None:
            if progress_callback:
                progress_callback(stage, detail)
            if should_cancel and should_cancel():
                raise JobCancelled(f"cancelled during {stage}")
        conv = self.conv_store.get_conversation(conversation_id, owner_id=owner_id)
        if not conv:
            conv = self.conv_store.create_conversation(
                conversation_id=conversation_id,
                title=generate_title_from_prompt(user_message),
                training_params=training_params,
                dataset_id=dataset_id,
                owner_id=owner_id or "local",
            )

        ws_dir = self.conv_store.workspaces_dir / conversation_id
        ws_dir.mkdir(parents=True, exist_ok=True)

        effective_dataset_id = dataset_id or conv.get("dataset_id") or getattr(dataset, "dataset_id", None)
        active_dataset = dataset
        if active_dataset is None and effective_dataset_id:
            try:
                from rl_ide.state.dataset_store import get_dataset_store

                active_dataset = get_dataset_store().get(
                    effective_dataset_id, owner_id=conv.get("owner_id") or "local"
                )
            except Exception:
                pass

        selected_opts = selected_options or {}
        t_params = dict(conv.get("training_params", {}) or {})
        if training_params:
            t_params.update(training_params)

        # Control directives (rounds / episodes / explicit mastery targets) in the message itself.
        meta_cmd = parse_nudge_meta_command(user_message)
        if meta_cmd:
            if meta_cmd.get("rounds") is not None:
                t_params["max_auto_tune_rounds"] = meta_cmd["rounds"]
            if meta_cmd.get("episodes") is not None:
                t_params["episodes_per_round"] = meta_cmd["episodes"]
            if meta_cmd.get("target") is not None:
                t_params["target_metric_threshold"] = meta_cmd["target"]

        blueprint_dict = conv.get("blueprint_spec")
        blueprint: AgentBlueprintSpec | None = None
        if blueprint_dict:
            try:
                blueprint = AgentBlueprintSpec.model_validate(blueprint_dict)
            except Exception:
                blueprint = None

        current_arena_html = conv.get("arena_html") or ""
        current_env_code = conv.get("env_code") or ""
        env_file = ws_dir / "env_module.py"
        if not current_env_code and env_file.exists():
            current_env_code = env_file.read_text(encoding="utf-8")

        arena_locked = bool(conv.get("arena_locked", False))
        status = conv.get("status", "clarifying")

        # Plan once, then derive intents from it (no duplicate LLM planning call).
        action_plan = self._plan_chat_action(
            user_message=user_message,
            conversation={**conv, "env_code": current_env_code},
            current_env_code=current_env_code,
        )
        intents = self.classify_intents(user_message, {**conv, "env_code": current_env_code}, force_action=force_action, plan=action_plan)

        history = list(conv.get("chat_history", []))
        history.append({
            "role": "user",
            "content": user_message,
            "timestamp": utc_now_iso(),
            "selected_options": selected_opts if selected_opts else None,
        })

        if blueprint and t_params:
            if "target_metric_threshold" in t_params:
                blueprint.target_metric_threshold = float(t_params["target_metric_threshold"])
            if "max_auto_tune_rounds" in t_params:
                blueprint.max_auto_tune_rounds = int(t_params["max_auto_tune_rounds"])
            if "episodes_per_round" in t_params:
                blueprint.episodes_per_round = int(t_params["episodes_per_round"])

        # Intake answers are applied mid-conversation too (not just on the first turn).
        if blueprint and selected_opts:
            blueprint = apply_answers_to_blueprint(blueprint, selected_opts)
            if any(k in selected_opts for k in ("arena_ui", "q_arena_ui")) and ExecutionTarget.ARENA_VISUAL_UI not in action_plan.targets:
                action_plan.targets = list(action_plan.targets) + [ExecutionTarget.ARENA_VISUAL_UI]
                action_plan.scope_classification = "both" if action_plan.requires_retrain else "arena_only"

        assistant_reply = ""
        questions: list[ClarifyingQuestion] = []
        quick_chips: list[str] = []
        notes: list[str] = []

        # Honesty guardrail: zero mutations, instant refusal.
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

        # ------------------------------------------------------------------
        # CASE 1: FIRST TURN — synthesize the environment, then its arena
        # ------------------------------------------------------------------
        if not current_arena_html or not blueprint:
            report("intake", "Interviewing for the task spec")
            intake_agent = IntakeArchitectAgent(self._llm_client)

            intake_resp = intake_agent.interview(
                user_prompt=user_message,
                dataset=active_dataset,
                conversation_history=history[-6:],
                user_answers=selected_opts,
                force_smart_defaults=False,
                training_params=t_params,
            )

            if intake_resp.blueprint_spec:
                blueprint = intake_resp.blueprint_spec
            else:
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

            questions = intake_resp.questions or intake_resp.clarifying_questions or []

            # Declare the arena as soon as the blueprint exists, before the 60-155 s environment
            # synthesis. The declared contract is enough to render a playable preview, so the user
            # is never staring at an empty stage — and "Lock Arena" is never blocked on a provider.
            if blueprint:
                report("arena_preview", "Declaring the instant arena preview")
                try:
                    early_preview = self.arena_synthesizer.preview_arena_html(
                        blueprint=blueprint,
                        sample_arena_state=self._neutral_sample_state(blueprint),
                    )
                    current_arena_html = early_preview
                    self.conv_store.save_or_update(
                        conversation_id=conversation_id,
                        status="arena_preview",
                        title=blueprint.task_title or conv.get("title", "New Agent"),
                        blueprint_spec=blueprint.model_dump(),
                        arena_html=early_preview,
                        training_params=t_params,
                        dataset_id=effective_dataset_id,
                        arena_source="preview",
                    )
                    if broadcast_callback:
                        broadcast_callback(
                            {
                                "event": "arena_updated",
                                "arena_html": early_preview,
                                "arena_kind": "preview",
                                "pending_bespoke": True,
                            }
                        )
                except Exception as exc:  # a preview is never worth failing the turn for
                    report("arena_preview", f"Preview unavailable: {exc}")

            # Synthesize + verify the environment first so the arena is bound to real state.
            sample_state: dict[str, Any] | None = None
            synthesis_error: str | None = None
            synthesis_warnings: list[str] = []
            relaxation_log: list[str] = []
            if active_dataset is not None and hasattr(active_dataset, "df"):
                save_records(ws_dir, active_dataset.df.to_dict(orient="records"))

            if blueprint:
                report("env_synthesis", f"Writing the environment for {blueprint.task_title}")
                try:
                    code_str, _, env_cls = self.env_synthesizer.synthesize(
                        blueprint=blueprint,
                        dataset=active_dataset,
                        workspace_dir=ws_dir,
                    )
                    current_env_code = code_str
                    report("verification", "Verifying the generated environment")
                    sample_state = self._render_sample_state(env_cls, active_dataset)
                    synthesis_warnings = list(getattr(self.env_synthesizer, "last_warnings", []) or [])
                    relaxation_log = list(getattr(self.env_synthesizer, "_relaxation_log", []) or [])
                except Exception as ex:
                    synthesis_error = str(ex)
                    sample_state = self._neutral_sample_state(blueprint)

            if synthesis_error:
                status = "synthesis_failed"
                assistant_reply = (
                    f"I could not build a verified environment for **{blueprint.task_title if blueprint else 'this task'}**: "
                    f"{synthesis_error}"
                )
                history.append({"role": "assistant", "content": assistant_reply, "timestamp": utc_now_iso()})
                self.conv_store.save_or_update(
                    conversation_id=conversation_id,
                    status=status,
                    chat_history=history,
                    blueprint_spec=blueprint.model_dump() if blueprint else None,
                    training_params=t_params,
                    dataset_id=effective_dataset_id,
                )
                return StudioChatResponse(
                    conversation_id=conversation_id,
                    status=status,
                    assistant_message=assistant_reply,
                    action_types=[StudioActionType.CLARIFY_OR_DISCUSS],
                    blueprint_spec=blueprint,
                    env_code=current_env_code,
                    arena_locked=False,
                    questions=questions,
                    training_params=t_params,
                    action_plan=action_plan,
                )

            # Declare the deterministic arena *first*: it is built from the declared contract in
            # milliseconds, so the studio (and "Lock Arena") never waits on a 30-155 s provider
            # call. The bespoke canvas replaces it in place when it arrives (G15).
            report("arena_preview", "Declaring the instant arena preview")
            preview_html = self.arena_synthesizer.preview_arena_html(
                blueprint=blueprint,
                sample_arena_state=sample_state,
            )
            current_arena_html = preview_html
            self.conv_store.save_or_update(
                conversation_id=conversation_id,
                # Status is left alone: the early save already set "arena_preview", and the user may
                # have locked and started training while this turn was still generating.
                title=blueprint.task_title or conv.get("title", "New Agent"),
                blueprint_spec=blueprint.model_dump(),
                arena_html=preview_html,
                env_code=current_env_code,
                # `None` preserves whatever the user chose: locking during generation must stick.
                arena_locked=None,
                training_params=t_params,
                dataset_id=effective_dataset_id,
                arena_source="preview",
            )
            if broadcast_callback:
                broadcast_callback(
                    {
                        "event": "arena_updated",
                        "arena_html": preview_html,
                        "arena_kind": "preview",
                        "pending_bespoke": True,
                    }
                )

            report("arena", "Designing the bespoke arena")
            arena_html_str, _ = self.arena_synthesizer.synthesize(
                blueprint=blueprint,
                sample_arena_state=sample_state,
                session_id=conversation_id,
                workspace_dir=ws_dir,
            )
            arena_source = "bespoke" if getattr(self.arena_synthesizer, "last_source", "fallback") == "llm" else "fallback"
            arena_error = getattr(self.arena_synthesizer, "last_error", None)
            if arena_source == "bespoke":
                current_arena_html = arena_html_str

            status = "arena_preview"
            arena_locked = False
            assistant_reply = (
                f"✨ Generated a live Arena preview for **{blueprint.task_title}** bound to the environment's real state. "
                "Ask me in chat to adjust the look & feel, or click **[ 🔒 Lock Arena & Start Training ]** when ready."
            )
            if arena_source != "bespoke":
                assistant_reply += (
                    "\n\n⚠️ The bespoke domain canvas could not be generated"
                    f"{f' ({arena_error})' if arena_error else ''}. "
                    "The arena shown is a schema-driven preview built from the declared actions and state — "
                    "it is fully playable, but ask me to *retry the arena* once the provider recovers for the tailored view."
                )
            if relaxation_log and any("now reachable" in line for line in relaxation_log):
                # Say the criteria changed: a silently relaxed KPI is a different task.
                assistant_reply += (
                    "\n\n🔧 The environment's success criteria were unreachable at first, so I "
                    "relaxed and re-verified them. Review them in **🔍 Inspect Code** if the "
                    "objective looks easier than you described."
                )
            if synthesis_warnings:
                # Surface verification uncertainty instead of silently training toward a 0% KPI.
                assistant_reply += "\n\n⚠️ " + " ".join(synthesis_warnings)
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

            # The user can lock (and start training) from the instant preview while the bespoke
            # canvas is still generating: never clobber that lock with this turn's stale state.
            latest = self.conv_store.get_conversation(conversation_id, owner_id=owner_id) or {}
            locked_during_turn = bool(latest.get("arena_locked"))
            status_at_lock = str(latest.get("status") or "")
            if locked_during_turn and status_at_lock not in ("", "arena_preview", "clarifying"):
                status = status_at_lock

            self.conv_store.save_or_update(
                conversation_id=conversation_id,
                status=status,
                title=blueprint.task_title or conv.get("title", "New Agent"),
                chat_history=history,
                blueprint_spec=blueprint.model_dump(),
                arena_html=current_arena_html,
                env_code=current_env_code,
                arena_locked=locked_during_turn or None,
                training_params=t_params,
                dataset_id=effective_dataset_id,
                arena_source=arena_source,
            )

            if broadcast_callback:
                broadcast_callback(
                    {
                        "event": "arena_updated",
                        "arena_html": current_arena_html,
                        "arena_kind": arena_source,
                        "pending_bespoke": False,
                        "error": arena_error,
                    }
                )

            return StudioChatResponse(
                conversation_id=conversation_id,
                status=status,
                assistant_message=assistant_reply,
                action_types=[StudioActionType.PREVIEW_OR_UPDATE_ARENA],
                blueprint_spec=blueprint,
                arena_html=current_arena_html,
                env_code=current_env_code,
                arena_locked=locked_during_turn,
                arena_source=arena_source,
                questions=questions,
                quick_chips=quick_chips,
                training_params=t_params,
                current_round=conv.get("current_round", 1),
                best_metric=conv.get("best_metric", 0.0),
                action_plan=action_plan,
            )

        # ------------------------------------------------------------------
        # CASE 2: MID / POST-TRAINING MULTI-TARGET EXECUTION
        # ------------------------------------------------------------------

        # Layer 0: domain re-anchoring
        if ExecutionTarget.REANCHOR_DOMAIN in action_plan.targets:
            try:
                dyn_bp = synthesize_blueprint_from_prompt(
                    user_message, dataset=active_dataset, client=self.client
                )
                blueprint = dyn_bp.to_agent_blueprint_spec()
                if t_params:
                    if "target_metric_threshold" in t_params:
                        blueprint.target_metric_threshold = float(t_params["target_metric_threshold"])
                    if "max_auto_tune_rounds" in t_params:
                        blueprint.max_auto_tune_rounds = int(t_params["max_auto_tune_rounds"])
                    if "episodes_per_round" in t_params:
                        blueprint.episodes_per_round = int(t_params["episodes_per_round"])

                report("env_synthesis", "Re-anchoring the environment to the new task")
                code_str, _, env_cls = self.env_synthesizer.synthesize(
                    blueprint=blueprint,
                    dataset=active_dataset,
                    workspace_dir=ws_dir,
                )
                current_env_code = code_str
                # A new task invalidates the previous policy.
                self.session_store.delete_model(conversation_id)
                sample_state = self._render_sample_state(env_cls, active_dataset)
                report("arena", "Rebuilding the arena for the new task")
                current_arena_html, _ = self.arena_synthesizer.synthesize(
                    blueprint=blueprint,
                    sample_arena_state=sample_state,
                    session_id=conversation_id,
                    workspace_dir=ws_dir,
                )
                notes.append(f"🔄 Domain re-anchored to **{blueprint.task_title}**; environment and arena re-synthesized.")
                if broadcast_callback:
                    broadcast_callback({"event": "arena_updated", "arena_html": current_arena_html})
            except Exception as ex:
                notes.append(f"⚠️ Re-anchoring failed: {ex}")

        # Layer 1: environment mechanics
        if ExecutionTarget.ENV_SIM_MECHANICS in action_plan.targets and blueprint is not None:
            report("env_synthesis", "Applying the requested mechanics change")
            try:
                blueprint.pending_directive = user_message
                code_str, _, env_cls = self.env_synthesizer.synthesize(
                    blueprint=blueprint,
                    dataset=active_dataset,
                    workspace_dir=ws_dir,
                )
                if current_env_code and code_str.strip() == current_env_code.strip():
                    notes.append(
                        "ℹ️ No mechanical change was applied: the environment code is unchanged "
                        "(a verified LLM is required for novel rule generation)."
                    )
                    raise RuntimeError("no-op environment update")
                kept, gate_note = self._gate_chat_mutation(
                    conversation_id,
                    before_code=current_env_code,
                    candidate_code=code_str,
                    kind="environment",
                    ws_dir=ws_dir,
                )
                if not kept:
                    raise RuntimeError(gate_note)
                if gate_note:
                    notes.append(gate_note)
                current_env_code = code_str

                previous_model = self.session_store.get_model(conversation_id)
                new_env = instantiate_env(
                    env_cls,
                    dataset_records=active_dataset.df.to_dict(orient="records")
                    if (active_dataset is not None and hasattr(active_dataset, "df")) else None,
                )
                self.session_store.set_env(conversation_id, new_env)
                if previous_model is not None:
                    if (
                        getattr(previous_model, "observation_space", None) == new_env.observation_space
                        and getattr(previous_model, "action_space", None) == new_env.action_space
                    ):
                        previous_model.set_env(new_env)
                    else:
                        self.session_store.delete_model(conversation_id)
                        notes.append("⚠️ Action/observation space changed, so the previous policy was reset.")
                notes.append("🛠️ Environment mechanics updated and verified via the sandbox smoke tester.")
            except RuntimeError as ex:
                if "no-op environment update" not in str(ex):
                    notes.append(f"⚠️ Environment update rejected (verification failed): {ex}")
            except Exception as ex:
                notes.append(f"⚠️ Environment update rejected (verification failed): {ex}")

        # Layer 2: reward shaping (only when the plan asks for it)
        if ExecutionTarget.REWARD_SHAPING in action_plan.targets and blueprint is not None:
            updated_code, applied, note = self._safe_reward_rewrite(current_env_code, user_message)
            if not applied:
                notes.append(f"ℹ️ Reward change not applied: {note}")
            else:
                try:
                    verified, healed_code, _ = self.smoke_tester.verify_and_heal(
                        code_str=updated_code,
                        blueprint=blueprint,
                        dataset=active_dataset,
                        workspace_dir=ws_dir,
                    )
                    if verified:
                        kept, gate_note = self._gate_chat_mutation(
                            conversation_id,
                            before_code=current_env_code,
                            candidate_code=healed_code,
                            kind="reward",
                            ws_dir=ws_dir,
                        )
                        if not kept:
                            notes.append(gate_note)
                            raise RuntimeError("reward change rejected by the verifier gate")
                        if gate_note:
                            notes.append(gate_note)
                        current_env_code = healed_code
                        env_file.write_text(healed_code, encoding="utf-8")
                        session = self.session_store.get_session(conversation_id)
                        if session:
                            session.env_module_code = healed_code
                        try:
                            updated_env_cls = load_synthesized_env_class(env_file)
                            active_env = instantiate_env(updated_env_cls)
                            self.session_store.set_env(conversation_id, active_env)
                        except Exception as ex:
                            print(f"Warning: env reload after reward update: {ex}")
                        notes.append(
                            "🧠 Reward formulation updated with a validated function-level transformation. "
                            "Existing policy weights were preserved."
                        )
                        if "is_placeholder = True" in healed_code:
                            notes.append(
                                "⚠️ This environment is a generic placeholder (no LLM-generated mechanics), "
                                "so the shaped reward has no domain meaning."
                            )
                    else:
                        notes.append("⚠️ Reward change rejected: updated code failed sandbox verification.")
                except Exception as ex:
                    notes.append(f"⚠️ Reward change rejected: {ex}")

        # Layer 3: inference strategy (interactive action selection)
        if ExecutionTarget.INFERENCE_STRATEGY in action_plan.targets:
            if not current_env_code and env_file.exists():
                current_env_code = env_file.read_text(encoding="utf-8")
            patched_code, applied = self._patch_interactive_inference(
                current_code=current_env_code,
                user_request=user_message,
                plan=action_plan,
            )
            if not applied:
                notes.append(
                    "ℹ️ Inference patch not applied: this environment's interactive_step() does not expose a "
                    "patchable policy call, so no code was changed."
                )
            else:
                try:
                    verified, healed_code, _ = self.smoke_tester.verify_and_heal(
                        code_str=patched_code,
                        blueprint=blueprint,
                        dataset=active_dataset,
                        workspace_dir=ws_dir,
                    )
                    if verified:
                        kept, gate_note = self._gate_chat_mutation(
                            conversation_id,
                            before_code=current_env_code,
                            candidate_code=healed_code,
                            kind="inference",
                            ws_dir=ws_dir,
                        )
                        if not kept:
                            notes.append(gate_note)
                            raise RuntimeError("inference patch rejected by the verifier gate")
                        if gate_note:
                            notes.append(gate_note)
                        current_env_code = healed_code
                        env_file.write_text(healed_code, encoding="utf-8")
                        session = self.session_store.get_session(conversation_id)
                        if session:
                            session.env_module_code = healed_code
                        try:
                            updated_env_cls = load_synthesized_env_class(env_file)
                            active_env = instantiate_env(updated_env_cls)
                            self.session_store.set_env(conversation_id, active_env)
                        except Exception as ex:
                            print(f"Warning: Env reload after inference patch: {ex}")
                        notes.append(
                            "🎯 Arena inference strategy updated: interactive_step() now samples among the best legal "
                            "actions for this action space."
                        )
                    else:
                        notes.append("⚠️ Inference patch rejected: updated code failed sandbox verification.")
                except Exception as ex:
                    notes.append(f"⚠️ Inference patch rejected: {ex}")

        # Layer 4: arena visuals
        if ExecutionTarget.ARENA_VISUAL_UI in action_plan.targets and blueprint is not None:
            report("arena", "Restyling the arena")
            sample_state = None
            try:
                if current_env_code and env_file.exists():
                    sample_state = self._render_sample_state(
                        load_synthesized_env_class(env_file),
                        active_dataset,
                    )
            except Exception:
                sample_state = None
            current_arena_html = self.arena_synthesizer.update_arena(
                blueprint=blueprint,
                current_arena_html=current_arena_html,
                user_visual_request=user_message,
                sample_arena_state=sample_state,
                workspace_dir=ws_dir,
            )
            notes.append("🎨 Visual Arena updated and hot-reloaded live. Neural network weights remain intact.")
            if broadcast_callback:
                broadcast_callback({"event": "arena_updated", "arena_html": current_arena_html})

        # Layer 5: training budget / run
        wants_training = ExecutionTarget.TRAINING_HYPERPARAMS in action_plan.targets
        coach_report = None
        if wants_training and blueprint is not None:
            # The coach reviews the exact budget that is about to be spent. In hard mode a
            # structural blocker (zero budget, impossible target, empty data) stops the run here
            # rather than after an hour of training.
            from rl_ide.coach import CoachStage, get_coach

            coach = get_coach()
            try:
                coach_report = coach.evaluate(
                    CoachStage.PARAMETERS,
                    conversation_id=conversation_id,
                    prompt=user_message,
                    blueprint=blueprint,
                    dataset=active_dataset,
                    params=t_params,
                )
            except Exception as coach_err:  # noqa: BLE001 - the coach must never break a turn
                logging.warning("Journey Coach parameter evaluation failed: %s", coach_err)
                coach_report = None
            if coach.hard_mode and coach_report is not None and coach_report.blockers:
                wants_training = False
                status = "blocked"
                action_plan.targets = [
                    target
                    for target in action_plan.targets
                    if target != ExecutionTarget.TRAINING_HYPERPARAMS
                ]
                notes.append(
                    "⛔ Training was not started. The Journey Coach found a blocking issue: "
                    + "; ".join(item.title for item in coach_report.blockers)
                    + ". Fix it and send the request again."
                )

        if wants_training and blueprint is not None:
            arena_locked = True
            if "target_metric_threshold" in t_params:
                blueprint.target_metric_threshold = float(t_params["target_metric_threshold"])
            if "max_auto_tune_rounds" in t_params:
                blueprint.max_auto_tune_rounds = int(t_params["max_auto_tune_rounds"])
            if "episodes_per_round" in t_params:
                blueprint.episodes_per_round = int(t_params["episodes_per_round"])

            try:
                if not env_file.exists() or not current_env_code:
                    code_str, _, _ = self.env_synthesizer.synthesize(
                        blueprint=blueprint, dataset=active_dataset, workspace_dir=ws_dir
                    )
                    current_env_code = code_str

                env_cls = load_synthesized_env_class(env_file)
                eps_per_burst = int(t_params.get("episodes_per_round", 50))
                run_rounds = int(t_params.get("max_auto_tune_rounds", 2))

                if run_sync:
                    trained_session = self.loop_controller.run_loop(
                        session_id=conversation_id,
                        blueprint=blueprint,
                        env_cls=env_cls,
                        dataset=active_dataset,
                        on_event_callback=broadcast_callback,
                        additional_rounds=run_rounds,
                        episodes_per_burst=eps_per_burst,
                    )
                    status = trained_session.status
                    notes.append(
                        f"🔄 Training completed through Round {trained_session.current_round} "
                        f"(metric {trained_session.current_metric}%). Ready for interactive play."
                    )
                else:
                    status = "training"
                    notes.append(
                        f"🚀 Arena locked. Training queued for {run_rounds} more round(s) "
                        f"({eps_per_burst} episodes/round)."
                    )
            except TrainingInProgressError:
                notes.append("⚠️ Training is already running for this session; the request was not duplicated.")
            except Exception as ex:
                notes.append(f"⚠️ Training could not start: {ex}")

        if not notes:
            assistant_reply = self._generate_diagnostic_explanation(conv, user_message)
        else:
            assistant_reply = "\n".join(notes)

        quick_chips = (
            [
                "🔄 Train 5 More Rounds (500 eps/round)",
                "🧠 Make Agent More Aggressive",
                "🛡️ Penalize Mistakes More Heavily",
                "🎨 Redesign Arena Look",
            ]
            if arena_locked
            else [
                "🎨 Neon Cyberpunk Theme",
                "📊 Add Detailed Telemetry HUD",
                "🔍 Make Board / Canvas Larger",
                "🔒 Lock Arena & Start Training",
            ]
        )

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
            ExecutionTarget.ARENA_VISUAL_UI: StudioActionType.PREVIEW_OR_UPDATE_ARENA,
            ExecutionTarget.INFERENCE_STRATEGY: StudioActionType.UPDATE_INFERENCE,
            ExecutionTarget.REWARD_SHAPING: StudioActionType.UPDATE_AGENT_CHARACTERISTICS,
            ExecutionTarget.ENV_SIM_MECHANICS: StudioActionType.UPDATE_ENV_MECHANICS,
            ExecutionTarget.TRAINING_HYPERPARAMS: StudioActionType.RUN_OR_CONTINUE_TRAINING,
        }
        for t in action_plan.targets:
            act = target_map.get(t)
            if act and act not in effective_actions:
                effective_actions.append(act)

        report("ready")
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
            guidance=coach_report.model_dump(mode="json") if coach_report else None,
        )

    # Generic reward concepts -> breakdown terms. These are English concepts, not domain templates.
    _REWARD_CONCEPTS: list[tuple[str, str, float]] = [
        (r"\b(quick|faster|fast|stalling|stall)\b", "speed_bonus", 2.0),
        (r"\b(aggress|aggressive|attack|offensive)\b", "aggression_bonus", 1.5),
        (r"\bcenter|centre\b", "center_control_bonus", 2.0),
        (r"\bcorner\b", "corner_bonus", 2.0),
        (r"\b(mistake|error|penal(ize|ise|ty|izing|ising)?)\b", "mistake_penalty", -2.5),
        (r"\b(safety|unsafe|collision|obstacles?)\b", "safety_penalty", -2.0),
        (r"\b(false positive|over-?block|too strict)\b", "false_positive_penalty", -3.0),
    ]


    def _gate_chat_mutation(
        self,
        conversation_id: str,
        *,
        before_code: str,
        candidate_code: str,
        kind: str,
        ws_dir: Path,
    ) -> tuple[bool, str]:
        """Judges a chat-authored code change before it is persisted.

        Smoke verification only proves the module imports. This asks the question the user
        actually cares about: does the trained policy still satisfy the held-out rows and the
        approved behavioural cases? On failure the previous module is left untouched, the
        candidate is discarded, and the triage reason is returned for the chat timeline.
        """
        model = self.session_store.get_model(conversation_id)
        env = self.session_store.get_env(conversation_id)
        if model is None or env is None:
            # Nothing has been trained yet: the next training round gates this change instead.
            session = self.session_store.get_session(conversation_id)
            if session is not None:
                session.mutation_pending = kind
            return True, ""

        corpus = EvalCorpusStore(self.session_store.base_dir)
        gate = VerifierGate()
        predict = greedy_predictor(model)

        try:
            split = AutonomousLoopController._selection_split(env) or "val"
            before_metric = offline_metric(env, predict, split=split)
            before_report = corpus.evaluate_policy(
                conversation_id, predict
            ) if corpus.corpus_path(conversation_id).exists() else None
        except Exception as exc:
            return True, f"Gate skipped ({type(exc).__name__}); the next training round will judge the change."

        candidate_path = ws_dir / "env_module_candidate.py"
        try:
            candidate_path.write_text(candidate_code, encoding="utf-8")
            candidate_env = instantiate_env(load_synthesized_env_class(candidate_path))
        except Exception as exc:
            candidate_path.unlink(missing_ok=True)
            return False, f"⚠️ Reward change rejected: the candidate module could not be instantiated ({exc})."

        if (
            getattr(model, "observation_space", None) != candidate_env.observation_space
            or getattr(model, "action_space", None) != candidate_env.action_space
        ):
            candidate_path.unlink(missing_ok=True)
            session = self.session_store.get_session(conversation_id)
            if session is not None:
                session.mutation_pending = kind
            return True, "ℹ️ Spaces changed, so the policy must be retrained; the next round will judge this change."

        try:
            after_metric = offline_metric(candidate_env, predict, split=split)
        except Exception:
            after_metric = None

        corpus_after = None
        if before_report is not None:
            # The policy weights did not change, so corpus assertions can only move if the new
            # module alters the observation space semantics; still checked for exactly that.
            corpus_after = before_report

        decision = gate.decide(
            VerifierSnapshot(
                mutation_kind=kind,
                session_id=conversation_id,
                env_code=before_code,
                val_metric=before_metric[0] if before_metric else None,
                val_outcomes=before_metric[1] if before_metric else [],
                corpus_passing_ids=sorted(
                    corpus.passing_case_ids(conversation_id, before_report)
                ) if before_report else [],
            ),
            val_metric_after=after_metric[0] if after_metric else None,
            val_outcomes_after=after_metric[1] if after_metric else [],
            corpus_passing_ids_after=sorted(
                corpus.passing_case_ids(conversation_id, corpus_after)
            ) if corpus_after else ([] if before_report is None else None),
            corpus_pass_rate_after=corpus_after.pass_rate_pct if corpus_after else None,
        )

        session = self.session_store.get_session(conversation_id)
        if session is not None:
            record_gate_decision(
                session,
                decision,
                round_num=int(getattr(session, "current_round", 0) or 0),
                origin="chat_mutation",
            )

        if decision.passed:
            candidate_path.unlink(missing_ok=True)
            if session is not None:
                session.mutation_pending = None
            return True, ""

        candidate_path.unlink(missing_ok=True)
        category = decision.triage.category if decision.triage else "unclassified"
        advice = decision.triage.advice if decision.triage else ""
        return False, (
            f"⚠️ Reward change rolled back ({category}): {decision.reason} {advice}"
        )

    def _safe_reward_rewrite(
        self,
        current_code: str,
        user_feedback: str,
    ) -> tuple[str, bool, str]:
        """Rewrites compute_reward() only when the transformation is provably safe.

        Returns ``(code, applied, note)``. The deterministic path refuses functions with early
        returns instead of producing broken or silently-zeroed reward code.
        """
        directives = self._reward_terms_from_text(user_feedback)
        one_line = " ".join(str(user_feedback).split())[:200]

        # 1. Validated full-module synthesis via the LLM, when available.
        if not isinstance(self.client, (FallbackCodeSynthesizer, UnconfiguredLLMClient)) and current_code:
            try:
                prompt = (
                    f"Update only `compute_reward(self, state, action, next_state, info)` to reflect this request:\n"
                    f"\"{user_feedback}\"\n\n"
                    f"Preserve every method signature and unrelated reward term. Clamp each term to [-10.0, 10.0].\n"
                    f"Return the complete updated Python module inside ```python ... ```.\n\n"
                    f"Current code:\n{current_code}"
                )
                res = self.client.generate(with_english_rule(prompt))
                match = re.search(r"```python\s*([\s\S]*?)```", res)
                if match and "def compute_reward" in match.group(1):
                    candidate = match.group(1).strip()
                    ast.parse(candidate)
                    return candidate, True, "Reward function rewritten by the LLM."
            except Exception as ex:
                logging.warning(f"LLM reward rewrite failed, using safe deterministic path: {ex}")

        if not current_code or "def compute_reward" not in current_code:
            return current_code, False, "No compute_reward() implementation was found to modify."
        if not directives:
            return current_code, False, "No concrete reward incentive was recognised in the request."

        try:
            tree = ast.parse(current_code)
        except SyntaxError as ex:
            return current_code, False, f"Current environment code is not parseable: {ex}"

        target_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "compute_reward":
                target_func = node
                break
        if target_func is None:
            return current_code, False, "compute_reward() implementation not found."

        returns = [n for n in target_func.body if isinstance(n, ast.Return)]
        if len(returns) != 1 or target_func.body[-1] is not returns[0]:
            return current_code, False, (
                "This reward function has multiple/early returns, so it cannot be transformed safely "
                "without an LLM. No changes were applied."
            )
        ret = returns[0]
        if not isinstance(ret.value, ast.Tuple) or len(ret.value.elts) != 2:
            return current_code, False, "compute_reward() must return (total_reward, breakdown_dict)."

        total_src = ast.get_source_segment(current_code, ret.value.elts[0]) or "0.0"
        bd_name = ast.get_source_segment(current_code, ret.value.elts[1]) or "bd"
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", bd_name):
            return current_code, False, "Reward breakdown is not a simple local variable."

        lines = current_code.splitlines()
        ret_lineno = ret.lineno - 1  # 0-based index of the return statement
        indent = " " * (ret.col_offset)
        new_keys = [name for name, _ in directives]
        injected = [f"{indent}# User directive: {one_line}"]
        for name, value in directives:
            injected.append(f"{indent}{bd_name}[{name!r}] = {value}")
        if "sum(" in total_src and bd_name in total_src:
            injected.append(f"{indent}total_reward = float(sum({bd_name}.values()))")
        else:
            injected.append(
                f"{indent}total_reward = float({total_src}) + float(sum({bd_name}[k] for k in {new_keys!r} if k in {bd_name}))"
            )
        injected.append(f"{indent}return total_reward, {bd_name}")

        rewritten = "\n".join(lines[:ret_lineno] + injected + lines[ret_lineno + 1:]) + "\n"
        try:
            ast.parse(rewritten)
        except SyntaxError as ex:
            return current_code, False, f"Reward transformation produced invalid code ({ex}); no changes applied."
        return rewritten, True, "Reward function updated with a validated function-level transformation."

    def _reward_terms_from_text(self, text: str) -> list[tuple[str, float]]:
        lowered = str(text or "").lower()
        terms: list[tuple[str, float]] = []
        for pattern, name, default_value in self._REWARD_CONCEPTS:
            if re.search(pattern, lowered):
                # A negated concept (e.g. "never penalize safety") flips the sign.
                value = default_value
                if re.search(r"\b(never|don't|do not|stop|avoid|reduce)\b[^.]{0,40}" + pattern, lowered):
                    value = -abs(default_value)
                terms.append((name, value))
        return terms

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
                res = self.client.generate(with_english_rule(prompt), timeout=15.0)
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
