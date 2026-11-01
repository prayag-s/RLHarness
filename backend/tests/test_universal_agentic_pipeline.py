"""Verification test suite for Universal Generative Pipeline & Agentic Chat Planner.
Tests strictly verify the elimination of hardcoded domain templates and static keyword lists:
1. Zero-hardcoding novel domain synthesis (Connect-4 / novel domain) without rover fallback.
2. Inference strategy patch for opening move randomization in interactive_step().
3. Multi-target code and UI synchronization across architectural layers.
4. Unsupported capability honest refusal with zero file mutations.
5. Codebase audit ensuring omni_agent routing delegates to agentic planner without keyword lists.
"""
import ast
import inspect
import subprocess
import uuid
from pathlib import Path
import numpy as np
import pytest
from rl_ide.architect.models import (
    AgentBlueprintSpec,
    DynamicTaskBlueprint,
    ExecutionTarget,
    ChatActionPlan,
)
from rl_ide.architect.intake import (
    synthesize_blueprint_from_prompt,
    IntakeArchitectAgent,
)
from rl_ide.synthesizer.env_builder import (
    EnvironmentSynthesisError,
    synthesize_universal_env,
    JITEnvironmentSynthesizer,
    instantiate_env,
    load_synthesized_env_class,
)
from rl_ide.synthesizer.arena_builder import (
    synthesize_universal_arena_html,
    JITArenaSynthesizer,
)
from rl_ide.synthesizer.verifier import SandboxSmokeTester
from rl_ide.architect.omni_agent import StudioOmniAgent, StudioActionType
from rl_ide.autopilot.loop_controller import (
    AutonomousLoopController,
    SessionStateStore,
    TrainingInProgressError,
    episode_metric_value,
)
from rl_ide.llm.client import FallbackCodeSynthesizer

# Trains, exports or spawns sandbox workers: part of the full suite, not the push gate.
pytestmark = pytest.mark.slow
# -----------------------------------------------------------------------------
# 1. Novel domains: no templates, no rover, real state bound to the arena
# -----------------------------------------------------------------------------
def test_zero_hardcoding_novel_domain_synthesis(tmp_path):
    blueprint = synthesize_blueprint_from_prompt("Train a Connect-4 grid agent that drops discs into 7 columns")
    assert isinstance(blueprint, DynamicTaskBlueprint)
    assert len(blueprint.actions) == 7
    assert "rover" not in blueprint.domain_title.lower()
    assert "state_pos_x" not in [f.name for f in blueprint.observation_features]
    env_code = synthesize_universal_env(blueprint, user_prompt="connect 4")
    assert "class SynthesizedGymEnv" in env_code
    verifier = SandboxSmokeTester()
    verified, _, _ = verifier.verify_and_heal(
        code_str=env_code, blueprint=blueprint.to_agent_blueprint_spec(), workspace_dir=tmp_path
    )
    assert verified is True
    env = load_synthesized_env_class(tmp_path / "env_module.py")()
    _, info = env.reset()
    sample_state = info["arena_state"]
    arena_html = synthesize_universal_arena_html(blueprint, sample_render_state=sample_state)
    assert "sim-canvas" not in arena_html
    assert "ARENA_UPDATE" in arena_html
    assert "startMockAnimationLoop" in arena_html
    assert "sendInteractiveAction" in arena_html
    for key in blueprint.state_dict_schema:
        assert key in arena_html
    novel = synthesize_blueprint_from_prompt("Train a warehouse elevator dispatcher managing 5 floors")
    assert len(novel.actions) == 5
    novel_code = synthesize_universal_env(novel, user_prompt="elevator dispatcher")
    verified2, _, _ = verifier.verify_and_heal(
        code_str=novel_code, blueprint=novel.to_agent_blueprint_spec(), workspace_dir=tmp_path / "novel"
    )
    assert verified2 is True
# -----------------------------------------------------------------------------
# 2. Placeholder environments must never report mastery
# -----------------------------------------------------------------------------
def test_placeholder_env_never_reports_mastery(tmp_path):
    blueprint = synthesize_blueprint_from_prompt("Train an agent for a made up task without any structure")
    code = synthesize_universal_env(blueprint, allow_placeholder=True)
    env = load_synthesized_env_class(_write(tmp_path / "placeholder.py", code))()
    assert getattr(env, "is_placeholder", False) is True
    _, info = env.reset()
    assert info["is_success"] is False
    for _ in range(env.episode_horizon):
        _, reward, terminated, _, step_info = env.step(env.action_space.sample())
        assert step_info["is_success"] is False
        assert reward == 0.0
        if terminated:
            break
    with pytest.raises(EnvironmentSynthesisError):
        synthesize_universal_env(blueprint, allow_placeholder=False)
# -----------------------------------------------------------------------------
# 3. Dataset metric is accuracy, not "any correct step"
# -----------------------------------------------------------------------------
def test_dataset_metric_is_accuracy_not_any_step(tmp_path):
    blueprint = AgentBlueprintSpec(
        task_title="Fraud Triage",
        domain_summary="Classify records",
        data_source_type="uploaded_dataset",
        observation_features=["amount", "risk"],
        action_labels=["approve", "flag", "block"],
        arena_visual_description="ledger",
        interactive_mode_description="review",
        success_metric_name="Accuracy (%)",
        label_column="is_fraud",
    )
    _, _, env_cls = JITEnvironmentSynthesizer().synthesize(blueprint=blueprint, workspace_dir=tmp_path)
    records = [{"amount": float(i), "risk": float(i % 3), "is_fraud": i % 3} for i in range(60)]
    env = instantiate_env(env_cls, dataset_records=records)
    assert getattr(env, "task_class", "") == "classification"
    steps = [{"is_correct": True}] + [{"is_correct": False}] * 24
    assert episode_metric_value(env, steps) == pytest.approx(4.0)
    mixed = [{"is_correct": True}] * 15 + [{"is_correct": False}] * 10
    assert episode_metric_value(env, mixed) == pytest.approx(60.0)
    assert episode_metric_value(env, [{"is_correct": None}] * 5) == 0.0
# -----------------------------------------------------------------------------
# 4. A random policy must not look like mastery on a dataset task
# -----------------------------------------------------------------------------
def test_random_policy_on_dataset_env_does_not_report_mastery(tmp_path):
    from stable_baselines3 import PPO
    blueprint = AgentBlueprintSpec(
        task_title="Policy Auditor",
        domain_summary="Classify records",
        data_source_type="uploaded_dataset",
        observation_features=["amount", "risk"],
        action_labels=["ok", "review", "violation"],
        arena_visual_description="ledger",
        interactive_mode_description="review",
        success_metric_name="Accuracy (%)",
        target_metric_threshold=90.0,
        max_auto_tune_rounds=1,
        label_column="label",
    )
    _, _, env_cls = JITEnvironmentSynthesizer().synthesize(
        blueprint=blueprint, workspace_dir=tmp_path / "workspaces" / "sess_metric"
    )
    records = [{"amount": float(i), "risk": float(i % 3), "label": i % 3} for i in range(80)]
    env = instantiate_env(env_cls, dataset_records=records)
    model = PPO("MlpPolicy", env, n_steps=16, batch_size=8, device="cpu", verbose=0)
    controller = AutonomousLoopController(store=SessionStateStore(base_dir=tmp_path / "workspaces"))
    metric, episodes, _ = controller.evaluate_policy_deterministic(env, model, n_eval_episodes=5)
    assert episodes and all(len(ep["steps"]) == 25 for ep in episodes)
    assert 0.0 <= metric < 100.0, (
        "An untrained policy must not be reported as mastery; the metric must aggregate "
        "per-step accuracy instead of 'at least one correct step'."
    )
# -----------------------------------------------------------------------------
# 5. Inference patch is action-space aware and never proposes illegal actions
# -----------------------------------------------------------------------------
def test_inference_patch_is_action_space_aware(tmp_path):
    agent = StudioOmniAgent()
    prompt = "if agent is first, then randomize its first move among good options"
    plan = agent.plan_chat_action(prompt)
    assert ExecutionTarget.INFERENCE_STRATEGY in plan.targets
    for prompt_text, expected_actions in [
        ("Train an agent to play tic-tac-toe on a 3x3 board", 9),
        ("Train a warehouse elevator dispatcher managing 5 floors", 5),
    ]:
        bp = IntakeArchitectAgent().interview(prompt_text, wants_defaults=True).blueprint_spec
        base_code, _, _ = JITEnvironmentSynthesizer().synthesize(bp, workspace_dir=tmp_path / uuid.uuid4().hex)
        patched_code, applied = agent._patch_interactive_inference(base_code, prompt, plan=plan)
        assert applied is True
        assert patched_code != base_code
        ast.parse(patched_code)
        env = load_synthesized_env_class(_write(tmp_path / f"{uuid.uuid4().hex}.py", patched_code))()
        assert env.action_space.n == expected_actions
        def peaked(obs, n=expected_actions):
            probs = np.zeros(n, dtype=np.float32)
            probs[0] = 1.0
            return 0, probs
        openings = []
        for _ in range(30):
            env.interactive_step({"reset": True, "human_first": False}, peaked)
            openings.append(int(env.last_move))
        assert all(0 <= m < expected_actions for m in openings), f"Illegal actions produced: {openings}"
        assert len(set(openings)) >= 2, "Opening randomization was not applied"
    unchanged, applied = agent._patch_interactive_inference("class X:\n    pass\n", prompt, plan=plan)
    assert applied is False
    assert unchanged == "class X:\n    pass\n"
# -----------------------------------------------------------------------------
# 6. Reward rewriting is safe and refuses unsafe transformations
# -----------------------------------------------------------------------------
def test_reward_rewrite_is_safe_and_preserves_early_returns():
    agent = StudioOmniAgent()
    early_return = '''class Env:
    def compute_reward(self, obs, action, next_obs, info):
        breakdown = {"step": -0.01}
        if info.get("done"):
            return 10.0, {"win": 10.0}
        return float(sum(breakdown.values())), breakdown
'''
    out, applied, _ = agent._safe_reward_rewrite(early_return, "penalize mistakes more heavily")
    assert applied is True
    ast.parse(out)
    assert "return 10.0, {\"win\": 10.0}" in out, "Nested early return must be preserved"
    assert out.index("mistake_penalty") < out.rindex("return total_reward, breakdown")
    # A breakdown that is not a named variable cannot be mutated safely.
    anonymous = '''class Env:
    def compute_reward(self, obs, action, next_obs, info):
        return 0.0, {}
'''
    out2, applied2, note2 = agent._safe_reward_rewrite(anonymous, "penalize mistakes")
    assert applied2 is False
    assert out2 == anonymous
    assert "breakdown" in note2.lower()
    # Dead code after the only return makes transformation unsafe.
    trailing = '''class Env:
    def compute_reward(self, obs, action, next_obs, info):
        bd = {}
        return 0.0, bd
        bd["never"] = 1.0
'''
    out3, applied3, _ = agent._safe_reward_rewrite(trailing, "reward corners")
    assert applied3 is False
    assert out3 == trailing
    # Real shaping is applied as a validated function-level transformation.
    safe = '''class Env:
    def compute_reward(self, obs, action, next_obs, info):
        breakdown = {"step": -0.01}
        total = float(sum(breakdown.values()))
        return total, breakdown
'''
    out4, applied4, _ = agent._safe_reward_rewrite(safe, "Reward corner openings and penalize stalling")
    assert applied4 is True
    ast.parse(out4)
    assert "corner_bonus" in out4 and "speed_bonus" in out4
    assert "return total_reward, breakdown" in out4
# -----------------------------------------------------------------------------
# 7. Inference-only requests leave the reward untouched
# -----------------------------------------------------------------------------
def test_inference_only_does_not_mutate_reward(tmp_path):
    agent = StudioOmniAgent(conv_store=_store(tmp_path))
    conv_id = f"test_inference_only_{uuid.uuid4().hex[:6]}"
    assert agent.process_message(conv_id, "Train an agent to play tic-tac-toe on a 3x3 board").status == "arena_preview"
    env_file = agent.conv_store.workspaces_dir / conv_id / "env_module.py"
    assert env_file.exists()
    def reward_source(code: str) -> str:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "compute_reward":
                return ast.get_source_segment(code, node) or ""
        return ""
    before_reward = reward_source(env_file.read_text(encoding="utf-8"))
    response = agent.process_message(conv_id, "if agent is first, then randomize its first move among good options")
    assert response.status != "unsupported"
    after_code = env_file.read_text(encoding="utf-8")
    assert reward_source(after_code) == before_reward, "Inference-only change must not modify compute_reward()"
    assert "policy_predict_fn" in after_code and "_good" in after_code
    assert response.action_plan is not None
    assert response.action_plan.requires_retrain is False
# -----------------------------------------------------------------------------
# 8. Compound (multi-target) request
# -----------------------------------------------------------------------------
def test_multi_target_code_and_ui_synchronization(tmp_path):
    agent = StudioOmniAgent(conv_store=_store(tmp_path))
    conv_id = f"test_multi_target_{uuid.uuid4().hex[:6]}"
    assert agent.process_message(conv_id, "Train an agent to play tic-tac-toe on a 3x3 board").status == "arena_preview"
    prompt = "Reward corner openings during training and also randomize the agent's first move, and make the arena show a golden scoreboard"
    response = agent.process_message(conv_id, prompt)
    assert response.action_plan is not None
    targets = set(response.action_plan.targets)
    assert ExecutionTarget.REWARD_SHAPING in targets
    assert ExecutionTarget.INFERENCE_STRATEGY in targets
    assert ExecutionTarget.ARENA_VISUAL_UI in targets
    assert response.action_plan.scope_classification == "both"
    code = (agent.conv_store.workspaces_dir / conv_id / "env_module.py").read_text(encoding="utf-8")
    assert "corner_bonus" in code
    assert "_good" in code
    assert "scoreboard" in (response.arena_html or "").lower()
# -----------------------------------------------------------------------------
# 9. Honest refusal, and no substring false positives
# -----------------------------------------------------------------------------
def test_unsupported_capability_honest_refusal(tmp_path):
    agent = StudioOmniAgent(conv_store=_store(tmp_path))
    conv_id = f"test_refusal_{uuid.uuid4().hex[:6]}"
    agent.process_message(conv_id, "Train an agent for tic-tac-toe on a 3x3 board")
    env_file = agent.conv_store.workspaces_dir / conv_id / "env_module.py"
    before = env_file.read_bytes()
    impossible = "Connect this agent to a live Discord voice channel and stream 3D WebGL animations"
    response = agent.process_message(conv_id, impossible)
    assert response.status == "unsupported"
    assert "I do not know how to do that yet" in response.assistant_message
    assert env_file.read_bytes() == before
    plan = agent.plan_chat_action("Train an agent to flag fraudulent invoices")
    assert plan.supported is True
    assert ExecutionTarget.UNSUPPORTED_CAPABILITY not in plan.targets
    pivot = agent.plan_chat_action("Actually train a poker agent instead", conversation={"env_code": "x = 1"})
    assert ExecutionTarget.REANCHOR_DOMAIN in pivot.targets
    mixed = agent.plan_chat_action("Why is the first move always the center? Randomize it among good options")
    assert ExecutionTarget.INFERENCE_STRATEGY in mixed.targets
# -----------------------------------------------------------------------------
# 10. Structural audit: no domain switchboards in the runtime
# -----------------------------------------------------------------------------
def test_no_domain_routing_tables_in_runtime():
    root = Path(__file__).resolve().parent.parent / "rl_ide"
    omni_src = (root / "architect" / "omni_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(omni_src)
    planner = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_plan_chat_action"), None
    )
    assert planner is not None
    planner_src = ast.get_source_segment(omni_src, planner) or ""
    for noun in [
        "blackjack", "tictactoe", "tic-tac-toe", "roulette", "connect4", "connect-4",
        "dealer", "felt", "chips", "poker", "solar", "cartpole", "hipaa", "burn-rate",
    ]:
        assert noun not in planner_src.lower(), f"Domain noun '{noun}' found in planner routing"
    classify = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "classify_intents"), None
    )
    assert classify is not None
    assert "_plan_chat_action" in (ast.get_source_segment(omni_src, classify) or "")
    arena_src = (root / "synthesizer" / "arena_builder.py").read_text(encoding="utf-8")
    for noun in ["hipaa", "burn-rate", "burn rate", "sim-canvas", "blackjack", "roulette"]:
        assert noun not in arena_src.lower(), f"Domain branch '{noun}' found in arena builder"
def test_zero_domain_hardcoding_in_codebase():
    forbidden_pattern = (
        r"is_blackjack|is_tictactoe|is_connect4|is_roulette|is_energy|is_rover|"
        r"_build_blackjack|_build_connect4|_build_russian_roulette|_build_tictactoe|"
        r"_build_cartpole|_build_energy_arbitrage|_build_general_sim|"
        r"cardiac_denial|is_verified_corporate|login_drop_pct|sim-canvas"
    )
    runtime = Path(__file__).resolve().parent.parent / "rl_ide"
    result = subprocess.run(
        ["grep", "-rnE", forbidden_pattern, str(runtime)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, f"HARDCODED DOMAIN BRANCHES FOUND:\n{result.stdout}"
# -----------------------------------------------------------------------------
# 11. Declared state contract is enforced
# -----------------------------------------------------------------------------
def test_state_dict_schema_contract_enforced(tmp_path):
    blueprint = AgentBlueprintSpec(
        task_title="Contract Env",
        domain_summary="Test contract",
        data_source_type="pure_simulation",
        observation_features=["a", "b"],
        action_labels=["left", "right"],
        arena_visual_description="stage",
        interactive_mode_description="keys",
        success_metric_name="Success (%)",
        state_dict_schema={"board": "list[int]", "current_player": "int"},
    )
    missing_key_env = _CONTRACT_ENV_TEMPLATE.replace('"current_player": 0', '"current_player": 0')
    verified, _, logs = SandboxSmokeTester().verify_and_heal(
        code_str=missing_key_env, blueprint=blueprint, workspace_dir=tmp_path / "missing"
    )
    assert verified is False
    assert any("board" in line for line in logs), "Contract violation must name the missing key"
# -----------------------------------------------------------------------------
# 12. Environment horizon is respected (long-horizon task can reach success)
# -----------------------------------------------------------------------------
def test_episode_horizon_respected(tmp_path):
    blueprint = AgentBlueprintSpec(
        task_title="Long Horizon",
        domain_summary="Requires 80 steps",
        data_source_type="pure_simulation",
        observation_features=["a", "b"],
        action_labels=["noop", "noop2"],
        arena_visual_description="stage",
        interactive_mode_description="none",
        success_metric_name="Success (%)",
        target_metric_threshold=100.0,
        max_auto_tune_rounds=1,
    )
    env_cls = load_synthesized_env_class(_write(tmp_path / "long_horizon.py", _LONG_HORIZON_ENV))
    controller = AutonomousLoopController(store=SessionStateStore(base_dir=tmp_path / "workspaces"))
    session = controller.run_loop(
        session_id="long_horizon",
        blueprint=blueprint,
        env_cls=env_cls,
        max_rounds=1,
        episodes_per_burst=2,
    )
    assert session.current_metric == 100.0, "A >50-step success must not be capped away"
# -----------------------------------------------------------------------------
# 13. Training is serialized per session
# -----------------------------------------------------------------------------
def test_training_lock_prevents_concurrent_runs(tmp_path):
    blueprint = synthesize_blueprint_from_prompt("Train an agent on a 3x3 board").to_agent_blueprint_spec()
    controller = AutonomousLoopController(store=SessionStateStore(base_dir=tmp_path / "workspaces"))
    _, _, env_cls = JITEnvironmentSynthesizer().synthesize(blueprint, workspace_dir=tmp_path / "ws")
    lock = controller.session_lock("sess_lock")
    assert lock.acquire(blocking=False)
    try:
        with pytest.raises(TrainingInProgressError):
            controller.run_loop(
                session_id="sess_lock",
                blueprint=blueprint,
                env_cls=env_cls,
                max_rounds=1,
                episodes_per_burst=1,
            )
    finally:
        lock.release()
# -----------------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------------
_CONTRACT_ENV_TEMPLATE = '''import gymnasium as gym
from gymnasium import spaces
import numpy as np
class SynthesizedGymEnv(gym.Env):
    action_labels = ["left", "right"]
    def __init__(self, dataset_records=None, config=None):
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.t = 0
    def _obs(self):
        return np.array([0.1 * self.t, 0.2], dtype=np.float32)
    def render_arena_state(self):
        return {"current_player": self.t % 2}
    def get_render_state(self):
        return self.render_arena_state()
    def reset(self, seed=None, options=None):
        self.t = 0
        return self._obs(), {"is_success": False, "reward_breakdown": {}, "arena_state": self.render_arena_state()}
    def step(self, action):
        self.t += 1
        return self._obs(), 0.0, self.t >= 3, False, {
            "is_success": False,
            "reward_breakdown": {},
            "arena_state": self.render_arena_state(),
        }
    def compute_reward(self, obs, action, next_obs, info):
        return 0.0, {}
    def interactive_step(self, payload, policy_predict_fn=None, mode="agent_vs_env"):
        return {
            "arena_state": self.render_arena_state(),
            "agent_action_label": "left",
            "action_probs": [0.5, 0.5],
            "status_text": "ok",
        }
    def get_demonstrations(self, num_samples=32):
        return np.zeros((5, 2), dtype=np.float32), np.zeros(5, dtype=np.int64)
'''
_LONG_HORIZON_ENV = '''import gymnasium as gym
from gymnasium import spaces
import numpy as np
class SynthesizedGymEnv(gym.Env):
    action_labels = ["noop", "noop2"]
    episode_horizon = 80
    task_class = "control"
    def __init__(self, dataset_records=None, config=None):
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.t = 0
    def render_arena_state(self):
        return {"progress": float(self.t), "step_count": self.t}
    def get_render_state(self):
        return self.render_arena_state()
    def reset(self, seed=None, options=None):
        self.t = 0
        return np.zeros(2, dtype=np.float32), {"is_success": False, "reward_breakdown": {}, "arena_state": self.render_arena_state()}
    def step(self, action):
        self.t += 1
        terminated = self.t >= self.episode_horizon
        return np.zeros(2, dtype=np.float32), (1.0 if terminated else 0.0), terminated, False, {
            "is_success": terminated,
            "reward_breakdown": {},
            "arena_state": self.render_arena_state(),
        }
    def compute_reward(self, obs, action, next_obs, info):
        return 0.0, {}
    def interactive_step(self, payload, policy_predict_fn=None, mode="agent_vs_env"):
        return {
            "arena_state": self.render_arena_state(),
            "agent_action_label": "noop",
            "action_probs": [0.5, 0.5],
            "status_text": "ok",
        }
    def get_demonstrations(self, num_samples=32):
        return np.zeros((5, 2), dtype=np.float32), np.zeros(5, dtype=np.int64)
'''
def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
def _store(tmp_path: Path):
    from rl_ide.state.conversation_store import ConversationStore
    return ConversationStore(db_path=tmp_path / "conversations.db", workspaces_dir=tmp_path / "workspaces")