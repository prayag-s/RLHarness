"""Regression tests for the product-friction fixes (G12, G13, G15).

* G12 — an environment whose success condition no policy can reach must be repaired, not merely
  reported. The same task produced 100 %, 62 %, 18 % and 0 % success across runs because the
  generated criteria were knife-edge.
* G13 — the studio must be usable for exploration without a provider, and a model-backed route must
  say so in a machine-readable way instead of only in prose.
* G15 — the deterministic arena is declared first (instant) and a failed bespoke canvas must never be
  presented as, or silently downgraded to, the final artefact.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rl_ide.api.server import LLM_KEY_REQUIRED, app
from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.llm.client import BaseLLMClient, FallbackCodeSynthesizer, LLMConfig
from rl_ide.synthesizer.arena_builder import JITArenaSynthesizer
from rl_ide.synthesizer.verifier import SandboxSmokeTester


def _blueprint() -> AgentBlueprintSpec:
    return AgentBlueprintSpec(
        task_title="Support Ticket Router",
        domain_summary="Route inbound support tickets to the right queue.",
        data_source_type="pure_simulation",
        observation_features=["queue_depth", "sla_risk"],
        action_labels=["Route to Tier 1", "Route to Senior", "Route to Retention"],
        arena_visual_description="A queue board with three lanes",
        interactive_mode_description="Click a lane to route the ticket",
        success_metric_name="Routing Accuracy",
        target_metric_threshold=85.0,
        max_auto_tune_rounds=2,
        initial_reward_components=[],
    )


UNREACHABLE_ENV = '''
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SynthesizedGymEnv(gym.Env):
    """Success is impossible: the threshold can never be met."""

    def __init__(self, dataset_records=None):
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.episode_horizon = 5
        self.step_count = 0
        self.last_metric = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.last_metric = 0.0
        return np.zeros(2, dtype=np.float32), {"arena_state": self.get_render_state()}

    def step(self, action):
        self.step_count += 1
        # A metric that never approaches the 0.999 success condition.
        self.last_metric = 0.5
        obs = np.array([float(action) / 3.0, self.last_metric], dtype=np.float32)
        info = {
            "arena_state": self.get_render_state(),
            "reward_breakdown": {"routing": 0.0},
            "is_success": self.last_metric >= 0.999,
        }
        terminated = self.step_count >= self.episode_horizon
        return obs, 0.0, terminated, False, info

    def compute_reward(self, obs, action, next_obs, info):
        return 0.0, {}

    def get_render_state(self):
        return {"step": self.step_count, "metric": self.last_metric}

    def render_arena_state(self):
        return self.get_render_state()

    def interactive_step(self, payload, policy_predict_fn=None, mode="agent_vs_env"):
        obs, _, _, _, info = self.step(int(payload.get("action", 0)) if isinstance(payload, dict) else 0)
        return {
            "arena_state": info["arena_state"],
            "agent_action_label": "Route to Tier 1",
            "action_probs": [0.34, 0.33, 0.33],
            "status_text": "routed",
        }

    def get_demonstrations(self, num_samples=32):
        X = np.random.rand(num_samples, 2).astype(np.float32)
        y = np.zeros(num_samples, dtype=np.int64)
        return X, y
'''


REACHABLE_ENV = UNREACHABLE_ENV.replace(
    '"is_success": self.last_metric >= 0.999,',
    '"is_success": self.last_metric >= 0.5,',
)


def _offline_client() -> FallbackCodeSynthesizer:
    """The zero-cost client the studio uses in tests: no provider, deterministic output."""
    return FallbackCodeSynthesizer(LLMConfig(provider="fallback"))


class _ScriptedRelaxer(BaseLLMClient):
    """Stands in for a provider: every prompt is answered with the given module.

    It deliberately does *not* subclass ``FallbackCodeSynthesizer`` — that class is the marker for
    "no model configured", and the relaxation pass must treat it as unavailable.
    """

    def __init__(self, reply: str):
        super().__init__(LLMConfig(provider="deepseek", api_key="sk-test-12345678"))
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt, system_prompt=None, **kwargs):  # noqa: ANN001, ANN003
        self.prompts.append(prompt)
        return f"```python\n{self.reply}\n```"


def test_unreachable_success_is_relaxed_and_reverified(tmp_path):
    """G12: a task that can never succeed is repaired, and the repair is reported."""
    client = _ScriptedRelaxer(REACHABLE_ENV)
    tester = SandboxSmokeTester(llm_client=client)

    verified, healed, _logs = tester.verify_and_heal(
        UNREACHABLE_ENV, blueprint=_blueprint(), workspace_dir=tmp_path
    )
    assert verified is True
    assert tester.last_success_observed is False

    fixed, code, relax_logs = tester.relax_unreachable_success(
        code_str=healed, blueprint=_blueprint(), workspace_dir=tmp_path
    )

    assert fixed is True, relax_logs
    assert tester.last_success_observed is True
    assert any("now reachable" in line for line in relax_logs)
    assert len(client.prompts) == 1, "one rewrite should be enough here"


def test_relaxation_is_capped_and_keeps_the_verified_code(tmp_path):
    """A provider that keeps returning unreachable code must not loop or corrupt the module."""
    client = _ScriptedRelaxer(UNREACHABLE_ENV)
    tester = SandboxSmokeTester(llm_client=client)

    fixed, code, relax_logs = tester.relax_unreachable_success(
        code_str=UNREACHABLE_ENV, blueprint=_blueprint(), workspace_dir=tmp_path, max_attempts=2
    )

    assert fixed is False
    assert len(client.prompts) == 2, "rewrites must be capped"
    assert "class SynthesizedGymEnv" in code
    assert any("still unreachable" in line for line in relax_logs)


def test_relaxation_never_calls_a_provider_offline(tmp_path):
    """Without a model the pass must not fabricate work: it reports and returns the input."""
    tester = SandboxSmokeTester(llm_client=_offline_client())
    fixed, code, logs = tester.relax_unreachable_success(
        code_str=UNREACHABLE_ENV, blueprint=_blueprint(), workspace_dir=tmp_path
    )
    assert fixed is False
    assert code == UNREACHABLE_ENV
    assert any("No model configured" in line for line in logs)


def test_synthesizer_hooks_relaxation_and_keeps_the_promise(tmp_path):
    """G12 end to end: `synthesize()` runs the repair and returns the repaired module.

    The synthesizer owns the hook, so this drives it with a provider stand-in that first returns an
    unreachable environment (the synthesis prompt) and then a reachable one (the relaxation prompt).
    """
    from rl_ide.synthesizer.env_builder import JITEnvironmentSynthesizer

    class _TwoStageProvider(BaseLLMClient):
        def __init__(self):
            super().__init__(LLMConfig(provider="deepseek", api_key="sk-test-12345678"))
            self.prompts: list[str] = []

        def generate(self, prompt, system_prompt=None, timeout=None):  # noqa: ANN001
            self.prompts.append(prompt)
            body = REACHABLE_ENV if "never signals success" in prompt else UNREACHABLE_ENV
            return f"```python\n{body}\n```"

    provider = _TwoStageProvider()
    synth = JITEnvironmentSynthesizer(llm_client=provider)
    code, path, env_cls = synth.synthesize(blueprint=_blueprint(), workspace_dir=tmp_path)

    assert len(provider.prompts) == 2, provider.prompts[:1]
    assert "now reachable" in " ".join(synth._relaxation_log)
    assert "<html" not in code  # it is python
    assert "self.last_metric >= 0.5" in code, "the relaxed criteria must be what gets persisted"
    assert path.read_text(encoding="utf-8") == code
    assert synth.last_warnings == [], "a repaired task must not still warn about an unreachable KPI"


def test_oracle_only_success_is_reported_as_low_achievable_rate(tmp_path):
    """Success reachable only through `get_critical_actions` must not look fully achievable."""
    code = UNREACHABLE_ENV.replace(
        "    def get_demonstrations(self, num_samples=32):",
        "    def get_critical_actions(self):\n        return [1]\n\n    def get_demonstrations(self, num_samples=32):",
    )
    tester = SandboxSmokeTester(llm_client=None)
    _verified, _healed, _logs = tester.verify_and_heal(code, blueprint=_blueprint(), workspace_dir=tmp_path)
    # This env cannot succeed even with the hint, so it is simply unreachable.
    assert tester.last_success_observed is False


def test_model_backed_routes_are_marked_and_exploration_is_not(monkeypatch):
    """G13: a 403 carries a reason header; upload and capabilities work without a provider."""
    import rl_ide.api.server as server

    monkeypatch.setattr(server, "is_llm_ready", lambda: False)
    client = TestClient(app)

    refused = client.post("/api/v4/conversations/probe_conv/chat", json={"user_message": "hi"})
    assert refused.status_code == 403
    assert refused.headers.get("X-RL-Reason") == LLM_KEY_REQUIRED
    assert "needs a model" in refused.json()["detail"]

    capabilities = client.get("/api/v4/capabilities")
    assert capabilities.status_code == 200
    body = capabilities.json()
    assert body["llm"]["verified"] is False
    assert "synthesize_environment" in body["needs_llm"]
    assert "upload_dataset" in body["works_without_llm"]

    models = client.get("/api/v4/conversations")
    assert models.status_code == 200, "exploration must not require a provider"


def test_arena_preview_is_instant_and_needs_no_provider():
    """G15: the deterministic arena is built from the contract alone, with no provider call."""
    synth = JITArenaSynthesizer(llm_client=_offline_client())
    html = synth.preview_arena_html(
        blueprint=_blueprint(),
        sample_arena_state={"queue_depth": 4, "sla_risk": 0.2},
    )
    assert "ARENA_UPDATE" in html
    assert "startMockAnimationLoop" in html
    assert "Route to Tier 1" in html


def test_failed_bespoke_canvas_is_labelled_not_hidden():
    """A silent fallback is what made a broken generator look like a working one."""
    class _Broken:
        config = type("Cfg", (), {"timeout": 1.0})()

        def generate(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("provider timed out")

    outcome: dict = {}
    from rl_ide.synthesizer.arena_builder import synthesize_universal_arena_html

    html = synthesize_universal_arena_html(
        blueprint=_blueprint(), client=_Broken(), outcome=outcome
    )
    assert outcome["source"] == "fallback"
    assert "timed out" in outcome["error"]
    assert html, "the user still gets a playable arena"


def test_restyle_failure_keeps_the_existing_canvas(monkeypatch):
    """A failed restyle must not downgrade a bespoke canvas to the generic preview."""
    class _Broken:
        config = type("Cfg", (), {"timeout": 1.0})()

        def generate(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("rate limited")

    synth = JITArenaSynthesizer(llm_client=_Broken())
    previous = "<html><body>bespoke canvas</body></html>"
    returned = synth.update_arena(
        blueprint=_blueprint(),
        current_arena_html=previous,
        user_visual_request="make it neon",
    )
    assert returned == previous
    assert synth.last_source == "fallback"
    assert synth.last_error and "rate limited" in synth.last_error


def test_arena_source_is_persisted_with_the_conversation(tmp_path):
    """The UI must be able to tell a preview from the bespoke canvas after a reload."""
    from rl_ide.state.conversation_store import ConversationStore

    store = ConversationStore(db_path=tmp_path / "s.db", workspaces_dir=tmp_path / "ws")
    store.create_conversation(conversation_id="c1", owner_id="local")
    store.save_or_update("c1", arena_html="<html/>", arena_source="preview")
    assert store.get_conversation("c1")["arena_source"] == "preview"
    store.save_or_update("c1", arena_html="<html>final</html>", arena_source="bespoke")
    assert store.get_conversation("c1")["arena_source"] == "bespoke"


def test_preview_is_declared_before_the_environment_exists(tmp_path):
    """G15: the arena appears as soon as the blueprint does, not after a 60-155 s synthesis.

    The user must be able to see (and lock) an arena while the environment is still generating,
    otherwise "Lock Arena" is blocked on a provider call for minutes.
    """
    from rl_ide.architect.omni_agent import StudioOmniAgent
    from rl_ide.state.conversation_store import ConversationStore

    store = ConversationStore(db_path=tmp_path / "p.db", workspaces_dir=tmp_path / "ws")
    agent = StudioOmniAgent(conv_store=store)
    conv_id = "conv_preview_order"

    events: list[dict] = []
    stages: list[str] = []

    def on_event(ev):
        events.append(ev)

    def on_stage(stage, detail=None):
        stages.append(stage)

    agent.process_message(
        conv_id,
        "Build an agent that routes support tickets to the right queue",
        broadcast_callback=on_event,
        progress_callback=on_stage,
    )

    # The preview is announced before the environment is written, and marked as a preview.
    preview_events = [e for e in events if e.get("event") == "arena_updated" and e.get("arena_kind") == "preview"]
    assert preview_events, events
    assert preview_events[0].get("pending_bespoke") is True
    assert stages.index("arena_preview") < stages.index("env_synthesis")
    assert store.get_conversation(conv_id)["arena_source"] in ("preview", "bespoke", "fallback")


def test_locking_during_generation_is_not_overwritten(tmp_path):
    """Locking from the instant preview must survive the turn's completion save."""
    from rl_ide.architect.omni_agent import StudioOmniAgent
    from rl_ide.state.conversation_store import ConversationStore

    store = ConversationStore(db_path=tmp_path / "l.db", workspaces_dir=tmp_path / "ws")
    agent = StudioOmniAgent(conv_store=store)
    conv_id = "conv_lock_race"

    def lock_from_the_preview(ev):
        # The user clicks "Lock Arena" the moment the preview is broadcast.
        if ev.get("event") == "arena_updated" and ev.get("arena_kind") == "preview":
            store.save_or_update(conv_id, arena_locked=True, status="training")

    resp = agent.process_message(
        conv_id,
        "Build an agent that routes support tickets to the right queue",
        broadcast_callback=lock_from_the_preview,
    )

    saved = store.get_conversation(conv_id)
    assert saved["arena_locked"] is True, "the lock was clobbered by the turn's save"
    assert resp.arena_locked is True
    assert saved["status"] == "training", "the turn must not reset a status the user moved past"
