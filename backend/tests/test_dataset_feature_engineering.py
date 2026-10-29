"""Regression tests: supervised environments must learn from real dataset columns.

These guard the commercial path (upload a CSV with a target column, train a policy that routes
records). Before the fix, observation names invented by the LLM rarely matched the CSV columns,
so nearly every observation dimension was silently zero and no demonstrations could be extracted.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.ingest import load_bytes
from rl_ide.llm.client import LLMConfig, FallbackCodeSynthesizer
from rl_ide.synthesizer.env_builder import (
    EnvironmentSynthesisError,
    JITEnvironmentSynthesizer,
    instantiate_env,
    load_synthesized_env_class,
)
from rl_ide.synthesizer.feature_engineering import derive_feature_ops, infer_label_map
from rl_ide.synthesizer.verifier import SandboxSmokeTester

ROUTING_LABELS = [
    "Route to Tier 1 Support",
    "Route to Senior Support",
    "Route to Retention Specialist",
    "Route to Engineering Escalation",
]

QUEUES = ["Tier 1 Support", "Senior Support", "Retention Specialist", "Engineering Escalation"]


def _routing_frame(rows: int = 120) -> pd.DataFrame:
    records = []
    for i in range(rows):
        queue = QUEUES[i % len(QUEUES)]
        records.append(
            {
                "channel": ["email", "chat", "phone"][i % 3],
                "monthly_spend": 40.0 + i * 3.5,
                "prior_tickets": float(i % 9),
                "sentiment": -1.0 + 0.1 * (i % 20),
                "resolution_queue": queue,
            }
        )
    return pd.DataFrame(records)


def _blueprint(**overrides) -> AgentBlueprintSpec:
    payload = dict(
        task_title="Support Triage",
        domain_summary="Route incoming tickets to the right queue",
        data_source_type="uploaded_dataset",
        # Deliberately wrong names: the env must ignore them and derive real ones.
        observation_features=["channel_email", "plan_starter", "spend_norm"],
        action_labels=list(ROUTING_LABELS),
        arena_visual_description="dispatch desk",
        interactive_mode_description="review a ticket",
        success_metric_name="Routing Accuracy (%)",
        target_metric_threshold=90.0,
    )
    payload.update(overrides)
    return AgentBlueprintSpec(**payload)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


class _ExplodingClient(FallbackCodeSynthesizer):
    """Any LLM call on the supervised path is a defect: it means the template was rejected."""

    def generate(self, *args, **kwargs):
        raise AssertionError("LLM repair must not be needed for a labelled tabular dataset")


# -----------------------------------------------------------------------------
# Feature derivation
# -----------------------------------------------------------------------------
def test_feature_ops_are_derived_from_real_columns():
    frame = _routing_frame()
    ops, names = derive_feature_ops(frame, "resolution_queue")

    columns = {op[1] for op in ops}
    assert columns == {"channel", "monthly_spend", "prior_tickets", "sentiment"}
    assert "resolution_queue" not in columns, "the target column must not leak into observations"
    assert len(names) == len(ops) == len(set(names))
    assert "channel=email" in names and "monthly_spend" in names


def test_feature_derivation_is_deterministic_across_process_boundaries():
    """Categorical encoding must never depend on Python's salted ``hash()``."""
    frame = _routing_frame()
    first = derive_feature_ops(frame, "resolution_queue")
    second = derive_feature_ops(frame.copy(), "resolution_queue")
    assert first == second

    blueprint = _blueprint(label_column="resolution_queue")
    dataset = load_bytes(filename="tickets.csv", content=_csv_bytes(frame))
    builder = JITEnvironmentSynthesizer.__new__(JITEnvironmentSynthesizer)
    builder._llm_client = None
    code_a, features_a = builder._build_dataset_code(blueprint, dataset)
    code_b, features_b = builder._build_dataset_code(blueprint, dataset)
    assert code_a == code_b and features_a == features_b


def test_label_map_uses_action_semantics_not_declaration_order():
    mapping = infer_label_map(QUEUES, ROUTING_LABELS)
    assert mapping["Engineering Escalation"] == 3
    assert mapping["Tier 1 Support"] == 0
    # Values that cannot be tied to an action stay unmapped rather than becoming an arbitrary class.
    assert "Totally Unrelated" not in infer_label_map(["Totally Unrelated", "Tier 1 Support"], ROUTING_LABELS)


# -----------------------------------------------------------------------------
# Generated environment
# -----------------------------------------------------------------------------
def test_synthetic_env_observes_real_values_and_produces_demonstrations(tmp_path):
    frame = _routing_frame()
    dataset = load_bytes(filename="tickets.csv", content=_csv_bytes(frame))
    blueprint = _blueprint(label_column="resolution_queue")

    _, env_path, env_cls = JITEnvironmentSynthesizer(_ExplodingClient(LLMConfig(provider="fallback"))).synthesize(
        blueprint=blueprint, dataset=dataset, workspace_dir=tmp_path
    )
    env = instantiate_env(env_cls, dataset_records=frame.to_dict(orient="records"))

    assert env.task_class == "classification"
    assert env.obs_dim == len(blueprint.observation_features)
    assert "resolution_queue" not in " ".join(blueprint.observation_features)

    observations = [env._extract_obs(record) for record in env.records]
    matrix = np.vstack(observations)
    assert np.all(matrix >= 0.0) and np.all(matrix <= 1.0)
    varying = int(np.sum(np.std(matrix, axis=0) > 1e-9))
    assert varying >= math.ceil(0.5 * env.obs_dim), "most observation dimensions must carry signal"

    features, targets = env.get_demonstrations(num_samples=200)
    assert features.shape == (200, env.obs_dim)
    assert set(np.unique(targets).tolist()) == {0, 1, 2, 3}


def test_supervised_env_passes_verification_without_llm_repair(tmp_path):
    frame = _routing_frame()
    dataset = load_bytes(filename="tickets.csv", content=_csv_bytes(frame))
    blueprint = _blueprint(label_column="resolution_queue")

    builder = JITEnvironmentSynthesizer.__new__(JITEnvironmentSynthesizer)
    builder._llm_client = None
    code, features = builder._build_dataset_code(blueprint, dataset)

    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    attempt_file = attempt_dir / "env_module.py"
    attempt_file.write_text(code, encoding="utf-8")
    env_cls = load_synthesized_env_class(attempt_file)

    result = SandboxSmokeTester(None).run_smoke_test(
        env_cls,
        dataset_records=frame.to_dict(orient="records"),
        expected_state_keys=list(blueprint.state_dict_schema or {}),
        expected_obs_dim=len(features),
        expected_n_actions=len(blueprint.action_labels),
    )
    assert result["verified"] is True
    assert result["is_placeholder"] is False


def test_labels_that_cannot_be_mapped_are_rejected(tmp_path):
    frame = _routing_frame()
    frame["resolution_queue"] = [f"queue_{i}" for i in range(len(frame))]
    dataset = load_bytes(filename="tickets.csv", content=_csv_bytes(frame))
    blueprint = _blueprint(label_column="resolution_queue")

    builder = JITEnvironmentSynthesizer.__new__(JITEnvironmentSynthesizer)
    builder._llm_client = None
    with pytest.raises(EnvironmentSynthesisError):
        builder._build_dataset_code(blueprint, dataset)


def test_dead_observation_vector_fails_verification(tmp_path):
    """An env whose features never change cannot teach a policy; verification must say so."""
    code = '''
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SynthesizedGymEnv(gym.Env):
    action_labels = ["a", "b"]
    observation_features = ["dead_1", "dead_2"]
    is_placeholder = False

    def __init__(self, dataset_records=None, config=None):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.step_count = 0

    def _obs(self):
        return np.zeros(2, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.step_count = 0
        return self._obs(), {"arena_state": {"step_count": 0, "tick": 0}, "is_success": False, "reward_breakdown": {}}

    def step(self, action):
        self.step_count += 1
        return (
            self._obs(),
            0.0,
            False,
            False,
            {"arena_state": {"step_count": self.step_count, "tick": self.step_count},
             "is_success": False, "reward_breakdown": {}},
        )

    def compute_reward(self, obs, action, next_obs, info):
        return 0.0, {}

    def render_arena_state(self):
        return {"step_count": self.step_count, "tick": self.step_count}

    def interactive_step(self, payload, policy_predict_fn=None, mode="agent_vs_env"):
        return {"arena_state": self.render_arena_state(), "agent_action_label": "a",
                "action_probs": [0.5, 0.5], "status_text": "ok"}

    def get_demonstrations(self, num_samples=32):
        return np.zeros((4, 2), dtype=np.float32), np.zeros(4, dtype=np.int64)
'''
    path = tmp_path / "dead_env.py"
    path.write_text(code, encoding="utf-8")
    env_cls = load_synthesized_env_class(path)

    with pytest.raises(ValueError, match="no learnable signal"):
        SandboxSmokeTester(None).run_smoke_test(env_cls)


# -----------------------------------------------------------------------------
# Target detection (a feature is not a label)
# -----------------------------------------------------------------------------
def test_continuous_float_feature_is_not_mistaken_for_a_target():
    from rl_ide.ingest.profiler import infer_label_column

    frame = pd.DataFrame(
        {
            "clause": [f"clause {i}" for i in range(30)],
            "sentiment_score": [0.10, 0.22, 0.35, 0.41] * 7 + [0.10, 0.22],
        }
    )
    assert infer_label_column(frame, 4, ["Standard Acceptable", "Flag Unlimited Liability"]) is None


def test_action_named_column_is_preferred_over_an_unrelated_discrete_column():
    from rl_ide.ingest.profiler import infer_label_column

    frame = pd.DataFrame(
        {
            "ticket_channel": ["email", "chat"] * 20,
            "monthly_spend": [float(i) for i in range(40)],
            "resolution_queue": ["Tier 1 Support", "Senior Support"] * 20,
        }
    )
    labels = ["Route to Tier 1 Support", "Route to Senior Support"]
    assert infer_label_column(frame, 2, labels) == "resolution_queue"


# -----------------------------------------------------------------------------
# Success-reachability probe
# -----------------------------------------------------------------------------
def _terminal_success_env(success_action: int | None):
    import gymnasium as gym
    from gymnasium import spaces

    class _Env(gym.Env):
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        action_space = spaces.Discrete(4)
        episode_horizon = 6
        is_placeholder = False

        def __init__(self, dataset_records=None, config=None):
            self.t = 0
            self.ok = False

        def reset(self, seed=None, options=None):
            self.t = 0
            self.ok = False
            return np.zeros(1, dtype=np.float32), {"is_success": False}

        def step(self, action):
            self.t += 1
            if success_action is not None and int(action) == success_action:
                self.ok = True
            done = self.t >= self.episode_horizon
            return np.zeros(1, dtype=np.float32), 0.0, done, False, {"is_success": self.ok and done}

        def get_critical_actions(self):
            return [success_action] if success_action is not None else [0]

    return _Env


def test_probe_observes_terminal_only_success():
    from rl_ide.synthesizer.verifier import SandboxSmokeTester

    env = _terminal_success_env(2)()
    assert SandboxSmokeTester._probe_success(env) is True


def test_probe_reports_unreachable_success():
    from rl_ide.synthesizer.verifier import SandboxSmokeTester

    env = _terminal_success_env(None)()
    env.get_critical_actions = lambda: [0]
    assert SandboxSmokeTester._probe_success(env) is False


def test_synthesis_surfaces_unreachable_success_warning(tmp_path):
    from rl_ide.synthesizer.verifier import SandboxSmokeTester

    env = _terminal_success_env(None)()
    path = tmp_path / "unreachable.py"
    path.write_text(
        "import gymnasium as gym\n"
        "import numpy as np\n"
        "from gymnasium import spaces\n\n"
        "class SynthesizedGymEnv(gym.Env):\n"
        "    action_labels = ['a', 'b']\n"
        "    observation_features = ['x']\n"
        "    episode_horizon = 6\n"
        "    is_placeholder = False\n"
        "    def __init__(self, dataset_records=None, config=None):\n"
        "        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)\n"
        "        self.action_space = spaces.Discrete(2)\n"
        "        self.t = 0\n"
        "    def reset(self, seed=None, options=None):\n"
        "        self.t = 0\n"
        "        return np.zeros(1, dtype=np.float32), {'is_success': False, 'arena_state': {'step_count': 0, 'tick': 0}, 'reward_breakdown': {}}\n"
        "    def step(self, action):\n"
        "        self.t += 1\n"
        "        done = self.t >= 6\n"
        "        return (np.array([self.t / 6.0], dtype=np.float32), 0.0, done, False,\n"
        "                {'is_success': False, 'arena_state': {'step_count': self.t, 'tick': self.t}, 'reward_breakdown': {}})\n"
        "    def compute_reward(self, obs, action, next_obs, info):\n"
        "        return 0.0, {}\n"
        "    def render_arena_state(self):\n"
        "        return {'step_count': self.t, 'tick': self.t}\n"
        "    def interactive_step(self, payload, policy_predict_fn=None, mode='agent_vs_env'):\n"
        "        return {'arena_state': self.render_arena_state(), 'agent_action_label': 'a', 'action_probs': [0.5, 0.5], 'status_text': 'ok'}\n"
        "    def get_demonstrations(self, num_samples=32):\n"
        "        return np.zeros((4, 1), dtype=np.float32), np.zeros(4, dtype=np.int64)\n",
        encoding="utf-8",
    )
    load_synthesized_env_class(path)
    tester = SandboxSmokeTester(None)
    blueprint = _blueprint(observation_features=["x"], action_labels=["a", "b"], label_column="target")
    ok, _, _ = tester.verify_and_heal(
        code_str=path.read_text(), blueprint=blueprint, workspace_dir=tmp_path / "heal"
    )
    assert ok is True
    assert any("Success signal never observed" in w for w in tester.last_warnings)
    assert env is not None
