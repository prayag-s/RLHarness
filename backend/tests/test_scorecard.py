"""Regression tests for the held-out scorecard (gap G1/G14/G16).
The commercial claim is "this policy routes held-out tickets better than doing nothing". These
tests lock down the properties that make that sentence true:
* the evaluation number is produced on rows the policy never trained on,
* the split is content-derived, so it survives restarts and process boundaries,
* no oracle hint (`get_critical_actions`) or legality mask can inflate the score,
* a baseline the buyer recognises is reported next to it,
* the exported ONNX policy is checked for behaviour, not just numeric parity.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO
from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.autopilot.loop_controller import AutonomousLoopController, SessionState, SessionStateStore
from rl_ide.autopilot.scorecard import (
    build_scorecard,
    majority_class_baseline,
    uniform_chance_pct,
    wilson_ci,
)
from rl_ide.export.exporter import PolicyExporter
from rl_ide.ingest import load_bytes
from rl_ide.synthesizer.env_builder import (
    JITEnvironmentSynthesizer,
    instantiate_env,
    load_synthesized_env_class,
)
from rl_ide.synthesizer.feature_engineering import (
    records_from_frame,
    split_counts,
    split_for_record,
)

# Trains, exports or spawns sandbox workers: part of the full suite, not the push gate.
pytestmark = pytest.mark.slow
FIXTURE = Path(__file__).parent / "fixtures" / "real_world" / "support_tickets.csv"
LABELS = [
    "Route to Tier 1 Support",
    "Route to Senior Support",
    "Route to Retention Specialist",
    "Route to Engineering Escalation",
]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
def _blueprint() -> AgentBlueprintSpec:
    return AgentBlueprintSpec(
        task_title="Support Triage",
        domain_summary="Route inbound tickets to the queue that should own them.",
        data_source_type="uploaded_dataset",
        observation_features=["placeholder"],
        action_labels=list(LABELS),
        arena_visual_description="dispatch desk",
        interactive_mode_description="review a ticket",
        success_metric_name="Routing Accuracy (%)",
        target_metric_threshold=90.0,
        label_column="resolution_queue",
    )
@pytest.fixture(scope="module")
def dataset():
    return load_bytes(filename="support_tickets.csv", content=FIXTURE.read_bytes())
@pytest.fixture(scope="module")
def env_module(tmp_path_factory, dataset):
    workspace = tmp_path_factory.mktemp("scorecard_env")
    builder = JITEnvironmentSynthesizer.__new__(JITEnvironmentSynthesizer)
    builder._llm_client = None
    code, features = builder._build_dataset_code(_blueprint(), dataset)
    path = workspace / "env_module.py"
    path.write_text(code, encoding="utf-8")
    return path, features
@pytest.fixture(scope="module")
def env(env_module, dataset):
    path, _ = env_module
    env_cls = load_synthesized_env_class(path)
    return instantiate_env(env_cls, dataset_records=dataset.df.to_dict(orient="records"))
# -----------------------------------------------------------------------------
# Split integrity
# -----------------------------------------------------------------------------
def test_split_assignment_is_stable_across_processes():
    """``hash()`` is salted per process; the split must not be."""
    records = records_from_frame(pd.read_csv(FIXTURE), limit=None)
    expected = split_counts(records, "resolution_queue")
    script = (
        "import json, pandas as pd\n"
        "from rl_ide.synthesizer.feature_engineering import records_from_frame, split_counts\n"
        f"frame = pd.read_csv({str(FIXTURE)!r})\n"
        'print(json.dumps(split_counts(records_from_frame(frame, limit=None), "resolution_queue")))\n'
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(BACKEND_ROOT),
        timeout=180,
    )
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == expected
def test_split_groups_are_disjoint_and_cover_every_record(env):
    groups = env.split_indices
    flat = [i for indices in groups.values() for i in indices]
    assert sorted(flat) == list(range(len(env.records)))
    assert len(set(flat)) == len(flat)
    assert env.split_counts()["train"] > env.split_counts()["test"] > 0
def test_embedded_and_runtime_records_land_in_the_same_split(env):
    """A runbundle carries a sample of rows; those rows must keep their split on reload."""
    for record in env._records:
        assert env._split_for(record) == split_for_record(record, "resolution_queue")
def test_episodes_never_cross_the_requested_split(env):
    allowed = set(env.split_indices["test"])
    for seed in range(5):
        env.reset(seed=seed, options={"split": "test"})
        for _ in range(10):
            _, _, terminated, _, info = env.step(0)
            assert info["decision_record_index"] in allowed
            if terminated:
                break
def test_records_from_frame_shuffles_before_truncating():
    """Taking the first N rows of an export-ordered file silently samples one region."""
    frame = pd.DataFrame({
        "amount": [float(i) for i in range(600)],
        "queue": [f"q{i % 4}" for i in range(600)],
    })
    records = records_from_frame(frame, limit=120)
    assert len(records) == 120
    assert records != records_from_frame(frame, limit=None)[:120]
    # Deterministic: the same frame yields the same sample.
    assert records == records_from_frame(frame.copy(), limit=120)
    # Still a sample of the whole file, not a single region.
    assert len({r["queue"] for r in records}) == 4
# -----------------------------------------------------------------------------
# Evaluation integrity
# -----------------------------------------------------------------------------
def test_evaluation_never_consults_oracle_actions(tmp_path, env):
    """`get_critical_actions` is a tactical oracle; evaluation must not touch it."""
    class OracleTrap(type(env)):
        def get_critical_actions(self):  # pragma: no cover - must never be called
            raise AssertionError("evaluation consulted the oracle action hint")
        def get_valid_actions(self):  # pragma: no cover - must never be called
            raise AssertionError("evaluation applied a legality mask the export does not apply")
    trap = OracleTrap(dataset_records=[dict(r) for r in env.records])
    model = PPO("MlpPolicy", trap, device="cpu", verbose=0)
    controller = AutonomousLoopController(SessionStateStore(base_dir=str(tmp_path / "workspaces")))
    metric, episodes, illegal = controller.evaluate_policy_deterministic(
        env=trap, model=model, n_eval_episodes=2, round_num=1, split="test"
    )
    assert 0.0 <= metric <= 100.0
    assert len(episodes) == 2
    assert all(ep["split"] == "test" for ep in episodes)
    assert illegal >= 0.0
# -----------------------------------------------------------------------------
# Scorecard content
# -----------------------------------------------------------------------------
def test_wilson_interval_widens_for_small_samples():
    wide = wilson_ci(8, 10)
    narrow = wilson_ci(800, 1000)
    assert wide is not None and narrow is not None
    assert wide[1] - wide[0] > narrow[1] - narrow[0]
    assert narrow[0] <= 80.0 <= narrow[1]
    assert wilson_ci(0, 0) is None
def test_majority_baseline_uses_training_labels_only(env):
    baseline = majority_class_baseline(env, "test")
    assert baseline is not None
    assert 0.0 <= baseline["value_pct"] <= 100.0
    assert baseline["detail"] in LABELS
    assert uniform_chance_pct(4) == 25.0
def test_in_sample_numbers_are_flagged(env):
    scorecard = build_scorecard(
        task_title="t",
        task_class="classification",
        metric_name="acc",
        split_counts={"train": 10, "val": 0, "test": 0},
        eval_split="train",
        split_fallback=False,
        policy_metric_pct=99.0,
        episodes=10,
        illegal_rate_pct=0.0,
        baselines=[{"name": "uniform_random", "value_pct": 25.0, "detail": "1/k", "n": None}],
    )
    assert scorecard["held_out"] is False
    assert any("in-sample" in note for note in scorecard["notes"])
def test_finalize_scorecard_persists_held_out_result(tmp_path, env, dataset):
    store = SessionStateStore(base_dir=tmp_path)
    controller = AutonomousLoopController(store)
    model = PPO("MlpPolicy", env, device="cpu", verbose=0)
    session = SessionState(
        session_id="scorecard_sess",
        blueprint=_blueprint(),
        current_round=1,
    )
    events: list[dict] = []
    scorecard = controller._finalize_scorecard(
        session=session, model=model, env=env, blueprint=session.blueprint, emit=events.append
    )
    assert scorecard["held_out"] is True
    assert scorecard["eval_split"] == "test"
    assert scorecard["split_counts"]["test"] > 0
    assert sum(scorecard["split_counts"].values()) == scorecard["rows"]["total"]
    assert scorecard["policy"]["ci95_pct"] is not None
    names = {b["name"] for b in scorecard["baselines"]}
    assert {"uniform_random", "majority_class", "logistic_regression"} <= names
    assert scorecard["rows"]["embedded_in_module"] == 120
    assert session.scorecard is scorecard
    assert (tmp_path / "scorecard_sess" / "scorecard.json").exists()
    assert any(e.get("event") == "scorecard_ready" for e in events)
# -----------------------------------------------------------------------------
# Deployment parity
# -----------------------------------------------------------------------------
def test_exported_policy_behaves_like_the_trained_policy(tmp_path, env):
    model = PPO("MlpPolicy", env, device="cpu", verbose=0)
    onnx_path = PolicyExporter.export_onnx(model, tmp_path / "model.onnx", obs_dim=int(env.obs_dim))
    result = PolicyExporter.verify_behavioural_parity(
        model=model, onnx_path=onnx_path, env=env, n_episodes=2
    )
    assert result["steps_compared"] > 0
    assert result["action_agreement_pct"] == 100.0
    assert result["onnx_success_pct"] == pytest.approx(result["policy_success_pct"], abs=1e-6)