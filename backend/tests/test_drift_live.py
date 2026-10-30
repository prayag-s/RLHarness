"""Live drift monitoring (G11).

`DriftDetector` shipped in v4.0 and was imported by exactly one module, so every session reported
"unmonitored" while a policy could silently degrade behind a green scorecard. Worse, the one caller
used an API that no longer existed inside a bare `except`, so exported bundles reported "healthy" for
any input at all.

These tests pin the behaviour that matters: a policy fed out-of-distribution observations is reported
as drifting, a policy fed its own distribution is not, and neither an empty window nor a missing
baseline can be mistaken for a clean bill of health.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rl_ide.drift import (
    DRIFTED,
    HEALTHY,
    UNMONITORED,
    WARMING_UP,
    DriftBaseline,
    DriftDetector,
    DriftMonitor,
    DriftStatusStore,
    fit_baseline_from_episodes,
)
from rl_ide.drift.monitor import DriftReport


def _sample_action(rng) -> tuple[int, list[float]]:
    """A consistent (action, probabilities) pair: the policy is right, so argmax matches."""
    action = int(rng.integers(0, 3))
    probs = [0.1, 0.1, 0.1]
    probs[action] = 0.8
    return action, probs


def _baseline(seed: int = 0, n: int = 200) -> DriftBaseline:
    rng = np.random.default_rng(seed)
    from rl_ide.contracts.adapter import EpisodeLog, StepResult

    steps: list[StepResult] = []
    probs: list[list[float]] = []
    for _ in range(n):
        action, p = _sample_action(rng)
        probs.append(p)
        steps.append(
            StepResult(
                obs_vector=[float(rng.normal(0.5, 0.05)), float(rng.normal(0.5, 0.05))],
                events={"action": {"action_idx": action}},
                terminated=False,
                truncated=False,
            )
        )
    return DriftDetector.fit_baseline(
        episodes=[EpisodeLog(episode_id=0, steps=steps)],
        feature_names=["amount", "risk"],
        action_probs=probs,
    )


def _in_distribution(baseline: DriftBaseline, n: int = 20, seed: int = 7):
    """Fresh draws from the same process that produced the baseline."""
    rng = np.random.default_rng(seed)
    obs = [[float(rng.normal(0.5, 0.05)), float(rng.normal(0.5, 0.05))] for _ in range(n)]
    probs = [_sample_action(rng)[1] for _ in range(n)]
    return obs, probs


def _out_of_distribution(baseline: DriftBaseline, n: int = 20):
    obs = [[9.0, -9.0] for _ in range(n)]
    probs = [[0.34, 0.33, 0.33] for _ in range(n)]
    return obs, probs


def _monitor(baseline: DriftBaseline, **kwargs) -> DriftMonitor:
    """A monitor that judges every window immediately, for unit-level assertions."""
    params = {"window": 20, "stride": 1, "confirmations": 1}
    params.update(kwargs)
    return DriftMonitor(baseline, **params)


def test_a_policy_fed_its_own_distribution_is_healthy():
    """In-distribution traffic does not raise the alarm at the calibrated live settings."""
    baseline = _baseline()
    monitor = _monitor(baseline, window=50, psi_threshold=0.25)
    obs, probs = _in_distribution(baseline, n=50, seed=2)

    report = None
    for o, p in zip(obs, probs):
        report = monitor.observe(o, p) or report

    assert report is not None
    assert report.is_drifted is False, report.reasons
    assert monitor.status == HEALTHY


def test_a_policy_fed_unseen_inputs_is_reported_as_drifting():
    baseline = _baseline()
    monitor = _monitor(baseline)
    obs, probs = _out_of_distribution(baseline)

    report = None
    for o, p in zip(obs, probs):
        report = monitor.observe(o, p) or report

    assert report is not None
    assert report.is_drifted is True
    assert monitor.status == DRIFTED
    assert report.max_psi > 0.25
    assert any("PSI" in r for r in report.reasons)
    assert "Retrain" in report.recommendation


def test_calibration_is_quiet_on_healthy_traffic_and_prompt_on_drift():
    """Pins the measured behaviour of the shipped defaults.

    Twenty fixed seeds of *unchanged* traffic (200 interactions each) may raise at most one alert;
    genuinely shifted traffic must be confirmed well before 100 interactions. Both bounds are what
    make the header badge worth looking at.
    """
    baseline = _baseline()

    flagged = 0
    for seed in range(20):
        obs, probs = _in_distribution(baseline, n=200, seed=500 + seed)
        monitor = DriftMonitor(baseline)
        for o, p in zip(obs, probs):
            monitor.observe(o, p)
        flagged += 1 if monitor.status == DRIFTED else 0
    assert flagged <= 1, f"{flagged}/20 unchanged sessions were flagged as drifting"

    obs, probs = _out_of_distribution(baseline, n=200)
    monitor = DriftMonitor(baseline)
    first_drifted = None
    for i, (o, p) in enumerate(zip(obs, probs), 1):
        monitor.observe(o, p)
        if monitor.status == DRIFTED and first_drifted is None:
            first_drifted = i
    assert first_drifted is not None, "a fully shifted stream was never flagged"
    assert first_drifted <= 100, f"took {first_drifted} interactions to notice real drift"


def test_one_crossing_is_not_an_incident():
    """A single noisy window is reported but does not flip the status on its own."""
    baseline = _baseline()
    monitor = DriftMonitor(baseline, window=50, stride=1, confirmations=2, psi_threshold=0.25)
    # 50 observations fill the first window; the next observation judges a second one.
    obs, probs = _out_of_distribution(baseline, n=60)

    judgements = 0
    for o, p in zip(obs, probs):
        if monitor.observe(o, p) is not None:
            judgements += 1
            if judgements == 1:
                # First window crossed, but one crossing is not yet an incident.
                assert monitor.status == HEALTHY
                assert monitor.snapshot()["unconfirmed_crossing"] is True
    assert monitor.status == DRIFTED, "the second consecutive window must confirm"


def test_a_window_too_small_to_judge_says_so_instead_of_guessing():
    baseline = _baseline()
    monitor = _monitor(baseline)
    obs, probs = _in_distribution(baseline, n=5)

    for o, p in zip(obs, probs):
        assert monitor.observe(o, p) is None
    assert monitor.status == WARMING_UP
    assert monitor.snapshot()["samples_seen"] == 5


def test_drift_clears_when_the_policy_comes_back_in_range():
    """A transient spike must not alarm forever."""
    baseline = _baseline()
    monitor = _monitor(baseline, window=50)

    obs, probs = _out_of_distribution(baseline, n=50)
    for o, p in zip(obs, probs):
        monitor.observe(o, p)
    assert monitor.status == DRIFTED

    good_obs, good_probs = _in_distribution(baseline, n=60, seed=11)
    for o, p in zip(good_obs, good_probs):
        monitor.observe(o, p)
    assert monitor.status == HEALTHY


def test_missing_probabilities_do_not_fabricate_confidence():
    """A uniform fallback keeps entropy neutral rather than inventing a confident policy."""
    monitor = _monitor(_baseline(), window=4)
    probs = DriftMonitor._as_distribution(None, 3)
    assert probs == [1 / 3, 1 / 3, 1 / 3]
    assert abs(sum(probs) - 1.0) < 1e-9
    # And it still accepts the step instead of raising.
    for _ in range(4):
        monitor.observe([0.5, 0.5], None, 3)
    assert monitor.status in (HEALTHY, DRIFTED)


def test_a_garbage_observation_is_ignored_not_fatal():
    monitor = _monitor(_baseline(), window=4)
    assert monitor.observe(None, [0.5, 0.5]) is None
    assert monitor.observe(["not", "numbers"], [0.5, 0.5]) is None


def test_status_survives_a_restart(tmp_path):
    """The status belongs to the conversation, not to the process that observed it."""
    store = DriftStatusStore()
    snapshot = _monitor(_baseline()).snapshot()
    snapshot["status"] = DRIFTED
    store.save(tmp_path, snapshot)

    reloaded = store.load(tmp_path)
    assert reloaded["status"] == DRIFTED
    assert json.loads((tmp_path / "drift" / "live_status.json").read_text())["status"] == DRIFTED
    assert store.load(tmp_path / "never_trained")["status"] == UNMONITORED


def test_baseline_fits_from_stored_training_episodes():
    """The live baseline comes from the episodes training actually produced."""
    episodes = [
        {
            "episode_id": 0,
            "steps": [
                {
                    "obs_vector": [0.1 * i, 0.2 * i],
                    "events": {"action": {"action_idx": i % 3}},
                    "terminated": False,
                    "truncated": False,
                    "info": {},
                }
                for i in range(50)
            ],
            "total_reward": 0.0,
            "breakdown": {},
            "step_rewards": [],
            "step_breakdowns": [],
            "locked": False,
            "metadata": {},
        }
    ]
    baseline = fit_baseline_from_episodes(episodes, ["amount", "risk"])
    assert baseline is not None
    assert baseline.feature_names == ["amount", "risk"]
    assert baseline.n_samples == 50
    assert set(baseline.action_distribution) == {"0", "1", "2"}


def test_no_episodes_means_no_baseline_rather_than_a_fake_one():
    assert fit_baseline_from_episodes([], ["amount"]) is None
    assert fit_baseline_from_episodes([{"episode_id": 0, "steps": []}], ["amount"]) is None


def test_baseline_loads_from_a_mapping_too():
    """The exporter holds parsed contents; a path-only loader silently broke every bundle check."""
    baseline = _baseline()
    loaded = DriftBaseline.load(baseline.model_dump())
    assert loaded.n_samples == baseline.n_samples
    assert loaded.feature_names == baseline.feature_names


def test_the_detector_reports_drift_with_the_public_field_names():
    """Guards the exporter contract: `check_drift` / `is_drifted` / `max_psi`, not invented names."""
    detector = DriftDetector(_baseline())
    obs, probs = _out_of_distribution(_baseline())
    report = detector.check_drift(obs, probs)
    assert isinstance(report, DriftReport)
    assert report.is_drifted is True
    assert report.max_psi > 0.0
    # The names the old exporter called, which never existed.
    assert not hasattr(report, "drift_detected")
    assert not hasattr(report, "feature_psi_max")


def test_interacting_scores_drift_and_warns_over_the_websocket(tmp_path, monkeypatch):
    """End to end: `/interact` scores the live step and the studio is told when it crosses over."""
    import rl_ide.api.server as server
    from rl_ide.drift import DriftStatusStore

    session_id = "conv_drift_e2e"
    ws = tmp_path / "workspaces" / session_id
    ws.mkdir(parents=True)
    _baseline().save(ws / "drift_baseline.json")
    monkeypatch.setattr(server, "_drift_workspace", lambda sid: ws)
    monkeypatch.setattr(server, "_DRIFT_WINDOW", 20)
    monkeypatch.setattr(server, "_DRIFT_STRIDE", 1)
    server.forget_drift_state(session_id)

    broadcasts: list[dict] = []
    monkeypatch.setattr(server, "_broadcast_threadsafe", lambda cid, ev: broadcasts.append(ev))

    obs, probs = _out_of_distribution(_baseline(), n=30)
    result: dict = {}
    for o, p in zip(obs, probs):
        result = server._with_drift(
            session_id, {"arena_state": {"step": 1}, "obs_vector": o, "action_probs": p}
        )

    assert "drift" in result, "every scored interaction must report its drift state"
    assert result["drift"]["status"] == DRIFTED
    assert any(ev.get("event") == "drift_warning" for ev in broadcasts)
    # The verdict outlives this process, because it belongs to the conversation.
    assert DriftStatusStore().load(ws)["status"] == DRIFTED
    server.forget_drift_state(session_id)


def test_monitoring_never_breaks_a_step_when_the_baseline_is_missing(monkeypatch, tmp_path):
    import rl_ide.api.server as server

    monkeypatch.setattr(server, "_drift_workspace", lambda sid: tmp_path / "nothing_here")
    server.forget_drift_state("conv_no_baseline")
    payload = {"arena_state": {"step": 1}, "obs_vector": [0.1, 0.2], "action_probs": [0.5, 0.5]}
    assert server._with_drift("conv_no_baseline", payload) == payload
    assert server.drift_snapshot("conv_no_baseline")["status"] == UNMONITORED


def test_training_baseline_refresh_accepts_dict_episodes_and_never_fails_the_run(monkeypatch, tmp_path):
    """`SessionState.episodes` are plain dicts; treating them as models failed a finished run."""
    import rl_ide.api.server as server

    monkeypatch.setattr(server, "_drift_workspace", lambda sid: tmp_path / "ws")

    class _Session:
        episodes = []          # a session that trained but kept no episodes
        total_episodes_trained = 0

    # No episodes -> no baseline, and crucially no exception.
    server._refresh_drift_baseline("conv_x", _Session(), _blueprint_stub())
    assert not (tmp_path / "ws" / "drift_baseline.json").exists()

    episode_dict = {
        "episode_id": 0,
        "steps": [
            {
                "obs_vector": [0.1 * i, 0.2 * i],
                "events": {"action": {"action_idx": i % 2}},
                "terminated": False,
                "truncated": False,
                "info": {},
            }
            for i in range(30)
        ],
        "total_reward": 0.0,
        "breakdown": {},
        "step_rewards": [],
        "step_breakdowns": [],
        "locked": False,
        "metadata": {},
    }

    class _Trained:
        episodes = [episode_dict]

    server._refresh_drift_baseline("conv_y", _Trained(), _blueprint_stub())
    saved = tmp_path / "ws" / "drift_baseline.json"
    assert saved.exists(), "a trained session must get a live baseline"
    assert json.loads(saved.read_text())["n_samples"] == 30


def _blueprint_stub():
    from rl_ide.architect.models import AgentBlueprintSpec

    return AgentBlueprintSpec(
        task_title="Router",
        domain_summary="Route work",
        data_source_type="pure_simulation",
        observation_features=["amount", "risk"],
        action_labels=["A", "B"],
        arena_visual_description="board",
        interactive_mode_description="click",
        success_metric_name="accuracy",
        target_metric_threshold=80.0,
        max_auto_tune_rounds=1,
        initial_reward_components=[],
    )


def test_a_baseline_without_policy_probabilities_monitors_inputs_only():
    """Stored episodes keep the chosen action, not the policy's probabilities.

    Comparing live entropy against a fabricated 0.25 baseline would invent drift, so the policy
    signals are switched off and only the input distribution is monitored.
    """
    from rl_ide.drift import fit_baseline_from_episodes

    episodes = [
        {
            "episode_id": 0,
            "steps": [
                {"obs": [float(x), float(-x)], "action": i % 2, "reward": 0.0}
                for i, x in enumerate([0.1 * k for k in range(100)])
            ],
        }
    ]
    baseline = fit_baseline_from_episodes(episodes, ["amount", "risk"])
    assert baseline is not None
    assert baseline.policy_signals is False

    # Same distribution the baseline was fitted on, in the stored `obs` order.
    in_range = [[0.1 * k, -0.1 * k] for k in range(100)]
    report = DriftDetector(baseline).check_drift(
        in_range, [[0.9, 0.1]] * 100, max_bins=5, psi_threshold=0.5
    )
    assert report.signals == ["psi"], report.signals
    assert report.entropy_zscore == 0.0
    assert report.action_shift_zscore == 0.0
    assert report.is_drifted is False


def test_input_only_calibration_is_quiet_and_still_detects_drift():
    """The production configuration (baseline from stored episodes) measured over 50 sessions."""
    from rl_ide.drift import fit_baseline_from_episodes

    source = _baseline(n=200)
    healthy_obs, _ = _in_distribution(source, n=200, seed=1)
    episodes = [
        {
            "episode_id": 0,
            "steps": [{"obs": o, "action": 0, "reward": 0.0} for o in healthy_obs],
        }
    ]
    baseline = fit_baseline_from_episodes(episodes, ["amount", "risk"])
    assert baseline is not None and baseline.policy_signals is False

    flagged = 0
    for seed in range(50):
        obs, _ = _in_distribution(source, n=200, seed=700 + seed)
        monitor = DriftMonitor(baseline)
        for o in obs:
            monitor.observe(o, [1 / 3, 1 / 3, 1 / 3], 3)
        flagged += 1 if monitor.status == DRIFTED else 0
    assert flagged == 0, f"{flagged}/50 unchanged sessions were flagged"

    shifted, _ = _out_of_distribution(source, n=200)
    monitor = DriftMonitor(baseline)
    first = None
    for i, o in enumerate(shifted, 1):
        monitor.observe(o, [1 / 3, 1 / 3, 1 / 3], 3)
        if monitor.status == DRIFTED and first is None:
            first = i
    assert first is not None and first <= 100, first


def test_observation_is_captured_even_when_the_policy_is_never_called():
    """A human click supplies the action, so `interactive_step` may not call `predict` at all."""
    from rl_ide.drift.live import capture_observations

    class _Env:
        def reset(self, **kwargs):
            return [0.25, 0.75], {}

        def step(self, action):
            return [float(action), 0.5], 0.0, False, False, {}

        def interactive_step(self, payload, predict, mode="agent_vs_env"):
            # Deliberately never calls `predict` — exactly like a human-driven step.
            obs, *_ = self.step(int(payload.get("action", 0)))
            return {"arena_state": {"step": 1}, "action_probs": [0.34, 0.33, 0.33]}

    env = _Env()
    captured = capture_observations(env)
    env.interactive_step({"action": 2}, lambda o: (0, [0.5, 0.5]))
    assert captured == [2.0, 0.5], captured


def test_the_last_verdict_stands_between_judgements():
    """Status must not revert to "warming_up" on every step between windows."""
    monitor = DriftMonitor(_baseline(), window=10, stride=5, confirmations=1, psi_threshold=0.25)
    obs, probs = _in_distribution(_baseline(), n=30, seed=2)
    statuses = []
    for o, p in zip(obs, probs):
        monitor.observe(o, p)
        statuses.append(monitor.status)
    judged = statuses[9]      # first window filled
    assert judged in (HEALTHY, DRIFTED)
    for status in statuses[10:14]:
        assert status == judged, "the verdict was replaced by warming_up between windows"
