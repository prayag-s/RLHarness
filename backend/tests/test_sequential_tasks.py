"""Sequential tasks (G10).

Every dataset environment used to be a contextual bandit: twenty-five independent rows with an
immediate reward. Renewal plays, claim journeys and multi-touch cadences are one case unfolding over
several decisions, where the outcome is only known at the end — modelling them as independent guesses
loses the thing that makes them hard.

These tests pin the difference: the same case requires *different* actions at different stages, the
reward is delayed, success is terminal-only, and the one-shot path is untouched.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rl_ide.architect.intake import apply_task_structuring, infer_task_structuring
from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.ingest import load_bytes
from rl_ide.synthesizer.env_builder import (
    JITEnvironmentSynthesizer,
    instantiate_env,
    load_synthesized_env_class,
)

ACTIONS = ["Email", "Call", "Discount"]


def _dataset():
    labels = ["email", "call", "discount"]
    rows = "\n".join(f"{i * 7},{i % 5},{(i * 3) % 7},{labels[i % 3]}" for i in range(40))
    return load_bytes(
        filename="cases.csv", content=("amount,tenure,score,label\n" + rows).encode()
    )


def _blueprint(**overrides) -> AgentBlueprintSpec:
    fields = dict(
        task_title="Renewal play",
        domain_summary="Run a multi-step renewal journey for at-risk accounts.",
        data_source_type="uploaded_dataset",
        observation_features=["amount", "tenure", "score"],
        action_labels=ACTIONS,
        arena_visual_description="a journey board",
        interactive_mode_description="advance one stage",
        success_metric_name="Journey Completion Rate (%)",
        target_metric_threshold=70.0,
        max_auto_tune_rounds=1,
        initial_reward_components=[],
        task_structuring="sequential",
        episode_length=4,
    )
    fields.update(overrides)
    return AgentBlueprintSpec(**fields)


def _build_env(blueprint: AgentBlueprintSpec, dataset=None):
    dataset = dataset or _dataset()
    code, features = JITEnvironmentSynthesizer(llm_client=None)._build_sequential_dataset_code(
        blueprint, dataset
    )
    directory = Path(tempfile.mkdtemp())
    module = directory / "env_module.py"
    module.write_text(code, encoding="utf-8")
    env = instantiate_env(
        load_synthesized_env_class(module),
        dataset_records=dataset.df.to_dict(orient="records"),
    )
    return env, code, features


# --------------------------------------------------------------------------- intake
@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("Build an agent to run a renewal play over the quarter for at-risk accounts", ("sequential", 5)),
        ("Run a 6-step onboarding flow for new customers", ("sequential", 6)),
        ("Design a 3 stage claim journey", ("sequential", 3)),
        ("Set up a multi-touch nurture cadence", ("sequential", 5)),
        ("Build an agent to triage support tickets", ("one_shot", 1)),
        ("Classify each invoice by risk", ("one_shot", 1)),
    ],
)
def test_intake_decides_whether_a_request_is_a_journey(prompt, expected):
    assert infer_task_structuring(prompt) == expected


def test_an_explicit_single_decision_wins_over_a_stray_mention_of_steps():
    """"Classify each ticket in 2 steps" is still a classifier, and must stay one."""
    assert infer_task_structuring("Classify each ticket, following the 3 step checklist") == ("one_shot", 1)


def test_structuring_survives_the_blueprint_conversions():
    blueprint = _blueprint()
    restored = blueprint.to_dynamic_task_blueprint().to_agent_blueprint_spec()
    assert restored.task_structuring == "sequential"
    assert restored.episode_length == 4


# ----------------------------------------------------------------------------- env
def test_one_episode_is_one_case_not_four_independent_decisions():
    env, _code, _features = _build_env(_blueprint())
    assert env.episode_horizon == 4, "the horizon is the journey length"
    assert env.task_class == "sequential"

    _obs, info = env.reset(seed=11)
    record = env.records[env.current_idx]
    # The same case stays put for the whole episode; only the stage advances.
    assert info["record_index"] == env.current_idx
    assert info["episode_length"] == 4

    plan = env._plan_for(record)
    assert len(plan) == 4
    assert len(set(plan)) > 1, "a journey whose stages all needed the same action is not sequential"


def test_the_state_carries_across_steps():
    env, _code, _features = _build_env(_blueprint())
    env.reset(seed=5)
    first = env._extract_obs(env.records[env.current_idx])
    plan = env._plan_for(env.records[env.current_idx])
    env.step(plan[0])
    second = env._extract_obs(env.records[env.current_idx])

    assert env.obs_dim == env.base_obs_dim + 3
    # Stage progress, accuracy-so-far and the previous action all ride in the observation.
    assert not (first == second).all()
    assert second[-3] > first[-3]


def test_the_reward_is_delayed_and_success_is_terminal_only():
    env, _code, _features = _build_env(_blueprint())
    env.reset(seed=3)
    plan = env._plan_for(env.records[env.current_idx])

    rewards = []
    for stage in range(4):
        _obs, reward, terminated, _truncated, info = env.step(plan[stage])
        rewards.append(reward)
        if stage < 3:
            assert info["is_success"] is False, "a journey is not successful halfway through"
            assert terminated is False

    assert rewards[:3] == [0.0, 0.0, 0.0], "an intermediate decision pays nothing"
    assert rewards[3] == 1.0
    assert info["is_success"] is True


def test_one_wrong_stage_loses_the_whole_journey():
    env, _code, _features = _build_env(_blueprint())
    env.reset(seed=3)
    plan = env._plan_for(env.records[env.current_idx])

    env.step((plan[0] + 1) % len(ACTIONS))
    for stage in range(1, 4):
        _obs, reward, terminated, _truncated, info = env.step(plan[stage])

    assert terminated is True
    assert info["is_success"] is False
    assert reward == -1.0


def test_demonstrations_cover_every_stage_of_the_plan():
    env, _code, _features = _build_env(_blueprint())
    features, targets = env.get_demonstrations(num_samples=200)

    assert features.shape[0] > 0
    assert features.shape[1] == env.obs_dim
    assert set(int(t) for t in targets).issubset(set(range(len(ACTIONS))))
    assert len(set(int(t) for t in targets)) > 1, "warm-start needs more than one action to imitate"


def test_episode_length_is_clamped_to_something_that_terminates():
    env, _code, _features = _build_env(_blueprint(episode_length=500))
    assert env.episode_horizon <= 12


def test_interactive_play_advances_the_journey_and_starts_a_new_case():
    env, _code, _features = _build_env(_blueprint(episode_length=2))
    env.reset(seed=9)
    out = env.interactive_step({}, lambda obs: (0, [0.5, 0.3, 0.2]))
    assert "Stage 1/2" in out["status_text"]
    env.interactive_step({}, lambda obs: (0, [0.5, 0.3, 0.2]))
    fresh = env.interactive_step({}, lambda obs: (0, [0.5, 0.3, 0.2]))
    assert "Stage 1/2" in fresh["status_text"], "a finished journey starts the next case"


def test_the_one_shot_template_is_unchanged():
    """Regression: a classifier must not silently become a journey."""
    dataset = _dataset()
    blueprint = _blueprint(task_structuring="one_shot", episode_length=1)
    code, _features = JITEnvironmentSynthesizer(llm_client=None)._build_dataset_code(blueprint, dataset)
    assert "task_class = \"classification\"" in code
    assert "episode_horizon = 25" in code


def test_applying_structuring_sets_the_metric_name():
    blueprint = _blueprint(task_structuring="one_shot", episode_length=1, success_metric_name="Accuracy (%)")
    updated = apply_task_structuring(blueprint, "run a renewal play over the quarter")
    assert updated.task_structuring == "sequential"
    assert "Journey" in updated.success_metric_name
