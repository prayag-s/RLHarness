"""Regression tests: the user's enumerated actions are a contract.

Observed live: a prompt naming four routing queues produced a generic two-action blueprint
("Approve / Safe", "Flag / Investigate") with `label_column="mentions_outage"`. Training then
reported 90 % on a task nobody asked for. Nothing errored — the degradation was silent.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rl_ide.architect.intake import (
    IntakeArchitectAgent,
    enforce_explicit_actions,
    extract_action_labels_from_prompt,
    synthesize_blueprint_from_prompt,
)
from rl_ide.architect.models import AgentBlueprintSpec, DynamicTaskBlueprint
from rl_ide.synthesizer.env_builder import JITEnvironmentSynthesizer
from rl_ide.synthesizer.feature_engineering import infer_label_map

SUPPORT_PROMPT = (
    "Train an agent that routes each incoming B2B support ticket to the correct queue: "
    "Tier 1 Support, Senior Support, Retention Specialist, or Engineering Escalation. "
    "Use the attached ticket history."
)
CLAIMS_PROMPT = (
    "Train an agent that routes each claim to Straight-Through Auto-Approve, Manual Adjuster Review, "
    "Special Investigations Unit, or Request Additional Documentation, using the attached claims history."
)


class _TwoActionLLM:
    """Stands in for an LLM that ignores the enumerated actions in the prompt."""

    def generate_json(self, *args, **kwargs):
        return {
            "domain_title": "Support Triage Agent",
            "domain_summary": "Classify tickets",
            "visual_metaphor": "a triage desk",
            "observation_features": [
                {"name": "a", "description": "a", "low": -1, "high": 1},
                {"name": "b", "description": "b", "low": -1, "high": 1},
            ],
            "actions": [
                {"index": 0, "label": "Approve / Safe", "mechanic_description": ""},
                {"index": 1, "label": "Flag / Investigate", "mechanic_description": ""},
            ],
            "state_dict_schema": {},
            "reward_components": {"goal_reward": "reward"},
        }

    def generate(self, *args, **kwargs):  # pragma: no cover - not used
        raise AssertionError("generate() should not be called")


def test_explicit_enumerations_are_extracted_without_quotes():
    assert extract_action_labels_from_prompt(SUPPORT_PROMPT) == [
        "Tier 1 Support",
        "Senior Support",
        "Retention Specialist",
        "Engineering Escalation",
    ]
    assert extract_action_labels_from_prompt(CLAIMS_PROMPT) == [
        "Straight-Through Auto-Approve",
        "Manual Adjuster Review",
        "Special Investigations Unit",
        "Request Additional Documentation",
    ]
    # Prose without an enumeration must not invent actions.
    assert extract_action_labels_from_prompt("Train a CartPole agent. Use a physics simulation.") is None


def test_llm_blueprint_is_overridden_by_the_users_action_list():
    blueprint = synthesize_blueprint_from_prompt(SUPPORT_PROMPT, client=_TwoActionLLM())
    assert [a.label for a in blueprint.actions] == ["Approve / Safe", "Flag / Investigate"]

    enforced = enforce_explicit_actions(blueprint, extract_action_labels_from_prompt(SUPPORT_PROMPT))
    assert [a.label for a in enforced.actions] == [
        "Tier 1 Support",
        "Senior Support",
        "Retention Specialist",
        "Engineering Escalation",
    ]
    assert [a.index for a in enforced.actions] == [0, 1, 2, 3]


def test_intake_keeps_all_declared_actions():
    agent = IntakeArchitectAgent(llm_client=_TwoActionLLM())
    response = agent.interview(user_prompt=SUPPORT_PROMPT, force_smart_defaults=True)
    assert response.blueprint_spec is not None
    assert len(response.blueprint_spec.action_labels) == 4


def test_declared_label_covering_fewer_actions_is_replaced_by_the_real_target(tmp_path):
    frame = pd.DataFrame(
        {
            "channel": ["email", "chat", "phone", "email"] * 30,
            "spend": [float(i) for i in range(120)],
            "mentions_outage": [i % 2 for i in range(120)],
            "resolution_queue": [
                "Tier 1 Support",
                "Senior Support",
                "Retention Specialist",
                "Engineering Escalation",
            ]
            * 30,
        }
    )
    action_labels = [
        "Route to Tier 1 Support",
        "Route to Senior Support",
        "Route to Retention Specialist",
        "Route to Engineering Escalation",
    ]
    blueprint = AgentBlueprintSpec(
        task_title="Ticket Router",
        domain_summary="Route tickets",
        data_source_type="uploaded_dataset",
        observation_features=["channel_email", "spend"],
        action_labels=list(action_labels),
        arena_visual_description="desk",
        interactive_mode_description="route",
        success_metric_name="Accuracy (%)",
        label_column="mentions_outage",  # what the LLM guessed: only 2 classes
    )

    chosen = JITEnvironmentSynthesizer._select_label_column(
        frame, blueprint.label_column, action_labels, len(action_labels)
    )
    assert chosen == "resolution_queue"
    covered = set(infer_label_map(list(frame[chosen].unique()), action_labels).values())
    assert covered == {0, 1, 2, 3}
