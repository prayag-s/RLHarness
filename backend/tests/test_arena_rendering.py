"""Regression tests for the generic arena renderer.

The arena draws any array-valued state field as a cell grid. That is right for boards and queue
loads, but wrong for arrays of labels: a support-routing arena rendered its four queues as
4-character fragments ("Tier", "Seni", "Rete", "Engi") in 40px squares.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.synthesizer.arena_builder import JITArenaSynthesizer

NODE = shutil.which("node") or "/home/devuser/.nvm/versions/node/v20.20.2/bin/node"

ACTION_LABELS = ["Tier 1 Support", "Senior Support", "Retention Specialist", "Engineering Escalation"]


def _blueprint() -> AgentBlueprintSpec:
    return AgentBlueprintSpec(
        task_title="Ticket Router",
        domain_summary="Route tickets",
        data_source_type="simulated",
        observation_features=["a", "b"],
        action_labels=list(ACTION_LABELS),
        arena_visual_description="dispatch desk",
        interactive_mode_description="route a ticket",
        success_metric_name="Routing Accuracy (%)",
    )


def _arena_html(sample_state: dict) -> str:
    return JITArenaSynthesizer()._generate_deterministic_arena_html(_blueprint(), sample_state)


def _render_with_node(html: str, state: dict) -> dict:
    script = re.search(r"<script>([\s\S]*)</script>", html)
    assert script, "arena must embed its script"
    harness = f"""
const stub = () => ({{
  innerHTML: '', innerText: '', textContent: '', style: {{}}, value: '',
  setAttribute() {{}}, getAttribute() {{ return null; }}, appendChild() {{}}, addEventListener() {{}},
  classList: {{ add() {{}}, remove() {{}}, toggle() {{}} }},
}});
const nodes = {{}};
global.document = {{
  getElementById: (id) => (nodes[id] = nodes[id] || stub()),
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener: () => {{}},
  body: stub(),
}};
global.window = {{ addEventListener: () => {{}}, parent: {{ postMessage: () => {{}} }}, matchMedia: () => ({{ matches: false }}) }};
global.setInterval = () => 0;
global.clearInterval = () => {{}};

{script.group(1)}

renderDomainState({json.dumps(state)});
console.log(JSON.stringify({{ grid: nodes['dynamic-state-grid'] ? nodes['dynamic-state-grid'].innerHTML : '' }}));
"""
    result = subprocess.run([NODE, "-e", harness], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not NODE or not shutil.which(NODE), reason="node is required to evaluate arena scripts")
def test_text_array_state_renders_readable_labels_not_fragments():
    html = _arena_html({"action_labels": list(ACTION_LABELS), "step_count": 0})
    rendered = _render_with_node(html, {"action_labels": list(ACTION_LABELS), "step_count": 0})
    assert "Retention Specialist" in rendered["grid"]
    assert "Rete<" not in rendered["grid"]


@pytest.mark.skipif(not NODE or not shutil.which(NODE), reason="node is required to evaluate arena scripts")
def test_numeric_array_state_still_renders_clickable_cells():
    html = _arena_html({"queue_load": [0.1, 0.5, 0.9, 0.2], "step_count": 0})
    rendered = _render_with_node(html, {"queue_load": [0.1, 0.5, 0.9, 0.2], "step_count": 0})
    assert "arena-cell" in rendered["grid"]
    assert "sendInteractiveAction" in rendered["grid"]


BOARD_LABELS = [
    "Place at Top-Left", "Place at Top-Center", "Place at Top-Right",
    "Place at Middle-Left", "Place at Center", "Place at Middle-Right",
    "Place at Bottom-Left", "Place at Bottom-Center", "Place at Bottom-Right",
]


def _board_blueprint() -> AgentBlueprintSpec:
    return AgentBlueprintSpec(
        task_title="Tic-Tac-Toe Tactician",
        domain_summary="Play tic-tac-toe",
        data_source_type="pure_simulation",
        observation_features=["board"],
        action_labels=list(BOARD_LABELS),
        arena_visual_description="A glowing 3x3 grid board",
        interactive_mode_description="play a move",
        success_metric_name="Task Success Rate (%)",
        state_dict_schema={
            "board": "list[int] of length 9, each in {-1, 0, 1} where 1=agent, -1=human, 0=empty",
            "agent_mark": "string: 'X' or 'O'",
            "move_count": "integer: number of moves made so far (0-9)",
        },
    )


@pytest.mark.skipif(not NODE or not shutil.which(NODE), reason="node is required to evaluate arena scripts")
def test_declared_board_length_draws_a_board_before_the_environment_exists():
    """An empty placeholder list draws no board at all, which is what the preview showed."""
    from rl_ide.architect.omni_agent import StudioOmniAgent

    blueprint = _board_blueprint()
    state = StudioOmniAgent.__new__(StudioOmniAgent)._neutral_sample_state(blueprint)
    assert state["board"] == [0] * 9

    arena = JITArenaSynthesizer()._generate_deterministic_arena_html(blueprint, state)
    rendered = _render_with_node(arena, state)
    assert rendered["grid"].count("arena-cell") == 9
    assert "sendInteractiveAction" in rendered["grid"]


@pytest.mark.parametrize(
    "described,expected",
    [
        ("list[int] of length 9, each in {-1, 0, 1}", 9),
        ("list of 9 integers: 1 for agent mark, 0 for empty", 9),
        ("array of 42 floats", 42),
        ("list of 3 integers or null: winning line", 3),
        ("list of 2D rows", 0),
        ("list of strings, e.g. 'X' or 'O'", 0),
    ],
)
def test_declared_collection_length_is_read_as_a_board_shape(described, expected):
    """The length is written many ways; only a sized numeric array is a board."""
    from rl_ide.architect.omni_agent import _declared_cells

    assert len(_declared_cells(described)) == expected
