# Specification for Antigravity: Zero-Hardcoding Universal Generative Pipeline & Agentic Chat Planner

> **CRITICAL ARCHITECTURAL DIRECTIVE FOR ANTIGRAVITY (READ FIRST):**
>
> **STOP HARDCODING DOMAIN TEMPLATES AND KEYWORD LISTS.**
> In the previous patch, when the user asked for Blackjack, you added `is_blackjack_intent()`, `_build_blackjack_code()`, a hardcoded Blackjack HTML template in `_generate_deterministic_arena_html()`, and hardcoded keywords (`"table"`, `"card"`, `"felt"`, `"chips"`) in `omni_agent.py`.
>
> **This is strictly the wrong architecture.** If the user asks for *Connect 4, Poker, Flappy Bird, Warehouse Forklift Routing, Hospital Triage, or Options Market Making* next, a keyword/template system will break and fall back to the 2D Rover again.
>
> **We want ZERO domain hardcoding and ZERO static keyword routing.** Every stage of the app—Intake (`intake.py`), Environment Code Synthesis (`env_builder.py`), Arena UI Synthesis (`arena_builder.py`), and Chat Modification (`omni_agent.py`)—must be **100% LLM-driven, generative, and domain-agnostic**, with self-healing sandbox verification.

---

## 1. The 4 Anti-Patterns to Eliminate Immediately

1. **Eliminate Domain Keyword Classifiers in `intake.py`:**
   - **Ban:** Functions like `is_blackjack_intent()`, `is_tictactoe_intent()`, `is_solar_intent()`, and fallback defaults that silently dump unrecognized prompts into a "2D Waypoint Rover" blueprint.
   - **Replace With:** A single **Universal LLM Domain Architect** (`synthesize_blueprint_from_prompt(prompt,chat_history)`) that dynamically designs the `TaskBlueprint` (observation schema, discrete/continuous actions, state variables, reward structure, UI visual paradigm, and domain-specific clarifying questions) for **any** prompt on Earth.

2. **Eliminate `_build_<domain>_code()` Templates in `env_builder.py`:**
   - **Ban:** Hardcoded environment generators like `_build_blackjack_code()` or falling back to a rover `SynthesizedGymEnv`.
   - **Replace With:** A single **Universal LLM Gymnasium Code Synthesizer** with a **3-Attempt Self-Healing Sandbox Loop**. The LLM writes the complete Python `SynthesizedGymEnv` (`__init__`, `reset`, `step`, `compute_reward`, `interactive_step`, `render_state`, `generate_demonstrations`) tailored to the `TaskBlueprint`. It runs `SandboxSmokeTester.run_smoke_test()`. If any error occurs, the traceback is fed back to the LLM to fix automatically.

3. **Eliminate Static HTML Fallbacks in `arena_builder.py`:**
   - **Ban:** `_generate_deterministic_arena_html()` branching on `if blackjack:` / `elif tictactoe:` / `else: rover_canvas`.
   - **Replace With:** **True JIT Generative UI Synthesis**. The LLM generates a bespoke, interactive HTML5/SVG/Canvas/Tailwind Arena component bound to the exact `render_state()` keys and `action_labels` of `SynthesizedGymEnv`. Every action button and interactive element in the generated HTML dynamically invokes `interactive_step(action_idx)`.

4. **Eliminate Keyword Lists in `omni_agent.py`:**
   - **Ban:** Checking `if word in ["table", "card", "felt", "randomize", ...]` or rigid enums (`UPDATE_ENV_MECHANICS` vs `UPDATE_AGENT_CHARACTERISTICS`).
   - **Replace With:** A **Claude Code / Codex-style Agentic Code Planner** (`_plan_chat_action`) that inspects the live session's current `TaskBlueprint`, `env_module.py` source code, and `arena.html`, and decides which exact files/functions to patch or re-synthesize.

---

## 2. Detailed Technical Specification by Module

### 2.1 Universal LLM Intake Architect (`backend/rl_ide/architect/intake.py`)

Remove all `is_<domain>_intent()` and `is_<domain>_spec()` switches. Every new prompt or domain pivot must pass through an LLM structured-output call returning a universal `DynamicTaskBlueprint`:

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ObservationFeatureSpec(BaseModel):
    name: str = Field(description="snake_case feature name exposed in obs vector and render_state")
    description: str
    low: float
    high: float

class ActionSpec(BaseModel):
    index: int
    label: str = Field(description="Human-readable action button label for the Arena UI")
    mechanic_description: str = Field(description="Exact effect of this action in step() and interactive_step()")

class ClarifyingQuestion(BaseModel):
    pillar: str
    question: str
    suggested_options: List[str]

class DynamicTaskBlueprint(BaseModel):
    domain_title: str
    domain_summary: str
    visual_metaphor: str = Field(
        description="Detailed description of what the bespoke Arena UI should look like (e.g., 'Connect 4 7x6 vertical drop grid', 'Elevator shaft building cross-section', 'Casino table', 'Orderbook depth ladder')"
    )
    observation_features: List[ObservationFeatureSpec]
    actions: List[ActionSpec]
    state_dict_schema: Dict[str, str] = Field(
        description="Exact JSON-serializable keys emitted by get_render_state() so env_builder and arena_builder share a strict contract"
    )
    reward_components: Dict[str, str]
    clarifying_questions: List[ClarifyingQuestion]
```

* **Zero Fallback to Rover:** The 2D Rover must **never** be used unless the user explicitly asks for a rover/navigation task.
* **Shared State Contract (`state_dict_schema`):** The `state_dict_schema` defined in the blueprint becomes the single source of truth shared between `env_builder.py` (which populates `info["render_state"]`) and `arena_builder.py` (which renders `state` in the browser).

---

### 2.2 Self-Healing Universal Environment Synthesizer (`backend/rl_ide/synthesizer/env_builder.py`)

Instead of calling hardcoded `_build_<domain>_code()` methods, implement `synthesize_universal_env(blueprint: DynamicTaskBlueprint, user_prompt: str) -> str`:

1. **Prompt the LLM Code Generator** with the `DynamicTaskBlueprint` and strict interface contract:
   - Class name `SynthesizedGymEnv(gym.Env)`
   - `observation_space = spaces.Box(...)` matching `len(blueprint.observation_features)`
   - `action_space = spaces.Discrete(len(blueprint.actions))`
   - `reset(self, seed=None, options=None)`
   - `step(self, action: int)`
   - `compute_reward(self, ...)` returning scalar float and breakdown dict
   - `get_render_state(self) -> dict` returning JSON-serializable keys matching `blueprint.state_dict_schema`
   - `interactive_step(self, user_action, policy_predict_fn=None, mode="agent_vs_env")`
   - `generate_demonstrations(self, num_episodes=20)` using a domain-appropriate heuristic teacher for Behavioral Cloning warm-start.
2. **3-Iteration Self-Healing Verification Loop:**
   - Run `SandboxSmokeTester.run_smoke_test(code)`.
   - Also verify that `get_render_state()` contains all keys promised in `blueprint.state_dict_schema`.
   - If syntax, runtime, shape, or JSON-serialization errors occur, pass the exact Python traceback back to the LLM and re-synthesize (up to 3 attempts).

---

### 2.3 Universal Generative Arena Builder (`backend/rl_ide/synthesizer/arena_builder.py`)

Eliminate hardcoded domain HTML templates as the primary path. Implement `synthesize_universal_arena_html(blueprint: DynamicTaskBlueprint, sample_render_state: dict) -> str`:

1. **Pass the Actual Sample Output of `env.get_render_state()`** (captured during the smoke test!) and `blueprint.actions` to the LLM UI Generator.
2. **Generate a Self-Contained Interactive HTML/JS/CSS Arena** that:
   - Renders the domain's true visual metaphor (cards, grids, pipes, queues, charts, physics objects, etc.) directly from the keys in `sample_render_state`.
   - Renders interactive controls mapped directly to `blueprint.actions` (`0 .. N-1`) and calls the host bridge (`window.parent.postMessage` / `sendInteractiveAction(actionIdx)`).
   - Includes a live `renderState(state)` function that updates the DOM/Canvas/SVG whenever `interactive_step()` or live training telemetry pushes a new state frame.

---

### 2.4 Code-Aware Agentic Chat Planner (`backend/rl_ide/architect/omni_agent.py`)

Delete all static keyword lists (`["table", "card", ...]`, `["randomize", ...]`) and rigid binary routing. Implement a **Claude Code / Codex-style Agentic Planner (`_plan_chat_action`)**:

#### Planner Input Context (Passed on Every Chat Turn):
1. The user's raw message and conversation history.
2. The active `DynamicTaskBlueprint` JSON.
3. The live Python source code of `env_module.py` (including `reset`, `step`, `compute_reward`, `interactive_step`, `get_render_state`).
4. A structural summary of the current `arena.html`.

#### Structured Plan Output (`ChatActionPlan`):

```python
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

class ExecutionTarget(str, Enum):
    REANCHOR_DOMAIN = "reanchor_domain"               # User pivoted to a completely new task/domain or current blueprint is wrong
    ENV_SIM_MECHANICS = "env_sim_mechanics"           # Patch reset(), step(), or opponent logic in env_module.py
    REWARD_SHAPING = "reward_shaping"                 # Patch compute_reward() / RewardDSL weights and rules
    INFERENCE_STRATEGY = "inference_strategy"         # Patch action selection inside interactive_step() (e.g. randomizing opening move among good options)
    ARENA_VISUAL_UI = "arena_visual_ui"               # Re-synthesize or patch the HTML/JS visual arena
    TRAINING_HYPERPARAMS = "training_hyperparams"     # Adjust PPO burst, entropy, or exploration settings
    UNSUPPORTED_CAPABILITY = "unsupported_capability" # Honest refusal when runtime genuinely lacks the capability

class ChatActionPlan(BaseModel):
    supported: bool = Field(description="True if executable within the platform; False if capability is not built.")
    unsupported_reason: Optional[str] = Field(default=None)
    targets: List[ExecutionTarget] = Field(description="Ordered list of architectural layers to modify.")
    requires_retrain: bool = Field(description="True if reward/env changes require a PPO training burst.")
    surgical_instructions: str = Field(description="Exact code/UI modifications to perform across the selected targets.")
    user_facing_summary: str = Field(description="Direct, helpful explanation of what was changed (or why it is unsupported).")
```

#### How the Agentic Planner Handles Any Prompt Without Keywords:
* **Case 1: *"train a connect 4 agent"* or *"train a blackjack agent"* or *"train an elevator scheduler"***
  - `intake.py` synthesizes the exact `DynamicTaskBlueprint`, `env_builder.py` generates and smoke-tests the custom Python Gym environment, and `arena_builder.py` generates the matching visual UI. Zero hardcoded domain templates required.
* **Case 2: *"if agent is first, then randomize its first move among good options"***
  - Planner inspects `env_module.py`, sees `action_idx = int(action_idx)` inside `interactive_step()`, selects `targets=[ExecutionTarget.INFERENCE_STRATEGY]`, and patches `interactive_step()` to sample stochastically among top-scoring legal opening moves on turn 1.
* **Case 3: *"make the arena show a dealer chip stack and animate card flips"***
  - Planner selects `targets=[ExecutionTarget.ARENA_VISUAL_UI]` (and `ENV_SIM_MECHANICS` if a new state key like `chip_stack` is needed in `get_render_state()`) without needing `"chip"` in a hardcoded Python list.
* **Case 4: *"connect this agent to a live Bloomberg terminal via FIX protocol"***
  - Planner recognizes the runtime cannot connect to external FIX hardware, sets `supported=False`, `targets=[ExecutionTarget.UNSUPPORTED_CAPABILITY]`, mutates **zero** files, and replies honestly: *"I do not know how to do that yet because live external FIX protocol connections are not supported in the sandbox runtime."*

---

## 3. Mandatory Verification & Testing Suite (`pytest`)

Create `backend/tests/test_universal_agentic_pipeline.py` and run the full test suite via `pytest -v`.

### Required Automated Tests:

1. **`test_zero_hardcoding_novel_domain_synthesis`**:
   - Pass a **completely novel prompt that has NO hardcoded template in the codebase** (e.g., *"Train a Connect-4 grid agent that drops discs into 7 columns"* or *"Train a Russian Roulette / Dice Pig / Warehouse Elevator dispatcher agent"*).
   - Assert that `intake.py` does **not** fall back to the 2D Waypoint Rover (`"state_pos_x"` / `"Move Left"` must NOT be present).
   - Assert that `env_builder.py` synthesizes a valid `SynthesizedGymEnv` that passes `SandboxSmokeTester.run_smoke_test()`.
   - Assert that `arena_builder.py` generates custom HTML bound to the environment's `get_render_state()` keys (and NOT the generic `<canvas id="sim-canvas">` rover fallback).

2. **`test_first_move_randomization_inference_patch`**:
   - Send the prompt: *"if agent is first, then randomize its first move among good options"*.
   - Assert that `_plan_chat_action` selects `ExecutionTarget.INFERENCE_STRATEGY`.
   - Assert that `interactive_step()` in `env_module.py` is surgically patched so that across 30 opening turns with the agent going first, at least 2 distinct good opening moves are chosen instead of a single deterministic `argmax` move.

3. **`test_multi_target_code_and_ui_synchronization`**:
   - Send a prompt that requires updating both environment state and arena visuals or reward + inference.
   - Assert that all targeted layers are patched and pass smoke tests.

4. **`test_unsupported_capability_honest_refusal`**:
   - Send an impossible/unbuilt prompt (e.g., *"Stream live 3D VR audio to a Discord voice channel"*).
   - Assert `plan.supported is False`, `ExecutionTarget.UNSUPPORTED_CAPABILITY` is returned, `env_module.py` is byte-for-byte unchanged, and the assistant explicitly states it does not know how to do that yet.

5. **Codebase Audit Test (`test_no_keyword_routing_lists_in_omni_agent`)**:
   - Inspect `omni_agent.py` to verify that intent routing goes through `_plan_chat_action` rather than static domain keyword lists.

---

## 4. Deliverables & `progress.md` Update Instructions

Upon completing the implementation, Antigravity **MUST**:
1. **Paste the Verbatim `pytest -v` Output:** Include the full terminal output showing all existing tests + `test_universal_agentic_pipeline.py` passing 100%.
2. **Update `progress.md`:** Record the removal of hardcoded domain/keyword fallbacks, the implementation of the Universal Generative Pipeline (`intake` $\rightarrow$ `env_builder` $\rightarrow$ `arena_builder`) and the Agentic Chat Planner (`omni_agent`), along with the updated test count.