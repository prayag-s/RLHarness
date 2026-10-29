# Specification for Antigravity: Agentic Chat Planner & Multi-Layer Code Executor

> **DIRECTIVE FOR ANTIGRAVITY:**
> 1. Implement the **Code-Aware Agentic Chat Planner & Multi-Layer Execution Engine** specified below, replacing the rigid enum-based intent switchboard (`UPDATE_ENV_MECHANICS` vs. `UPDATE_AGENT_CHARACTERISTICS`).
> 2. Write and execute the automated verification suite (`pytest`) covering all new capabilities and guardrails.
> 3. **MANDATORY REPORTING:** Output the complete, verbatim `pytest` test report in your response so it can be reviewed, and update `progress.md` to record the completion of this milestone, test counts, and architectural changes.

---

## 1. Problem Statement & Architectural Shift

### Current Bottleneck
Currently, when a user sends a chat prompt like *"If agent is first, randomize its first move among good options"*:
1. `_classify_with_llm` forces the prompt into coarse, static buckets (`UPDATE_AGENT_CHARACTERISTICS` or `UPDATE_ENV_MECHANICS`).
2. `UPDATE_AGENT_CHARACTERISTICS` only rewrites the scalar reward function (`_rewrite_reward_function()`), which does **not** alter deterministic `np.argmax(probs)` action selection during Arena play.
3. `UPDATE_ENV_MECHANICS` regenerates `env_module.py` focusing on Gym simulation rules (`reset()`/`step()`), leaving the hardcoded greedy `action_idx = int(action_idx)` inside `interactive_step()` untouched.
4. Furthermore, if a user asks for a capability that is genuinely not built into the platform, the classifier still tries to force it into one of the handlers rather than honestly stating that it does not know how to do that yet.

### Target Behavior (Claude Code / Codex Style Agentic Execution)
The Studio Chat must operate as a **Code-Aware Agentic Planner**:
- It inspects the actual live methods of `env_module.py` (`reset`, `step`, `interactive_step`, `compute_reward`) and available training/adapter hooks.
- It determines **which exact architectural layer(s)** must be modified (Arena Inference, Environment Mechanics, Reward Shaping, Training Hyperparameters, or a combination).
- It applies surgical code patches to the exact function(s) needed (e.g., patching `interactive_step()` to sample stochastically among top opening candidates on turn 1).
- **Honesty Guardrail:** If the user requests something outside the platform's supported capabilities, the planner must set `supported=False`, make **zero** code mutations, and reply clearly: *"I do not know how to do that yet because [reason]."*

---

## 2. Technical Specification

### 2.1 Structured Planner Schema (`ChatActionPlan`)

Replace the rigid single-enum classification in `_classify_with_llm` with a code-aware planner (`_plan_chat_action`) that returns a structured `ChatActionPlan`:

```python
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

class ExecutionTarget(str, Enum):
    REWARD_SHAPING = "reward_shaping"                 # Modifies RewardDSL / compute_reward (Training incentive)
    INFERENCE_STRATEGY = "inference_strategy"         # Modifies action selection in interactive_step() / policy wrapper (Arena/Eval behavior)
    ENV_SIM_MECHANICS = "env_sim_mechanics"           # Modifies reset() / step() / opponent logic in env_module.py (Training + Arena physics/rules)
    TRAINING_HYPERPARAMS = "training_hyperparams"     # Modifies entropy coef, training burst length, or stochastic exploration
    UNSUPPORTED_CAPABILITY = "unsupported_capability" # Capability not built or impossible in current runtime

class ChatActionPlan(BaseModel):
    supported: bool = Field(
        description="True if the platform can execute this accurately; False if the capability is not built."
    )
    unsupported_reason: Optional[str] = Field(
        default=None,
        description="If supported=False, clear explanation of what capability is missing."
    )
    scope_classification: str = Field(
        description="'arena_only' | 'training_only' | 'both' | 'unsupported'"
    )
    targets: List[ExecutionTarget] = Field(
        description="One or more execution layers required to fulfill the user prompt completely."
    )
    requires_retrain: bool = Field(
        description="True if reward/env/training changes require a PPO training burst; False if inference-only."
    )
    surgical_plan: str = Field(
        description="Exact functions/lines in env_module.py, adapter, or trainer to modify."
    )
    user_facing_summary: str = Field(
        description="Clear, concise response explaining the exact changes applied (or why it is unsupported)."
    )
```

---

### 2.2 Context-Aware Planning Prompt (`_plan_chat_action`)

When a user submits a chat message, pass the following context to the LLM planner:
1. The user's prompt.
2. The current source code (or method bodies) of `env_module.py` (`reset`, `step`, `interactive_step`, `compute_reward`).
3. The list of supported runtime capabilities:
   - Modifying `interactive_step()` action selection / inference rules (e.g., temperature sampling, top-$k$ filtering, turn-1 opening randomization, masking, fallback heuristics).
   - Modifying `reset()` and `step()` environment state transitions, board/physics rules, or sparring opponent strategies.
   - Modifying `compute_reward()` / `RewardDSL` rules and weights.
   - Triggering training bursts or adjusting exploration/entropy hyperparameters.

**Strict System Rule for Planner:**
> *"Inspect the live code before planning. If the user asks to change how the agent selects moves when playing in the Arena (such as randomizing its opening move among good options), route to `INFERENCE_STRATEGY` so `interactive_step()` is patched directly—do NOT route solely to `REWARD_SHAPING`. If the user wants this behavior in both self-play training and live play, include both `INFERENCE_STRATEGY` and `ENV_SIM_MECHANICS`. If the user asks for a feature that does not exist in our runtime (e.g., multiplayer networking, external API webhooks, 3D shader rendering, audio synthesis), set `supported=False` and `targets=[UNSUPPORTED_CAPABILITY]`."*

---

### 2.3 Surgical Execution Handlers

#### Handler A: `INFERENCE_STRATEGY` (`_patch_interactive_inference`)
- **Purpose:** Modifies how the agent selects actions during `interactive_step()` (and any inference helper in `env_module.py`) without wiping out the trained policy or resetting environment rules.
- **Example Execution for *"If agent is first, randomize its first move among good options"*:**
  In `env_module.py` $\rightarrow$ `interactive_step()`, replace the rigid deterministic `argmax` block:
  ```python
  # BEFORE
  action_idx, probs = policy_predict_fn(agent_obs)
  action_idx = int(action_idx)
  ```
  with turn-aware stochastic candidate selection:
  ```python
  # AFTER
  action_idx, probs = policy_predict_fn(agent_obs)
  is_first_move = (np.count_nonzero(self.board) == 0)

  if is_first_move and probs is not None:
      legal_indices = [i for i in range(len(probs)) if self.board[i] == 0]
      candidate_probs = np.array([probs[i] for i in legal_indices], dtype=np.float64)
      max_p = np.max(candidate_probs) if len(candidate_probs) > 0 else 0.0
      # Select moves within 70% of the highest probability move
      good_moves = [legal_indices[i] for i, p in enumerate(candidate_probs) if p >= 0.7 * max_p]
      # Fallback: if policy is overly peaked on a single cell (e.g., center 4),
      # include strong strategic openings (center + corners)
      if len(good_moves) <= 1:
          strategic_openings = [i for i in [0, 2, 4, 6, 8] if i in legal_indices]
          if len(strategic_openings) > 1:
              good_moves = strategic_openings
      action_idx = int(np.random.choice(good_moves))
  else:
      action_idx = int(action_idx)
  ```
- **Validation & Hot-Reload:**
  1. Parse patched `env_module.py` via `ast.parse()` to verify syntax safety.
  2. Execute a dry-run smoke test calling `env.reset()` and `env.interactive_step(...)` with a mock `policy_predict_fn` to ensure no runtime exceptions occur.
  3. Hot-reload the patched module in memory so the Arena immediately reflects the new behavior without requiring a server restart.

#### Handler B: Multi-Target Chaining (`ENV_SIM_MECHANICS` + `REWARD_SHAPING` + `INFERENCE_STRATEGY`)
- Allow the execution loop to iterate through `plan.targets` in order:
  1. `ENV_SIM_MECHANICS`: Patch `reset()` / `step()` in `env_module.py`.
  2. `REWARD_SHAPING`: Patch `compute_reward()` / `RewardDSL`.
  3. `INFERENCE_STRATEGY`: Patch `interactive_step()` in `env_module.py`.
  4. `TRAINING_HYPERPARAMS` / `requires_retrain`: Run the PPO training burst only when `plan.requires_retrain` is `True`.

#### Handler C: `UNSUPPORTED_CAPABILITY` Guardrail
- When `plan.supported is False` or `ExecutionTarget.UNSUPPORTED_CAPABILITY in plan.targets`:
  - Perform **zero** code edits and **zero** training bursts.
  - Return a response with status `"unsupported"` and a transparent message:
    `"I do not know how to do that yet: {plan.unsupported_reason}"`

---

## 3. Mandatory Verification & Testing Suite

Create a dedicated test file (e.g., `backend/tests/test_agentic_chat_planner.py`) and run the entire test suite via `pytest -v`.

### Required Test Cases:
1. **`test_first_move_randomization_inference_patch`**:
   - Send the exact prompt: *"if agent is first, then randomize its first move among good options"*.
   - Assert that the planner sets `supported=True` and includes `ExecutionTarget.INFERENCE_STRATEGY` in `plan.targets`.
   - Assert that `interactive_step()` in `env_module.py` is updated.
   - Run 30 consecutive simulated opening turns where the agent moves first (using a peaked mock policy that prefers cell `4`) and assert that **at least 2 distinct opening cells** from `{0, 2, 4, 6, 8}` are selected (proving `argmax` determinism on turn 1 was replaced by good-move randomization).
2. **`test_multi_layer_arena_and_training_patch`**:
   - Send a compound prompt requesting both a reward/mechanic change and an inference change (e.g., *"Reward corner openings during training and also randomize the agent's first move in the arena"*).
   - Assert that both `REWARD_SHAPING` and `INFERENCE_STRATEGY` are executed and verified.
3. **`test_unsupported_capability_honest_refusal`**:
   - Send an out-of-scope prompt (e.g., *"Connect this agent to a live Discord voice channel and stream 3D WebGL animations"*).
   - Assert that `plan.supported is False`.
   - Assert that `env_module.py` and the reward function remain **100% unchanged** (byte-for-byte identical before and after the call).
   - Assert that the chat reply explicitly states it does not know how to / cannot do that yet.
4. **Full Regression Suite**:
   - Run `pytest` across all existing test files to confirm 0 regressions.

---

## 4. Deliverables & `progress.md` Update Instructions

Upon completing the implementation and running the tests, you **MUST** provide:
1. **Verbatim Test Report:** Paste the full console output of `pytest -v` showing all existing tests plus the new `test_agentic_chat_planner.py` tests passing.
2. **Diff / Summary of Patched Code:** Show the updated planner and execution loop code.
3. **`progress.md` Update:** Update `progress.md` with a new section documenting:
   - The transition from static intent classification to the **Code-Aware Agentic Chat Planner (`ChatActionPlan`)**.
   - Support for `INFERENCE_STRATEGY` surgical patching in `interactive_step()`.
   - The `UNSUPPORTED_CAPABILITY` honesty guardrail.
   - Updated total passing test count and test execution summary.