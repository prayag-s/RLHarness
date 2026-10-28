# RL-Native Agent IDE (v4.0) — Zero-Adapter JIT Synthesis, Conversational Architect & Autonomous Closed-Loop Training

**Paradigm Shift (v3.0 → v4.0):**
We are removing **all** pre-shipped domain adapters (`CSVStreamAdapter`, `TextClassifyAdapter`, `AutoDataAdapter`, `DynamicSimAdapter`), the rigid YAML `RewardDSL`, and all hardcoded UI renderers (`PolymorphicArena.tsx`).
In v4.0, the IDE works like **Claude Code / Codex / Lovable for RL Agents**:
1. **Nothing domain-specific is shipped.**
2. **Conversational Intake Architect:** The user describes what they want in natural language (and optionally drops in files/SQL). The IDE checks the **5 Pillars** (Objective, Data/Simulator, Visual Arena Preference, Success Target & Auto-Tune Rounds, Interactive Play Mode) and asks at most 2–3 clarifying questions with clickable suggestion pills (plus a 1-click *"Use Smart Defaults & Start"* button).
3. **Just-In-Time (JIT) Code Synthesis & Self-Healing:** Once expectations are locked, the IDE synthesizes:
   - `env_module.py`: A complete, custom `gymnasium.Env` + modular Python `compute_reward()` + `interactive_step()` + `get_demonstrations()` tailored to the exact problem and uploaded data.
   - `arena.html`: A custom, interactive HTML5/Canvas/SVG/Tailwind visualization & playable mini-app rendered inside a sandboxed `<iframe>` via `postMessage`.
   - **Self-Healing Verifier:** Automatically runs a headless smoke test on `env_module.py` and repairs any runtime/shape errors before training starts.
4. **Autonomous Closed-Loop Training & Self-Tuning:** The IDE runs the entire multi-round loop autonomously—Warm-Start $\rightarrow$ PPO Training Burst $\rightarrow$ Observing Metrics & Failure Episodes $\rightarrow$ Rewriting Reward Weights & Hyperparameters $\rightarrow$ Re-Training—until the user's target mastery expectation is met, while streaming live visual frames and plain-English progress logs to the UI.
5. **Owned Local Artifact:** Exports the final trained policy into a self-contained ONNX `.runbundle` (`model.onnx` + `env_module.py` + `arena.html` + `run_local.py`)[cite: 3, 4].

---

## 1. Clean-Slate Deprecation & Deletion Manifest

In Phase 4A, **DELETE** the following legacy hardcoded domain files[cite: 3, 4]:
- `backend/rl_ide/adapters/stream_adapter.py`[cite: 3, 4]
- `backend/rl_ide/adapters/text_classify_adapter.py`[cite: 3, 4]
- `backend/rl_ide/adapters/auto_data_adapter.py`[cite: 4]
- `backend/rl_ide/adapters/dynamic_sim_adapter.py`[cite: 4]
- `backend/rl_ide/reward/dsl.py`[cite: 3, 4]
- `backend/rl_ide/reward/evaluator.py`[cite: 3, 4]
- `backend/rl_ide/compiler/nl_to_config.py`[cite: 3, 4]
- `backend/rl_ide/compiler/critique_engine.py`[cite: 3, 4]
- `backend/rl_ide/vibe/translator.py`[cite: 4]
- `backend/rl_ide/vibe/orchestrator.py`[cite: 4]
- `frontend/src/components/PolymorphicArena.tsx` (replace with universal `SandboxedArenaIframe.tsx`)[cite: 4]
- Delete obsolete adapter-specific test files (`test_adapter_conformance.py`, `test_phase_0a.py` through `test_phase_3c.py`)[cite: 4] and replace them with the new `v4.0` test suite (`test_v4_0a.py` through `test_v4_0d.py`).

**KEEP & UPGRADE** these domain-agnostic core modules[cite: 3, 4]:
- `backend/rl_ide/ingest/` (`loader.py`, `profiler.py`, `sql_connector.py` — used as a clean helper library so generated environments can load any CSV, XLSX, PDF, DOCX, or SQL query in 1 line of code)[cite: 4].
- `backend/rl_ide/drift/monitor.py` (Unsupervised PSI / Entropy / Action-Rate drift detector)[cite: 3, 4].
- `backend/rl_ide/export/exporter.py` & `registry.py` (ONNX exporter, parity verifier, `.runbundle` packager, and standalone `run_local.py` daemon)[cite: 3, 4].

---

## 2. v4.0 End-to-End Architecture

~~~text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        v4.0 FRONTEND (React + Tailwind)                     │
│  ┌───────────────────────┬─────────────────────────┬─────────────────────┐  │
│  │ 1. ARCHITECT & LOG    │ 2. DYNAMIC ARENA IFRAME │ 3. LIVE METRICS &   │  │
│  │ - Clarifying Q&A Chat │ - Renders synthesized   │    INTERACT PANEL   │  │
│  │ - Clickable Option    │   `arena.html` via      │ - Target Goal Meter │  │
│  │   Pills               │   sandboxed iframe      │ - Auto-Tune Round   │  │
│  │ - Autonomous Loop     │ - Live Episode Stream   │   History & Diffs   │  │
│  │   Activity Feed       │ - [🎮 Play / Interact]  │ - Post-Training     │  │
│  │ - Mid-Flight Nudge    │   bi-directional bridge │   Export (.onnx)    │  │
│  └───────────────────────┴─────────────────────────┴─────────────────────┘  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ REST + WebSocket (/ws/v4/{session_id})
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                   v4.0 SYNTHESIS & AUTOPILOT ENGINE (FastAPI)               │
│                                                                             │
│  [Stage 1: Intake Architect Agent]                                          │
│     ├─ Inspects prompt + uploaded files/SQL schema                          │
│     ├─ Checks 5 Pillars (Objective, Data, Arena UI, Target Metric, Play UI) │
│     └─ Emits either `ClarificationQuestions` OR locked `AgentBlueprintSpec` │
│                                                                             │
│  [Stage 2: JIT Environment & Arena Synthesizer + Self-Healing Verifier]     │
│     ├─ Generates `workspaces/<id>/env_module.py` (`SynthesizedGymEnv`)      │
│     ├─ Generates `workspaces/<id>/arena.html` (HTML5/Canvas/JS UI)          │
│     └─ Runs `SandboxSmokeTester` ( resets, steps 20x, tests reward & JSON ) │
│        └─ If traceback -> auto-repairs code with LLM (up to 3 retries)      │
│                                                                             │
│  [Stage 3: Autonomous Closed-Loop Trainer & Self-Tuner]                     │
│     └─ For Round 1..MaxRounds:                                              │
│          1. Warm-Start Behavioral Cloning (from `get_demonstrations()`)     │
│          2. Run PPO Burst -> Stream live state frames to `arena.html`       │
│          3. Evaluate `compute_success_metric(episodes)` vs User Target      │
│          4. If Target Met -> Freeze & Celebrate!                            │
│          5. If Not Met -> `AutoTunerAgent` inspects failure trajectories,   │
│             rewrites `compute_reward()` & hyperparams, runs Critic Warmup,  │
│             and starts Round N+1 automatically!                             │
└─────────────────────────────────────────────────────────────────────────────┘
~~~

---

## 3. Core Contracts & System Prompts (The Brains of v4.0)

### 3.1 Multi-Provider LLM Client (`backend/rl_ide/llm/client.py`)
Must support user-configurable BYOK and default environment variables with first-class support for **DeepSeek**:
- **DeepSeek (`provider="deepseek"`):** Base URL `https://api.deepseek.com/v1`, default model `deepseek-chat` (or `deepseek-reasoner`), uses `DEEPSEEK_API_KEY`.
- **Anthropic (`provider="anthropic"`):** Claude 3.5/3.7 Sonnet (`ANTHROPIC_API_KEY`)[cite: 3, 4].
- **OpenAI (`provider="openai"`):** GPT-4o / o3-mini (`OPENAI_API_KEY`)[cite: 3, 4].
- **Local (`provider="ollama"`):** Local Ollama / vLLM endpoint (`http://localhost:11434/v1`)[cite: 3, 4].
- **Deterministic Fallback Synthesizer (`FallbackCodeSynthesizer`):** When running unit tests in CI/offline mode without an active API key, dynamically constructs valid Python `env_module.py` and `arena.html` files from the `AgentBlueprintSpec` so `pytest` always runs 100% deterministically and offline[cite: 3, 4].

---

### 3.2 Stage 1 Contract: `IntakeArchitectAgent` (`backend/rl_ide/architect/intake.py`)

Before starting synthesis, the app must ensure all **5 Pillars** are clear:
1. `objective_and_actions`: What decisions/moves does the agent make, and what is the goal?
2. `environment_and_data`: Is it using uploaded files/SQL tables (`dataset_id`) or generating a simulated world/game from scratch?
3. `arena_visualization`: How does the user want to visually see the training and behavior in the center panel?
4. `expectations_and_target`: What is the target success metric (e.g., *"Win/Draw rate >= 90%"*, *"Zero collisions"*, *"Classification accuracy >= 95%"*) and max auto-tuning rounds (default `3–5`)?
5. `interactive_mode`: How should the user be able to interact/play with the agent once trained?

```python
class ClarifyingQuestion(BaseModel):
    pillar: str                  # "objective" | "data" | "arena_ui" | "expectations" | "interaction"
    question: str                # Friendly 1-sentence question
    suggested_options: list[str] # 3-4 clickable pills in the UI
    default_option: str          # Used if user clicks "Use Smart Defaults"

class AgentBlueprintSpec(BaseModel):
    task_title: str
    domain_summary: str
    data_source_type: str        # "pure_simulation" | "uploaded_dataset" | "hybrid_doc_and_sim"
    dataset_id: str | None = None
    observation_features: list[str]
    action_labels: list[str]
    arena_visual_description: str
    interactive_mode_description: str
    success_metric_name: str     # e.g., "Non-Loss Rate (%)", "Target Docking Rate (%)", "Audit Accuracy (%)"
    target_metric_threshold: float # e.g., 85.0 ( meaning 85% )
    max_auto_tune_rounds: int = 4
    initial_reward_components: list[dict[str, Any]]

class IntakeResponse(BaseModel):
    status: str                  # "needs_clarification" | "ready_to_build"
    assistant_message: str
    questions: list[ClarifyingQuestion] = []
    blueprint_spec: AgentBlueprintSpec | None = None
```

#### System Prompt for `IntakeArchitectAgent`:
```text
You are the Lead AI Agent Architect for an autonomous Reinforcement Learning IDE.
Your job is to interview the user briefly to lock down a complete `AgentBlueprintSpec` before our code synthesizer builds their custom RL environment, visual Arena UI, and autonomous training loop.

Check if the user's prompt (plus any attached dataset schema) clearly specifies the 5 Pillars:
1. Objective & Actions: What the agent controls and what good behavior looks like.
2. Data / Environment Source: Uploaded data/SQL/PDFs vs. a synthesized game/simulation.
3. Visual Arena Design: How they want to visually watch the agent on screen (e.g., 2D grid, canvas physics, card inspector, dashboard).
4. Target Expectations: The target success metric threshold (0-100%) and how many autonomous self-tuning rounds to run.
5. Live Interaction Mode: How the human can play against or test the agent interactively.

RULES:
- If critical details (especially Visual Arena preference or Target Expectations) are missing on the first turn, set `status="needs_clarification"` and ask at most 2 to 3 concise, high-impact questions.
- Every question MUST include 3-4 clickable `suggested_options` and a sensible `default_option`.
- If the user answers your questions OR says "use defaults / build now", immediately set `status="ready_to_build"` and output the complete `AgentBlueprintSpec` in strict JSON.
```

---

### 3.3 Stage 2 Contract: `JITEnvironmentSynthesizer` (`backend/rl_ide/synthesizer/env_builder.py`)

Generates a self-contained Python module `workspaces/<session_id>/env_module.py` that defines `SynthesizedGymEnv(gymnasium.Env)`.

#### Required Interface Inside Every Generated `env_module.py`:
```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SynthesizedGymEnv(gym.Env):
    """Auto-synthesized Gymnasium environment for the user's specific task."""
    def __init__(self, dataset_records=None, config=None):
        super().__init__()
        # Must define self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(N,), dtype=np.float32)
        # Must define self.action_space = spaces.Discrete(M)
        # Must define self.action_labels = [...]
        ...

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        ...

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Calls self.compute_reward(state_dict, int(action), next_state_dict, info)
        # Populates info["reward_breakdown"] = {"term_a": val, ...}
        # Populates info["is_success"] = bool (used to compute 0-100% success metric)
        # Populates info["arena_state"] = self.render_arena_state()
        ...

    def compute_reward(self, state: dict, action: int, next_state: dict, info: dict) -> tuple[float, dict[str, float]]:
        """Modular reward function. Returns (total_reward, breakdown_dict).
        The Autonomous Auto-Tuner rewrites this method across rounds to improve behavior."""
        ...

    def render_arena_state(self) -> dict:
        """Returns JSON-serializable dictionary sent via WebSocket/postMessage to arena.html."""
        ...

    def interactive_step(self, payload: dict, policy_predict_fn) -> dict:
        """Handles live human interaction from arena.html (e.g., human clicks a square, moves a target, or submits a custom record) and returns updated arena_state + agent action + action probabilities."""
        ...

    def get_demonstrations(self, num_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Returns heuristic (obs_batch, action_batch) pairs for rapid Warm-Start Behavioral Cloning."""
        ...
```

#### System Prompt for `JITEnvironmentSynthesizer`:
```text
You are an Expert RL Environment Engineer. Given an `AgentBlueprintSpec` (and optional `DatasetProfile`), write a single, self-contained, bug-free Python module defining `SynthesizedGymEnv(gymnasium.Env)`.

STRICT ENGINEERING REQUIREMENTS:
1. `self.observation_space` MUST be `spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)`. Always return `np.asarray(obs, dtype=np.float32)` with exact shape `(obs_dim,)` and no NaNs/Infs.
2. `self.action_space` MUST be `spaces.Discrete(n_actions)`.
3. Keep `compute_reward(self, state, action, next_state, info)` modular, returning `(float(total_reward), dict_of_term_floats)`.
4. Every `step()` MUST set `info["reward_breakdown"]`, `info["is_success"]` (boolean indicating whether the step/episode achieved the user's goal), and `info["arena_state"] = self.render_arena_state()`.
5. Implement `get_demonstrations(self, num_samples=200)` returning `(X_obs, y_actions)` using sensible domain heuristics so Warm-Start Behavioral Cloning gives the agent an immediate head start.
6. Implement `interactive_step(self, payload, policy_predict_fn)` so when a human clicks or interacts in the UI, it updates the state, queries `action_idx, action_probs = policy_predict_fn(obs)`, applies the agent's response, and returns a JSON dict with `arena_state`, `agent_action_label`, `action_probs`, and `status_text`.
7. Allowed imports ONLY: `math`, `random`, `numpy`, `pandas`, `gymnasium`, `typing`, ` copy`, `collections`. Never use network calls, `subprocess`, `os.system`, or external game libraries like `pygame`.
```

#### Self-Healing Sandbox Verifier (`backend/rl_ide/synthesizer/verifier.py`):
Before starting training, `verify_and_heal_env(code_str, dataset_records)` executes:
1. AST security scan (blocks `os.system`, `subprocess`, `eval`, `exec` of untrusted strings, socket/network calls).
2. Instantiates `env = SynthesizedGymEnv(dataset_records)`.
3. Runs `obs, info = env.reset()`, verifies `obs.shape == env.observation_space.shape` and `np.all(np.isfinite(obs))`.
4. Steps through 25 random actions, verifies `reward` is float, `info["arena_state"]` is JSON-serializable (`json.dumps`), and `env.get_demonstrations(32)` returns valid arrays.
5. Tests `env.interactive_step({}, dummy_policy_fn)`.
6. **If any check raises an Exception:** Captures the exact Python traceback and calls the LLM repair prompt (up to 3 retries) until the environment passes 100% of verification checks!

---

### 3.4 Stage 2B Contract: `JITArenaSynthesizer` (`backend/rl_ide/synthesizer/arena_builder.py`)

Generates `workspaces/<session_id>/arena.html`—a rich, self-contained HTML5/Tailwind/Canvas/JS visualization and interactive playground tailored to `env.render_arena_state()`.

#### Standard `window.postMessage` Protocol Between Parent IDE and `arena.html` Iframe:
1. **Parent IDE $\rightarrow$ Iframe (`type: "ARENA_UPDATE"`):**
   ```json
   {
     "type": "ARENA_UPDATE",
     "mode": "training_stream" | "interactive_play",
     "round": 2,
     "episode_id": 14,
     "step": 8,
     "total_reward": 12.5,
     "reward_breakdown": {"win_bonus": 10.0, "center_control": 2.5},
     "arena_state": { ... },
     "action_probs": [0.05, 0.80, 0.15]
   }
   ```
2. **Iframe $\rightarrow$ Parent IDE (`type: "ARENA_INTERACT"`):**
   When the user clicks inside `arena.html` (e.g., clicking a game cell, dragging a waypoint, toggling a traffic light, editing a customer row):
   ```json
   {
     "type": "ARENA_INTERACT",
     "payload": { "user_action": 4, "custom_params": {} }
   }
   ```
   The Parent IDE forwards `payload` to `POST /api/v4/sessions/{id}/interact`, which runs `env.interactive_step(payload, policy_fn)` and posts back the updated `ARENA_UPDATE` message in `<30ms`!

#### System Prompt for `JITArenaSynthesizer`:
```text
You are a Creative Frontend & Simulation UI Engineer.
Given an `AgentBlueprintSpec` and the exact JSON structure returned by `SynthesizedGymEnv.render_arena_state()`, generate a complete, self-contained, single-file `arena.html` document.

REQUIREMENTS:
1. Use dark-mode modern aesthetics (`bg-slate-950 text-slate-100`, Tailwind CSS via CDN or embedded CSS, clean SVG/HTML5 Canvas animations).
2. Listen for `window.addEventListener("message", (event) => ...)` handling `event.data.type === "ARENA_UPDATE"`.
3. Render the domain state visually and unmistakably based on `event.data.arena_state` (e.g., an interactive game board, a 2D physics/robotics radar canvas, a visual shift calendar, a live financial/sensor chart, or an interactive card table).
4. Render live telemetry pills at the top (Episode ID, Step, Total Reward, and a horizontal bar strip for `reward_breakdown`).
5. Include an interactive mode: when the user clicks interactive elements in the arena (or clicks a "Reset / New Interactive Turn" button), emit `window.parent.postMessage({ type: "ARENA_INTERACT", payload: {...} }, "*")` so the user can play against or test the trained agent directly inside the arena!
6. Handle missing/initial state gracefully with a sleek "Waiting for first episode stream..." skeleton state.
```

---

### 3.5 Stage 3 Contract: `AutonomousLoopController` & `AutoTunerAgent` (`backend/rl_ide/autopilot/loop_controller.py`)

Once `env_module.py` and `arena.html` are synthesized and verified, the **Autonomous Closed-Loop Trainer** runs **without requiring manual user intervention**:

```text
For Round r = 1 to blueprint.max_auto_tune_rounds:
  1. Emit WebSocket Event: `{"event": "round_started", "round": r, "message": "Running Warm-Start & Training Burst..."}`
  2. If r == 1: Run `warmstart_from_demonstrations(env, model)` (20 epochs BC on `env.get_demonstrations()`).
  3. If r > 1: Run `critic_warmup(model, buffer)` (15 epochs value-network calibration on new reward function).
  4. Run PPO Burst (`n_episodes = 25`), streaming live `ARENA_UPDATE` frames over WebSocket to `arena.html`.
  5. Compute `current_metric = evaluate_success_rate(episodes)` (0.0 to 100.0%).
  6. Emit WebSocket Event: `{"event": "round_evaluated", "round": r, "metric": current_metric, "target": target_threshold}`.
  7. Check Stopping Criteria:
     - If `current_metric >= blueprint.target_metric_threshold`:
       Emit `{"event": "target_achieved", "message": "Target mastery reached! Agent is ready for interactive play & ONNX export."}` -> BREAK.
  8. Autonomous Self-Tuning Step (`AutoTunerAgent`):
     - Collect summary statistics of the last 25 episodes (top 3 best episodes, top 3 worst failure episodes, average reward per term, action distribution entropy).
     - Call `AutoTunerAgent` to diagnose why the agent fell short of `target_threshold` and rewrite `compute_reward()` (and optionally `get_demonstrations()` or `ent_coef`/`learning_rate`).
     - Verify the updated `compute_reward()` via `SandboxSmokeTester`.
     - Hot-reload the updated reward function into `SynthesizedGymEnv` and save checkpoint snapshot `v{r}`.
     - Emit WebSocket Event: `{"event": "auto_tuned", "round": r, "diagnosis": "...", "changes_summary": "..."}` -> Continue to Round r+1!
```

*Note: At any time during or after the autonomous loop, the user can also type a natural-language nudge (e.g., "Make it play more aggressively" or "Never pick option C when value > 100") which triggers one additional targeted tuning round.*

---

## 4. Sequential Build Phases for Antigravity

> **DIRECTIVE FOR ANTIGRAVITY:**
> Execute strictly ONE phase at a time (`Phase 4A` -> `Phase 4B` -> `Phase 4C` -> `Phase 4D`).
> At the end of each phase, run `pytest` (and `npm run build` in Phase 4D) and ensure 100% pass before proceeding.

---

### 🚀 PHASE 4A: Clean-Slate Adapter Removal, Multi-Provider DeepSeek/BYOK Client & Conversational Intake Architect

**Objective:** Remove all legacy hardcoded adapters/DSL, upgrade the LLM layer with first-class DeepSeek + Claude + OpenAI + Offline Fallback support, and build the **Stage 1 Conversational Intake Architect**.

**Tasks for Antigravity:**
1. **Delete Legacy Hardcoded Adapters & DSL:**
   - Remove `backend/rl_ide/adapters/`, `backend/rl_ide/reward/`, `backend/rl_ide/compiler/`, `backend/rl_ide/vibe/`, and old adapter test files (`test_adapter_conformance.py`, `test_phase_0a.py` .. `test_phase_3c.py`)[cite: 3, 4].
   - Keep `backend/rl_ide/ingest/` (file/SQL/PDF/DOCX loaders), `backend/rl_ide/drift/monitor.py`, and `backend/rl_ide/export/`[cite: 3, 4].
2. **Build `backend/rl_ide/llm/client.py`:**
   - Support `provider="deepseek"` (`https://api.deepseek.com/v1`, model `deepseek-chat` / `deepseek-reasoner`), `openai`, `anthropic`, `ollama`, and a smart `FallbackCodeSynthesizer` for offline testing[cite: 3, 4].
   - Support connection validation (`validate_llm_connection(config)`).
3. **Build `backend/rl_ide/architect/models.py` & `intake.py`:**
   - Implement `ClarifyingQuestion`, `AgentBlueprintSpec`, `IntakeResponse`, and `IntakeArchitectAgent`.
   - Given a user's initial natural-language prompt (and optional `IngestedDataset` profile from `ingest/`), analyze completeness across the **5 Pillars**:
     1. Objective & Actions
     2. Data / Simulator Source
     3. Custom Visual Arena Design
     4. Target Success Metric & Max Auto-Tune Rounds
     5. Interactive Play Mode
   - If the initial prompt is underspecified (e.g., *"I want to train a tic-tac-toe agent"* or *"Train an agent on this CSV"* without specifying how the arena should look or the target win-rate expectation), return `status="needs_clarification"` with 2–3 targeted `ClarifyingQuestion` items (each with 3–4 clickable option pills and a smart default).
   - Once the user submits answers (or passes `force_smart_defaults=True`), return `status="ready_to_build"` with the complete `AgentBlueprintSpec`.
4. **Write Verification Suite (`backend/tests/test_v4_0a.py`):**
   - Test 1: Underspecified prompt (*"Train an agent to play tic-tac-toe"*) returns `status="needs_clarification"` asking about Arena visual style and Target Win/Draw expectation, with clickable options.
   - Test 2: Answering the clarifying questions (or clicking *"Use Smart Defaults"*) produces a complete, valid `AgentBlueprintSpec`.
   - Test 3: Uploading a custom CSV/PDF via `ingest/` + prompt produces a data-aware `AgentBlueprintSpec` referencing the ingested columns.
   - Test 4: Verify DeepSeek, OpenAI, Anthropic, and Fallback provider routing in `llm/client.py`.

**Completion Gate 4A:** Run `pytest backend/tests/test_v4_0a.py` — must pass 100%.

---

### 🚀 PHASE 4B: JIT Environment Synthesizer (`env_module.py`), Self-Healing Verifier & Dynamic Arena Synthesizer (`arena.html`)

**Objective:** Build **Stage 2**—the code generators that write `SynthesizedGymEnv` and `arena.html` on the fly for *any* `AgentBlueprintSpec`, plus the `SandboxSmokeTester` that verifies and auto-heals generated code.

**Tasks for Antigravity:**
1. **Build `backend/rl_ide/synthesizer/env_builder.py` (`JITEnvironmentSynthesizer`):**
   - Takes `AgentBlueprintSpec` (and optional `IngestedDataset`) and synthesizes `workspaces/<session_id>/env_module.py` implementing `SynthesizedGymEnv(gymnasium.Env)`.
   - Must dynamically load `env_module.py` via `importlib.util.spec_from_file_location`.
2. **Build `backend/rl_ide/synthesizer/verifier.py` (`SandboxSmokeTester`):**
   - Performs AST security scan on generated Python code (blocking `os.system`, `subprocess`, sockets, file deletion).
   - Instantiates `SynthesizedGymEnv`, checks `observation_space` and `action_space`, runs `reset()` and 25 `step()` calls, verifies `compute_reward()` returns `(float, dict)`, verifies `render_arena_state()` is JSON-serializable, verifies `get_demonstrations(32)` returns valid `(X, y)` numpy arrays, and tests `interactive_step()`.
   - **Self-Healing Loop:** If any error or shape mismatch occurs, automatically feeds the traceback + broken code to the LLM repair prompt (up to 3 attempts) until verification passes 100%.
3. **Build `backend/rl_ide/synthesizer/arena_builder.py` (`JITArenaSynthesizer`):**
   - Inspects `AgentBlueprintSpec` and a real sample dictionary from `env.render_arena_state()`.
   - Generates `workspaces/<session_id>/arena.html` implementing the `ARENA_UPDATE` and `ARENA_INTERACT` `window.postMessage` protocol, live reward breakdown bars, and interactive play controls.
4. **Write Verification Suite (`backend/tests/test_v4_0b.py`):**
   - Test 1: Synthesize a game/simulation environment (e.g., Tic-Tac-Toe or Grid Catcher) from an `AgentBlueprintSpec` -> verify `SandboxSmokeTester` passes 100% and `interactive_step` works.
   - Test 2: Deliberately inject a bug into generated environment code (e.g., wrong numpy shape or division by zero in `compute_reward`) -> verify `SandboxSmokeTester` catches the traceback and self-heals the code automatically!
   - Test 3: Synthesize a custom data-driven environment from an ingested CSV/SQL dataset -> verify `SynthesizedGymEnv` steps through records and generates a valid `arena.html` containing the `ARENA_UPDATE` and `ARENA_INTERACT` bridges.

**Completion Gate 4B:** Run `pytest backend/tests/test_v4_0a.py backend/tests/test_v4_0b.py` — must pass 100%.

---

### 🚀 PHASE 4C: Autonomous Closed-Loop Trainer, Self-Tuner (`AutoTunerAgent`), FastAPI Server & ONNX Export

**Objective:** Build **Stage 3**—the hands-free autonomous training loop that trains, observes behavior, rewrites reward functions/hyperparameters across rounds until expectations are met, streams everything over WebSockets, and exports the final owned `.runbundle`.

**Tasks for Antigravity:**
1. **Build `backend/rl_ide/autopilot/auto_tuner.py` (`AutoTunerAgent`):**
   - Inspects episode telemetry from Round $r$: current success metric vs. `target_metric_threshold`, per-term reward averages, action distribution, and 3 failed episode traces.
   - Generates a plain-English `diagnosis` (e.g., *"Agent was choosing illegal/edge moves 22% of the time; increased illegal-action penalty from -2 to -10 and added +3 bonus for blocking opponent threats"*), rewrites `compute_reward()` (and updated heuristic demonstrations if needed), verifies via `SandboxSmokeTester`, and hot-swaps into `env_module.py`.
2. **Build `backend/rl_ide/autopilot/loop_controller.py` (`AutonomousLoopController`):**
   - Orchestrates the full multi-round autonomous loop (`Round 1 .. max_auto_tune_rounds`):
     - Round 1: Warm-Start Behavioral Cloning (`get_demonstrations()`) + PPO Burst 1.
     - Evaluates success metric (`mean(info["is_success"]) * 100`).
     - If below target: invokes `AutoTunerAgent`, runs 15-epoch Critic Warmup on re-scored replay buffer, and runs PPO Burst $r+1$.
     - Streams real-time events (`round_started`, `episode_step`, `round_evaluated`, `auto_tuned`, `target_achieved`) over WebSocket `/ws/v4/{session_id}` and stores history in `SessionStateStore`.
   - Also supports `nudge_session(session_id, user_feedback_text)` so the user can give mid-flight or post-training English feedback to trigger an extra tuning burst.
3. **Update `backend/rl_ide/export/exporter.py` for v4.0:**
   - Export the trained SB3 policy to `model.onnx` (verifying parity $< 10^{-5}$ with `onnxruntime`)[cite: 3, 4].
   - Package `.runbundle` containing `model.onnx`, `blueprint_spec.json`, `env_module.py`, `arena.html`, `drift_baseline.json`, and standalone `run_local.py` (supporting `--sample`, `--input`, and `--serve --port 9000` microservice daemon)[cite: 4].
4. **Build Clean v4.0 FastAPI Server (`backend/rl_ide/api/server.py`):**
   - `POST /api/v4/ingest/upload` & `POST /api/v4/ingest/sql` — universal file/SQL ingestion[cite: 4].
   - `POST /api/v4/architect/chat` — conversational intake interview (returns clarifying questions or locked `AgentBlueprintSpec`).
   - `POST /api/v4/sessions/start` — triggers JIT synthesis (`env_module.py` + `arena.html`), runs self-healing verification, and starts the Autonomous Closed-Loop Training in the background (or synchronously in test mode).
   - `GET /api/v4/sessions/{id}` — returns current session state, synthesized code, round history, and metrics.
   - `GET /api/v4/sessions/{id}/arena` — serves the synthesized `arena.html` for the sandboxed iframe.
   - `POST /api/v4/sessions/{id}/interact` — executes `env.interactive_step(payload, policy_fn)` for live human play/testing inside `arena.html`.
   - `POST /api/v4/sessions/{id}/nudge` — applies user feedback and runs an additional tuning round.
   - `POST /api/v4/sessions/{id}/export` — packages and downloads the `.runbundle` + `.zip`[cite: 3, 4].
   - `POST /api/v4/settings/validate` — tests DeepSeek / Claude / OpenAI / Ollama connectivity[cite: 4].
   - `WS /ws/v4/{session_id}` — streams live training frames, autonomous tuning logs, and round metrics.
5. **Write Verification Suite (`backend/tests/test_v4_0c.py`):**
   - Test 1: Full autonomous multi-round loop on a synthesized game task where Round 1 starts with a weak reward function, `AutoTunerAgent` diagnoses and rewrites `compute_reward()`, and Round 2 achieves the target success threshold.
   - Test 2: Full FastAPI flow (`/api/v4/architect/chat` -> `/api/v4/sessions/start` -> `/api/v4/sessions/{id}/interact` -> `/api/v4/sessions/{id}/export` -> `python run_local.py --sample`).

**Completion Gate 4C:** Run `pytest backend/tests/` — all v4 tests must pass 100%.

---

### 🚀 PHASE 4D: The v4.0 Autonomous Synthesis Frontend (Intake Architect Chat, Sandboxed Arena Iframe & Live Autopilot Timeline)

**Objective:** Rebuild `frontend/` around the v4.0 workflow: **Converse & Clarify $\rightarrow$ Watch JIT Synthesis & Autonomous Multi-Round Training in Custom Iframe Arena $\rightarrow$ Play/Interact Live $\rightarrow$ Export Owned Policy**.

**Tasks for Antigravity:**
1. **Header Bar (`frontend/src/components/Header.tsx`):**
   - Brand: **RL-Native Agent Studio v4.0 (Autonomous JIT Synthesis)**.
   - LLM Provider pill showing active engine (`DeepSeek V3 / R1`, `Claude 3.7`, `GPT-4o`, `Ollama`) + `[⚙️ LLM Settings]` modal (with DeepSeek as top preset) + `[📦 Policy Registry]` + `[⬇️ Export Owned Agent (.onnx)]`.
2. **Left Panel — Conversational Architect & Autonomous Activity Feed (`frontend/src/components/ArchitectPanel.tsx`):**
   - **Universal Attachment Bar:** Drag-and-drop any file (`.csv`, `.xlsx`, `.pdf`, `.docx`, `.sqlite`, `.json`) or click `[🗄️ Connect SQL]`[cite: 4].
   - **Intake Architect Chat:**
     - User describes what they want to build in plain English.
     - When the Architect responds with `ClarifyingQuestion` cards, render **interactive clickable option pills** under each question so the user can answer with 1 click, plus a **`[⚡ Use Smart Defaults & Build Agent Now]`** express button!
   - **Autonomous Loop Activity Timeline:**
     - Once building starts, streams live step-by-step cards:
       - 🛠️ *Synthesized `SynthesizedGymEnv` (Verified 25/25 smoke steps ✅)*
       - 🎨 *Synthesized Custom Visual Arena (`arena.html`)*
       - 🔄 *Round 1: Warm-Start + 25 Episodes → Success Rate: 58% (Target: 85%)*
       - 🧠 *Auto-Tuner Diagnosis: Boosted tactical block reward & adjusted entropy → Re-calibrating Critic...*
       - 🌟 *Round 2: Success Rate reached 92% (Target: 85% Met!)*
   - **Mid-Flight / Post-Training Nudge Bar:** Allows the user to type any refinement (*"Make the agent more conservative"* or *"Add a penalty for taking too many steps"*) at any time.
3. **Center Panel — Universal Sandboxed Arena Iframe (`frontend/src/components/SandboxedArenaIframe.tsx`):**
   - Hosts `<iframe sandbox="allow-scripts" srcDoc={arenaHtml} />` taking up the full center stage.
   - Top bar above the iframe:
     - Mode Switcher: **`[ 📺 Watch Live Training ]`** vs **`[ 🎮 Interact / Play Live ]`**.
     - Episode Replay Scrub Bar: Click any past episode to replay its trajectory inside `arena.html`.
     - **`[ 🔍 Inspect Generated Code ]` Drawer Toggle:** Lets curious users peek at the synthesized `env_module.py`, `compute_reward()`, and `arena.html` code (read-only or editable).
   - Bi-directional `postMessage` bridge: forwards WebSocket `ARENA_UPDATE` frames into the iframe, and listens for `ARENA_INTERACT` events from user clicks inside the iframe, calling `POST /api/v4/sessions/{id}/interact` and updating the iframe immediately.
4. **Right Panel — Expectations, Mastery Telemetry & Round Diffs (`frontend/src/components/TelemetryPanel.tsx`):**
   - **Expectation Progress Card:** Shows user's target metric (e.g., `Target Non-Loss Rate: 85%`) vs current achieved metric (`92% ✅`), current Auto-Tune Round (`Round 2 / 4`), and Drift Health status.
   - **Reward Evolution Log:** Shows how `AutoTunerAgent` evolved the reward breakdown and hyperparameters across rounds, with a 1-click `[⏪ Rollback to Round N]` button.
   - **Live Reward & Success Sparklines:** Plots episode return and success rate across all rounds.

**Completion Gate 4D:**
- Run `npm run build` (`tsc && vite build`) in `frontend/` and verify 0 errors.
- Run full backend `pytest` suite (`test_v4_0a.py`, `test_v4_0b.py`, `test_v4_0c.py`) and verify 100% green.
- Update `progress.md` with the v4.0 architecture summary and test results.