# RL-Native Agent IDE — Master Plan & Antigravity Build Specification (v2.0 Hardened)

**A Claude Code alternative that is agent-centric, not code-centric.**[cite: 1, 2]

*Users describe a task in natural language, watch an agent learn it, critique its behavior in English, and walk away with a policy they own — no per-call LLM fees at inference, no data leaving their boundary.*[cite: 1, 2]

---

## Table of Contents

1. [Philosophy](#1-philosophy)
2. [The Core Insight](#2-the-core-insight)
3. [What We Are Building (and Not Building)](#3-what-we-are-building-and-not-building)
4. [The Product in One Screen](#4-the-product-in-one-screen)
5. [Hardened Architecture](#5-hardened-architecture)
6. [The Four Loops](#6-the-four-loops)
7. [The Adapter Contract (Hardened)](#7-the-adapter-contract-hardened)
8. [The Reward DSL & AST Sandbox](#8-the-reward-dsl--ast-sandbox)
9. [The Critique→Diff Engine & Two-Stage Preview](#9-the-critiquediff-engine--two-stage-preview)
10. [What We Black-Box (and What We Don't)](#10-what-we-black-box-and-what-we-dont)
11. [Unsupervised Drift Detection Specification](#11-unsupervised-drift-detection-specification)
12. [Monetization](#12-monetization)
13. [Competitive Landscape & Moats](#13-competitive-landscape--moats)
14. [Risks & Mitigations](#14-risks--mitigations)
15. [Go-to-Market](#15-go-to-market)
16. [Go-Ground Roadmap & Success Metrics](#16-go-ground-roadmap--success-metrics)
17. [Antigravity Execution Blueprint (Phase-by-Phase Build Instructions)](#17-antigravity-execution-blueprint-phase-by-phase-build-instructions)
18. [Appendix A — Glossary](#appendix-a--glossary)
19. [Appendix B — What We Will Not Compromise On](#appendix-b--what-we-will-not-compromise-on)

---

## 1. Philosophy

### 1.1 Intelligence should be owned, not rented
Claude Code, Cursor, and every frontier-model agent tie the user to a per-token meter[cite: 1, 2]. The user pays forever, the data leaves their boundary, and every improvement accrues to the vendor[cite: 1, 2]. We believe a large class of recurring, well-defined tasks should be **capitalized, not expensed**: train once, own forever, run locally[cite: 1, 2].

### 1.2 Behavior is the artifact, not code
Code-centric tools ask the user to review **diffs**[cite: 1, 2]. We ask the user to review **behavior**[cite: 1, 2]. The user's job is not to read code — it is to know what "good" looks like in their domain, watch the agent, and say what's wrong[cite: 1, 2]. The IDE translates that into reward changes[cite: 1, 2].

### 1.3 The human is the alignment mechanism
Reward hacking is not a bug to eliminate — it is the central risk of any RL system[cite: 1, 2]. Our defense is structural: **short training bursts between critiques**, always-visible reward breakdowns, and a human who watches every episode[cite: 1, 2]. The critique loop *is* the alignment[cite: 1, 2].

### 1.4 Narrow and deep beats broad and shallow
We are not building a general reasoning agent[cite: 1, 2]. We are building a workbench for **recurring, well-defined tasks with an observable or verifiable notion of correctness**[cite: 1, 2]. We say this loudly[cite: 1, 2]. Our pitch is "we're better *for this class of task*," not "we replace Claude Code."[cite: 1, 2]

### 1.5 The user's accumulation is the moat
The IDE's defensibility does not come from secrecy[cite: 1, 2]. It comes from **switching cost**: after three months, the user has accumulated task configs, critique logs, trained policies in production, and institutional knowledge encoded in reward functions[cite: 1, 2]. That is not portable[cite: 1, 2]. That is the moat[cite: 1, 2].

### 1.6 Everything inspectable, nothing magic
The NL compiler always shows editable diffs[cite: 1, 2]. The reward DSL is visible[cite: 1, 2]. The training is standard (PPO/Contextual-Bandit via SB3/CleanRL with warm-start behavioral cloning)[cite: 1, 2]. The export is ONNX[cite: 1, 2]. The user can see, edit, roll back, and re-run everything[cite: 1, 2]. We sell a workbench, not an oracle[cite: 1, 2].

---

## 2. The Core Insight

**For narrow recurring tasks, a trained small policy beats a frontier model on cost, latency, and fit.**[cite: 1, 2]

The frontier model's generality is waste the user is paying for[cite: 1, 2]. A compact policy (e.g., a fast MLP/Transformer head over frozen local features/embeddings) trained on 500 examples of *the user's* definition of correctness, running locally at zero marginal cost, outperforms a general model on that specific task — and never sends a byte outside the boundary[cite: 1, 2].

| Dimension | Claude Code / Frontier APIs | This IDE |
|---|---|---|
| Cost model | Per-token, forever (OpEx)[cite: 1, 2] | One-time training + local inference (CapEx)[cite: 1, 2] |
| Data | Leaves boundary[cite: 1, 2] | Never leaves boundary (Local/VPC BYOK)[cite: 1, 2] |
| Improvement | Vendor's roadmap[cite: 1, 2] | Yours — retrain when you want[cite: 1, 2] |
| Failure mode | Model update breaks you[cite: 1, 2] | You own weights, control drift[cite: 1, 2] |
| Latency | Network + queue (500ms–5s)[cite: 1, 2] | Local, deterministic (<5ms ONNX)[cite: 1, 2] |
| Task fit | General, "good enough"[cite: 1, 2] | Narrow, tuned to *your* correctness[cite: 1, 2] |

---

## 3. What We Are Building (and Not Building)

### We ARE building
- A **desktop/web IDE** with three panels: Task, Live Agent, Critique[cite: 1, 2].
- An **adapter SDK** + two reference adapters (`CSVStreamAdapter`, `TextClassifyAdapter`) proving the abstraction generalizes[cite: 1, 2].
- A **NL→Config compiler** that turns prose into runnable task configs[cite: 1, 2].
- A **reward DSL** — a small, inspectable, AST-validated, diffable language for reward terms[cite: 1, 2].
- A **critique→diff engine** that turns English critique into reward diffs with a Two-Stage Sanity Preview, always shown editable[cite: 1, 2].
- A **training orchestrator** running Warm-Start Behavioral Cloning + SB3 PPO in the background, streaming episodes[cite: 1, 2].
- A **policy export pipeline** (ONNX + task config + drift baseline → self-contained local artifact)[cite: 1, 2].
- A **run bundle format** — reproducible, shareable, versionable[cite: 1, 2].

### We are NOT building
- ❌ A marketplace (distraction, requires scale we don't have)[cite: 1, 2].
- ❌ A frontier-model wrapper (the whole point is to escape it)[cite: 1, 2].
- ❌ A general reasoning agent (that's Claude Code's territory)[cite: 1, 2].
- ❌ A model-hosting service (users run training/inference on their own hardware)[cite: 1, 2].
- ❌ A prompt-engineering tool (we're building a training workbench, not a prompt IDE)[cite: 1, 2].
- ❌ A general RL research platform (we're narrow: recurring knowledge-work tasks)[cite: 1, 2].

---

## 4. The Product in One Screen

~~~text
┌──────────────────────┬────────────────────────┬──────────────────────┐
│  1. TASK             │  2. LIVE AGENT         │  3. CRITIQUE         │
│  (config & DSL)      │  (behavior stream)     │  (chat & diff card)  │
│                      │                        │                      │
│  Adapter: CSV Stream │  [Episode #47 Replay]  │  You: "it's flagging │
│  Horizon: MDP (0.90) │  [Live Reward Curve]   │  normal volatility   │
│                      │  [Per-Term Breakdown]  │  as anomalies"       │
│  Reward terms:       │                        │                      │
│  + accurate: +1.0    │  Watching episode      │  IDE: Reward Diff    │
│  - false_pos: -5.0   │  47 of 100...          │  ~ false_pos -5 → -8 │
│  - miss: -3.0        │                        │  + high_vol_context  │
│                      │  [Pause] [Lock Ep #47] │  Preview: 14/15 flip │
│  [NL Edit] [v1 ▾]    │  [Run 50-Ep Burst]     │  [Apply] [Tweak]     │
└──────────────────────┴────────────────────────┴──────────────────────┘
~~~

**Four user actions, one loop:**[cite: 2]
1. Describe the task (NL → config, adapter selection, warm-start)[cite: 2].
2. Watch episodes (behavior and per-term reward breakdown, not raw logs)[cite: 2].
3. Critique in English (LLM → reward diff + two-stage preview, editable)[cite: 2].
4. Export the policy (ONNX + config + drift detector, runs locally)[cite: 2].

---

## 5. Hardened Architecture

~~~text
┌──────────────────────────────────────────────────────────────────────┐
│                         IDE FRONTEND (React + Vite)                  │
│  (Three panels: Task Config + Live Agent Stream + Critique Diffs)    │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ REST + WebSocket (/ws/stream)
┌──────────────────────────────────▼───────────────────────────────────┐
│                      ORCHESTRATOR (FastAPI + Python)                 │
│  ┌────────────────────┐  ┌───────────────────┐  ┌─────────────────┐  │
│  │ NL→Config Compiler │  │ Task State Store  │  │ Training Runner │  │
│  │ (BYOK / Ollama)    │──│ (SQLite + Bundles)│──│ (Warmstart+PPO) │  │
│  └────────────────────┘  └───────────────────┘  └─────────────────┘  │
│  ┌────────────────────┐  ┌───────────────────┐  ┌─────────────────┐  │
│  │ Critique→Diff Eng. │  │ Reward Evaluator  │  │ Policy Exporter │  │
│  │ + 2-Stage Preview  │──│ (Safe AST Engine) │──│ (ONNX + Drift)  │  │
│  └────────────────────┘  └───────────────────┘  └─────────────────┘  │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ DomainAdapter Protocol
┌──────────────────────────────────▼───────────────────────────────────┐
│                           DOMAIN ADAPTERS                            │
│  ┌────────────────────────────────┐ ┌──────────────────────────────┐ │
│  │ CSVStreamAdapter               │ │ TextClassifyAdapter          │ │
│  │ (Time-series, stateful cooldown│ │ (Frozen local embedder/SVD + │ │
│  │  & alert budget MDP, gamma=0.9)│ │  metadata, bandit gamma=0.0) │ │
│  └────────────────────────────────┘ └──────────────────────────────┘ │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────┐
│         RL ENGINE (Stable-Baselines3 PPO + Critic Calibration)       │
│                    Gymnasium Environment Wrapper                     │
└──────────────────────────────────────────────────────────────────────┘
~~~

**Key Architectural Invariants:**
- **BYOK or Local Ollama for all LLM calls:** Supports OpenAI, Anthropic, local Ollama/vLLM, plus a deterministic `MockRulesLLMClient` so unit tests and offline demos work without API keys[cite: 2].
- **Strict Separation of Environment Events and Rewards:** Adapters emit raw `events` (`obs`, `action`, `state`, optional `truth`)[cite: 2]. The `RewardEvaluator` scores them[cite: 2]. Editing the Reward DSL re-scores historical episodes in <5ms without re-running the adapter.
- **Warm-Start + Critic Calibration:**
  1. **Warm-Start Behavioral Cloning (Phase 0 before Episode 1):** Pre-trains the policy for 2 seconds on initial high-confidence rules so the user doesn't watch 200 episodes of random actions.
  2. **Critic Warmup on Diff Apply:** Whenever the Reward DSL changes, freeze the Actor ($\pi_\theta$) and train the Critic ($V_\phi$) for 15 epochs on the re-scored replay buffer, plus enforce a KL-divergence leash ($\beta \cdot D_{\text{KL}}(\pi_{\text{new}} \parallel \pi_{\text{prev}})$) so PPO does not thrash or forget previous rules.

---

## 6. The Four Loops

### Loop 1 — Authoring (NL → runnable task)[cite: 2]
`User prose → NL Compiler → structured TaskConfig + RewardDSL → AST Validator → editable diff → confirm & warm-start`[cite: 2]
- **Deliverable:** A portable, versionable `TaskConfig`[cite: 2].

### Loop 2 — Training (background, streaming)[cite: 2]
`Policy trains in micro-bursts → episodes stream to UI → reward breakdowns per term → user watches & locks good episodes`[cite: 2]
- **Deliverable:** A live episode buffer and versioned policy checkpoints (`v1`, `v2`, ...)[cite: 2].

### Loop 3 — Human Critique (the novel alignment loop)[cite: 2]
`User clicks bad episode → types critique → Critique Engine inspects (DSL, episode events, critique, past diffs) → outputs RewardDiff + 2-Stage Preview → user confirms → Critic warmup + short training burst → back to watching`[cite: 2]
- **Deliverable:** The interactive critique→diff→replay cycle[cite: 2].

### Loop 4 — Export (policy ownership)[cite: 2]
`User clicks "Export Owned Policy" → freeze weights → export model.onnx + verify parity → compute drift_baseline.json → package .runbundle with standalone run_local.py`[cite: 2]
- **Deliverable:** A zero-dependency policy artifact the user owns forever[cite: 2].

---

## 7. The Adapter Contract (Hardened)

Every domain plugs in through a single strict protocol (`backend/rl_ide/contracts/adapter.py`)[cite: 2]:

~~~python
from typing import Protocol, Any
from pydantic import BaseModel

class ObservationSpec(BaseModel):
    vector_dim: int
    feature_names: list[str]
    horizon_type: str       # "stateful_mdp" | "contextual_bandit"
    default_gamma: float    # 0.90 for stateful stream, 0.0 for independent records

class ActionSpec(BaseModel):
    action_type: str        # "discrete"
    labels: list[str]       # e.g., ["normal", "anomaly"] or ["urgent", "normal", "spam", "escalate"]

class StepResult(BaseModel):
    obs_vector: list[float]
    events: dict[str, Any]  # {"obs": {...}, "action": {...}, "state": {...}, "truth": {...}}
    terminated: bool
    truncated: bool
    info: dict[str, Any]

class DomainAdapter(Protocol):
    def observations(self) -> ObservationSpec: ...
    def actions(self) -> ActionSpec: ...
    def event_schema(self) -> dict[str, str]:
        """Returns {field_path: type_str} of all variables legal in Reward DSL."""
        ...
    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, Any]]: ...
    def step(self, action_idx: int) -> StepResult: ...
    def render_frame(
        self,
        episode_id: int,
        step_result: StepResult,
        total_reward: float,
        breakdown: dict[str, float]
    ) -> dict[str, Any]: ...
~~~

### Reference Adapters

| Adapter | Horizon | Observation Pipeline | Reference Use |
|---|---|---|---|
| **CSVStreamAdapter** | `stateful_mdp` ($\gamma = 0.90$) | Sliding window numeric features (`price`, `return_1m`, `volume`, `volatility`, `rolling_vol_90p`, `z_score`) + stateful agent tracking (`cooldown`, `steps_since_last_alert`, `rolling_alert_rate`) | Stock ticker anomaly detection, log spike monitoring, sensor streams[cite: 2] |
| **TextClassifyAdapter** | `contextual_bandit` ($\gamma = 0.0$) | Fast local dense embedding/SVD vector + interpretable tabular metadata (`word_count`, `char_len`, `urgency_keyword_count`, `sender_is_vip`, `has_attachment`, `queue_load`) | Email triage, support ticket routing, content moderation[cite: 2] |

---

## 8. The Reward DSL & AST Sandbox

**Design principle:** The LLM emits declarative YAML DSL, never raw Python[cite: 2]. The evaluator validates all expressions against an AST whitelist and the adapter's `event_schema()`[cite: 2].

~~~yaml
version: "1.0"
horizon:
  gamma: 0.90
  critic_warmup_epochs_on_diff: 15
  kl_coef: 0.1
reward:
  terms:
    - name: accurate_match
      when: "truth.has_label == True and action.label == truth.label"
      value: 1.0
    - name: false_positive_low_vol
      when: "action.label == 'anomaly' and obs.volatility <= obs.rolling_vol_90p"
      value: -8.0
    - name: high_vol_anomaly_bonus
      when: "action.label == 'anomaly' and obs.volatility > obs.rolling_vol_90p and state.cooldown == 0"
      value: 1.5
    - name: alert_spam_penalty
      when: "action.label == 'anomaly' and state.rolling_alert_rate > 0.15"
      value: -4.0
  pinned_episodes:
    - episode_id: 47
      forbid_action: "anomaly"
      penalty: -10.0
  aggregate: sum
  clamp: [-100.0, 100.0]
~~~

**AST Sandbox Rules (`RewardEvaluator`):**
1. Parse `when` and `value` using `ast.parse(expr, mode="eval")`.
2. **Allowed AST Nodes Only:** `Expression`, `BoolOp`, `BinOp`, `UnaryOp`, `Compare`, `Name`, `Attribute`, `Constant`, `Load`, and operators `And`, `Or`, `Not`, `Eq`, `NotEq`, `Lt`, `LtE`, `Gt`, `GtE`, `In`, `NotIn`, `Add`, `Sub`, `Mult`, `Div`, `USub`, `UAdd`.
3. **Forbidden Nodes:** `Call`, `Import`, `ImportFrom`, `Lambda`, `Subscript` with dynamic code, or any attribute starting with `_`.
4. **Static Variable Verification:** Every `namespace.field` reference (e.g., `obs.volatility`, `action.label`, `state.cooldown`, `truth.label`) must exist in `adapter.event_schema()`[cite: 2]. Undefined fields raise a `RewardValidationError` before training starts[cite: 2].

---

## 9. The Critique→Diff Engine & Two-Stage Preview

**Inputs to the Engine:**[cite: 2]
- Current `RewardDSL` YAML[cite: 2].
- Target `EpisodeLog` (the episode the user clicked and critiqued) + last 20 episodes in `EpisodeBuffer`[cite: 2].
- Locked/Pinned "Golden Episodes" that must not regress.
- User's plain-English critique[cite: 2].
- Past applied diffs[cite: 2].

**Outputs (`RewardDiffProposal`):**[cite: 2]
1. **Structured YAML Diff:** Terms added, modified, or removed, plus optional `pinned_episodes` constraint[cite: 2].
2. **One-Line Rationale:** Plain-English explanation per modified term[cite: 2].
3. **Conflict Detection:** Flags if a proposed term contradicts an existing term or violates a locked episode[cite: 2].
4. **Two-Stage Sanity Preview:**
   - **Stage 1 — Instant Counterfactual Score Preview (<20ms):** Re-scores the last 20 cached episodes using the proposed `RewardDSL` without retraining. Shows exact reward shift on the critiqued episode (e.g., *"Episode #47 score changes from +1.0 to -7.0"*).
   - **Stage 2 — Speculative Forked Micro-Burst (~2 seconds):** Clones the current policy in memory (`policy_fork`), runs Critic warmup + 10 fast PPO updates on the buffer with the new reward, and reports behavioral action flips (e.g., *"Forked policy changes action on Episode #47 from `anomaly` → `normal` and preserves 100% of locked episodes"*).

---

## 10. What We Black-Box (and What We Don't)

**Core Rule:** We never compromise the user's data boundary[cite: 1, 2]. Data never leaves the user's machine/VPC[cite: 1, 2].

| Component | Protection Strategy | Rationale |
|---|---|---|
| Critique→Diff prompt engineering, conflict solver & AST edge-case evaluator | Compiled native binary wheel (Cython / Rust PyO3) executing locally using the user's BYOK key or local Ollama | Protects IP without sending user data to our servers |
| Warm-start + Critic-recalibration + KL-leash schedule | Compiled inside trainer core | Prevents competitors from easily replicating smooth 3-second behavioral adaptation |
| Adapter SDK interface, Reward DSL syntax, TaskConfig & ONNX export | 100% Open & Inspectable[cite: 2] | Required for user trust, third-party adapters, and zero-lock-in inference ownership[cite: 2] |

---

## 11. Unsupervised Drift Detection Specification

Because production inference runs without ground-truth labels (`truth.label` is absent in live streams)[cite: 2], `DriftDetector` exports `drift_baseline.json` during packaging and monitors three unsupervised signals in `run_local.py` and the IDE:

1. **Input Feature PSI (Population Stability Index):**
   - Stores 10-bucket quantile boundaries per input feature from training episodes.
   - At inference, computes $\text{PSI} = \sum (P_i - Q_i) \ln(P_i / Q_i)$. Flags drift if $\max(\text{PSI}) > 0.25$.
2. **Policy Action-Probability Entropy Shift:**
   - Computes Shannon entropy $H(\pi(\cdot|s)) = -\sum p(a|s)\log p(a|s)$ from ONNX logits. Flags uncertainty drift if rolling mean entropy shifts by $> 2.5\sigma$ from training baseline.
3. **Action-Rate Distribution Shift:**
   - Tracks rolling frequency of each discrete action (e.g., baseline `anomaly` rate = 5.0%; live rate = 28.0%). Flags drift if Z-score $> 3.0$.

---

## 12. Monetization

| Tier | Includes | Indicative Price |
|---|---|---|
| **Free** | One adapter, short training runs (<10k steps), watermarked export, BYOK[cite: 2] | $0[cite: 2] |
| **Pro** | All adapters, unlimited training, full export, drift monitoring, run bundle sharing[cite: 2] | $X/mo per seat[cite: 2] |
| **Team** | Shared task configs, internal policy registry, audit logs, SSO[cite: 2] | $Y/mo per seat[cite: 2] |

*Pitch: Pay for the workbench. Use your own keys (or local Ollama) during authoring. Pay nothing for inference, forever.*[cite: 2]

---

## 13. Competitive Landscape & Moats

- **Code-centric agents (Claude Code, Cursor):** Write code, review code diffs; tied to per-call frontier LLM costs[cite: 2].
- **RL libraries (SB3, CleanRL):** Raw algorithms; no UI, no NL compiler, no human critique loop[cite: 2].
- **Agent frameworks (LangGraph, CrewAI):** Prompt wrappers tied to frontier APIs; no owned local policy artifact[cite: 2].
- **Our Moat:** Unified `DomainAdapter` schema + interactive `Critique→RewardDiff` loop with Two-Stage Preview + accumulated user `RunBundles` and institutional reward functions[cite: 2].

---

## 14. Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **Reward hacking**[cite: 2] | Critical[cite: 2] | Short training bursts; visible per-term breakdowns; pinned episode constraints; KL leash to prior checkpoint[cite: 2]. |
| **PPO Critic thrashing after reward edits** | High | 15-epoch Critic-only value recalibration on re-scored replay buffer before updating Actor weights. |
| **Cold-start random behavior** | High | 2-second Warm-Start Behavioral Cloning pre-pass before streaming Episode 1. |
| **Small model silent failure out-of-distribution**[cite: 2] | High[cite: 2] | 3-signal Unsupervised Drift Monitor (Input PSI + Policy Entropy + Action-Rate shift) packaged into every export. |
| **LLM misinterprets critique**[cite: 2] | High[cite: 2] | Always show editable YAML diff + conflict detector + Two-Stage Sanity Preview + 1-click version rollback[cite: 2]. |

---

## 15. Go-to-Market

- **Positioning:** *"Claude Code writes your code. We grow your agent — then you own it, run it locally, and never pay per-call again."*[cite: 2]
- **Target Segments:** (1) Ops/analysts with high-volume recurring triage/monitoring tasks; (2) Small teams sharing internal workflows; (3) Privacy-bound orgs (FinTech, Healthcare, Legal, SecOps) where data cannot leave the boundary[cite: 2].

---

## 16. Go-Ground Roadmap & Success Metrics

- **Phase 0 (Weeks 1–4 — Proof of Loop):** Adapter SDK, Safe Reward DSL, `CSVStreamAdapter`, NL→Config compiler, Warm-Start + SB3 PPO runner, Critique→Diff engine with 2-stage preview, ONNX export, and 3-panel React UI[cite: 2].
- **Phase 1 (Weeks 5–8 — Second Adapter & Drift):** `TextClassifyAdapter` (email/ticket triage), `DriftDetector` v1, and cross-adapter zero-core-change conformance suite[cite: 2].
- **Phase 2 (Weeks 9–16 — Productization):** Run bundle sharing, multi-provider BYOK hardening, team policy registry, design partner onboarding[cite: 2].
- **North Star Metric:** **Policies exported and running in production.**[cite: 2]

---

## 17. Antigravity Execution Blueprint (Phase-by-Phase Build Instructions)

> **DIRECTIVE FOR ANTIGRAVITY (AUTONOMOUS CODING AGENT):**
> You are building this repository phase by phase.
> - **Rule 1:** Execute ONLY one phase at a time.
> - **Rule 2:** Write production-grade, modular, strictly typed code (Python 3.11+ with Pydantic v2 in `backend/`, React 18 + TypeScript + Vite + Tailwind in `frontend/`).
> - **Rule 3:** Every phase has a mandatory `pytest` or build verification gate. Run the tests, fix any errors until 100% pass, and stop for user review before starting the next phase.

---

### 🚀 PHASE 0A: Monorepo Foundation, Adapter SDK, AST-Safe Reward DSL & `CSVStreamAdapter`

**Objective:** Build the core contracts, the sandboxed AST `RewardEvaluator`, and the first reference adapter (`CSVStreamAdapter`) with synthetic data generation. No LLM calls or UI yet.

**Tasks for Antigravity:**
1. **Project Structure:**
   Initialize `backend/pyproject.toml` (using `uv` or `pip`/`venv`) with dependencies:
   `fastapi`, `uvicorn[standard]`, `pydantic>=2.0`, `pyyaml`, `numpy`, `pandas`, `gymnasium`, `stable-baselines3`, `torch`, `onnx`, `onnxruntime`, `scikit-learn`, `httpx`, `pytest`, `pytest-asyncio`.
2. **Adapter Contract (`backend/rl_ide/contracts/adapter.py`):**
   Implement `ObservationSpec`, `ActionSpec`, `StepResult`, `EpisodeLog`, and the `DomainAdapter` protocol exactly as specified in Section 7.
3. **Reward DSL & AST Evaluator (`backend/rl_ide/reward/dsl.py` & `evaluator.py`):**
   - Parse the YAML `RewardDSL` model (`horizon`, `reward.terms`, `reward.pinned_episodes`, `aggregate`, `clamp`)[cite: 2].
   - Implement `RewardValidator`: parses `when` and `value` strings with Python `ast.parse(..., mode='eval')`, enforces the AST node whitelist (rejecting `ast.Call`, `ast.Import`, dunders, etc.), and verifies every `namespace.field` reference against `adapter.event_schema()`[cite: 2].
   - Implement `RewardEvaluator.evaluate(episode_id, events) -> tuple[float, dict[str, float]]` returning `(total_clamped_reward, per_term_breakdown)`.
   - Implement `RewardEvaluator.rescore_episodes(episodes: list[EpisodeLog])` to retroactively recompute rewards across historical episodes in memory.
4. **Reference Adapter 1 (`backend/rl_ide/adapters/stream_adapter.py`):**
   - Implement `CSVStreamAdapter` for financial ticker / sensor anomaly detection[cite: 2].
   - Build a deterministic synthetic dataset generator (`generate_synthetic_stream(n_rows=1000, seed=42)`) producing columns: `timestamp`, `price`, `return_1m`, `volume`, `volatility`, `rolling_vol_90p`, `z_score`, and `true_anomaly` (where high volatility spikes are normal unless accompanied by extreme volume/z-score anomalies).
   - Track internal MDP state (`cooldown`, `steps_since_last_alert`, `rolling_alert_rate`) inside `events["state"]`.
5. **Verification Suite (`backend/tests/test_phase_0a.py`):**
   - Test 1: Valid YAML DSL evaluates accurately and returns per-term breakdown.
   - Test 2: Undefined variable (e.g., `obs.non_existent_field > 5`) raises `RewardValidationError` during static check[cite: 2].
   - Test 3: Malicious code injection (e.g., `__import__('os').system('ls')`) raises `RewardSecurityError`.
   - Test 4: `CSVStreamAdapter` resets, steps through an episode, and `rescore_episodes()` re-evaluates 100 logged steps in `<10ms` when a term weight changes.

**Completion Gate 0A:** Run `pytest backend/tests/test_phase_0a.py` and confirm 100% pass.

---

### 🚀 PHASE 0B: BYOK LLM Compatibility Layer, `NLToConfigCompiler` & Versioned `TaskStateStore`

**Objective:** Build Loop 1 (Authoring)[cite: 2] and the versioned persistence engine so tasks and reward diffs can be compiled from English and rolled back at any time[cite: 2].

**Tasks for Antigravity:**
1. **BYOK LLM Client (`backend/rl_ide/compiler/llm_client.py`):**
   - Create `LLMClient` supporting OpenAI API, Anthropic API, and local Ollama/vLLM OpenAI-compatible endpoints via user-configured keys/URLs[cite: 2].
   - Implement `MockRulesLLMClient` as the default fallback when no API key is provided: it uses deterministic keyword/pattern synthesis to generate valid `TaskConfig` and `RewardDiffProposal` objects so all automated tests and local demos work offline at zero cost.
2. **NL→Config Compiler (`backend/rl_ide/compiler/nl_to_config.py`):**
   - Takes `user_prompt: str` and `adapter_name: str`[cite: 2].
   - Inspects the adapter's `event_schema()`, `observations()`, and `actions()`[cite: 2].
   - Emits a validated `TaskConfig` containing adapter config, hyperparameters, and initial `RewardDSL`[cite: 2].
   - Runs `RewardValidator` automatically; if validation fails, feeds the error back to the LLM for up to 2 automatic repair attempts.
3. **Versioned State Store (`backend/rl_ide/state/store.py`):**
   - Backed by SQLite + a local `runs/<task_id>/` directory.
   - Stores versioned snapshots (`v1`, `v2`, ...): each snapshot records `task_config.yaml`, `reward_dsl.yaml`, `applied_critique`, `policy_checkpoint_path`, and `locked_episode_ids`[cite: 2].
   - Implements `rollback(task_id, target_version)` restoring both the `RewardDSL` and the saved model checkpoint[cite: 2].
4. **Verification Suite (`backend/tests/test_phase_0b.py`):**
   - Test compiling English task prose into a valid `CSVStreamAdapter` config.
   - Test automatic validator retry when an invalid field name is proposed.
   - Test saving 3 sequential DSL versions and rolling back from `v3` to `v1`.

**Completion Gate 0B:** Run `pytest backend/tests/test_phase_0b.py` and confirm 100% pass.

---

### 🚀 PHASE 0C: Streaming Training Runner (Warm-Start + SB3 PPO + Critic Warmup) & Critique→Diff Engine

**Objective:** Build Loop 2 (Training) and Loop 3 (Critique→Diff with Two-Stage Sanity Preview)[cite: 2].

**Tasks for Antigravity:**
1. **Gymnasium Wrapper (`backend/rl_ide/trainer/gym_env.py`):**
   - Implement `AdapterGymEnv(gymnasium.Env)` wrapping any `DomainAdapter` and `RewardEvaluator`[cite: 2].
   - Support hot-swapping `RewardEvaluator` between bursts without re-initializing the environment.
2. **Training Runner (`backend/rl_ide/trainer/runner.py`):**
   - Implement `warmstart_policy(epochs=15)`: fast supervised behavioral cloning pass on initial heuristic rules so Episode 1 starts at ~65%+ accuracy.
   - Implement `run_burst(n_episodes=30, on_episode_callback=...)`: runs SB3 `PPO` (`MlpPolicy`), stores full `EpisodeLog` items in `EpisodeBuffer` (max 200 recent episodes), and streams each completed episode to `on_episode_callback`[cite: 2].
   - Implement `apply_reward_diff(new_dsl)`:
     1. Saves current policy & DSL snapshot to `TaskStateStore`.
     2. Hot-swaps `new_dsl` into `RewardEvaluator` and re-scores all episodes in `EpisodeBuffer`.
     3. **Critic Warmup:** Trains only the PPO value network (`vf`) for 15 mini-epochs on the re-scored buffer so value predictions match the new reward scale.
3. **Critique→Diff Engine (`backend/rl_ide/compiler/critique_engine.py`):**
   - Input: `current_dsl`, `target_episode` (clicked by user), `recent_episodes`, `locked_episodes`, `critique_text`, `past_diffs`[cite: 2].
   - Output: `RewardDiffProposal` containing:
     - `new_dsl` YAML and unified text diff[cite: 2].
     - `rationale`: 1-line explanation per change[cite: 2].
     - `conflicts`: list of any detected conflicts with existing terms or locked episodes[cite: 2].
     - `stage1_counterfactual_preview`: before/after reward scores on `target_episode` and the last 20 episodes.
     - `stage2_forked_behavior_preview`: clones policy in memory, runs a 2-second speculative micro-update on the re-scored buffer, and reports how many actions flip in the desired direction (and whether `target_episode` flips).
4. **Verification Suite (`backend/tests/test_phase_0c.py`):**
   - Run warm-start + Burst 1 on `CSVStreamAdapter` (where initial reward over-flags volatility spikes as anomalies).
   - Submit critique: *"It is flagging normal volatility as anomalies; penalize false positives when volatility is below rolling_vol_90p"*[cite: 2].
   - Verify `CritiqueEngine` produces a valid diff, Stage 1 preview drops reward on false-positive episodes, Stage 2 preview flips the action on the critiqued episode, and Burst 2 reduces false-positive rate by >40%.

**Completion Gate 0C:** Run `pytest backend/tests/test_phase_0c.py` and confirm 100% pass.

---

### 🚀 PHASE 0D: ONNX Policy Exporter, Drift Detector & FastAPI + WebSocket Server

**Objective:** Build Loop 4 (Policy Ownership Export + Drift Detection)[cite: 2] and expose all loops over FastAPI REST + WebSockets[cite: 2].

**Tasks for Antigravity:**
1. **Unsupervised Drift Monitor (`backend/rl_ide/drift/monitor.py`):**
   - Implement `fit_baseline(episodes)` -> generates `drift_baseline.json` (feature quantile buckets for PSI, baseline policy entropy mean/std, and baseline action distribution).
   - Implement `check_drift(obs_batch, action_probs_batch)` -> returns `DriftReport` (`is_drifted: bool`, `max_psi: float`, `entropy_zscore: float`, `action_shift_zscore: float`, `recommendation: str`).
2. **ONNX Policy Exporter (`backend/rl_ide/export/exporter.py`):**
   - Export the trained SB3 PyTorch policy actor network to `model.onnx` (`opset_version=17`)[cite: 2].
   - Verify numerical parity between PyTorch and `onnxruntime.InferenceSession` across 32 test observations (`np.allclose(..., atol=1e-5)`).
   - Generate a self-contained `.runbundle` folder (and `.zip` archive) containing:
     - `model.onnx`[cite: 2]
     - `task_config.yaml` & `reward_dsl.yaml`[cite: 2]
     - `drift_baseline.json`
     - `run_local.py`: a standalone, zero-LLM, zero-SB3 Python script (requiring only `onnxruntime` and `numpy`) that loads `model.onnx`, runs local inference on new data, and checks `drift_baseline.json`[cite: 2].
3. **FastAPI + WebSocket Server (`backend/rl_ide/api/server.py`):**
   - `GET /api/adapters` — list registered adapters and schemas.
   - `POST /api/tasks/compile` — compile NL description into `TaskConfig` + warm-start policy[cite: 2].
   - `POST /api/train/burst` — trigger an async training burst streaming frames over `/ws/stream`[cite: 2].
   - `POST /api/episodes/{ep_id}/lock` — toggle golden episode lock.
   - `POST /api/critique/preview` — run `CritiqueToDiffEngine` + Two-Stage Preview[cite: 2].
   - `POST /api/critique/apply` — apply confirmed diff, run Critic warmup, and trigger next burst[cite: 2].
   - `POST /api/tasks/rollback` — revert to version `vN`[cite: 2].
   - `POST /api/export` — export `.runbundle` and return download path + parity check report[cite: 2].
   - `WS /ws/stream` — real-time episode frames, reward curve points, and per-term breakdown bars[cite: 2].
4. **Verification Suite (`backend/tests/test_phase_0d.py`):**
   - Test all REST endpoints via `TestClient`, verify ONNX export numerical parity, and execute `subprocess.run(["python", "run_local.py", "--sample"])` inside the exported bundle directory to prove standalone local inference works.

**Completion Gate 0D:** Run `pytest` across all backend tests and confirm 100% pass.

---

### 🚀 PHASE 0E: The Three-Panel IDE Frontend ("The Product in One Screen")

**Objective:** Build the React + TypeScript + Tailwind single-screen workbench (`frontend/`) implementing Section 4[cite: 2].

**Tasks for Antigravity:**
1. **Scaffold Frontend (`frontend/`):**
   Initialize Vite + React + TypeScript + Tailwind CSS + `lucide-react`.
2. **Build the 3-Panel Single-Screen Layout:**[cite: 2]
   - **Header Bar:** App brand (`RL-Native Agent IDE`), active Adapter selector (`CSV Stream` / `Text Triage`), Version badge (`v1`, `v2` with 1-click Rollback dropdown), Drift Status indicator, BYOK Settings modal, and primary CTA `[Export Owned Policy (ONNX)]`[cite: 2].
   - **Left Panel — TASK & REWARD DSL (28% width):**[cite: 2]
     - Natural language task description input + `[Compile & Warm-Start]` button[cite: 2].
     - Editable, syntax-styled `RewardDSL` YAML inspector showing active terms, weights, and `when` conditions[cite: 2].
     - Version history timeline showing past diffs applied[cite: 2].
   - **Center Panel — LIVE AGENT (44% width):**[cite: 2]
     - Playback controls: `[Run 30-Ep Burst]`, `[Pause]`, `[Lock Episode ★]`[cite: 2].
     - **Domain Visualizer:** Interactive time-series chart for `CSVStreamAdapter` (showing price, 90th-percentile volatility band, and agent anomaly flags) + clickable Episode History strip[cite: 2].
     - **Live Reward Curve:** Sparkline tracking episode total rewards across bursts[cite: 2].
     - **Per-Term Reward Breakdown:** Horizontal stacked bar chart showing exact positive/negative contribution of every DSL term for the currently inspected episode[cite: 2].
   - **Right Panel — CRITIQUE & DIFF WORKBENCH (28% width):**[cite: 2]
     - Displays the currently selected Episode (#ID, key telemetry values, action taken)[cite: 2].
     - English critique input box (with 3 quick-critique chips for instant demo testing)[cite: 2].
     - **Interactive Diff Card:** Shows proposed YAML diff (`+` / `~` / `-` terms), 1-line rationale, Conflict warning badge, **Stage 1 Counterfactual Score Delta**, and **Stage 2 Forked Policy Flip Rate**, with `[Apply Diff & Run Burst]` and `[Discard]` buttons[cite: 2].
3. **Export Modal:**
   When `[Export Owned Policy (ONNX)]` is clicked, show the ONNX numerical parity check (`Max diff: 1.2e-7 ✅`), bundle contents (`model.onnx`, `task_config.yaml`, `drift_baseline.json`, `run_local.py`), and copyable CLI command to run inference locally[cite: 2].

**Completion Gate 0E:** Run `npm run build` in `frontend/` with zero TypeScript or bundle errors, and verify full end-to-end UX against the FastAPI backend.

---

### 🚀 PHASE 1: Second Reference Adapter (`TextClassifyAdapter`) & Cross-Domain Conformance

**Objective:** Prove the `DomainAdapter` contract holds across a maximally different domain (Email / Support Ticket Triage) without changing a single line of the core training runner or critique engine[cite: 2].

**Tasks for Antigravity:**
1. **Implement `TextClassifyAdapter` (`backend/rl_ide/adapters/text_classify_adapter.py`):**
   - Domain: Support ticket & email routing (`urgent`, `normal`, `spam`, `escalate`)[cite: 2].
   - Synthetic generator creating 500 realistic records with subject, body text, sender tier (`vip`, `free`, `internal`), `word_count`, `urgency_keyword_count`, ` angry_tone_score`, `has_attachment`, and `queue_load`.
   - Observation pipeline: combines a fast deterministic 64-dim hashed/SVD text embedding with normalized metadata features so SB3 `MlpPolicy` trains in <3 seconds on CPU.
   - Exposes clean human-readable fields (`obs.sender_is_vip`, `obs.word_count`, `obs.urgency_keyword_count`, `obs.angry_tone_score`, `state.queue_load`, `action.label`) in `event_schema()` for the `RewardDSL`[cite: 2].
2. **Frontend Text Episode Renderer:**
   - Update Center Panel to render an Email/Ticket Triage card when `TextClassifyAdapter` is active (displaying sender badge, email text preview, extracted signals, and agent routing badge).
3. **Cross-Domain Conformance Test (`backend/tests/test_adapter_conformance.py`):**
   - Automated parameterized `pytest` suite running the exact same pipeline (`NLToConfigCompiler` → `warmstart` → `run_burst` → `CritiqueToDiffEngine` → `apply_reward_diff` → `PolicyExporter` → `DriftDetector`) across `[CSVStreamAdapter, TextClassifyAdapter]`[cite: 2].
   - Assert zero core orchestrator modifications and 100% ONNX export parity for both adapters[cite: 2].

**Completion Gate 1:** Run `pytest` across the entire repository (`test_phase_0a.py` through `test_adapter_conformance.py`) and `npm run build` in `frontend/`. Confirm 100% green.

---

## Appendix A — Glossary

- **Adapter**: A domain plugin implementing the `DomainAdapter` protocol that emits structured events[cite: 2].
- **Task Config**: The portable specification containing adapter settings, hyperparameters, and the `RewardDSL`[cite: 2].
- **Reward DSL**: A declarative, AST-validated YAML language for reward terms[cite: 2].
- **Reward Diff**: A structured patch to the `RewardDSL` produced by the `CritiqueToDiffEngine`[cite: 2].
- **Two-Stage Preview**: Stage 1 (Counterfactual re-scoring of cached episodes) + Stage 2 (Speculative forked micro-burst showing action flips).
- **Run Bundle (`.runbundle`)**: Self-contained directory with `model.onnx`, `task_config.yaml`, `reward_dsl.yaml`, `drift_baseline.json`, and `run_local.py`[cite: 2].

## Appendix B — What We Will Not Compromise On

1. **The diff is always visible.** No black-box reward changes[cite: 2].
2. **The user confirms every applied change.** No silent updates[cite: 2].
3. **The exported policy runs locally.** No vendor dependency at inference[cite: 2].
4. **BYOK or local models for authoring.** We never touch the per-token meter[cite: 2].
5. **Data never leaves the user's boundary.** No telemetry or episode exfiltration[cite: 1, 2].
6. **The human is in the loop.** Always[cite: 2].

---

## 18. Reality alignment (2026-10-31, `v5.5`)

This document is the *intent* spec. The repository diverged from it in four places during Phases
0–P3. Auditing the spec instead of the code produced a false bill of indictment in the Crossfire
memo (claim 4 in particular), so the divergences are recorded here rather than silently kept.

| Spec says (§) | What is actually built | Consequence for claims |
|---|---|---|
| §8 **Reward DSL**: a declarative YAML language, LLM emits YAML never Python; safety rests on an AST whitelist | **Generated Python** `compute_reward()` rewritten by AST-validated transformation; the safety boundary is the `bwrap` OS sandbox, not the AST filter | The memo's objection "AST filtering is not a security boundary" is correct *about the design* and **closed at runtime** by `security/sandbox.py` (`--unshare-net`, secret stripping, `RL_IDE_REQUIRE_ISOLATION=1`). A declarative evaluator remains an option, not a requirement. |
| §9 **Two-stage preview**: <20 ms counterfactual rescore + forked micro-burst with action flips, shown *before* the user commits | **Not implemented.** The product applies, trains briefly, then either retains or rolls back | The promise changes from "preview then commit" to "commit then independent gate". `verifier_gate.py` supplies the gate; the pre-commit preview is an open UX gap, tracked in `progress.md` §20. |
| §3/§7 **Two reference adapters** (`CSVStreamAdapter`, `TextClassifyAdapter`) behind a `DomainAdapter` protocol | **Universal zero-adapter synthesis**: environments are generated per task from a dataset or a plain-English brief, sandbox-verified, and rendered into a generated arena | The adapter SDK is not the abstraction users see. Cross-domain support is delivered through synthesis + conformance tests (`test_sequential_tasks.py`, `test_universal_agentic_pipeline.py`) instead of two hand-written adapters. |
| §1.5 **Moats**: accumulated configs, critique logs and policies create switching cost | Exportable diffs, readable reward code and standard ONNX make the accumulated artifact **portable** | **The moat claim is withdrawn** (Crossfire claim 5). Retention now rests on production integration, labelled history and retraining cadence; the eval corpus is a *quality* asset, not a lock-in. |

### 18.1 Data boundary, stated precisely

- **Inference** never leaves the boundary: the shipped artifact is ONNX plus a local runner.
- **Authoring** uses BYOK or a local model. A hosted provider receives prompts, and between sandboxed
  rounds it receives **redacted** traces (raw observations replaced by per-dimension ranges, see
  `autopilot/traces.py`).
- Tenants that require boundary-local authoring set `RL_IDE_ALLOW_HOST_LLM=0`; host-side tuning is then
  refused with `auto_tune_host_disabled` and the reward function is retained.

### 18.2 Phase P3 deliverables (see `PHASE_P3_V5_5_REFINED_EXECUTION_PLAN.md`)

`crossfire_protocol.py` (pre-registration guard), `eval_corpus.py` (golden regression vault),
`verifier_gate.py` (paired gate + deterministic triage), `traces.py` (redaction),
`ingest/interaction_log.py` (passive harvest), per-round sandbox orchestration, best-on-val
selection, and the two-basis economics ledger.

Operator-facing usage of these controls (endpoints, payloads, switches, troubleshooting) is in
`CROSSFIRE_CONTROLS_USER_GUIDE.md`; the UI that exposes them is specified in
`TRUST_DRAWER_SPEC_AND_IMPLEMENTATION.md`.
