# EXECUTIVE DECISION MEMO
**Date:** October 30, 2026  
**Subject:** Stress-Test Analysis: "i am building and IDE./harness for agents by reinforcement learning simply by in vibe coding."  
**Context:** Attached Documents / OCR Transcripts:
[Notes: MASTER_SPEC_AND_BUILD_PLAN.md]
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

## 1. Verdict

> **Survives only as scoped MVP**  
> The surviving mechanism is a single-task critique-to-reward-diff training loop that keeps a short burst only when an independent executable verifier passes, then exports the compact policy via ONNX.
>
> After evaluation, we found that all five claims were weakened rather than refuted, leaving the harness viable only with named scope cuts. The main errors are that 'train once' hides ongoing reward specification, verifier maintenance and drift monitoring while API pricing falls; the fit gain and critique-to-reward compile lack a disjoint held-out test; and the six-subsystem build is gated by security review, a named ML owner and customer IT approvals rather than code volume. Exportability also inverts the artifact moat, so retention must rest on a validated eval corpus rather than configs or…

### What decided it
- User must produce and verify a few hundred correctness examples before the first policy beats a frontier model; deadline-pressed developers skip this and keep the API. [unverified]

### Before you commit
- Choose one recurring task that already has an executable verifier (unit-tested code, SQL against a fixed schema, or format-constrained extraction), compute break-even episode volume net of verifier maintenance and drift monitoring, and record the payback threshold before any training run. _(claim 1, 2)_
- Run the falsification test on that task: collect ~300 examples plus >=100 disjoint held-out critiques authored before training, pre-register the rubric and threshold, train a compact open-weight policy, and retain reward diffs only when the independent verifier passes; treat the resulting validated eval corpus, not exported configs, as the retention asset. _(claim 2, 3, 5)_
- Cut the build to one named ML/security owner: allowlisted reward-expression evaluator in a subprocess/container with no host secrets, one training backend (PPO or BC), single pilot adapter, NL-to-config compiler, and ONNX export; defer multi-customer adapters and drift detection until IT approval and production traffic exist. _(claim 4)_

### Scoreboard
- **Total Assumptions Audited:** 5
- **Survived:** 0
- **Weakened:** 5
- **Broken:** 0
- **Unresolved:** 0

---

## 2. What the idea actually is

### market premise
- There is a sufficiently large class of recurring, well-defined knowledge-work tasks with observable or verifiable correctness that users would rather train a policy once than call a frontier model repeatedly. _(claim 1)_

### small policy beats frontier
- A compact policy trained locally on a few hundred examples of a user's definition of correctness outperforms a general frontier model on that specific narrow task on cost, latency and fit. _(claim 2)_

### critique-to-reward diff
- A non-expert user's plain-English critique can be compiled into reward-function diffs accurate enough that repeated short training bursts converge on the behavior the user intended, without the user writing code. _(claim 3)_

### vibe-coded build
- The full IDE harness — adapter SDK, NL-to-config compiler, AST-sandboxed reward DSL, PPO/behavioral-cloning training orchestrator, ONNX export and drift detection — can be built mainly through AI-assisted vibe coding by a small team in a realistic timeframe. _(claim 4)_

### switching-cost moat
- Accumulated task configs, critique logs and trained policies create switching costs high enough that users stay rather than returning to frontier-model tools. _(claim 5)_

**What the terms meant:**  
- **frontier model** — A top-end general-purpose LLM accessed per token via API, such as Claude or GPT-class models.
- **IDE** — Integrated development environment: the desktop/web application with Task, Live Agent and Critique panels.
- **harness** — An orchestration and evaluation scaffold that runs agent episodes, adapters, training and export around a model.
- **vibe coding** — Building software primarily by describing intent to an AI coding assistant and accepting generated code, rather than hand-writing it.
- **PPO** — Proximal Policy Optimization, the standard on-policy RL algorithm from Stable-Baselines3 or CleanRL.

---

## 3. What survives, what dies

| # | Claim | Status | Why |
| --- | --- | --- | --- |
| 1 | There is a sufficiently large class of recurring, well-defined knowledge-work tasks with observable or verifiable correctness that users would rather train a policy once than call a frontier model repeatedly. | WEAKENED (CONTESTED) | The claim conflates 'train one policy' with 'eliminate recurring human effort': the reward specification, episode labeling, verifier maintenance and drift monitoring that the qualifying task class requires are continuing costs, so the 'train once vs. call repeatedly' comparison misstates both sides of the ledger — and the repeated-call baseline is a per-token meter whose price is falling, further compressing the amortization window the claim never states. |
| 2 | A compact policy trained locally on a few hundred examples of a user's definition of correctness outperforms a general frontier model on that specific narrow task on cost, latency and fit. | WEAKENED (CONTESTED) | 'Fit' superiority is measured against the training signal itself: absent a held-out critique set and a break-even run count, convergence on a few hundred self-authored examples cannot separate learning the task from memorizing the user's stated words — and those few hundred verified critiques are an upfront cost incurred before any of the three claimed gains exist. |
| 3 | A non-expert user's plain-English critique can be compiled into reward-function diffs accurate enough that repeated short training bursts converge on the behavior the user intended, without the user writing code. | WEAKENED (CONTESTED) | There is no verifier independent of the critic-derived reward model: repeated optimization against it hacks the reward within a few bursts, and because 'accurate enough' is defined by the very convergence it is supposed to measure, the claim cannot distinguish a bad compile from a vague critique — the success criterion is circular and the orthogonality of reward terms it silently depends on is never established. |
| 4 | The full IDE harness — adapter SDK, NL-to-config compiler, AST-sandboxed reward DSL, PPO/behavioral-cloning training orchestrator, ONNX export and drift detection — can be built mainly through AI-assisted vibe coding by a small team in a realistic timeframe. | WEAKENED (CONTESTED) | 'Mainly vibe coding' targets code volume, but every named blocker on the six-subsystem scope is a review, isolation, or third-party-approval gate: the reward DSL's security boundary cannot rest on AST filtering, the PPO/behavioral-cloning loop needs a named ML owner's validation, each adapter is gated by that customer's IT, and drift detection requires production traffic absent at build time. |
| 5 | Accumulated task configs, critique logs and trained policies create switching costs high enough that users stay rather than returning to frontier-model tools. | WEAKENED (CONTESTED) | The claim's lock-in mechanism depends on artifacts being non-portable, but the product's own transparency posture (§1.6 exportable diffs, visible reward DSL, standard PPO/ONNX exports) makes the accumulated policy re-derivable in-context by a frontier model — so switching cost is lowest for exactly the users who have accumulated the most, inverting the retention claim; and per Operator the confound is that the cost that does exist is deferred behind reward-DSL authoring and episode review that many users never complete. |

---

## 4. The load-bearing kills

#### Claim 2. [WEAKENED (CONTESTED)] A compact policy trained locally on a few hundred examples of a user's definition of correctness outperforms a general frontier model on that specific narrow task on cost, latency and fit.
- **Impact Level:** HIGH

**Strategic Importance:**  
Core value theorem: if a compact local policy does not beat frontier models on narrow-task cost, latency, and fit, the economic rationale for the RL-native IDE weakens enough to force a pivot.

**Contradictions & Counterarguments:**  
> **Counter-evidence:** User must produce and verify a few hundred correctness examples before the first policy beats a frontier model; deadline-pressed developers skip this and keep the API. [unverified]
> **Counter-evidence:** The superiority claim on 'fit' is measured against the training signal itself; convergence on a few hundred self-authored critiques cannot distinguish a policy that learned the task from one that learned the user's phrasing, and no held-out evaluation or payback-volume threshold is specified.

**What to change:**  
> Don't ship the RL-trained compact policy as the default path that users must hand-label their way into; instead make the harness passively harvest the critiques users already emit while they use a frontier model (accepts/rejects, edit diffs, review comments), and only train+downgrade to the local policy once ~300 such signals exist — with the 'fit' claim gated on a pre-registered rubric scored against a held-out critique set authored before training. The superiority claim narrows from 'beats frontier on your narrow task' to 'beats frontier on a task where you already produce critiques', which…

**Next Validation Experiment (Smallest Real-World Test):**  
> **Next validation:** This week: take one developer's existing review comments/commit edits for a single narrow task (e.g. writing tests for one file), train one LoRA on ~300 of them, and score it against 100 comments withheld and rubric-frozen before the run, alongside the frontier model at the listed token prices — logging wall-clock latency and calls-to-payback for each.

**Stop if:** Stop if the compact policy fails to win on at least 60 of the 100 held-out critiques (i.e.

**Steel Man Re-Architecture (Break to Rebuild):**  
> **Fatal Flaw:** 'Fit' superiority is measured against the training signal itself: absent a held-out critique set and a break-even run count, convergence on a few hundred self-authored examples cannot separate learning the task from memorizing the user's stated words — and those few hundred verified critiques are an upfront cost incurred before any of the three claimed gains exist.  
> **Salvaged Claim:** Restrict the decision to a task where the user already produces critiques in the course of work; train a compact open-weight policy (LoRA/DPO) on ~300 examples and score it ONLY against a disjoint held-out set of >=100 critiques authored before training began, under a rubric pre-registered before the run and with a stated break-even threshold (payback within N inference calls / M weeks of saved frontier-API spend). Cost and latency may be claimed unconditionally on the measured token pricing; fit may be claimed only if the held-out margin clears the pre-set threshold, and the claim is void…  
> **Trade-off Acknowledged:** The fit advantage becomes conditional and post-hoc rather than definitional, and the addressable case narrows to teams that already emit labeled critiques — cold-start users are excluded outright, and Operator's labeling/retraining burden is budgeted rather than removed, so the fix buys falsifiability at the cost of a smaller deployable market and an extra ~100-example eval budget.  

#### Claim 3. [WEAKENED (CONTESTED)] A non-expert user's plain-English critique can be compiled into reward-function diffs accurate enough that repeated short training bursts converge on the behavior the user intended, without the user writing code.
- **Impact Level:** HIGH

**Strategic Importance:**  
Core product loop: the non-expert English-critique-to-reward-diff mechanism is the key differentiator; if it cannot reliably converge without code, the UX and target user materially change.

**Contradictions & Counterarguments:**  
> **Counter-evidence:** The compiler's fidelity is validated only by whether behavior converges, so the claim cannot distinguish 'the critique was compiled badly' from 'the user's critique was vague' — the central premise is unfalsifiable as written.
> **Counter-evidence:** Failed training bursts give the non-expert no way to tell whether the critique, the compiler, or RL variance is at fault, so every miss costs hours of manual episode review before the next attempt. [unverified]
> **Counter-evidence:** [Day-1000 Blocker] Repeated optimization against a critic-generated reward model with no independent verifier hacks the reward within a few bursts; MVP survives only where executable checks or user output rankings bound it. [unverified]

**What to change:**  
> Ship only a verifier-gated loop: each short training burst must pass an independent check the user never critiques (existing test suite, simulator assertion, or blind A/B ranking of outputs) before its reward diff is kept, and failing bursts auto-roll back with the verifier log used to triage whether the critique, the compiler, or variance caused the miss — instead of marketing the product as converging on 'the behavior the user intended' from prose alone.

**Next Validation Experiment (Smallest Real-World Test):**  
> **Next validation:** This week, take 10 real tasks from 2–3 prospective users' own repos that already have runnable tests, and for each run the full loop — plain-English critique → reward diff → one short LoRA burst → run the tests plus a previously-passing test — counting how many bursts pass without the human hand-editing the reward diff.

**Stop if:** Stop if fewer than 6 of 10 bursts pass their verifier within two critiques, or if more than 2 of 10 kept reward diffs regress a previously passing test (cross-term interference), or if any user declines to supply the blind A/B rankings when no test exists.

**Steel Man Re-Architecture (Break to Rebuild):**  
> **Fatal Flaw:** There is no verifier independent of the critic-derived reward model: repeated optimization against it hacks the reward within a few bursts, and because 'accurate enough' is defined by the very convergence it is supposed to measure, the claim cannot distinguish a bad compile from a vague critique — the success criterion is circular and the orthogonality of reward terms it silently depends on is never established.  
> **Salvaged Claim:** Plain-English critique is compiled into a diffable reward-diff that the user accepts or rejects as text (no code authored by the user), but only for tasks that already carry an independent verifier — an existing test suite, a simulator check, or a held-out A/B ranking of candidate outputs the user never critiques. Each short training burst must pass that verifier before its reward diff is retained; failing bursts are rolled back, and the failed critique is triaged automatically against the verifier log instead of being charged to the user's phrasing.  
> **Trade-off Acknowledged:** Scope collapses from 'the behavior the user intended' to 'behaviors that already have an oracle', so the no-code promise holds only where someone else wrote the checks or where the user pays a per-burst ranking cost — and the user still supplies judgments, just as rankings rather than prose. Frequent verifier-ambiguous aborts replace silent reward drift, which is cheaper per failure but visibly slower and more interrupt-driven than the 30-second diff the operator benchmarks against, and the untrusted-critique injection path into the reward-spec prompt still needs hardening.  

#### Claim 4. [WEAKENED (CONTESTED)] The full IDE harness — adapter SDK, NL-to-config compiler, AST-sandboxed reward DSL, PPO/behavioral-cloning training orchestrator, ONNX export and drift detection — can be built mainly through AI-assisted vibe coding by a small team in a realistic timeframe.
- **Impact Level:** HIGH

**Strategic Importance:**  
Most load-bearing: the proposal is specifically to build the harness via AI-assisted vibe coding; if a small team cannot realistically deliver this full stack that way, the build decision, scope, and staffing change immediately.

**Contradictions & Counterarguments:**  
> **Counter-evidence:** The claim's method (fast code generation) targets the cheap axis while its own evidence shows the expensive axis was specification revision — v2.0 'Hardened' — and it depends on a reviewer competence it simultaneously devalues. [unverified]
> **Counter-evidence:** Per-customer adapter integration requires that customer's IT approval and cannot be vibe-coded in advance; unreviewed AI code for the reward DSL and training loop adds manual validation load the small-team timeline omits. [unverified]
> **Counter-evidence:** [Day-1 Blocker] Reward DSL must not rely on AST filtering alone; untrusted expressions need subprocess/container isolation with no host credentials. [unverified]

**What to change:**  
> Cut the build to a scoped MVP — a declarative, allowlisted reward-expression evaluator running in a subprocess/container with no host secrets, one training backend (PPO or BC, not both), the NL-to-config compiler, ONNX export, and the adapter SDK against a single consenting pilot customer — and put a named senior ML/security owner with budgeted review hours on the schedule, since the thing that failed is not code volume but the review, isolation, and third-party-approval gates that 'vibe coding' does not remove. Defer drift detection and multi-customer adapters until a pilot's IT approval is…

**Next Validation Experiment (Smallest Real-World Test):**  
> **Next validation:** This week, have one engineer who did not write the code spend two days attacking the reward-expression filter on the host-credentialed build with 10 hostile expressions (dunder/attribute traversal, imports, infinite loops, memory exhaustion), and in the same week get a written yes/no plus a date from the pilot customer's IT lead on adapter integration.

**Stop if:** Stop if even one of the 10 hostile expressions escapes the filter into host execution or reaches host credentials — one is fatal — or if the pilot customer's IT lead cannot give a written approval date within two weeks.

**Steel Man Re-Architecture (Break to Rebuild):**  
> **Fatal Flaw:** 'Mainly vibe coding' targets code volume, but every named blocker on the six-subsystem scope is a review, isolation, or third-party-approval gate: the reward DSL's security boundary cannot rest on AST filtering, the PPO/behavioral-cloning loop needs a named ML owner's validation, each adapter is gated by that customer's IT, and drift detection requires production traffic absent at build time.  
> **Salvaged Claim:** Build a scoped MVP by AI-assisted coding under one named ML/security owner: a declarative, allowlisted reward-expression evaluator running in a subprocess/container with no host secrets, one training backend (PPO or BC, not both), the adapter SDK against a single internal/consenting pilot customer, the NL-to-config compiler, and ONNX export. Defer multi-customer adapters and drift detection until a pilot customer's IT approval is signed and production traffic exists to define drift against.  
> **Trade-off Acknowledged:** Ships something real in the timeframe but is not the full six-subsystem harness: drift detection is unvalidated until traffic exists, adapters scale only at the speed of customer IT approvals the team does not control, and adding a senior ML/security reviewer consumes budget and calendar that the 'vibe coding' premise assumed you would not spend — the deep-and-narrow constraint in §1.4 wins over parallel subsystem delivery.  

---

## 5. The version I'd build

> A scoped IDE/harness for one recurring task with an existing executable verifier. The user supplies plain-English critique; the system compiles it into a diffable reward-expression that the user accepts or rejects as text, then runs a short training burst for a compact open-weight policy using one backend (PPO or BC, not both).

- **What it omits:** The full six-subsystem harness and the 'train once eliminates recurring effort' framing are omitted: multi-customer adapters and drift detection are deferred because customer IT approval and production traffic do not exist at build time; dual training backends are cut to one because the PPO/behavioral-cloning loop needs a named ML owner's validation; AST-only sandboxing is not used as the security boundary because it cannot carry that boundary; and the artifact-accumulation moat is dropped because exportable diffs, visible reward DSL and standard ONNX exports make the policy re-derivable by a…
- **Demo path:** Demo on one pilot task that already has a verifier: an existing test suite, simulator check, or held-out A/B ranking. Show the user writing a plain-English critique, the compiler emitting a text reward-diff, the user accepting or rejecting it, a short training burst passing or failing the independent verifier, rollback of failed diffs, and ONNX export.

---

## 6. Cheapest next experiment

> Take one recurring task where the user already produces critiques and where an executable verifier exists. Collect ~300 training examples and >=100 disjoint held-out critiques authored before training; pre-register the rubric and a break-even threshold (payback within N inference calls or M weeks of saved frontier-API spend); train the compact policy and compare held-out pass rate, cost and latency against the frontier-model baseline.

---

## 7. Sources

- ["LLM API Pricing Trends & Updates (October 2026) | BenchLM.ai"](https://benchlm.ai/llm-pricing-trends) _(verified)_: "LLM API Pricing Trends As of October 29, 2026, the frontier token price index is 15, 85 % below its March 2023 base. The median flagship model costs $ 5.63 per million blended tokens across 37 active constituents, and 5 entries, exits, or price changes landed in the latest snapshot."
- ["LLM API Pricing Comparison & Calculator (October 2026)"](https://benchlm.ai/llm-pricing) _(verified)_: "The cheapest production-grade model (score 70+) is GPT-6.1 Terra at $1.00/$5.00."
- ["Local Police Departments, Procedures, Policies, and Technology, 2020 ..."](https://bjs.ojp.gov/library/publications/local-police-departments-procedures-policies-and-technology-2020-statistical) _(verified)_: "This report provides data on authorized equipment and techniques, body-worn cameras, and K-9 units in local police departments. It also presents tables on training, policies, and procedures. Additionally, the report describes the prevalence of community policing plans."
- ["[2607.29516] From Code Review to Code Critique: Intent, Drift, and ..."](https://arxiv.org/abs/2607.29516) _(verified)_: "AI coding agents are generating code at volumes that exceed the capacity of traditional peer review. At the same time, existing AI code review tools over-index on low-value suggestions such as style and best practices while under-indexing on the concerns human reviewers prioritize most: correctness, security, and performance. We present ARCTIC, an AI-powered Code Critique system that reframes ..."
- ["GitHub - nurettincoban/diff2ai: Turn Git diffs into high-signal AI code ..."](https://github.com/nurettincoban/diff2ai) _(verified)_: "Pasting a raw diff into a chat window gets you noisy, unstructured feedback with made-up line numbers. diff2ai builds a focused prompt from your actual change — line-numbered diff, commit messages, file summary — with a strict output schema, so the review you get back has severities, exact file/line references, and concrete fixes. Then it can run the AI for you, cross-check several ..."
- ["Sup4ikX/smart-diff-explainer - GitHub"](https://github.com/Sup4ikX/smart-diff-explainer) _(verified)_: "Smart Diff Explainer A skill for AI coding assistants — takes a git diff and turns it into a plain-English explanation of what changed and why."
- ["Agent Lightning: Lightweight RL agent-training framework"](https://www.microsoft.com/en-us/research/blog/agent-lightning-v1-0-a-3500-line-lightweight-agentic-rl-framework-for-training-agents-with-real-harnesses/) _(verified)_: "At a glance Harnessed Agentic RL: Microsoft Research Asia introduces a training paradigm in which the same agent harness used in deployment participates directly in reinforcement learning, removing the need to reimplement the agent inside the training framework."
- ["Data and statistics | USPTO"](https://www.uspto.gov/learning-and-resources/data-and-statistics) _(verified)_: "It does not include the legacy patent statistic reports known as the Patent Technology Monitoring Team (PTMT) Reports and Technology Assessment and Forecast (TAF) Reports. If you're looking for the metrics and data found in these reports, please visit the IP5 statistic reports and the WIPO IP Statistics Data Center ."
- ["AgentFlow/assets/doc/logs.md at main · lupantech/AgentFlow"](https://github.com/lupantech/AgentFlow/blob/main/assets/doc/logs.md) _(verified)_: "During training, training output logs are automatically saved with IP-based organization:"
- ["Log Operations — Writeup | by CyberWithManos - Medium"](https://medium.com/@cyberwithmanos/tryhackme-log-operations-writeup-0d476ddcc0b5) _(verified)_: "Answer: "Debug" Task 3: Where To Start After Deciding the Logging Purpose Q1: You are a consultant working for a growing startup."
- ["Create diagnostic settings at scale by using built-in Azure policies"](https://learn.microsoft.com/en-us/azure/azure-monitor/data-collection/diagnostic-settings-policy-built-in) _(verified)_: "This article describes how to use a set of built-in policies to direct resource logs for supported resources to Log Analytics workspaces, event hubs, and storage accounts. To create a custom policy definition for a resource type that doesn't have a built-in policy, see Create diagnostic settings at scale by using custom Azure policies."
- ["Use the Policy Configuration Agent to create policies - Microsoft ..."](https://learn.microsoft.com/en-us/intune/copilot/agents/manage-policy-configuration-agent) _(verified)_: "Learn how to use the Policy Configuration Agent in Microsoft Intune to create policies. You can create policies on files you uploaded, natural language prompts, or industry baselines."

---

*Generated via Crossfire Decision Testing Memo: Autonomous Adversarial Stress Testing*
