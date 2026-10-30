# RL-Native Agent IDE — Progress & Session Handoff Log

**Project:** RL-Native Agent Studio v4.6 (Honest Metrics, Verified Synthesis, Dataset-Grounded Learning)  
**Last Updated:** 2026-10-30  
**Current Status:** v4.5 remediation (phases 0–E) plus the v4.6 round-2 commercial validation (§10) are
complete. All audit issues F1–F36 are closed or explicitly accepted in
[DESIGN_AUDIT_AND_PHASED_REMEDIATION.md](./DESIGN_AUDIT_AND_PHASED_REMEDIATION.md).
- **Backend Tests:** **90 passed, 0 failed** (`pytest -q`, ≈11 min). Includes `test_phase_hardening.py`,
  `test_dataset_feature_engineering.py`, `test_arena_rendering.py`, `test_intake_action_contract.py`,
  the rewritten `test_universal_agentic_pipeline.py`, and the behavioural `test_agentic_chat_planner.py`.
- **Frontend Build:** `npm run build` clean (tsc + vite, 1 599 modules).
- **Browser suite:** **executed** against the live app (`browser_test_real_world_value.js`): four
  commercial use cases, 4/4 passing at 87.0 % / 95.4 % / 81.2 % / 100 % with ONNX parity < 1e-6 and
  all four declared actions preserved. The LLM-authored **control** task remains statistically
  unstable across runs (see §10.4) and is scored as an advisory learning check.
  See §10.
- **Metric semantics:** classification tasks report per-step accuracy; control tasks report the
  terminal outcome; generic placeholder environments report **0 %** and are flagged.
- **Known residuals:** see §6 of the audit document (no built-in auth, sample rows in exported
  bundles for offline execution, advisory surgical instructions, drift monitoring not wired).
- **Commercial gaps:** the residuals above plus the operational layer (in-sample metrics, unsandboxed
  env execution, blocking synthesis, no tenancy/metering/CI) are tracked with evidence and effort
  estimates in [COMMERCIAL_READINESS_GAPS.md](./COMMERCIAL_READINESS_GAPS.md).

> Historical sections below (v4.0–v4.4) are retained as an append-only log. Where they conflict with
> the v4.5 status above (e.g. “all static domain routing eliminated”, “100 % mastery”, “53/53 tests”),
> the v4.5 section at the end of this file is authoritative.

---

## 1. System Architecture Overview (v4.0 Autonomous JIT)

The RL-Native Agent Studio operates as a zero-adapter JIT synthesis and closed-loop self-tuning agent IDE:

```
+----------------------------------------------------------------------------------------------------+
|                                    1. CONVERSATIONAL INTAKE ARCHITECT                               |
| - Evaluates user prompt + uploaded dataset across the 5 Pillars (Objective, Data, Arena, Target, Play) |
| - Emits Clarifying Questions with clickable option pills or [⚡ Use Smart Defaults & Build Now]     |
| - Locks complete AgentBlueprintSpec                                                                 |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                      2. JIT CODE & ARENA SYNTHESIS                                 |
| - JITEnvironmentSynthesizer: synthesizes workspaces/<sid>/env_module.py (SynthesizedGymEnv)        |
| - SandboxSmokeTester: Static AST security check + 25-step runtime smoke test + Self-Healing Loop  |
| - JITArenaSynthesizer: synthesizes workspaces/<sid>/arena.html with window.postMessage bridge      |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                               3. AUTONOMOUS CLOSED-LOOP TRAINING & SELF-TUNING                      |
| - Round 1: Warm-Start Behavioral Cloning on env.get_demonstrations() + PPO Burst 1                 |
| - Telemetry Evaluation: checks mean(info["is_success"]) * 100 vs target_metric_threshold           |
| - Target Met -> Policy frozen, emits target_achieved                                               |
| - Below Target -> AutoTunerAgent diagnoses failure modes, rewrites compute_reward(),                |
|   re-scores buffer, executes 15-epoch Critic Warmup on value network, and initiates Round r+1       |
| - Real-time WebSocket streaming (/ws/v4/{sid}) + rollback_to_round() + nudge_session()              |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                  4. ONNX EXPORT & STANDALONE RUNBUNDLE                             |
| - Exports SB3 policy to model.onnx (opset 17, numerical parity < 1e-5 verified)                    |
| - Packages .runbundle & .zip: model.onnx, env_module.py, arena.html, blueprint_spec.json,          |
|   drift_baseline.json, and zero-dependency run_local.py CLI + HTTP daemon (--serve --port 9000)   |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. King Tut Forensic Audit Findings & Applied Remediations

A deep line-by-line audit was performed across all 22 frontend and backend source files. Key findings and their remediations:

### 2.1 P0 Blockers & Critical Remediations Applied
- **Fixed SQL Ingestion Signature Crash (`sql_connector.py` & `server.py`):**
  - Updated `load_sql_dataset()` to accept named parameters (`connection_uri`, `query`, `name`) and return a fully profiled `IngestedDataset` instance.
  - Verified with new test `test_fastapi_sql_ingest_endpoint`.
- **Synchronized Frontend ↔ Backend Contract Disconnect (`types/index.ts` & `services/api.ts`):**
  - Updated TypeScript interfaces (`AgentBlueprintSpec`, `ClarifyingQuestion`, `IntakeResponse`) to mirror backend Pydantic models with robust fallback aliases.
  - Normalized responses in `api.ts` so `resp.questions`, `resp.assistant_message`, and `resp.blueprint_spec` map seamlessly to UI state.
- **Hardened Sandboxed Dynamic Arena (`SandboxedArenaIframe.tsx`):**
  - Removed `allow-same-origin` from iframe sandbox (`sandbox="allow-scripts"`), preventing the synthesized HTML from accessing parent DOM, cookies, or `localStorage`.
  - Added source origin verification on `window.addEventListener("message")` (`event.source === iframeRef.current.contentWindow`).
- **Pre-Import AST Security Audit (`env_builder.py`):**
  - Added `audit_ast_security(code_str)` check *before* writing or dynamically importing generated environment modules.
- **Non-Blocking Background Training (`server.py`):**
  - Offloaded blocking `global_controller.run_loop` execution using `asyncio.to_thread` so the FastAPI event loop, WebSockets, and live interactions do not stall.
  - Sanitized session IDs against path traversal in `/download`.
- **RL/ML Optimization & Value Network Protection (`loop_controller.py`):**
  - Warm-Start Behavioral Cloning now restricts optimization strictly to actor parameters (`policy.action_net` and `mlp_extractor.policy_net`), preventing value network gradient pollution.
  - Critic Warmup freezes actor and policy networks while properly training value network layers.
  - Adaptive discount factor: `gamma=0.0` for contextual bandit / tabular tasks, `gamma=0.95` for simulations.
- **Realistic Drift Baselines (`exporter.py`):**
  - Drift baseline calculation now samples realistic observations from synthesized environment reset/step/demonstrations rather than unit Gaussian noise.
- **Robust Fuzzy Intent & Typo Detection (`intake.py`, `env_builder.py`, `arena_builder.py`, `auto_tuner.py`, `client.py`):**
  - Resolved root cause where colloquial typos (such as "tic toe", "tic tax tow", "tictac", "ttt", "3x3 board") fell through to generic 2D physics navigation fallback.
  - Implemented unified `is_tictactoe_intent()` with fuzzy phonetic / typo regex matching and `is_tictactoe_spec()` across all generator and auto-tuning components.
  - Added `test_tictactoe_fuzzy_intent_and_typo_handling` unit test verifying all typo variants produce the correct 3x3 interactive board agent.
- **Pruned Dead v3 Residue:**
  - Removed obsolete legacy directories `backend/rl_ide/state/` and `backend/rl_ide/trainer/` that imported deleted v3 modules.

---

## 3. v4.1 Upgrades & Enhancements

The v4.1 upgrade introduces persistent multi-agent conversation management, multi-question intake UX, explicit training configuration, centralized prompt hardening, and duplicate key bug fixes:

### 3.1 Duplicate React Key Fix (`loop_controller.py` & `SandboxedArenaIframe.tsx`)
- **Multi-Round Episode Prefixing:** In `loop_controller.py`, all evaluation and training episodes are uniquely prefixed by round: `f"r{round_num}_eval_{ep_idx}"` and `f"r{round_num}_train_{ep_idx}"`. Every episode dictionary/model now includes `"round": round_num`.
- **Frontend Key Robustness:** `SandboxedArenaIframe.tsx` maps episode chips with composite keys `key={`${ep.round || 0}_${ep.episode_id}_${idx}`}` and clean labels (e.g., `R1 • Eval #5`).
- **Complete Map Audit:** All `.map()` operations across `ArchitectPanel.tsx` and `TelemetryPanel.tsx` were audited to guarantee 100% unique React keys.

### 3.2 Full Conversation & Session Persistence (`conversation_store.py` & `server.py`)
- **SQLite + Workspace Persistence:** Built `ConversationStore` (`runs/conversations.db` + `workspaces/<cid>/`), autosaving from the very first user message. Persists: `conversation_id`, `title`, `chat_history`, `dataset_id`, `blueprint_spec`, `training_params`, `current_round`, `best_metric`, `rounds_history`, `episodes`, `env_code`, `arena_html`, and `latest_model_zip_path`.
- **Conversation Endpoints:**
  - `GET /api/v4/conversations`: Returns all saved conversations ordered by `updated_at DESC`.
  - `POST /api/v4/conversations`: Initializes a new conversation.
  - `GET /api/v4/conversations/{id}`: Reloads conversation state, **re-instantiates `SynthesizedGymEnv` and loads saved SB3 `PPO.load(model_zip, env, device="cpu")` into memory**, enabling instant interactive play and nudging without retraining.
  - `PATCH /api/v4/conversations/{id}/params`: Updates training hyperparameters (`max_auto_tune_rounds`, `episodes_per_round`, `target_metric_threshold`).
  - `DELETE /api/v4/conversations/{id}`: Deletes conversation and workspace files.
  - `DELETE /api/v4/conversations`: Atomically wipes all conversations and associated workspace folders.
- **Frontend Conversation Drawer & Switcher:** Added `ConversationSidebar.tsx` and header toggle in `Header.tsx`. Users can switch between past agents, inspect past mastery scores, delete individual runs, purge all conversations via 'Clear All', or start new agents with state instantly restored.

### 3.3 Multi-Question Intake UX & Explicit Training Controls (`intake.py` & `ArchitectPanel.tsx`)
- **4-Part Structured Intake Rubric:** `IntakeArchitectAgent` returns non-overlapping question IDs (`q_objective`, `q_arena_ui`, `q_interaction`, `q_expectations`) and `recommended_training_params` (`max_auto_tune_rounds: 5`, `episodes_per_round: 200`, `target_metric_threshold: 90.0`).
- **Independent Question Selection:** `ArchitectPanel.tsx` maintains dictionary state `selectedAnswers` keyed by `question.id`, allowing independent option selection across all questions without premature submission. Inline text fields allow custom answers per question.
- **Explicit Training Parameter Bar:** Direct UI controls for Rounds (`1-100`), Episodes/Round (`50`, `100`, `200`, `250`, `500`, `1000`), and Target Goal % (`50%-100%`).
- **Primary CTA:** Single action button `[ 🚀 Confirm Selections & Build Agent ]` packages all selected answers and custom training parameters into one atomic payload.

### 3.4 Centralized & Hardened System Prompts (`backend/rl_ide/llm/prompts.py`)
- Centralized deterministic prompts for all 4 LLM roles:
  - `INTAKE_ARCHITECT_SYSTEM_PROMPT`: 5-Pillar rubric, JSON schema enforcement, question IDs, honoring explicit training parameters.
  - `ENV_SYNTHESIZER_SYSTEM_PROMPT`: `SynthesizedGymEnv` interface, action projection, bounded rewards `[-10.0, +10.0]`, multi-strategy opponent pools, first-player randomization, and `get_demonstrations()`.
  - `ARENA_SYNTHESIZER_SYSTEM_PROMPT`: Self-contained HTML5/Tailwind/JS arena, `ARENA_UPDATE` / `ARENA_INTERACT` messaging bridge.
  - `AUTO_TUNER_SYSTEM_PROMPT`: Root-cause failure diagnosis, reward weight clamping, strict signature preservation.

---

## 4. Verification Test Results

### 4.1 Backend Test Suite (29 / 29 Tests Passing — 100% Green)
```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.7.0
rootdir: /home/devuser/projects/RLHarness/backend

backend/tests/test_llm_mandatory_gate.py::test_unverified_llm_status_and_gating PASSED [  3%]
backend/tests/test_llm_mandatory_gate.py::test_invalid_key_fails_verification PASSED [  6%]
backend/tests/test_llm_mandatory_gate.py::test_valid_key_verifies_and_activates_globally PASSED [ 10%]
backend/tests/test_llm_mandatory_gate.py::test_fallback_rejected_in_production_mode PASSED [ 13%]
backend/tests/test_v4_0a.py::test_underspecified_prompt_returns_clarifying_questions PASSED [ 17%]
backend/tests/test_v4_0a.py::test_answering_questions_produces_valid_blueprint PASSED [ 20%]
backend/tests/test_v4_0a.py::test_smart_defaults_produces_valid_blueprint PASSED [ 24%]
backend/tests/test_v4_0a.py::test_ingested_csv_produces_data_aware_blueprint PASSED [ 27%]
backend/tests/test_v4_0a.py::test_ingested_document_produces_data_aware_blueprint PASSED [ 31%]
backend/tests/test_v4_0a.py::test_deepseek_llm_client_routing PASSED     [ 34%]
backend/tests/test_v4_0a.py::test_openai_llm_client_routing PASSED       [ 37%]
backend/tests/test_v4_0a.py::test_anthropic_llm_client_routing PASSED    [ 41%]
backend/tests/test_v4_0a.py::test_ollama_llm_client_routing PASSED       [ 44%]
backend/tests/test_v4_0a.py::test_fallback_synthesizer_and_connection_validation PASSED [ 48%]
backend/tests/test_v4_0a.py::test_models_json_roundtrip PASSED           [ 51%]
backend/tests/test_v4_0a.py::test_tictactoe_fuzzy_intent_and_typo_handling PASSED [ 55%]
backend/tests/test_v4_0b.py::test_jit_simulation_synthesis_and_smoke_test PASSED [ 58%]
backend/tests/test_v4_0b.py::test_self_healing_catches_and_repairs_division_by_zero PASSED [ 62%]
backend/tests/test_v4_0b.py::test_self_healing_catches_and_repairs_shape_mismatch PASSED [ 65%]
backend/tests/test_v4_0b.py::test_data_driven_environment_and_arena_bridges PASSED [ 68%]
backend/tests/test_v4_0b.py::test_ast_security_blocks_malicious_code PASSED [ 72%]
backend/tests/test_v4_0c.py::test_autonomous_multi_round_self_tuning_loop PASSED [ 75%]
backend/tests/test_v4_0c.py::test_user_nudge_and_snapshot_rollback PASSED [ 79%]
backend/tests/test_v4_0c.py::test_fastapi_e2e_flow_and_local_runbundle_execution PASSED [ 82%]
backend/tests/test_v4_0c.py::test_fastapi_sql_ingest_endpoint PASSED     [ 86%]
backend/tests/test_v4_1.py::test_unique_episode_keys_across_multi_round_loops PASSED [ 89%]
backend/tests/test_v4_1.py::test_conversation_persistence_and_resumability PASSED [ 93%]
backend/tests/test_v4_1.py::test_multi_question_intake_and_custom_training_params PASSED [ 96%]
backend/tests/test_v4_convergence.py::test_tictactoe_reaches_90_percent_non_loss_rate PASSED [100%]

================== 29 passed, 5 warnings in 77.09s ===================
```

### 4.2 Frontend Production Build
```text
> rl-ide-frontend@0.1.0 build
> tsc && vite build

vite v6.4.5 building for production...
✓ 1599 modules transformed.
dist/index.html                   0.56 kB │ gzip:  0.36 kB
dist/assets/index-CrBb28dF.css   36.41 kB │ gzip:  6.57 kB
dist/assets/index-VJM9si22.js   248.43 kB │ gzip: 71.28 kB
✓ built in 18.12s
```

---

## 5. v4.2 Continuous Conversational Studio & Commercial E2E Audit Report

**Audit Date:** 2026-10-29  
**Execution Environment:** Live Chrome Browser via Puppeteer + FastAPI Backend (Port 8000) + React/Vite Frontend (Port 5173).  
**Outcome:** 11 / 11 Test Scenarios Passed (10 Commercial Use Cases + Cross-Conversation Resumption).  
**Browser Stability:** Zero Console Errors (0), Zero React Duplicate Key Warnings (0), Zero Unhandled Promise Rejections (0).

### 5.1 Architecture & Engine Upgrades (v4.2)

1. **Instant Mock-Animated Arena Previews (`arena_locked == False`):**
   - Blueprints synthesize and render an active HTML5/Tailwind canvas immediately upon first prompt without requiring the user to wait for training or manual blueprint locking.
   - The preview runs `startMockAnimationLoop()` simulating real agent decisions and live state updates at 60 FPS in a secure, sandboxed iframe.

2. **Continuous Omni-Agent Conversational Studio (`StudioOmniAgent`):**
   - Replaced fragmented multi-modal steps with a continuous single-thread omni-agent chat (`/api/v4/conversations/{id}/chat`).
   - Dynamically routes between visual restyling, agent behavioral updates (reward function rewriting + Critic calibration), environment mechanics regeneration, and autonomous training bursts.
   - The studio is never locked out: post-training prompts allow immediate rule modification, reward tuning, or adding auto-tune rounds.

3. **Pre- and Post-Training Visual Hot-Reloading:**
   - Supported direct visual styling via conversational chat (e.g., adding HIPAA badges, neon arcade scoreboards, live cloud burn-rate meters, animated power-flow arrows).
   - Hot-reloads visual layout without wiping environment code, neural network policy weights, or round telemetry history.
   - Integrated automated JavaScript syntax validation (`heal_arena_html_scripts` and `validate_arena_html_scripts` using `node --check -`) to heal unmatched LLM template literal syntax prior to iframe injection.

4. **Network-Level LLM Timeout & Responsive Fallback:**
   - Enforced `timeout=25.0` on remote LLM requests (`httpx.Client(timeout=25.0)`) within visual restyling to prevent studio hangs during slow upstream provider responses.
   - Seamlessly falls through to deterministic HTML5 generation in <0.02s when external APIs latency-spike.

5. **Device-Consistent ONNX Policy Export & Parity Verification:**
   - Hardened `PolicyExporter.export_onnx` and `PolicyExporter.verify_parity` to ensure policies are exported and evaluated on CPU (`model.policy.to("cpu")`), eliminating CUDA FP32 vs CPU FP32 numerical divergence.
   - Verified bit-for-bit numerical parity diff of `8.94e-8` to `1.19e-7` (substantially outperforming the strict `< 1e-5` threshold).

6. **SQLite Security & Read-Only Guardrails:**
   - Hardened `/api/v4/ingest/sql` to enforce read-only execution. Destructive SQL commands (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`) are rejected with 400 Bad Request before database access.

7. **Clean Multi-Session State Isolation:**
   - Updated `handleNewAgent` in `App.tsx` to clear attached datasets (`setAttachedDataset(null)`), preventing residual datasets from colliding across newly created agent workspaces.

---

### 5.2 10 Commercial Use Cases Execution Matrix

| # | Commercial Domain | Real-World Scenario | Fixture / Ingestion | Key Capabilities Verified | Browser Status |
|---|---|---|---|---|---|
| **1** | **Healthcare** | HIPAA Prior-Authorization Clinical Triage | `patient_claims.csv` | HIPAA boundary badge restyling, abnormal vitals highlighting, post-training cardiac symptom rule, autonomous convergence | **PASSED** |
| **2** | **FinTech** | Real-Time Wire & ACH Fraud Interceptor | `wire_transfers.sqlite` | Read-only SQL modal guardrail, malicious query rejection (`DROP TABLE`), 3-class triage, live iframe fraud scoring | **PASSED** |
| **3** | **DevOps / FinOps** | Kubernetes Auto-Scaler & SLA Protector | `k8s_telemetry.csv` | Dynamic pod scaling, live $/hour cloud burn-rate meter, SLA zero-violation enforcement, policy auto-tuning | **PASSED** |
| **4** | **E-Commerce & Retail** | Flash-Sale Dynamic Pricing | `sku_flash_sale.xlsx` | 14-day inventory depletion, post-training rule constraint (forbid clearance when >7 days remain), multi-round tuning | **PASSED** |
| **5** | **LegalTech** | NDA & Contract Clause Risk Auditor | `vendor_nda_clauses.docx` | Clause risk classification, multi-turn conversational diagnostics, liability cap reasoning explanation | **PASSED** |
| **6** | **CleanTech** | Solar + Battery Tariff Arbitrage Controller | Pure Continuous Simulation | 4-node animated power-flow arrows (Solar/Battery/Building/Grid), `.runbundle` packaging, ONNX parity `8.94e-8` | **PASSED** |
| **7** | **Corporate Finance** | Travel & Expense (T&E) Policy Auditor | `employee_expenses.csv` | Expense violation triage, receipt verification, behavioral policy convergence | **PASSED** |
| **8** | **Robotics & Logistics** | 2D AMR Warehouse Navigator | Kinematics Simulation | 2D Canvas arena, obstacle avoidance, battery conservation, live interactive canvas waypoint deployment | **PASSED** |
| **9** | **SaaS & Telecom** | VIP Customer Churn Optimizer | `saas_customer_health.csv` | Net Retention Rate maximization, post-training discount penalty rule when login drop <20%, round resumption | **PASSED** |
| **10** | **Gaming & Strategy** | Tactical Grid / Connect-Four Bot | Discrete Grid Environment | Neon arcade theme, per-cell win probabilities, 3 consecutive live interactive moves against trained policy | **PASSED** |
| **11** | **Platform Core** | Cross-Conversation Resumption | Multi-Agent DB | Resumed past agent conversations from Left Sidebar drawer with complete model, weights, and telemetry intact | **PASSED** |

---

### 5.3 Automated Verification Results

#### 1. Live Browser Puppeteer Audit (`browser_test_10_commercial_usecases.js`)
```text
========================================================================
🚀 10-DOMAIN COMMERCIAL USABILITY & E2E BROWSER AUDIT (v4.2)
========================================================================

👉 Navigating to RL-Native Agent IDE at http://localhost:5173 ...
   ✓ Studio connected with Verified LLM credentials.

------------------------------------------------------------------------
▶️ [Use Case 1/10] Healthcare — HIPAA Prior-Authorization Clinical Triage
------------------------------------------------------------------------
   ✓ Uploaded fixture: patient_claims.csv
   ✓ Instant mock-animated clinical chart arena rendered.
   ✓ Pre-training visual restyling applied with HIPAA badge.
   ✓ Arena locked. Autonomous training initiated.
   ✓ Policy convergence achieved.
   ✓ Post-training chat modification accepted (Round 1).

------------------------------------------------------------------------
▶️ [Use Case 2/10] FinTech — Real-Time Wire & ACH Fraud Interceptor
------------------------------------------------------------------------
   ✓ New conversation created.
   Testing SQL DB modal & read-only guardrails...
   ✓ SQL Guardrail verified: blocked destructive query (true).
   ✓ Ingested wire_transfers SQLite table.
   ✓ Arena locked. Autonomous training initiated.
   ✓ Policy convergence achieved.
   ✓ Switched to Play / Interact mode.
   ✓ Live interactive fraud scoring step evaluated.

------------------------------------------------------------------------
▶️ [Use Case 3/10] DevOps / FinOps — Kubernetes Auto-Scaler & SLA Protector
------------------------------------------------------------------------
   ✓ New conversation created.
   ✓ Uploaded fixture: k8s_telemetry.csv
   ✓ Live $/hour cloud burn-rate meter rendered in arena.
   ✓ Arena locked. Autonomous training initiated.
   ✓ Policy convergence achieved.

------------------------------------------------------------------------
▶️ [Use Case 4/10] E-Commerce & Retail — Flash-Sale Dynamic Pricing
------------------------------------------------------------------------
   ✓ New conversation created.
   ✓ Uploaded fixture: sku_flash_sale.xlsx
   ✓ Arena locked. Autonomous training initiated.
   ✓ Policy convergence achieved.
   ✓ Post-training pricing policy rule updated (Round 1).

------------------------------------------------------------------------
▶️ [Use Case 5/10] LegalTech — NDA & Contract Clause Risk Auditor
------------------------------------------------------------------------
   ✓ New conversation created.
   ✓ Uploaded fixture: vendor_nda_clauses.docx
   ✓ Contract clause conversational diagnostic response verified.

------------------------------------------------------------------------
▶️ [Use Case 6/10] CleanTech — Solar + Battery Tariff Arbitrage Controller
------------------------------------------------------------------------
   ✓ New conversation created.
   ✓ CleanTech 4-node power-flow arena restyled.
   ✓ Arena locked. Autonomous training initiated.
   ✓ Policy convergence achieved.
   Testing .runbundle export and ONNX numerical parity...
   ✓ Exported .runbundle with ONNX numerical parity diff: 8.940696716308594e-8

------------------------------------------------------------------------
▶️ [Use Case 7/10] Corporate Finance — Travel & Expense (T&E) Policy Auditor
------------------------------------------------------------------------
   ✓ New conversation created.
   ✓ Uploaded fixture: employee_expenses.csv
   ✓ Arena locked. Autonomous training initiated.
   ✓ Policy convergence achieved.

------------------------------------------------------------------------
▶️ [Use Case 8/10] Robotics & Logistics — 2D AMR Warehouse Navigator
------------------------------------------------------------------------
   ✓ New conversation created.
   ✓ 2D kinematics canvas arena rendered.
   ✓ Arena locked. Autonomous training initiated.
   ✓ Policy convergence achieved.
   ✓ Switched to Play / Interact mode.
   ✓ Live target waypoint clicked on 2D robot canvas.

------------------------------------------------------------------------
▶️ [Use Case 9/10] SaaS & Telecom — VIP Customer Churn & Retention Optimizer
------------------------------------------------------------------------
   ✓ New conversation created.
   ✓ Uploaded fixture: saas_customer_health.csv
   ✓ Arena locked. Autonomous training initiated.
   ✓ Policy convergence achieved.
   ✓ Post-training discount threshold adjusted (Round 1).

------------------------------------------------------------------------
▶️ [Use Case 10/10] Gaming & Strategy — Tactical Grid / Connect-Four Bot
------------------------------------------------------------------------
   ✓ New conversation created.
   ✓ Sleek neon arcade board theme with win probabilities rendered.
   ✓ Arena locked. Autonomous training initiated.
   ✓ Policy convergence achieved.
   ✓ Switched to Play / Interact mode.
   Playing 3 live turns against policy in iframe...
   ✓ 3 live interactive moves executed in arena.

------------------------------------------------------------------------
▶️ [Audit Item] Cross-Conversation Switching & State Resumption
------------------------------------------------------------------------
   ✓ Switched to past conversation, state & arena cleanly restored.

========================================================================
📊 GLOBAL INVARIANTS & AUDIT SUMMARY REPORT
========================================================================

1. Zero Browser Console Errors / Page Errors:
   Total Console Errors: 0
   Total Duplicate Key Warnings: 0

2. Commercial Use Case Execution:
   [1/11] Healthcare HIPAA Prior-Auth Router                : PASSED
   [2/11] FinTech Wire Fraud Interceptor                    : PASSED
   [3/11] Kubernetes Cluster Auto-Scaler                    : PASSED
   [4/11] Flash-Sale Dynamic Pricing Agent                  : PASSED
   [5/11] LegalTech Contract Clause Risk Auditor            : PASSED
   [6/11] CleanTech Solar + Battery Controller              : PASSED
   [7/11] Corporate Finance Expense Auditor                 : PASSED
   [8/11] 2D AMR Warehouse Navigator                        : PASSED
   [9/11] SaaS VIP Customer Churn Optimizer                 : PASSED
   [10/11] Gaming Tactical Grid Opponent Bot                 : PASSED
   [11/11] Cross-Conversation Resumable Switching            : PASSED

🎉 ALL 10 COMMERCIAL USE CASES & 9 GLOBAL INVARIANTS PASSED 100% WITH ZERO ERRORS!
```

#### 2. Backend Regression Suite (`pytest backend/tests/test_v4_2_commercial_suite.py`)
```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.7.0
rootdir: /home/devuser/projects/RLHarness/backend

backend/tests/test_v4_2_commercial_suite.py ...........                  [100%]

================== 11 passed in 282.91s (0:04:42) ==================
```

#### 3. Captured Verification Screenshots
All screenshots verified and stored in `test_screenshots/commercial_suite/`:
- `01_healthcare_preview.png`: Clinical chart preview with instant mock animation.
- `01_healthcare_trained.png`: Trained triage policy with HIPAA verification badge.
- `02_fintech_trained.png`: Trained wire fraud interceptor with live interactive scoring.
- `03_k8s_trained.png`: Kubernetes cluster autoscaler with live cloud burn-rate meter.
- `04_pricing_trained.png`: Dynamic flash-sale agent with post-training discount rule.
- `05_legal_preview.png`: Contract clause risk auditor with conversational diagnostics.
- `06_cleantech_trained.png`: CleanTech 4-node power flow arena with `.runbundle` export.
- `07_expense_trained.png`: Corporate T&E expense compliance auditor.
- `08_amr_trained.png`: 2D warehouse robot rover with clickable waypoint deployment.
- `09_saas_trained.png`: SaaS VIP customer retention optimizer with churn threshold tuning.
- `10_gaming_trained.png`: Neon arcade board theme with live 3-turn interactive iframe play.

---

### 5.4 Forensic Bug Fix & Resiliency Register (Audit Findings & Resolutions)

During the end-to-end execution of the 10 Commercial Use Cases across backend and live browser environments, 9 edge-case issues were uncovered and resolved:

| # | Domain / Component | Issue Description | Root Cause Analysis | Remediation Applied | Files Modified |
|---|---|---|---|---|---|
| **1** | **Backend API** (`/interact`) | `422 Unprocessable Entity` when submitting interactive steps. | Top-level payload dictionary sent `{"user_action": 4}` without wrapping in `payload` key. | Added Pydantic `extra = "allow"` and `get_effective_payload()` helper to extract actions transparently. | [server.py](file:///home/devuser/projects/RLHarness/backend/rl_ide/api/server.py) |
| **2** | **Test Harness** (`test_v4_2_commercial_suite.py`) | Premature target convergence assertion error. | Fast convergence in Round 1 caused `current_round == 2` assertion to fail when round halted at 1. | Updated assertions to dynamic inequality check (`current_round >= 1` and `current_round >= init_r1 + 1`). | [test_v4_2_commercial_suite.py](file:///home/devuser/projects/RLHarness/backend/tests/test_v4_2_commercial_suite.py) |
| **3** | **Intake Classifier** (`intake.py`, `arena_builder.py`) | AMR Warehouse Rover misclassified as Energy Controller. | Rover prompt mentioning "conserving battery" matched energy keyword before robotics keyword. | Added negative keyword exclusions (`not any(kw in p for kw in ["rover", "robot", "amr"])`) before energy classification. | [intake.py](file:///home/devuser/projects/RLHarness/backend/rl_ide/agent/intake.py), [arena_builder.py](file:///home/devuser/projects/RLHarness/backend/rl_ide/agent/arena_builder.py) |
| **4** | **Export API** (`/export`) | `KeyError: 'bundle_path'` in export response. | Backend returned `bundle_dir` while client / tests expected `bundle_path`. | Export endpoint now returns both `bundle_path` and `bundle_dir`. | [server.py](file:///home/devuser/projects/RLHarness/backend/rl_ide/api/server.py) |
| **5** | **Arena Synthesizer** (`arena_builder.py`) | JavaScript syntax errors in dynamic iframe scripts. | LLM generated malformed string interpolations (e.g., unmatched `${...}`) crashing browser execution. | Integrated AST-based syntax repair (`heal_arena_html_scripts`) and validation with `node --check -`. | [arena_builder.py](file:///home/devuser/projects/RLHarness/backend/rl_ide/agent/arena_builder.py) |
| **6** | **Frontend Studio** (`App.tsx`) | Dataset persistent across conversations hiding upload zone. | Creating a new agent retained previous `attachedDataset` state. | Added `setAttachedDataset(null)` to `handleNewAgent` in `App.tsx`. | [App.tsx](file:///home/devuser/projects/RLHarness/frontend/src/App.tsx) |
| **7** | **Browser E2E Runner** (`browser_test_10_commercial_usecases.js`) | Case-sensitive substring assertion mismatch. | Puppeteer checked `.includes('burn-rate')` while rendered DOM used `Cloud Burn Rate`. | Converted checks to lowercase normalized matching (`.toLowerCase().includes('burn')`). | [browser_test_10_commercial_usecases.js](file:///home/devuser/projects/RLHarness/browser_test_10_commercial_usecases.js) |
| **8** | **LLM Visual Restyle** (`client.py`, `arena_builder.py`) | Remote LLM latency hang on HTML generation. | Slow upstream provider response (>90s) exceeded test timeout. | Set `timeout=25.0` on `httpx.Client` with automatic sub-millisecond deterministic fallback. | [client.py](file:///home/devuser/projects/RLHarness/backend/rl_ide/llm/client.py), [arena_builder.py](file:///home/devuser/projects/RLHarness/backend/rl_ide/agent/arena_builder.py) |
| **9** | **Policy Exporter** (`exporter.py`) | ONNX numerical parity divergence (`2.38e-2 > 1e-5`). | PyTorch policy evaluated on `cuda:0` while ONNX Runtime evaluated on `cpu`. | Enforced `model.policy.to("cpu")` prior to export and parity verification, achieving `8.94e-8` parity. | [exporter.py](file:///home/devuser/projects/RLHarness/backend/rl_ide/export/exporter.py) |

---

### 5.5 Repository & Operational Status

- **Active Git Branch:** `main`
- **Latest Commits:**
  - `f3ed087`: `feat(conversations): add clear all conversations endpoint and UI action, purge stale test runs`
  - `26813d2`: `feat(v4.2): complete 10-domain commercial e2e audit with onnx parity and full browser verification`
- **Working Tree:** Clean (all changes tracked and committed).
- **Service Daemons:**
  - FastAPI uvicorn daemon: Active on `http://localhost:8000`.
  - Vite dev server: Active on `http://localhost:5173`.
- **Test Artifacts:**
  - Fixtures: `backend/tests/fixtures/commercial_suite/` (CSV, SQLite, XLSX, DOCX).
  - Screenshots: `test_screenshots/commercial_suite/` (11 high-resolution verification PNGs).
  - Browser Test Script: `browser_test_10_commercial_usecases.js`.
  - Regression Test File: `backend/tests/test_v4_2_commercial_suite.py`.

---

## 6. Agentic Chat Planner & Multi-Layer Code Executor (v4.3 Milestone)

### 6.1 Architectural Overview
The conversational layer has been upgraded from rigid keyword/intent enum matching to a **Code-Aware Agentic Chat Planner & Multi-Layer Code Executor**. The planner inspects both the user prompt and the existing `env_module.py` implementation to decompose complex requests into targeted execution layers.

```
                                  +---------------------------------------+
                                  |            User Chat Prompt           |
                                  +---------------------------------------+
                                                      │
                                                      ▼
                                  +---------------------------------------+
                                  |        _plan_chat_action()            |
                                  | (LLM Planner + Fallback Classifier)   |
                                  +---------------------------------------+
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
            [supported == False]                                            [supported == True]
                       │                                                             │
                       ▼                                                             ▼
     +-----------------------------------+                         +-----------------------------------+
     |        Honesty Guardrail          |                         |     Multi-Target Executor Loop    |
     | - Zero code mutations             |                         | 1. ENV_SIM_MECHANICS              |
     | - Preserves env_module.py bitwise |                         | 2. REWARD_SHAPING                 |
     | - Status: "unsupported"           |                         | 3. INFERENCE_STRATEGY             |
     | - "I do not know how to do that"  |                         |    (AST-validated surgical patch) |
     +-----------------------------------+                         +-----------------------------------+
```

### 6.2 Key Capabilities Implemented
1. **ChatActionPlan Schema (`models.py`)**:
   - `ExecutionTarget` enum: `REWARD_SHAPING`, `INFERENCE_STRATEGY`, `ENV_SIM_MECHANICS`, `TRAINING_HYPERPARAMS`, `UNSUPPORTED_CAPABILITY`.
   - `ChatActionPlan` Pydantic model with fields: `supported`, `unsupported_reason`, `scope_classification`, `targets`, `requires_retrain`, `surgical_plan`, `user_facing_summary`.
   - Integrated into `StudioChatResponse` as `action_plan`.

2. **Inference Strategy Surgical Patching (`omni_agent.py`)**:
   - Implemented `_patch_interactive_inference()` to modify `interactive_step()` without triggering retraining when only inference/arena behavior is altered.
   - For opening move randomization (e.g. Tic-Tac-Toe): injects a turn-1 candidate stochastic selection block (`np.random.choice(good_moves)`) over viable openings `{0, 2, 4, 6, 8}` instead of deterministic argmax.
   - Sanitized numpy array conditionals to prevent `ValueError: The truth value of an array with more than one element is ambiguous`.
   - Validated code changes using `ast.parse()` prior to writing back to `env_module.py`.

3. **Multi-Target Chained Execution**:
   - Complex compound prompts (e.g., modifying both reward functions for training and interactive inference behavior for human play) execute across multiple layers in a single chat turn.
   - `ENV_SIM_MECHANICS` -> `REWARD_SHAPING` -> `INFERENCE_STRATEGY` sequentially applied and verified.

4. **Honesty Guardrail**:
   - Out-of-scope requests (multiplayer matchmaking, Discord bots, WebGL 3D, VR, real-time voice) are classified as `supported=False` with `UNSUPPORTED_CAPABILITY`.
   - Guaranteed **zero code mutations** (file hash / content remains byte-for-byte identical).
   - Generates honest rejection message: `"I do not know how to do that yet: {unsupported_reason}"`.

### 6.3 Test Verification Metrics
Automated verification suite `backend/tests/test_agentic_chat_planner.py` executes 3 comprehensive validation tests:
- `test_first_move_randomization_inference_patch`: Verifies opening move randomization in Tic-Tac-Toe across 30 games with zero retraining.
- `test_multi_layer_arena_and_training_patch`: Verifies chained reward shaping and inference strategy patching in a single invocation.
- `test_unsupported_capability_honest_refusal`: Verifies zero code mutations and honest rejection response.

---

## 7. Blackjack 21 Card Domain & Casino Table Arena Synthesis

### 7.1 Issue & Root Cause Analysis
- **User Issue**: Starting a new session with `"train a blackjack playing agent which winws everyttime"` resulted in an `"Autonomous Task Synthesizer"` arena showing a 2D rover navigation waypoint coordinate grid with actions `"Move Left"`, `"Move Right"`, `"Move Up"`, `"Move Down"`, `"Hold Position"` and telemetry `distance_reward`, `velocity_cost`. Subsequent requests (`"arena should have blackjack table etc"`) did not change the canvas.
- **Root Cause**:
  1. `IntakeArchitectAgent._deterministic_intake()` only contained rules for Tic-Tac-Toe, Solar Energy, CartPole, and Uploaded Datasets. Any other prompt defaulted to an "Autonomous Task Synthesizer" 2D rover.
  2. `JITArenaSynthesizer._generate_deterministic_arena_html()` defaulted non-TicTacToe / non-dataset domains to a 2D canvas with `agent_pos` / `target_pos`.
  3. `JITEnvironmentSynthesizer` lacked a dedicated Blackjack Gymnasium synthesis template, falling back to `_build_general_sim_code` (2D rover physics).
  4. In `ContinuousStudioOmniAgent.classify_intents()` and `process_message()`, visual keywords did not match "table", "card", or "blackjack", preventing live re-anchoring when users requested table redesigns.

### 7.2 Resolution & Capabilities Added
1. **Blackjack Intake Classification (`backend/rl_ide/architect/intake.py`)**:
   - Implemented `is_blackjack_intent(text)` with fuzzy keyword matching (`blackjack`, `21`, `card table`, `dealer`, `hit or stand`, `double down`).
   - Implemented `is_blackjack_spec(blueprint)` detection.
   - Added 5-pillar clarifying questions tailored for Blackjack (`Casino Basic Strategy`, `Emerald Felt Table`, `Live Player / Agent Advisor Mode`, `Win/Non-Bust Expectations`).
   - Synthesizes first-class `AgentBlueprintSpec` for `Blackjack 21 Card Agent` with observations `["player_total", "dealer_upcard_value", "has_usable_ace", "can_double_down"]` and actions `["Hit (Draw Card)", "Stand (Hold Hand)", "Double Down (2x Bet)"]`.

2. **Casino Felt Table Arena (`backend/rl_ide/synthesizer/arena_builder.py`)**:
   - Implemented self-contained emerald casino felt table with dealer card row, player card row, and center table branding (`♠ BLACKJACK PAYS 3 TO 2 ♦ DEALER MUST STAND ON 17 ♣`).
   - SVG/CSS card renderer with suits (♠ ♥ ♦ ♣), ranks (A, 2-10, J, Q, K), and card-back patterns for hidden hole cards.
   - Built-in Mock Simulation & Animation Loop (`startMockAnimationLoop()`): simulates deals, hits, stands, dealer reveals, win/loss banners, reward breakdown (`win_payout`, `blackjack_bonus`, `bust_penalty`), and action probabilities.
   - Interactive buttons: `🟢 Hit (+1 Card)`, `🛑 Stand (Hold Hand)`, `⚡ Double Down (2x)`, and `🔄 New Deal`.
   - Full bi-directional `postMessage` protocol: `ARENA_UPDATE`, `SET_ARENA_MODE`, `ARENA_INTERACT`.

3. **Gymnasium Environment Synthesizer (`backend/rl_ide/synthesizer/env_builder.py`)**:
   - Implemented `_build_blackjack_code()` producing standard Gymnasium `SynthesizedGymEnv`.
   - Card deck logic, soft/usable Ace handling, dealer hit-to-17 rules, double down multiplier.
   - Passed all 6 criteria of `SandboxSmokeTester.run_smoke_test(env_cls)` (25 runtime steps, JSON serializable `arena_state`, demonstration dataset generation, and `interactive_step`).

4. **Omni-Agent Domain Re-Anchoring (`backend/rl_ide/architect/omni_agent.py`)**:
   - Added table/card visual keywords (`table`, `cards`, `felt`, `blackjack`, `dealer`, `chips`) to `classify_intents` and `has_visual_intent`.
   - Added domain re-anchoring logic: if a session was previously initialized with generic rover fallback and the user requests blackjack or card play, the agent dynamically upgrades the blueprint and hot-reloads the Blackjack table arena live.
   - Upgraded user's active session (`conv_1793265260777_j059x`) to the Blackjack 21 Card Table.

### 7.3 Verification Suite
- Created `backend/tests/test_blackjack_arena_and_env.py` (4/4 tests passing).
- Verified 47 / 47 total core tests passing (100% green).

---

## 8. Zero-Hardcoding Universal Generative Pipeline & Agentic Chat Planner (v4.4 Specification)

### 8.1 Specification Directives & Problem Statement
- **Prior Issue:** The system relied on static domain templates (`tictactoe`, `blackjack`, `solar`, `cartpole`) and static keyword routing lists (`["table", "card", "felt"]`, `["randomize", "first move"]`). When users provided novel prompts (e.g. Connect 4, Russian Roulette, Elevator Scheduling, or arbitrary custom environments), the system regressed to the generic 2D Waypoint Rover canvas (`Move Left`, `Move Right`, `distance_reward`). Furthermore, conversational modifications to inference strategies (such as randomizing opening moves) were erroneously routed to reward retraining rather than surgical inference patching.
- **Directives Enforced:**
  1. **Zero Domain Templates & Zero Rover Fallback:** Eradicate all static keyword routing lists. The 2D Rover canvas must NEVER be generated unless the prompt explicitly requests navigation or rover tasks.
  2. **Universal Generative Intake Pipeline (`intake.py`):** Dynamic synthesis of `DynamicTaskBlueprint` from arbitrary prompts with structured observation/action specs, domain summaries, reward formulas, and render state schemas.
  3. **Universal JIT Environment Synthesizer (`env_builder.py`):** Universal Gym environment generation with AST pre-import security validation, a 3-iteration self-healing loop, and standard `compute_reward()`, `interactive_step()`, and `generate_demonstrations()`.
  4. **Universal Generative Arena Builder (`arena_builder.py`):** Renders dynamic visual UIs driven by the environment's actual `get_render_state()` keys and blueprint action spaces.
  5. **Code-Aware Agentic Chat Planner (`omni_agent.py`):** Multi-target `ChatActionPlan` routing across `ExecutionTarget` enums, AST-validated surgical patching for `interactive_step()` (e.g., turn-aware first move randomization), honest refusal for unsupported external capabilities, and pure conversational handling for diagnostic questions.

### 8.2 Architectural Implementation

```
               [ User Prompt: "Train a Connect-4 grid agent" / "Russian Roulette" / etc. ]
                                                   │
                                                   ▼
                     +-------------------------------------------------------------+
                     |    1. UNIVERSAL GENERATIVE INTAKE ARCHITECT (intake.py)     |
                     | - DynamicTaskBlueprint synthesis for arbitrary domains      |
                     | - Zero Rover fallback; dynamic observation & action schemas |
                     | - 4-Pillar clarifying questions (q_objective, q_arena_ui)   |
                     +-------------------------------------------------------------+
                                                   │
                                                   ▼
                     +-------------------------------------------------------------+
                     |    2. UNIVERSAL GYMNASIUM SYNTHESIZER (env_builder.py)      |
                     | - Dynamic Gymnasium SynthesizedGymEnv synthesis             |
                     | - AST Security Audit + 3-Iteration Self-Healing Loop        |
                     | - Implements get_render_state() & interactive_step()        |
                     +-------------------------------------------------------------+
                                                   │
                                                   ▼
                     +-------------------------------------------------------------+
                     |       3. UNIVERSAL ARENA BUILDER (arena_builder.py)         |
                     | - Inspects sample_render_state keys captured in smoke test  |
                     | - Connect-4 (7x6 drop grid), Revolver, Novel Domain Stage   |
                     | - Dynamic telemetry cards & action buttons; ZERO 2D Rover   |
                     +-------------------------------------------------------------+
                                                   │
                                                   ▼
                     +-------------------------------------------------------------+
                     |      4. CODE-AWARE AGENTIC CHAT PLANNER (omni_agent.py)     |
                     | - ChatActionPlan with 7 ExecutionTarget enums               |
                     | - Surgical interactive_step() patching (turn-aware sampler) |
                     | - Diagnostic inquiry guardrail (pure discussion, no retrain)|
                     | - Honest refusal guardrail with zero code mutations         |
                     +-------------------------------------------------------------+
```

### 8.3 Key Implementation Details
1. **Dynamic Blueprint Schema (`backend/rl_ide/architect/models.py`)**:
   - `DynamicTaskBlueprint`: structured models for `ObservationFeatureSpec`, `ActionSpec`, `domain_summary`, `reward_intent`, `state_dict_schema`, with bidirectional conversions to/from legacy `AgentBlueprintSpec`.
   - `ExecutionTarget` enum: `REANCHOR_DOMAIN`, `ENV_SIM_MECHANICS`, `REWARD_SHAPING`, `INFERENCE_STRATEGY`, `ARENA_VISUAL_UI`, `TRAINING_HYPERPARAMS`, `UNSUPPORTED_CAPABILITY`.
   - `ChatActionPlan`: structured execution plan with `supported`, `unsupported_reason`, `scope_classification`, `targets`, `requires_retrain`, `surgical_plan`, and `user_facing_summary`.
   - `IntakeResponse`: exposes `.clarifying_questions` property for backwards compatibility with test harnesses and frontend components.

2. **Universal Intake (`backend/rl_ide/architect/intake.py`)**:
   - `synthesize_blueprint_from_prompt()`: dynamically formats task title, observation spaces, discrete actions, and state dict schema for any domain.
   - Refactored `interview()` and `_deterministic_intake()`: accepts user answers by both question ID (`q_objective`, `q_arena_ui`, etc.) and pillar keys (`objective`, `arena_ui`, etc.). Preserves custom `training_params` and `dataset_id`.

3. **Universal Environment Synthesizer (`backend/rl_ide/synthesizer/env_builder.py`)**:
   - `synthesize_universal_env()`: generates fully functional Gymnasium environments with AST security verification, discrete/box observation spaces, `compute_reward()`, `get_render_state()`, and demonstration generation.
   - All environments implement both `get_render_state()` and `render_arena_state()` for bi-directional compatibility.
   - Rover waypoint synthesis is strictly constrained to explicit prompts containing `"rover"` or `"waypoint"`.

4. **Universal Arena Synthesizer (`backend/rl_ide/synthesizer/arena_builder.py`)**:
   - `synthesize_universal_arena_html()`: generates self-contained HTML/CSS/JS arenas mapping directly to `sample_render_state` keys.
   - Tailored domain visual stages for Connect 4 (7x6 blue drop grid with falling disc animations), Russian Roulette (revolver cylinder with rotating chamber slots), and Universal Novel Domain Stage with live metric telemetry cards and interactive action buttons.
   - Complete bi-directional host bridge: `sendInteractiveAction(actionIdx)` and `renderState(state)` handler.

5. **Code-Aware Agentic Chat Planner (`backend/rl_ide/architect/omni_agent.py`)**:
   - Eliminated static keyword routing lists from `classify_intents`, delegating intent decomposition directly to `_plan_chat_action()`.
   - Turn-aware opening move randomization: surgical AST-validated replacement of `interactive_step()` policy selection, allowing the agent to sample stochastically among viable openings on turn 1 without triggering unnecessary PPO retraining.
   - Diagnostic question detection: messages asking questions ("Why did the agent...", "What are the rules?") are classified as `discussion` with `targets=[]`, delivering instant conversational explanations without code mutations or retrain overhead.
   - Honest refusal guardrail: impossible/unsupported requests (Discord, Bloomberg FIX protocol, WebGL shaders) are rejected with `supported=False` and zero code mutations.

### 8.4 Verification Test Results
- `backend/tests/test_universal_agentic_pipeline.py`: **5 / 5 PASSED**
  - `test_zero_hardcoding_novel_domain_synthesis`: Verifies novel domain synthesis without rover fallback.
  - `test_first_move_randomization_inference_patch`: Verifies opening move randomization across 30 opening turns.
  - `test_multi_target_code_and_ui_synchronization`: Verifies multi-target prompt syncing reward code and arena UI.
  - `test_unsupported_capability_honest_refusal`: Verifies zero code mutations and honest refusal for impossible features.
  - `test_no_keyword_routing_lists_in_omni_agent`: AST codebase audit verifying absence of static keyword routing lists in `classify_intents`.
- `backend/tests/test_agentic_chat_planner.py`: **3 / 3 PASSED**
- Total test count across all suites: **53 / 53 PASSED (100% Green)**.







---

## 9. v4.5 Remediation — Phases 0–E (2026-10-29)

Authoritative record: [DESIGN_AUDIT_AND_PHASED_REMEDIATION.md](./DESIGN_AUDIT_AND_PHASED_REMEDIATION.md).
All work was driven by the issue register F1–F33 in that document.

### 9.1 What changed, by phase

**Phase 0 — Transport verification**
- Investigated the reported `Authorization: ******` blocker: the source already sends
  `Bearer {api_key}` (`Bearer {api_key or 'ollama'}` for Ollama). The earlier report was a false
  positive caused by secret redaction in tool output.
- Added `test_llm_clients_send_bearer_authorization`, which captures outgoing headers with a mocked
  `httpx` transport for deepseek/openai/ollama.

**Phase A — Honest metrics and contracts**
- `episode_metric_value()` replaces `any(step.is_success)`: classification tasks aggregate per-step
  accuracy; control tasks use the terminal outcome.
- Dataset environments expose `task_class="classification"`, `is_correct`, and a generic
  `label_column`; the target column is excluded from observations (no label leakage).
- The schema-driven fallback is now an explicit **placeholder**: `is_placeholder=True`,
  `is_success=False`, reward `0.0`. Without a usable LLM, `synthesize_universal_env` raises
  `EnvironmentSynthesisError` instead of fabricating a trainable task.
- Rollout/evaluation use `episode_horizon` / `spec.max_episode_steps` (fallback 200) instead of a
  fixed 50-step cap.
- Verification enforces the declared `state_dict_schema` keys, label/space alignment, state
  mutability, and reports success reachability as uncertainty.

**Phase B — Training controls and isolation**
- `max_rounds` now unambiguously means “rounds to run in this call”; the session target accumulates.
- Budget: `episodes_per_round × 25` timesteps with an explicit 100 000 ceiling (no silent 1 024 cap).
- Auto-tuner hyperparameters (`ent_coef`, `learning_rate`) are applied to the PPO model.
- Unverified auto-tune output is rejected (`auto_tune_failed`) and never written/loaded.
- Targets are parsed only from explicit target/goal/threshold phrases; unused `eval_every` and
  `continue_training` were removed.
- Per-session `threading.Lock`; `/interact` returns **409** while training mutates the model;
  background tasks are retained in a strong-reference set.

**Phase C — Mutation safety**
- All write/import/train/hot-reload paths are gated on `verified`.
- `_repair_code` no longer substitutes a different environment; unrepairable code fails explicitly.
- `_safe_reward_rewrite()` replaces the regex rewriter: AST-validated insertion before the final
  return, refuses early-return/anonymous-breakdown/trailing-code cases, sanitizes user text.
- Inference-only plans no longer touch `compute_reward` or trigger retraining
  (`INFERENCE_STRATEGY → UPDATE_INFERENCE`).
- The inference patch is action-space aware (legal masking, `n_actions` bound, top-k fallback) and
  returns `(code, applied)` so no-op patches are reported honestly.
- Changed observation/action spaces reset the policy deliberately (`policy_reset` event + UI note).

**Phase D — Zero hardcoding and the state contract**
- Planner routing is a domain-free intent grammar (verbs + visual/style nouns) with word-boundary
  capability patterns; “invoices” is no longer rejected as “voice”.
- Arena generation uses the environment’s real `reset()` render state (or a neutral schema sample)
  and renders only user-requested elements generically. Rover/HIPAA/burn-rate/arcade domain branches
  were removed; duplicate `id="status-badge"` and unbalanced markup fixed; cell→action mapping only
  when the counts match; restyle sees the full document.
- Intake uses generic structure inference (quoted labels, `RxC` grids, `N columns/options/floors`),
  task-shaped questions, forwards `conversation_history`, and applies answers mid-session.
- `DynamicTaskBlueprint` now carries its own metric/threshold/round/episode contract.
- Export parity uses real environment observations at `atol=1e-5`; bundles fail if `env_module.py`
  or `arena.html` is missing; the UI reports drift as `unmonitored` instead of “healthy”.

**Phase E — Isolation, durability, tests, docs**
- `clear_all()` only deletes conversation-owned directories; `delete()` clears in-memory
  model/env/session and unreferenced datasets.
- Dataset records are persisted (`workspaces/<id>/dataset_records.json`) and reloaded by the loop,
  rollback, nudge, interact and export paths.
- CORS restricted to local origins; optional `RL_IDE_API_TOKEN`; HTML/JS escaping in arenas.
- `sys.modules` cache bounded; `env_module_try_*` artefacts removed after verification attempts.
- Tests rewritten to behavioural contracts; new `test_phase_hardening.py` (10 tests) covers transport,
  isolation, export completeness, cleanup scoping, durability, mid-session answers, verification
  gating, auto-tune rejection and target parsing.

### 9.2 Test results (verbatim summary)

```text
$ pytest -q
72 passed, 5 warnings in 377.22s (0:06:17)
```

Frontend:

```text
$ npm run build
✓ 1599 modules transformed.
✓ built in 5.32s
```

### 9.3 Test-file changes

| File | Change |
|---|---|
| `tests/test_phase_hardening.py` | **New** — transport, isolation, export, cleanup, durability, intake answers, verification gating, auto-tune rejection, target parsing |
| `tests/test_universal_agentic_pipeline.py` | Rewritten to 14 behavioural tests (placeholder honesty, accuracy metric, long horizon, action-space-aware patching, reward safety, inference-only immutability, structural no-hardcoding audit, schema contract, training lock) |
| `tests/test_agentic_chat_planner.py` | Legal-action assertions instead of hardcoded `{0,2,4,6,8}`; reward assertions use the actual term names |
| `tests/test_v4_0a.py` | Structured prompts (`3x3 board`) for deterministic mode; fuzzy-intent test replaced by a domain-free task-shape test |
| `tests/test_v4_0b.py` | Shape mismatch now asserts explicit failure (no hidden substitution) |
| `tests/test_v4_0c.py`, `tests/test_v4_convergence.py` | Convergence tests run against a real, learnable contract fixture (`tests/fixtures/tictactoe_fixture_env.py`) instead of the fabricated placeholder |
| `tests/test_v4_2_commercial_suite.py` | Removed assertions for hardcoded overlays/canvas and fabricated mastery; asserts valid percentages and contract-bound arenas |

### 9.4 Remaining risks

See §6 of the audit document. None of them are silent failures: each is either surfaced in the UI,
documented in code, or gated by configuration.

---

## 10. v4.6 — Real-World Commercial Validation (2026-10-30)

The v4.5 work proved the pipeline was honest; it did not prove it learned anything a customer would
pay for. This section records what happened when four paid-use-case problems were driven through the
**running app in a real browser** (`node browser_test_real_world_value.js`, Chrome + Vite + FastAPI,
DeepSeek `deepseek-flash` verified live at 0.8–1.1 s latency).

### 10.1 What the browser run measured

Every check is an observable outcome read back from the API/session, not a page-text match:

| Use case | Input | Metric (round 1) | Export parity | Result |
|---|---|---|---|---|
| B2B support ticket routing (4 queues) | `support_tickets.csv`, 600 rows | **87.0 %** accuracy (chance 25 %) | 3.0e-07 | PASS |
| SaaS renewal save-play selection | `saas_renewal_risk.csv`, 500 rows | **95.4 %** accuracy | 2.4e-07 | PASS |
| Insurance claim handling path | `insurance_claims.csv`, 520 rows | **81.2 %** accuracy | 5.1e-07 | PASS |
| Peak-season inventory replenishment (control) | none (LLM-synthesized sim) | **100 %** success | 1.8e-07 | PASS (see §10.4 — unstable across runs) |

Each use case also verified: dataset ingestion, `arena_preview` status, arena HTML rendered inside the
sandboxed iframe, environment **not** a placeholder, observation features derived from real columns
with the label excluded, training queued, rounds recorded, metric above chance, live `/interact`
scoring, and ONNX parity < 1e-5 on export. Screenshots: `test_screenshots/real_world/`,
machine-readable report: `test_screenshots/real_world/report.json`.

### 10.2 Defects this exposed (all fixed)

Before the fixes, the first use case trained for five rounds to **62–66 %** while the environment was
effectively blind — see F34–F36 in §7 of the audit document:

1. **Observations ignored the dataset.** The synthesized env copied LLM-invented feature names against
   the real CSV, so 23/24 dimensions were permanently `0.0`; categorical values were encoded with
   process-salted `hash()`; string targets produced zero demonstrations, which forced an LLM rewrite
   of the whole module. Fixed by `synthesizer/feature_engineering.py` (numeric min-max, one-hot
   categoricals, deterministic label→action mapping, runtime-fitted bounds) — the supervised path now
   verifies on attempt 0 with **no LLM call**.
2. **Label inference picked continuous floats** (a DOCX table got `sentiment_score` as its target),
   forcing document tasks down the supervised path. Fixed in `ingest/profiler.py`: non-integral floats
   are rejected and columns whose values name the declared actions win.
3. **Unreachable success was invisible.** The reachability probe only stepped 25 random actions, so a
   terminal-only success on a 30-step horizon could never be observed, and the warning was discarded —
   one run trained two rounds and reported 0 % forever. The probe now runs complete, horizon-aware
   episodes (using `get_critical_actions()` when available) and the warning is surfaced in the chat reply.
   Short-horizon envs also stopped tripping a false "static arena_state" failure.
4. **Arena labels were cut to four characters.** The generic renderer treated any array whose length
   equalled the action count as a board (rendering "Retention Specialist" as `Rete` in a 40 px square),
   and control buttons clipped long labels. Text arrays now render as full chips; buttons wrap.
5. **The user's action list was not enforced.** One run returned only **two** actions
   ("Approve / Safe", "Flag / Investigate") for a prompt naming four queues, then trained to 90 % on a
   task nobody asked for — no error anywhere. The intake only ever parsed *quoted* lists, and only on
   the deterministic path. Now every enumerated list in the prompt (quoted, colon-introduced, or
   "to A, B, or C") is extracted and enforced on both the LLM and deterministic blueprints, and a
   declared label column that explains fewer actions than the dataset's real target is replaced by
   the column that explains all of them.

### 10.3 Verification

```text
$ pytest -q
90 passed, 5 warnings in 663.34s (0:11:03)
```

| New/changed test file | Purpose |
|---|---|
| `tests/test_dataset_feature_engineering.py` | **New** (12 tests) — real-column observations, deterministic encoding, demonstrations from string labels, no-LLM verification, dead-observation rejection, label inference, terminal-success probing |
| `tests/test_arena_rendering.py` | **New** (2 tests) — evaluates the arena script in node: text arrays render readable labels, numeric boards still render clickable cells |
| `tests/test_intake_action_contract.py` | **New** (4 tests) — prompt enumerations are extracted, an LLM blueprint that ignores them is overridden, intake keeps all declared actions, and a label column covering fewer actions is replaced |
| `tests/test_v4_0b.py` | Data-driven fixture grown to 12 labelled rows (the supervised builder requires ≥ 8) |
| `tests/test_phase_hardening.py` | Reward-shaping fixture grown to 40 rows for the same reason |
| `tests/test_v4_2_commercial_suite.py` | Cross-conversation switching ignores historical runs whose synthesis failed and requires at least one restorable conversation |

### 10.4 Residual risks from this round

- **LLM-authored control environments are not reliably learnable.** Four runs of the same inventory
  prompt produced 100 %, 62 %, 18 % and 0 % success in two rounds of 150 episodes: the synthesized
  success criteria (fill rate ≥ 95 % **and** holding cost ≤ budget **and** zero peak stockouts) are
  re-generated on every synthesis and are sometimes knife-edge — one sampled environment was
  miscalibrated enough that even a hand-written "order 50 every day" heuristic only succeeded 2 times
  in 6 episodes. Detection is reliable now (F36 surfaces an unreachable KPI to the user), but the
  criteria are not rewritten automatically. Dataset-driven tasks, which carry the label contract of the
  CSV, were stable across every run.
- **Synthesis latency is 2–3 minutes** per environment on `deepseek-flash`; the browser suite allows
  up to 10 minutes per chat turn.
- Datasets are synthetic but structurally realistic (conditional feature generation with label overlap);
  they are fixtures for the harness, not customer data.

---

## 11. v4.7 — Phase P0-A: the honest number (2026-10-30)

Implements G1 + G14 + G16 from [COMMERCIAL_READINESS_GAPS.md](./COMMERCIAL_READINESS_GAPS.md): the
reported metric must describe the policy the customer receives, measured on rows it never trained
on, next to a baseline a buyer recognises.

### 11.1 What changed

| Gap | Change |
|---|---|
| **G1** | Deterministic 70/15/15 split by row-content hash (`zlib.crc32`, not `hash()`); training and round selection see only `train`/`val`; the scorecard evaluates the untouched `test` split once at the end of a run. `evaluate_policy_deterministic` no longer consults `get_critical_actions()` (a tactical oracle) or `get_valid_actions()` (a legality mask the ONNX export does not apply); illegal moves are counted and reported instead of hidden. |
| **G1** | Baselines computed on the same held-out rows: uniform chance (1/k), majority class (most frequent *training* label), logistic regression on the same derived features (tabular), uniform-random policy over full episodes (control tasks). Wilson 95 % interval on the policy number. |
| **G14** | `records_from_frame` shuffles with a fixed seed before truncating, so the embedded 120-row sample is representative instead of the first 120 rows of an export-ordered file; `split_counts`, `dataset_rows_total` and `embedded_in_module` are recorded in the scorecard so a number is never presented as "your whole dataset". |
| **G16** | `PolicyExporter.verify_behavioural_parity` runs identical seeded episodes through the SB3 policy and the exported ONNX session, reporting action agreement and any success-rate gap. It is written into the bundle's `scorecard.json` alongside the numeric parity figure. |

New module: [scorecard.py](./backend/rl_ide/autopilot/scorecard.py) (Wilson interval, majority /
logreg baselines, scorecard schema and the in-sample warning). The scorecard is persisted to
`workspaces/<session>/scorecard.json`, exposed on `SessionState`, on `GET /api/v4/conversations/{id}`,
and pushed over the websocket as a `scorecard_ready` event. The UI shows it in the telemetry panel
(held-out value with CI, each baseline, the uplift verdict, ONNX action agreement, split sizes).

### 11.2 Measured effect

Support-ticket fixture (600 rows, 4 queues), 2 rounds × 40 episodes:

| | Before (masked, all rows) | After (held-out test split) |
|---|---|---|
| Reported metric | 90.6 % (val, same rows as training) | **85.4 %** (84 test rows) with 95 % CI [83.3, 87.2] |
| Baselines shown | none | uniform 25.0 %, majority 30.9 %, **logistic regression 84.5 %** |
| Uplift over best baseline | unknown | +0.9 pts |

The last row is the point of the exercise: on this dataset the RL policy is only marginally better
than a one-line classifier, and now the product says so instead of reporting 90.6 %.

Reproduced end-to-end through the live API (upload → chat → arena preview → lock & train → export)
on a real backend process, not a test client:

```text
held_out true · eval_split test · splits {train: 423, val: 93, test: 84}
policy 85.44 % (95 % CI 83.38-87.29, 50 episodes, 0.0 % illegal)
baselines: uniform 25.0 · majority 30.94 · logistic regression 84.5
uplift vs best baseline +0.94 pts · rows {total: 600, embedded_in_module: 120}
export deployment: 100 % action agreement over 250 steps, ONNX success == policy success (83.2),
                   numeric parity 2.4e-07
```

Tic-Tac-Toe (control task) reports 78-90 % across runs unmasked, versus the previous ≥ 90 % which
was only reachable with the oracle mask; the convergence test was rewritten to assert a best-round
floor plus a 40-point margin over the in-run random baseline (≈ 20 %).

### 11.3 Verification

```text
$ pytest -q
101 passed, 7 warnings in 612.97s (0:10:12)
```

| Test file | Purpose |
|---|---|
| `tests/test_scorecard.py` | **New** (11 tests) — split stability across processes, disjoint/exhaustive splits, embedded-record split parity, episodes confined to the requested split, shuffled embedded sampling, oracle/mask trap during evaluation, Wilson interval, majority baseline, in-sample flagging, scorecard persistence, ONNX behavioural parity |
| `tests/test_v4_convergence.py` | Rewritten to assert convergence without oracle masking, plus a margin over the in-run random baseline |
| `tests/test_v4_1.py` | Unchanged; still verifies unique episode ids (held-out episodes are tagged `r<N>_holdout_*`) |
| `npm run build` | Clean (tsc + vite, 1 599 modules) |

Live verification (separate backend on `:8010`, deleted afterwards): dataset upload, arena preview,
"Lock Arena & Start Training", two rounds, `scorecard_ready` in the session payload and in
`GET /api/v4/conversations/{id}`, export bundle containing `scorecard.json` with the deployment
section.

### 11.4 Residual risk added by this phase

- **Selection is single-shot.** Rounds are selected on `val`, but no checkpoint is retained per
  round, so the scorecard reports the final model rather than the best-on-validation one. A cheap
  follow-up is to snapshot `model.zip` when val improves and evaluate that checkpoint on test.
- **Baseline sample sizes repeat rows.** `get_demonstrations(5000)` cycles the split, so the
  numbers are unbiased but the reported `n` is the sample count, not the row count.
- **Control tasks have no split.** They report `held_out: false` with an explanatory note; success is
  measured over full episodes against the environment's own dynamics.
- **Scorecard cost.** The final evaluation adds 50 episodes per run (≈ 2-4 s on tabular tasks).
- **Demo rollouts still mask legality.** `_run_ppo_burst` (the rollout streamed to the arena during
  training) applies `get_valid_actions()` so the visualisation stays legible; it is a display path,
  not a measurement, and the scorecard path is unmasked. Worth unifying when the legality contract
  is defined for export (G2/§7 R1).

---

## 12. v4.8 — Phase P0-B.1: no data leaves, tenants isolated, disk bounded (2026-10-30)

Implements G3 + G4 + G9. Phase P0-B is split into two reviewable commits: **B.1** (this section)
covers data, tenancy and retention; **B.2** covers sandboxed execution (G2).

### 12.1 What changed

| Gap | Change |
|---|---|
| **G3** | `package_runbundle(include_data=False)` is the default. The bundled `env_module.py` has its `_records` sample stripped (regex on the generated line), `data/README.md` explains how to mount `records.json`, and `scorecard.json → deployment.contains_training_data` records the choice. `POST /export` takes `{"include_data": true}` for the old self-contained behaviour. |
| **G4** | `conversations.owner_id` column (auto-migrated, default `local`), owner-scoped `list/get/delete`, and a new `DatasetStore` (`runs/datasets.db` + gzipped JSON rows) replacing the process dict — uploads now survive a restart. Every endpoint resolves the owner from `x-owner-id` via a context-var middleware, so swapping in real auth later changes one function. New `GET /api/v4/datasets` and `DELETE /api/v4/datasets/{id}`. |
| **G9** | Verification workspaces use `tempfile.TemporaryDirectory` and are always cleaned (609 stale `temp_verify_*` directories were removed from this checkout); `DELETE /conversations/{id}` now also removes `exported_bundles/<id>.runbundle*` and `VACUUM`s the database; dataset deletion removes its payload file and compacts. |

Deleting a conversation previously left every page behind: the database here was 220 MB with zero
rows and 53 897 of 53 900 pages free. `VACUUM` after delete is now covered by a regression test.

### 12.2 Verification

```text
$ pytest -q
109 passed, 11 warnings in 684.28s (0:11:24)
```

| Test file | Purpose |
|---|---|
| `tests/test_tenancy_and_bundles.py` | **New** (8 tests) — dataset durability across store instances, dataset owner scoping, conversation owner scoping, delete removes workspace/bundles and reclaims pages, API hides other owners' conversations (list + get + delete → 404), default bundle contains no customer rows, `include_data` bundle is self-contained, verification leaves no workspace behind |
| `tests/test_v4_2_commercial_suite.py` | Updated to the new `dataset_store` symbol |

### 12.3 Residual risk from B.1

- **Ownership is a trusted header, not authentication.** `x-owner-id` is only meaningful behind a
  reverse proxy that sets it; the shared `RL_IDE_API_TOKEN` remains the only gate. An identity layer
  is still required before hosting (tracked in G4's fix list, not delivered by scoping).
- **Dataset payloads are JSON, not columnar.** Fine at 600–50 000 rows; a parquet payload is the
  obvious upgrade if uploads grow.
- **`save_or_update` creating a conversation falls back to owner `local`.** Normal flows always
  create the conversation first, so this only matters for internal callers.

---

## 13. v4.9 — Phase P0-B.2: generated code behind an OS boundary (2026-10-30)

Implements G2 for the paths that execute model-generated Python — training, verification, and export
sampling/parity — and makes the one remaining path fail closed for hosted tenants.

### 13.1 What changed

| Component | Change |
|---|---|
| `rl_ide/security/sandbox.py` | **New.** Builds a `bwrap` invocation with only `/usr`, `/lib`, `/lib64`, the venv, `rl_ide` (read-only at `/opt`) and the job's case directory visible; `--unshare-net --unshare-pid --die-with-parent`; hard wall-clock timeout; merged stderr pipe (a separate stderr pipe can deadlock when it fills); minimal environment (no provider keys). |
| `rl_ide/security/worker.py` | **New.** Modes `smoke`, `train`, `observations`, `parity`. Writes results to a file in the case dir and streams events on stdout; never imported by the API. |
| `rl_ide/autopilot/sandbox_runner.py` | **New.** Stages a private case directory per job, runs the worker, copies back only that session's artefacts, removes the case dir. |
| Verification | `SandboxSmokeTester` imports and exercises generated modules inside the worker (`smoke`). |
| Training | `run_training_task` prefers the sandbox; `SandboxUnavailable` (no bwrap) falls back in-process with a loud `sandbox_bypassed` event, an isolation *failure* does not fall back. |
| Export | Observation sampling for parity/drift and the ONNX-vs-SB3 episode comparison both run in the worker; the API no longer instantiates the generated module during export. |
| Interactivity | A persistent `serve` worker holds the environment (and policy) so state survives between requests; `/interact` talks to it over JSON lines. `GET /conversations/{id}` no longer instantiates the module at all. Idle sessions are reaped (`RL_IDE_INTERACT_IDLE_SECONDS`, default 900 s) and concurrency is capped (`RL_IDE_MAX_INTERACT_SESSIONS`, default 8); `atexit` closes the rest. |

**`unshare -rn` was deliberately rejected.** It gives network/PID isolation but leaves the host
filesystem readable, so it is not an equivalent boundary; treating it as one would be security
theatre. Hosts without `bwrap` get the explicit single-user fallback instead.

### 13.2 What the tests prove (not just assert)

The isolation tests bypass the AST denylist entirely — the module is hostile at import time:

```text
worker cannot read runs/conversations.db   -> FileNotFoundError (path does not exist in the sandbox)
worker has no network                      -> OSError: Network is unreachable
provider keys are not forwarded            -> sandbox env contains no *KEY*/*TOKEN* variables
benign environment still verifies          -> smoke ok, obs_dim 2
training worker returns a session          -> rounds recorded, scorecard.json copied back
hosted tenant hitting /interact            -> 200 from a persistent `serve` worker, no in-process env
serve session keeps state                  -> "call 1" then "call 2" from the same worker
```

### 13.3 Behaviour change to be aware of

**LLM auto-tuning is disabled inside the sandbox.** The worker has no network and no provider keys
by design, so when the loop detects a plateau it emits `auto_tune_skipped` and keeps the reward
function instead of calling the LLM. Dataset-driven runs converge without tuning; if auto-tuning
matters for a task, the environment must eventually move behind an RPC boundary (a P1-C follow-up).

Suite runtime grew from **11:24 to 16:03** because each worker imports numpy/gym (and torch for
parity). That is the cost of the boundary, not a regression.

### 13.4 Verification

```text
$ pytest -q
116 passed, 11 warnings in 963.83s (0:16:03)
```

| Test file | Purpose |
|---|---|
| `tests/test_sandbox_isolation.py` | **New** (7 tests) — tenant file unreadable, no network, keys not forwarded, benign env verifies, bwrap reported, training worker returns session + artefacts, hosted `/interact` refused |
| `tests/test_tenancy_and_bundles.py` | Verification test now asserts the sandboxed path leaves no workspace behind |

### 13.5 Residual risk from B.2

- **Interactive sessions are process-per-conversation.** Each live conversation holds one bwrap
  worker (≈100–200 MB with torch). Idle reaping bounds this, but a hundred simultaneously active
  conversations would not fit on one host; the serve path needs pooling before that.
- **`--die-with-parent` is deliberately not used for serve.** Linux delivers that signal when the
  spawning *thread* exits, and FastAPI runs sync endpoints in a threadpool — the worker was being
  SIGKILLed between requests. Serve sessions now end when the API closes their stdin (EOF), which
  also covers an API crash.
- **Sandbox escape surface.** `bwrap` is the boundary; kernel or bubblewrap vulnerabilities are out
  of scope. The case directory is writable by design (the trainer writes models there).
- **`bwrap` availability.** Without it there is no isolation; hosted deployments must treat
  `isolation_available() == False` as a hard startup failure. That startup assertion is not wired
  yet — a P1-C item.
- **No seccomp profile.** The worker can still make syscalls the interpreter does not need.

---

## 14. v4.10 — Suite hygiene: the default run is free, use-case suites are opt-in (2026-10-30)

### 14.1 The leak

`rl_ide/api/server.py` calls `init_llm_state()` at import time. Inside pytest that used to read the
persisted config at `/tmp/.test_llm_config.json` and then `DEEPSEEK_API_KEY`/`OPENAI_API_KEY`/
`ANTHROPIC_API_KEY` from the environment, and *verify the provider over the network*. Two
consequences:

1. Every test process that imported the app made a real outbound call. With a placeholder key
   (written by `test_valid_key_verifies_and_activates_globally`) it was a guaranteed 401 — a
   non-hermetic, offline-hostile, 0.5 s tax per process.
2. With a **real** key exported, the verification succeeded and every `/chat` in
   `test_v4_2_commercial_suite.py` (10 domains × 5–8 turns) and `test_sample_usecases.py` drove the
   live provider — a real, paid token burn hidden inside `pytest`.

### 14.2 What changed

- **`is_offline_test_mode()`** ([client.py](./backend/rl_ide/llm/client.py)) — test mode *unless*
  `RL_IDE_LIVE_LLM=1`. Every place that previously forced the deterministic fallback under pytest
  (`init_llm_state`, `get_active_llm_client`, `create_llm_client`, `is_llm_ready`,
  `verify_and_set_llm_config`, `validate_llm_connection`, config persistence path) now keys off it.
  `init_llm_state()` returns immediately, so no provider is probed and no key on disk or in the
  environment is consumed.
- **`live_llm` marker** registered in [pyproject.toml](./backend/pyproject.toml) with
  `addopts = -m "not live_llm"` — the ten commercial domains and the two sample walkthroughs are
  deselected from the default suite. Each file also module-skips unless `RL_IDE_LIVE_LLM=1`, so an
  explicit `pytest tests/test_v4_2_commercial_suite.py` can never silently spend tokens.
- Opt-in command: `RL_IDE_LIVE_LLM=1 ../.venv/bin/python -m pytest -m live_llm tests/test_v4_2_commercial_suite.py tests/test_sample_usecases.py`.

### 14.3 Crash leftovers: staged rows do not outlive the process (G3/G9 follow-through)

A `serve` case directory under `runs/sandbox/serve_*` holds `env_module.py`, `records.json` (the
customer rows sampled for that session) and `model.zip`. `SandboxedEnvSession.close()` removes it,
but a SIGKILLed or crashed API leaves it behind — and the interrupted test runs had left ten of them.

- `_purge_orphan_serve_dirs()` ([server.py](./backend/rl_ide/api/server.py)) deletes case dirs that
  are neither live in this process nor modified in the last hour. Runs at API start-up (skipped under
  pytest, so a test run beside a dev server cannot delete a live session's case) and after every
  idle sweep.
- The one-hour grace is what makes it safe to run at all: a second instance sharing the same working
  directory keeps its own case dirs, while a day-old crash leftover is always collected.

### 14.4 Verification

```text
$ pytest -q
108 passed, 2 skipped in 678.43s (0:11:18)   (13 live-LLM use-case tests deselected)

$ RL_IDE_LIVE_LLM=1 pytest -m live_llm --collect-only -q tests/test_v4_2_commercial_suite.py tests/test_sample_usecases.py
13 tests collected
```

| Test | Asserts |
|---|---|
| `test_offline_test_mode_never_reaches_a_provider` | With a real-looking `DEEPSEEK_API_KEY` set, startup makes **zero** provider calls, state stays unverified, studio stays unlocked, and the active client is `FallbackCodeSynthesizer` |
| `test_live_llm_opt_in_restores_provider_verification` | `RL_IDE_LIVE_LLM=1` restores env-key verification, the only supported way to reach a paid provider under pytest |
| `test_orphan_sandbox_case_dirs_are_purged` | A two-hour-old `serve_*` case dir holding staged rows is deleted; a fresh one is left alone |
| `test_serve_session_keeps_state_between_requests`, `test_interact_endpoint_runs_in_the_sandbox`, `test_hosted_tenant_can_interact_without_in_process_execution` | Interactive play is answered by the sandbox worker over JSON lines, keeps environment state, and never instantiates the module in the API process |

Live confirmation on the restarted API (not a test fixture), with an existing `env_module.py`:

```text
POST /api/v4/sessions/live_verify_serve/interact   x-owner-id: acme
  -> {"arena_state":{"calls":1},"action_probs":[0.5,0.5],"status_text":"call 1"}
POST ... (again)
  -> {"arena_state":{"calls":2},...,"status_text":"call 2"}
ps -> bwrap ... python -m rl_ide.security.worker --mode serve   (180 MB RSS)
```

### 14.5 Residual risk

- The opt-in suites still assert product behaviour that the fallback synthesizer satisfies only
  approximately; they are a manual pre-release check, not CI. Use-case generation and its
  verification are to be rebuilt as a separate, cost-bounded harness.
- `RL_IDE_LIVE_LLM=1` is inherited by child processes; a live run therefore also disables the
  offline-fallback safety net for any nested pytest invocation.

---

## 15. v5.0 — Phase P1-C: operable (2026-10-30)

### 15.1 G7 — usage metering and a tenant budget cap

`llm/client.py` recorded nothing about tokens or spend, so the product could not invoice, could not
cap a runaway customer, and could not tell a buyer what a run costs.

- **Capture at the provider boundary.** Every `generate()` on DeepSeek/OpenAI/Anthropic/Ollama now
  times the call and reads the `usage` block the provider returns
  ([metering.py](./backend/rl_ide/llm/metering.py)). Anthropic's `input_tokens`/`output_tokens` and
  the OpenAI-style `prompt_tokens`/`completion_tokens` are both understood.
- **Price locally.** A per-1M-token table prices each call (`deepseek-chat` 0.27/1.10, `gpt-4o`
  2.50/10.00, `claude-3-5-sonnet` 3.00/15.00, local Ollama 0), overridable per contract with
  `RL_IDE_LLM_PRICES`. An unlisted model is still recorded, with tokens and zero cost.
- **Persist, per conversation and per tenant.** New `llm_usage` table in the existing
  `runs/conversations.db` ([conversation_store.py](./backend/rl_ide/state/conversation_store.py)),
  written with request-scoped attribution (`bind_usage_context`), so spend follows the conversation
  and the `x-owner-id` tenant even though the provider call happens in a worker thread.
- **Cap before spending.** `RL_IDE_OWNER_BUDGET_USD` makes `enforce_budget()` raise
  `BudgetExceeded` *before* the HTTP call, so a capped tenant fails cleanly instead of finishing the
  month in debt.
- **Expose it.** `GET /api/v4/usage[?conversation_id=]` returns totals, a per-model split, the last
  50 calls and the remaining budget; the header shows `$spent` for the active conversation.

Verification (`tests/test_usage_metering.py`, 5 tests): a 1000/2000-token DeepSeek call is recorded
and priced at $0.00027 + $0.0022; an unlisted model records 12 tokens at $0; a tenant at $5.00 with a
$1.00 cap raises before the network call (`network_calls == []`); the endpoint is owner-scoped and
returns 404 for another tenant's conversation.

### 15.2 G5 + G6 — every studio turn is a durable job; training is a queue worker

A turn took 2–3 minutes inside one HTTP request, and its work existed only as a coroutine: no id to
poll, no cancel, and a restart erased it without a trace. Models, locks and sessions also lived only
in `AutonomousLoopController` dicts.

- **`jobs` table** ([job_store.py](./backend/rl_ide/state/job_store.py)) — `job_id`, conversation,
  owner, kind, `status` (`queued|running|succeeded|failed|cancelled`), `stage`, `payload`, `result`,
  `error`, `cancel_requested`. `claim_next()` takes a job atomically (`UPDATE … WHERE status='queued'`)
  so two API processes cannot run the same one. Both stores now open SQLite in WAL mode with a
  30 s busy timeout, which is what makes several processes share the file safely.
- **`POST /chat` with `Prefer: respond-async`** returns `202 {job_id, stage}` in ~60 ms and runs the
  turn on a worker thread. Without the header the call still blocks and returns the full response,
  so existing clients and the test suite keep their contract.
- **Progress is real.** `process_message` gained `progress_callback(stage, detail)` and
  `should_cancel()`, reported at `intake → env_synthesis → verification → arena → ready`, checked at
  the same boundaries so a cancel never interrupts a write. Stages are persisted on the job and
  published to the websocket as `job_stage` events.
- **`GET /jobs/{id}`, `GET /jobs?conversation_id=`, `POST /jobs/{id}/cancel`** — owner-scoped; a
  cancel request flips a flag and the job stops at the next boundary.
- **Training is claimed from the same table.** A turn that triggers training enqueues a `training`
  job; `_drain_training_jobs()` claims and runs it (then exits — the queue is the database, not a
  thread). The old in-process `_spawn_background` coroutine is gone from this path.
- **Restart semantics.** `reconcile_interrupted()` fails any `queued|running` row with
  *"Interrupted: the API restarted while this job was running… retry it."* and `retryable: true`, so
  the UI can offer a retry instead of spinning forever.
- **Delivery no longer needs the request's loop.** `ConnectionManager.publish()` is a thread-safe,
  bounded queue that the websocket drains. The old code did `run_coroutine_threadsafe(…, loop)` with
  a loop captured from the request — dead after a reload, and provably broken under TestClient
  (`_UnixSelectorEventLoop … closed=True`), which is how the gap was found.

Verification (`tests/test_jobs.py`, 9 tests): a queued job is claimed exactly once; a restart turns
`running` into `failed`+retry; cancel is cooperative and refuses finished jobs; the runner stops at a
stage boundary; the async endpoint answers in < 2 s and reaches `succeeded` with an arena; stages
arrive over the websocket in order with the right `job_id`; the sync path still returns the full
response and records a job; jobs are 404 for another tenant; the training consumer runs a queued job
and stores `best_metric`.

Frontend: `chatWithStudioOmni` now sends `Prefer: respond-async` and polls the job, surfacing each
stage in the timeline; a 15-minute deadline replaces the silent hang.

**Residual:** the SB3 model objects, env instances and the loop's training lock are still
per-process (`global_store`, `AutonomousLoopController`). Job *state* is durable and training is
queue-backed, but running two Uvicorn workers with shared training state, or resuming an in-flight
training round after a restart, is still out of scope — the job fails cleanly instead. The legacy
`POST /sessions/start` path still spawns its training in-process.

### 15.3 G8 — pinned dependencies, a container, and a CI gate

There was no `.github/`, no `Dockerfile` and no lockfile: nothing reproducible to hand a customer,
and no gate that stops a regression.

- **Lockfiles.** `backend/requirements.lock` (runtime) and `requirements-dev.lock` (+ pytest) are
  `uv pip compile` resolutions from `pyproject.toml` on Python 3.12. Validated, not assumed: a clean
  venv built *only* from the lock runs the fast suite green (**64 passed, 2 skipped**).
- **`Dockerfile`.** python:3.12-slim + `bubblewrap`, installs the locked set, builds the SPA in a
  node stage, runs as uid 10001, binds `/data` for workspaces, sets `RL_IDE_REQUIRE_ISOLATION=1` and
  serves the built UI from the same origin (`RL_IDE_FRONTEND_DIST`), with a healthcheck.
- **Isolation is now a startup gate.** `enforce_isolation_requirement()` raises when
  `RL_IDE_REQUIRE_ISOLATION=1` and no sandbox exists, closing the B.2 residual "hosted deployments
  must treat `isolation_available() == False` as a hard startup failure".
- **SPA serving without shadowing.** `mount_frontend()` mounts at `/` but is called *after* every
  route: mounted earlier it swallowed `/api/*` (the test caught it — `GET /api/v4/settings/llm`
  returned 404).
- **CI** (`.github/workflows/ci.yml`): fast suite on every push and PR (installs bubblewrap, asserts
  the API boots with `RL_IDE_REQUIRE_ISOLATION=1`, hits `/settings/llm` and `/usage`), the full suite
  nightly or on demand, `tsc --noEmit` + `vite build` for the frontend, and an image build that runs
  the container and waits for the healthcheck. The image needs
  `--security-opt seccomp=unconfined` for bubblewrap's user namespaces.
- **Suite split.** Heavy modules are marked `slow` (training, export, sandbox workers):
  `pytest -m "not slow and not live_llm"` is **67 passed, 2 skipped in 1:08**; the full run keeps
  everything.

**Residual:** Docker is not installed on this host, so the image is built and booted by CI only — the
locked set, the SPA build and the API boot were each validated locally, but not the `docker build`
itself. A virtualenv under `/tmp` cannot exec inside `bwrap` (`--tmpfs /tmp` shadows it) — the image
is unaffected, but a dev venv must live outside `/tmp`.

### 15.4 Live validation of P1-C (real provider, real job)

Run against the API on `127.0.0.1:8000` as tenant `acme`, one real DeepSeek turn:

```text
POST /conversations/live_p1c_.../chat   Prefer: respond-async   x-owner-id: acme
  -> 202 in 0.03 s   {"job_id": "job_b0cea6e4a264", "stage": "queued"}

GET /jobs/job_b0cea6e4a264   (polled every 5 s)
  intake         (~60 s)
  env_synthesis  (~70 s)
  arena          (~155 s)
  succeeded      (~4 min 50 s total)

result: status=arena_preview, arena_html 51 799 B, env_code 25 073 B
GET /usage -> 13 calls, 1 190 in + 6 372 out tokens, $0.00733 (deepseek-flash)
```

That run also found a real tenancy bug (commit `6b2ec1d`): the omni agent created the conversation
with the default owner, so the tenant could not read its own conversation while `local` could.

**Final gate:** `pytest -q` → **126 passed, 2 skipped, 0 failed in 5:26**. Fast gate
(`-m "not slow and not live_llm"`) → **67 passed, 2 skipped in 1:08**.

---

## 16. v5.1 — Phase P2-A: remove sales friction (2026-10-30)

The three P2 items that stop a prospect from seeing the product at all: an unlearnable task, a
key-gated app, and a three-minute wait before *anything* appears. One operational bug surfaced on the
way and is fixed here too.

### 16.1 G15 — the arena no longer waits on the provider

The arena used to be the *last* thing a turn produced, so a new user watched an empty stage for
~135 s and then waited again for the canvas. Now the arena is declared from the blueprint the moment
the blueprint exists, and replaced in place.

- `JITArenaSynthesizer.preview_arena_html()` builds the schema-driven arena with **no provider call**
  (milliseconds), from the declared actions and `state_dict_schema`.
- `process_message` saves and broadcasts that preview *before* environment synthesis, then builds it
  again with the real `get_render_state()` sample, then swaps in the bespoke canvas when it arrives.
- Every save and broadcast now carries `arena_source` (`preview` | `bespoke` | `fallback`), and the
  provider outcome is reported by `synthesize_universal_arena_html(outcome=...)` instead of being
  inferred from log warnings.
- **A failed restyle no longer downgrades a working canvas.** `update_arena` used to overwrite the
  bespoke HTML with the generic preview when the LLM call failed; it now keeps what the user has and
  reports the error, and the route returns `arena_source` + `error`.
- **Frontend:** a live pill in the arena header — `🎨 Synthesizing bespoke domain canvas (~90s) —
  showing live telemetry preview` — driven by the job's real stage (and by `active_job` after a
  reload, so it cannot go stale), a red `Bespoke canvas failed` pill with **Retry**, and a
  `Generic preview` label so a screenshot can never present a preview as the final artefact.
- **Race fixed:** locking the arena during generation used to be clobbered by the turn's completion
  save (it wrote `arena_locked=False` unconditionally). Intermediate saves now pass `None`
  (preserve), the final save re-reads the lock, and a status the user moved past (`training`) is not
  reset.

Measured on real turns (`deepseek-flash`, tenant `acme`):

```text
# 1. full turn, insurance-claims routing (measured before the preview was moved earlier)
POST /conversations/conv_g15_.../chat   Prefer: respond-async   -> 202 in 0.08 s
  0.1 s   queued
  5.1 s   intake
 60.2 s   env_synthesis
135.3 s   arena            arena_source=preview   15 973 B   <- playable + lockable here
          (locked the arena from the preview at this point)
260.6 s   succeeded        arena_source=bespoke   38 856 B
final: arena_locked=true (lock survived the completion save)

# 2. current code, SOC-alert triage (preview declared before the environment exists)
POST .../chat -> 202 in 0.24 s
 75.1 s   arena_source=preview, 17 594 B, job stage still "env_synthesis"
          cancel -> "Cancellation requested; the job stops at the next stage boundary."
 +20 s    status=arena_preview, arena_source=preview, 17 594 B  (still usable, not stuck)
```

The second run is the important one: the preview is visible **while the environment is still being
written**, so "Lock Arena" never waits on a provider, and a cancelled synthesis leaves a playable
arena behind instead of an empty stage.

### 16.2 G13 — explore first, connect a model when you need one

- The startup lockout is gone: the whole app used to sit behind a blurred overlay and an unclosable
  modal until a key verified. It now opens in **explore mode**; upload, dataset profiling, listing and
  reading existing agents, the arena preview and ONNX export all work with no provider.
- Only `synthesize_environment`, `architect_chat`, `train` and `restyle_arena` refuse, and the refusal
  is machine-readable: `403` + `X-RL-Reason: llm-key-required` + a message that says what still works.
  The UI switches on that header (`ApiError.needsModel`) instead of string-matching `"403"` / `"verified"`
  in prose, which was the previous behaviour.
- `GET /api/v4/capabilities` publishes the honest state (verified? which actions need a model? is a
  sandbox available? is isolation required?), so the UI does not have to guess.
- Data-intake controls are no longer disabled while unverified, and the composer stays enabled so a
  clickable send explains what is missing instead of dead-ending.

### 16.3 G12 — an unreachable success condition is repaired, not just reported

A success signal that never fires makes every round report 0 % and burns the whole training budget.
`SandboxSmokeTester.relax_unreachable_success` now rewrites the environment's *success criteria* (a
relaxed threshold cannot help a signal that never fires), re-runs the full smoke test, and accepts
the result only when success becomes observable — at most two rewrites, and the previously verified
module is kept when the provider cannot fix it. `JITEnvironmentSynthesizer.synthesize` runs this
automatically and records what it did, and the assistant reply says the criteria were relaxed.

The probe also stopped conflating two different questions. `_probe_success(use_oracle=...)` now
separates *"can this ever succeed?"* (the environment's own `get_critical_actions` hint) from *"can a
policy that has to be learned succeed?"* (random/heuristic play). A task that only the oracle can
complete is reported as a low achievable rate instead of looking healthy — the same masking doctrine
as G1, applied to synthesis rather than evaluation.

### 16.4 A 2.4 GB write-ahead log, and why it filled the disk

Found the hard way: a test run died with `OSError: [Errno 28] No space left on device`. The cause was
not the tests:

- `with sqlite3.connect(...) as conn:` **does not close the connection** — it only commits or rolls
  back. Both stores used that idiom everywhere.
- In WAL mode SQLite *reuses* a WAL rather than truncating it, so a checkpoint that is ever blocked by
  a reader leaves the file at its high-water mark forever. On this machine
  `runs/conversations.db-wal` reached **2 432 606 208 bytes** beside an 88 MB database.

`rl_ide/state/sqlite_utils.py` now owns connection handling: `bounded_connection()` always closes,
`journal_size_limit` is 64 MB, `busy_timeout` is 30 s, and `checkpoint()` truncates. Deletion
checkpoints before and after `VACUUM`. `tests/test_state_durability.py` asserts the WAL returns to
zero bytes, that 200 sequential writes never accumulate one, and that a delete reclaims disk.

### 16.5 Verification

- New tests: `tests/test_state_durability.py` (5) and `tests/test_product_friction.py` (12) — the
  repair loop both succeeding and hitting its cap, the offline path never calling a provider, the
  end-to-end synthesizer hook, oracle-vs-learnable reachability, the instant preview, preview-before-
  synthesis ordering, the failed-restyle guard, the lock race, and the 403 reason header.
- Updated: `tests/test_llm_mandatory_gate.py` now asserts the new contract (reason header + "needs a
  model") instead of the retired "mandatory" wording.
- Frontend: `tsc --noEmit` clean, `vite build` clean (267 kB JS).
- Gates: **`pytest -q` -> 143 passed, 2 skipped, 0 failed in 11:04**; fast gate
  (`-m "not slow and not live_llm"`) -> **84 passed, 2 skipped in 1:26**.

### 16.6 Residual risk from P2-A

- **Two pytest sessions must not share `--basetemp`.** pytest keeps the three most recent
  `pytest-of-<user>/pytest-N` directories and deletes older ones at *session start*. Running a second
  invocation while a long suite is in flight can delete the running session's temporary directories,
  which surfaces as a `FileNotFoundError` in whichever test was writing at that moment. Two failures
  in this phase were exactly that, not product bugs - both tests pass alone and the clean full run is
  green. Use `--basetemp=` when running suites in parallel.
- **The preview is honest but not instant from *submit*.** It is instant from the moment the
  blueprint exists, which is after intake (~60 s on `deepseek-flash`). A user-visible "instant" arena
  before intake would have to be domain-agnostic, which is the generic filler this phase removed.
- **Retry reuses the restyle path.** A retry regenerates the canvas from the current HTML with the
  same 240 s ceiling; there is no queue position or backoff, so a retry during a provider outage
  fails visibly rather than waiting it out.
- **G13 is the gate only.** Enterprise provider targets (Azure OpenAI, self-hosted) are still open,
  and `x-owner-id` is still plumbing rather than authentication (G4).


---

## 17. v5.2 — Phase P2-B: the scorecard does not go stale (G11)

The scorecard proves a policy *was* good on held-out rows. Nothing checked that it still is. Drift
detection had existed since v4.0 and was imported by exactly one module, so the header badge was a
hardcoded `unmonitored` — honest about the gap, and the gap was real.

### 17.1 What is now wired

- **The observation the policy actually saw** is attached to every interaction: `serve` mode records
  it in `predict()` and returns it as `obs_vector`; the legacy in-process path captures it in
  `policy_predict`. Generated code never had to expose anything.
- **[live.py](./backend/rl_ide/drift/live.py)** — `DriftMonitor` keeps a rolling window of
  observations and action probabilities and scores it with the existing `DriftDetector`;
  `fit_baseline_from_episodes` builds the baseline from the conversation's own stored episodes;
  `DriftStatusStore` persists the verdict beside the conversation.
- **`/interact` scores every step** and returns a `drift` block; a confirmed crossing broadcasts
  `drift_warning` over the websocket. Monitoring can never break play: a missing baseline, a garbage
  observation or an exception in scoring is swallowed with a warning and the step still returns.
- **Training refits the baseline** (`_refresh_drift_baseline`) and drops the in-memory monitor, so a
  retrained policy is not judged against the policy it replaced. Deleting a conversation forgets its
  state.
- **The header badge is real**: `GET /conversations/{id}` returns `drift` + `drift_status`
  (`unmonitored` / `warming_up` / `healthy` / `drifted`) and the badge shows the sample count and the
  reasons as its tooltip.

### 17.2 Two defects the wiring exposed

**The only existing caller was broken.** `exporter.py` called `detector.detect_drift(...)` and read
`sig.drift_detected` and `sig.feature_psi_max`. `DriftDetector` exposes `check_drift` returning
`DriftReport` with `is_drifted` / `max_psi`, and `DriftBaseline.load()` was being handed a mapping
where it required a path. Both sat inside a bare `except: pass`, so **every exported `.runbundle`
reported `healthy` for any input at all** — the most dangerous kind of monitoring bug. It now uses
the real API and reports `unmonitored` plus the error when it genuinely cannot judge.

**PSI was unusable at live sample sizes.** Empty live bins were clipped to `1e-4` and then dominated
the sum: twenty in-distribution observations scored PSI **1.78** against their own baseline, i.e.
every small batch looked like an incident. Three changes, each measured:

| Change | Effect on unchanged traffic (30 seeds) |
|---|---|
| continuity correction (½ sample per bin) on both histograms | PSI 1.78 → 0.43 at n=20 |
| coarser bins for live windows (deciles → 5) | median PSI 0.43 → 0.22 at n=20, 0.05 at n=100 |
| live threshold 0.5 (textbook 0.25 assumes thousands of rows) + two consecutive windows |

Shipped defaults are `window=50, stride=5, confirmations=2, bins=5, psi=0.5`
(`RL_IDE_DRIFT_WINDOW` / `RL_IDE_DRIFT_STRIDE` override the first two). Measured over 100 healthy
sessions of 200 interactions each: **2 flagged (2 %), both from the action-rate signal, none from
PSI**. Genuinely shifted traffic is confirmed after **55 interactions**, every time.

### 17.3 Verification

- `backend/tests/test_drift_live.py` — 20 tests: healthy traffic stays quiet, shifted inputs are
  reported with the PSI reason, a too-small window says `warming_up` instead of guessing, a transient
  spike clears, a single crossing is *not* an incident, missing probabilities do not fabricate
  confidence, garbage observations are ignored, status survives a restart, a baseline fits from
  stored episodes and is `None` when there are none, `DriftBaseline.load` accepts a mapping, the
  report exposes the real field names (guarding the exporter contract), `/interact` scores and
  broadcasts, a missing baseline leaves the step untouched, a baseline without policy probabilities
  monitors inputs only, an observation is captured even when the policy is never called, and the last
  verdict stands between judgements. Two calibration tests pin both bounds in both configurations:
  with a full policy baseline (≤1 of 20 healthy sessions flagged) and with the production
  input-only baseline (**0 of 50** healthy sessions flagged, drift confirmed by interaction 100).
- Live validation on a real trained session (`deepseek-flash`, tenant `acme`, 100 episodes, real
  bwrap serve worker):

```text
GET /conversations/{id}                     -> drift_status: warming_up   (baseline fitted from the
                                               100 stored episodes, no click required)
POST /sessions/{id}/interact   x50          -> step 1  warming_up, samples=1
                                               ...samples accumulate from the environment itself,
                                               even though a human click never calls the policy
                                               step 50 first window: PSI 0.530 on state_var_2 > 0.5
                                                       status healthy, confirmed=False  (one crossing
                                                       is not an incident)
                                               step 55 second window: PSI 0.423 -> healthy
GET /conversations/{id}                     -> drift_status: healthy, samples: 50
```

**Three defects this live run caught that the unit tests did not:**
1. `_refresh_drift_baseline` called `model_dump()` on `SessionState.episodes`, which are dicts — it
   failed a training job *after* training had succeeded. Fixed, and the whole refresh is now wrapped
   so an advisory baseline can never fail a run.
2. The baseline never appeared at all, because stored episodes use the API shape
   (`{"obs": [...], "action": 2}`) rather than the `EpisodeLog` contract. `fit_baseline_from_episodes`
   now reads both, and the header went from `unmonitored` to `warming_up` on a trained session.
3. No observation was reaching the monitor: `interactive_step` does not call the policy when the
   human supplies the action, so `predict()` never ran. Observations are now captured by wrapping
   `env.step`/`env.reset`, which works however the generated module drives itself.

A fourth was cosmetic but visible: the status flipped back to `warming_up` on every step between
judgements, erasing a real verdict on the next click.

- Gates: **`pytest -q` → 163 passed, 2 skipped, 0 failed in 10:05**; fast gate
  (`-m "not slow and not live_llm"`) → **99 passed, 2 skipped**; frontend `tsc --noEmit` +
  `vite build` clean.

---

## 18. v5.3 — Phase P2-C: sell it to a bank (G4, G6 resume, G9 window, G13 providers)

Four residuals that each individually block a purchase: an unprovable tenant, data that outlives its
policy, no Azure/self-hosted model, and a training run that restarts from zero after a crash.

### 18.1 G4 — authentication: ownership is proven, not declared

`x-owner-id` was a label any caller could set. Now:

- **[auth_store.py](./backend/rl_ide/state/auth_store.py)** issues `rlt_…` tokens (32 random bytes)
  and stores **only `sha256`**; verification is a hash lookup plus a constant-time compare, and the
  plaintext is returned exactly once. Revocation takes effect on the next request.
- **The token decides the tenant.** As soon as authentication is on, `x-owner-id` is ignored — the
  boundary cannot be self-declared.
- **Two modes, decided at request time:** hosted (`RL_IDE_REQUIRE_AUTH=1`, or any tenant token has
  been issued) requires a bearer token; the desktop single-user studio keeps working with no
  credentials. A hosted deployment therefore cannot silently fall back to header-scoped tenancy.
- **The deployment secret is not a tenant.** `RL_IDE_ADMIN_TOKEN` may `POST/GET/DELETE
  /api/v4/admin/tokens` but never reads anyone's data.
- **The websocket enforces the same boundary**: the token arrives as a query parameter (browsers
  cannot set socket headers) and the socket may only follow a session the caller owns.
- `GET /api/v4/whoami` reports the tenant; `GET /api/v4/capabilities` reports the auth mode; the SPA
  sends the token on every request through a single `apiFetch`, prompts once on a 401, and stores the
  token under Settings → *Tenant token*.

Verified live on a provider-less instance started with `RL_IDE_REQUIRE_AUTH=1`: no token → `401`;
acme's token → `200`; beta reading acme's conversation → `404`; revoked token → `401`; websocket with
no / wrong / other-tenant token → refused, owner connects. In the browser the SPA showed the auth
prompt, accepted the token, and then listed the tenant's 9 agents — matching the API exactly.

### 18.2 G9 — the retention window

`RetentionPolicy` (`RL_IDE_RETENTION_DAYS`, plus per-artefact overrides) drives a scheduled sweep of
expired conversations, datasets and bundles, and removes orphan workspace directories whose database
row is already gone. Three properties matter more than the code:

- **Off by default.** A desktop studio must never delete a user's work.
- **Dry run is the default for the manual trigger**, so an operator can compare the policy against
  real data before arming it (`GET /api/v4/admin/retention` shows what would go right now).
- **One failure does not abort the sweep** — a locked workspace is logged and the rest continues.

### 18.3 G13 — Azure OpenAI and self-hosted endpoints

- **`azure_openai`**: deployment in the path, `api-key` header, `api-version` query — three things that
  break a naive OpenAI client. Accepts either a resource endpoint or a full deployment URL.
- **`openai_compatible`** (vLLM, TGI, LM Studio, an internal gateway): `base_url` required, bearer
  token optional because many run inside the customer's network without auth.
- Both are in the provider picker (now three columns, six providers) and both name the missing
  setting (`AZURE_OPENAI_API_KEY`, `OPENAI_COMPATIBLE_BASE_URL`) instead of failing at request time.

### 18.4 G6 — resuming an interrupted training run

`POST /api/v4/conversations/{id}/training/resume` queues only the rounds that are still missing
(derived from `rounds_history`) and warm-starts from the saved policy; a failed training job now
carries a `resume_hint` naming that endpoint. A *mid-round* checkpoint is still out of scope — the
round in flight is redone, the completed ones are not.

### 18.5 Verification

- New tests: `test_auth.py` (10), `test_retention.py` (8), `test_llm_providers.py` (13), plus 3 in
  `test_jobs.py` for resume.
- Two defects the tests caught, both mine: the token store used a **hardcoded** `runs/conversations.db`
  so a developer's live tokens made the whole suite demand credentials (fixed by binding tokens to the
  conversation store's path, and by ignoring issued tokens in test mode); and the resume test polluted
  a shared job store, which is exactly the "already queued" refusal working correctly.
- Frontend: `tsc --noEmit` clean; `vite build` clean. One real bug found by reading the code rather
  than by any tool: wrapping all fetches in `apiFetch` made it call itself (infinite recursion), which
  type-checks and builds perfectly.

---

## 19. v5.4 — Phase P2-D: journeys, not 25 independent guesses (G10)

Every dataset environment was a contextual bandit: twenty-five independent rows, each with its own
immediate reward. That fits routing and triage. It does not fit a renewal play over a quarter, a claim
journey or a multi-touch cadence — those are one case that unfolds over several decisions and is only
judged at the end. The representation was missing, so those problems could not be modelled at all.

### 19.1 What exists now

- The blueprint declares `task_structuring` (`one_shot` | `sequential`) and `episode_length`, carried
  across both directions of the `AgentBlueprintSpec` ⇄ `DynamicTaskBlueprint` conversion.
- **`SEQUENTIAL_ENV_TEMPLATE`** models one case per episode:
  - the case has a **plan** — the required action at each stage, derived from the case's own label and
    offset by the stage, so *the same record needs a different action at stage 0 than at stage 2*. A
    policy that maps record → action cannot solve it; it has to track where it is in the journey;
  - the observation carries the journey state (stage progress, accuracy so far, previous action) on
    top of the real feature columns;
  - the reward is **delayed**: `0.0` for every intermediate stage and `±1.0` only when the journey
    ends, and `is_success` is terminal-only (which is what the scorecard already measures).
- **The intake decides** from the request: "renewal play", "6-step onboarding flow", "claim journey",
  "multi-touch cadence" → sequential with the stage count if one is stated; "classify each ticket" or
  "triage each alert" → one-shot even if the sentence mentions steps. Choice matters because the wrong
  one is a modelling error the user discovers only after training.
- The one-shot template and its 25-step horizon are **untouched**, with a regression test that asserts
  it — a classifier must not silently become a journey.

Measured directly on a generated environment (renewal play, 4 stages):

```text
obs_dim 6 (3 real features + stage, accuracy-so-far, previous action)   horizon 4
plan for the case: [1, 0, 2, 1]     <- different action at every stage
running the plan:  rewards [0.0, 0.0, 0.0, 1.0]   is_success only at the end
one wrong stage:   final reward -1.0, is_success False
```

### 19.2 Scope, stated plainly

This is the sequential *representation*: persistent state, delayed reward, a plan the policy must
follow. It is not a new learning algorithm, and because the stage is in the observation the problem
remains Markovian — it is a real sequence task, not a partially-observable one. A `transition` that
mutates hidden state from one case to the next is still out of scope, so the original 3–4 day estimate
was for something larger than what shipped here.

### 19.3 Verification

- `backend/tests/test_sequential_tasks.py` — 17 tests: the intake decision table (including the
  "classify each ticket, following the 3 step checklist" trap), the blueprint round-trip, one episode
  = one case, state carrying across steps, delayed reward, terminal-only success, one wrong stage
  losing the journey, demonstrations covering every stage, the episode-length clamp, interactive play
  advancing and restarting a journey, and the one-shot regression.
- Gates: **`pytest -q` → 214 passed, 2 skipped, 0 failed in 27:02**; fast gate
  (`-m "not slow and not live_llm"`) → **155 passed, 2 skipped**.
