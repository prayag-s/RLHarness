# RLHarness — Crossfire Adversarial Audit Analysis & Phase P3 (`v5.5`) Remediation Specification

**Date:** October 30, 2026  
**Baseline Audited:** `main` at `v5.4` (Phases 0–E and Commercial Readiness Phases P0-A through P2-D complete; `163+` backend tests passing)  
**Input Documents:**
1. `Crossfire_Decision_Memo.md` (Adversarial Stress-Test of Core Product & Technical Claims 1–5)
2. `COMMERCIAL_READINESS_GAPS.md` (Operational & Trust Layer Gap Register `G1`–`G16`)
3. `progress.md` (§9–§19 Authoritative Implementation Log through `v5.4`)

---

## 1. Executive Summary & Synthesis

The Crossfire Decision Memo stress-tested five foundational hypotheses of the RL-Native Agent IDE:
1. **Claim 1 (Market Premise & Payback Economics):** Whether "train once, run locally" beats per-token frontier API calls once ongoing reward tuning, verifier maintenance, and falling API prices ($85\%$ drop to $\$1.00\text{--}\$5.63$ per $1\text{M}$ tokens) are accounted for.
2. **Claim 2 (Small Policy Fit vs. Frontier Model):** Whether a compact local policy trained on a few hundred examples genuinely beats a frontier model on **fit, cost, and latency** when evaluated on a disjoint held-out set authored before training, without requiring prohibitive cold-start manual labeling.
3. **Claim 3 (Plain-English Critique $\to$ Reward Diff Convergence):** Whether non-expert English critiques compiled into reward diffs converge on intended behavior without hacking the reward or regressing previously passing behaviors (cross-term interference).
4. **Claim 4 (Vibe-Coded Build & Execution Security Boundary):** Whether the multi-subsystem harness can rely on AST filtering and unverified LLM-generated code without an OS-level security boundary and strict ML verification gates.
5. **Claim 5 (Switching-Cost Moat Inversion):** Whether transparent diffs, readable reward rules, and standard ONNX exports invert the lock-in moat by making the user's accumulated rules easy to paste back into a frontier LLM.

### High-Level Verdict Against `v5.4`
Between `v4.5` and `v5.4`, DeepSeek systematically closed **all 36 correctness issues (`F1`–`F36`)** and **all 16 operational gaps (`G1`–`G16`)**. As a result, **Claim 4 is $100\%$ resolved** and **Claim 2 is $85\%$ resolved**. 

However, **five specific mechanisms**—concentrated in **Claim 1 (Payback Economics)**, **Claim 2 (Best-on-Val Selection & Passive Critique Log Harvesting)**, **Claim 3 (Verifier-Gated Critique Loop & Failure Triage)**, **Claim 4/3 (Sandbox-Safe Host-Orchestrated Auto-Tuning)**, and **Claim 5 (The Golden Eval Corpus Vault)**—remain open and must be implemented in **Phase P3 (`v5.5`)** to make the platform completely bulletproof against Crossfire's falsification criteria.

---

## 2. Master Traceability Matrix: Crossfire Claims vs. `v5.4` Codebase State

| Crossfire Claim | Crossfire Contradiction / Failure Mode | What RLHarness `v4.7`–`v5.4` Already Built | What Remains Open in `v5.4` | Target `v5.5` Deliverable |
| :--- | :--- | :--- | :--- | :--- |
| **Claim 1:** Market Premise & Payback Economics | Frontier token prices fell $85\%$ ($\$1\text{--}\$5 / 1\text{M}$ tokens). "Train once" hides ongoing reward tuning and drift maintenance; no break-even call volume or payback threshold is computed. | **Partially Solved (`v5.0` §15.1 & `v5.2` §17):** Built `llm/metering.py` (`llm_usage` table tracking exact per-conversation tokens and USD spend) and `drift/live.py` (calibrated PSI/action-rate live drift monitor). | `scorecard.json` does not connect to `llm_usage` or measure ONNX inference latency ($p_{50}/p_{99}$), so **Calls-to-Payback** and **Latency Speedup vs. Frontier** are never reported. | **Deliverable 5:** Break-Even & Payback Calculator in `scorecard.py` + UI Scorecard Economics card. |
| **Claim 2:** Small Policy Beats Frontier on Cost, Latency & Fit | Measuring "fit" in-sample against the training signal cannot separate learning from memorization. Also, requiring users to manually write $300$ examples upfront causes cold-start abandonment; teams need passive harvesting of existing frontier logs. | **Mostly Solved (`v4.7` §11):** Built deterministic $70/15/15$ content-hash split (`zlib.crc32`), stripped oracle (`get_critical_actions`) and legality masks from evaluation, added Wilson $95\%$ CI, and added `LogisticRegression` / majority / random baselines in `scorecard.py`. | 1. **Single-shot checkpointing (§11.4 residual):** Final model is evaluated instead of `best_val_model.zip`.<br>2. **No passive critique log ingestor:** Upload profiler does not parse frontier accept/reject/edit logs into training + held-out critique sets. | **Deliverable 4:** `best_val_model.zip` checkpoint selection + Passive Critique/Interaction Log Ingestor in `profiler.py` & `feature_engineering.py`. |
| **Claim 3:** Critique $\to$ Reward Diff Convergence Without Code | Optimizing against a critic-derived reward with no independent verifier hacks the reward within a few bursts, cannot separate a bad compile from a vague critique or RL variance, and regresses prior tests (cross-term interference). | **Partially Solved (`v4.5` §9.1 & `v5.1` §16.3):** Built `_safe_reward_rewrite` (AST-validated reward editing), snapshot storage, and `relax_unreachable_success` probe. | 1. **Unconditional retention:** Chat reward edits and bursts are kept even if `val` accuracy drops or locked cases regress.<br>2. **No automated failure triage:** Failed bursts do not diagnose *why* (0% term activation vs. cross-term interference vs. RL variance). | **Deliverable 1:** Verifier-Gated Critique Loop with Auto-Rollback & 3-Way Automated Triage (`verifier_gate.py` & `omni_agent.py`). |
| **Claim 3 & 4 Interaction:** Autonomous Reward Tuning Inside OS Sandbox | Sandboxing must isolate untrusted code with zero host secrets/network, while the training loop still needs iterative verifier-gated refinement. | **Sandbox Solved (`v4.9` §13), Auto-Tune Regressed (§13.3):** `bwrap` isolates `--unshare-net` and strips API keys, but because the whole multi-round loop runs inside one worker call, LLM auto-tuning is completely disabled (`auto_tune_skipped`). | In `v5.4`, autonomous multi-round reward self-tuning is a hard no-op whenever `bwrap` sandboxing is active (`progress.md` §13.3). | **Deliverable 2:** Host-Orchestrated Per-Round `bwrap` Execution + Verifier-Gated Auto-Tuning in `sandbox_runner.py`. |
| **Claim 4:** Vibe-Coded Build & AST Security Boundary | AST filtering is not a security boundary against object-graph traversal or file reads (`numpy.fromfile`); unverified code and unpinned builds fail enterprise review. | **$100\%$ Solved (`v4.9`–`v5.3`):** `bwrap` OS sandbox (`rl_ide/security/sandbox.py`), `RL_IDE_REQUIRE_ISOLATION=1` startup gate, SHA-256 tenant auth (`auth_store.py`), schema-only `.runbundle` default, pinned lockfiles, `Dockerfile`, and GitHub Actions CI. | **None.** Claim 4's security and operational gates are fully implemented and verified by `test_sandbox_isolation.py` and `test_auth.py`. | Maintain existing regression gates (`test_sandbox_isolation.py`, `test_auth.py`). |
| **Claim 5:** Switching-Cost Moat Inversion | Because diffs, reward functions, and ONNX exports are transparent, users can paste accumulated rules into a frontier LLM—inverting the config moat. Retention must rest on a **validated evaluation corpus**, not configs. | **Not Yet Solved:** Conversations persist `episodes` and `rounds_history`, but there is no cumulative, assertion-enforced regression test corpus (`eval_corpus.json`) that locks user critiques and edge cases across retrains. | Users can export configs, and nothing accumulates their domain critiques into a mandatory behavioral regression suite that gates future policy changes. | **Deliverable 3:** Persistent `EvalCorpusStore` (`eval_corpus.py`) — The "Golden Eval Corpus" Regression Vault. |

---

## 3. Part 1: What DeepSeek (`v4.7`–`v5.4`) Already Solved

Before detailing the 5 remaining builds, it is important to recognize the Crossfire blockers that have **already been permanently neutralized** in the repository:

1. **OS-Level Sandbox Replacing AST-Only Filtering (Crossfire Claim 4 — Day-1 Blocker):**
   - **Crossfire Objection:** *"Reward DSL must not rely on AST filtering alone; untrusted expressions need subprocess/container isolation with no host credentials."*
   - **Implemented in `v4.9` (§13) & `v5.0` (§15.3):** `rl_ide/security/sandbox.py` and `rl_ide/security/worker.py` execute all smoke verification, training, observation sampling, parity checks, and live `/interact` sessions inside a Linux `bubblewrap` (`bwrap`) container with `--unshare-net --unshare-pid`, a read-only mount of `/usr`, `/lib`, and `/opt`, an isolated case directory, and all `*KEY*`/`*TOKEN*` environment variables stripped. `test_sandbox_isolation.py` proves that hostile import-time code cannot read `runs/conversations.db` (`FileNotFoundError`), cannot open sockets (`OSError: Network is unreachable`), and cannot read API keys.

2. **Held-Out Split, Unmasked Evaluation, and Baseline Comparison (Crossfire Claim 2):**
   - **Crossfire Objection:** *"'Fit' superiority is measured against the training signal itself; absent a held-out set... convergence cannot separate learning the task from memorizing."*
   - **Implemented in `v4.7` (§11):** `autopilot/scorecard.py` partitions uploaded datasets deterministically into $70\%$ `train`, $15\%$ `val`, and $15\%$ `test` using `zlib.crc32` row-content hashes (stable across processes). Training and round selection never touch `test`. `evaluate_policy_deterministic()` evaluates without `get_critical_actions()` or `get_valid_actions()` oracle masks, computes a Wilson $95\%$ confidence interval, and compares the policy directly against three baselines on the exact same held-out rows: Uniform Chance ($1/k$), Majority Class, and a fitted `LogisticRegression` classifier.

3. **Zero-Exfiltration `.runbundle` Packaging & Multi-Tenant Auth (Crossfire Claim 4):**
   - **Implemented in `v4.8` (§12) & `v5.3` (§18):** `package_runbundle(include_data=False)` strips raw customer rows by default and ships only fitted feature bounds, `model.onnx` (verified for both numerical parity $< 10^{-5}$ and episode-level behavioral action agreement), and `data/README.md`. `auth_store.py` enforces SHA-256 bearer tokens (`rlt_...`) across REST and WebSockets, and `RetentionPolicy` bounds local disk growth while `sqlite_utils.py` caps WAL files at $64\text{ MB}$.

---

## 4. Part 2: Detailed Analysis & Remediation of the 5 Remaining Crossfire Gaps

Below is the exhaustive breakdown of each remaining gap: **(1) The Crossfire Finding**, **(2) Current State in `v5.4`**, and **(3) Exact Technical Remediation**.

---

### Issue 1: Unverified Critique-to-Reward Retention & Missing Failure Triage (Crossfire Claim 3)

#### 1.1 Crossfire Finding & Adversarial Counter-Evidence
- **Contradiction:** Crossfire flagged that when a non-expert user submits a plain-English critique (e.g., *"Penalize false positives more heavily"* or *"Stop stalling and move toward the extraction zone faster"*), the system compiles it into a `compute_reward()` modification and runs a short PPO burst. If the system judges success solely by whether the *new shaped reward increased*, the premise is circular:
  1. **Reward Hacking:** Repeated optimization against a critic-modified reward function with no independent verifier causes the policy to hack the new penalty within $2\text{--}3$ bursts (e.g., never predicting an anomaly again to avoid the false-positive penalty).
  2. **Cross-Term Interference:** Adding Term $B$ frequently breaks a behavior previously stabilized by Term $A$, regressing previously passing episodes.
  3. **Opaque Failure Attribution:** When a training burst fails to produce the behavior the user wanted, the non-expert user has no way to tell whether **(a)** the LLM compiled their critique into a dead/wrong condition (`compiler_mismatch`), **(b)** the new reward term conflicted with existing rules (`cross_term_interference`), or **(c)** PPO simply needed more steps / lower entropy (`rl_variance`).

#### 1.2 Current State in RLHarness `v5.4`
- In `backend/rl_ide/architect/omni_agent.py`, `_safe_reward_rewrite()` safely inserts the new reward term into `compute_reward()` using AST validation (Phase C, `v4.5`), calibrates the critic, and triggers a training burst.
- **The Gap:** Once `_safe_reward_rewrite()` passes syntax/smoke verification and runs the training burst, **the new `env_module.py` and updated `model.zip` are kept unconditionally**—even if validation accuracy (`val` split) plummets from $88\%$ to $52\%$ or previously locked episodes flip from pass to fail! Furthermore, no post-burst diagnostic compares term activation rates against validation deltas.

#### 1.3 Elaborate Technical Remediation (`backend/rl_ide/autopilot/verifier_gate.py` & `omni_agent.py`)
We will introduce a strict **Independent Verifier Gate** around every post-initial reward mutation (both user chat critiques and autonomous auto-tuner rewrites):

1. **Pre-Mutation Snapshot (`VerifierSnapshot`):**
   Before applying any `REWARD_SHAPING` or `AutoTunerAgent` modification to an already-trained session:
   - Capture `pre_env_code: str` (current `env_module.py`).
   - Copy current `model.zip` to `workspaces/<id>/pre_critique_model.zip`.
   - Evaluate (or read cached) independent verifier metrics on the `val` split (`val_metric_before: float`) and the session's `EvalCorpusStore` (`corpus_pass_rate_before: float`, `passing_case_ids_before: set[str]`).

2. **Term Activation Instrumentation During Burst:**
   - During the post-critique PPO burst and validation rollout, aggregate the `info["reward_breakdown"]` dictionary across all rollout steps.
   - For the newly added/modified reward term key(s) $k_{\text{new}}$, compute:
     $$\text{activation\_rate}(k_{\text{new}}) = \frac{\text{count}(\lvert \text{reward\_breakdown}[k_{\text{new}}] \rvert > 10^{-6})}{\text{total\_rollout\_steps}}$$
     and `mean_term_value` $=\mathbb{E}[\text{reward\_breakdown}[k_{\text{new}}]]$.

3. **Post-Burst Independent Verifier Gate:**
   - Evaluate the post-burst policy on the **untouched `val` split** (`val_metric_after`) and **`EvalCorpusStore`** (`corpus_pass_rate_after`, `regressed_case_ids = passing_case_ids_before - passing_case_ids_after`).
   - **Retention Rule:** The reward diff and updated policy are **RETAINED** if and only if:
     1. $\text{val\_metric\_after} \ge \text{val\_metric\_before} - \epsilon_{\text{tol}}$ (where $\epsilon_{\text{tol}} = 1.5$ percentage points to allow minor stochastic noise when trading off precision/recall), **AND**
     2. $\lvert \text{regressed\_case\_ids} \rvert == 0$ (zero regressions on locked Golden Eval Corpus cases, unless the user's critique explicitly overrode that specific case).

4. **Automated Rollback & 3-Way Failure Triage (`CritiqueTriageReport`):**
   - If the Verifier Gate **FAILS**:
     - Automatically restore `pre_env_code` to `workspaces/<id>/env_module.py` and `pre_critique_model.zip` to `model.zip`.
     - Classify the root cause deterministically into one of three triage categories:
       * **`"compiler_mismatch"`:** $\text{activation\_rate}(k_{\text{new}}) == 0.0$. The compiled reward condition never triggered on any state in the rollout. *Actionable advice:* Show the exact generated expression and state variable ranges so the user/compiler can adjust the threshold.
       * **`"cross_term_interference"`:** $\text{activation\_rate}(k_{\text{new}}) > 0.0$, the new term's mean shaped contribution improved, **but** $\text{val\_metric\_after} < \text{val\_metric\_before} - \epsilon_{\text{tol}}$ or $\lvert \text{regressed\_case\_ids} \rvert > 0$. The new penalty over-dominated existing terms and caused collateral regressions. *Actionable advice:* Report the exact regressed golden case IDs and suggest reducing the new term's weight by $50\%$.
       * **`"rl_variance"`:** $\text{activation\_rate}(k_{\text{new}}) > 0.0$, no golden cases were contradicted in intent, but policy entropy remained high and `val` did not converge within the short burst budget. *Actionable advice:* Offer a 1-click retry with $2\times$ burst episodes and Critic warmup.

---

### Issue 2: Multi-Round Auto-Tuning Disabled Inside the `bwrap` Sandbox (`progress.md` §13.3 & Crossfire Claim 3)

#### 2.1 Crossfire Finding & `progress.md` §13.3 Residual
- In `progress.md` §13.3, DeepSeek documented a critical side effect of moving `run_training_task` inside the network-less `bwrap` worker:
  > *"LLM auto-tuning is disabled inside the sandbox. The worker has no network and no provider keys by design, so when the loop detects a plateau it emits `auto_tune_skipped` and keeps the reward function instead of calling the LLM."*
- **Why this matters:** When a control or sequential task plateaus in Round 1 below the target threshold, the autonomous self-tuning loop (`AutoTunerAgent`) is supposed to inspect the failure traces, adjust `compute_reward()` and hyperparameters (`ent_coef`, `learning_rate`), verify the new environment, and run Round 2. Because the entire multi-round loop was placed inside a single `bwrap --mode train` subprocess call, multi-round LLM reward tuning is dead whenever OS sandboxing is active.

#### 2.2 Current State in RLHarness `v5.4`
- `sandbox_runner.py` launches `python -m rl_ide.security.worker --mode train` once for all rounds `1..max_rounds`. Inside `worker.py`, there is no network (`--unshare-net`) and no API key, so `AutoTunerAgent` cannot reach DeepSeek/OpenAI/Anthropic.

#### 2.3 Elaborate Technical Remediation (`backend/rl_ide/autopilot/sandbox_runner.py` & `loop_controller.py`)
We will refactor the boundary between the **Host Orchestrator** (`sandbox_runner.py`) and the **Isolated Worker** (`worker.py`) so that **the worker executes one round at a time**, while **the host orchestrates between-round LLM auto-tuning and Verifier Gating**:

1. **Per-Round Sandboxed Execution Loop:**
   - Instead of running all $R$ rounds inside one monolithic `worker.py` invocation, the host loop in `sandbox_runner.py` iterates $r = 1 \dots \text{max\_rounds}$:
     - **Step A (Sandboxed Round Execution):** Host invokes `bwrap ... python -m rl_ide.security.worker --mode train_round --round-idx r` inside the private case directory (passing in `env_module.py`, `records.json`, and prior `model.zip` if $r > 1$).
     - The worker runs Warm-Start BC (if $r == 1$ and no prior model) or Critic Warmup (if $r > 1$ and reward changed), runs the PPO burst for Round $r$, evaluates on `val` and `eval_corpus.json`, saves `round_{r}_model.zip`, and writes `round_result.json` (containing `val_metric`, `corpus_pass_rate`, `sample_failure_trajectories`, `action_entropy`, and `reward_breakdown_stats`).
2. **Best-on-Validation Checkpoint Tracking:**
   - The host compares `round_result.val_metric` against `best_val_metric`. If improved (and `corpus_pass_rate` did not regress), the host copies `round_{r}_model.zip` $\to$ `best_val_model.zip` and `env_module.py` $\to$ `best_val_env_module.py`.
3. **Host-Side LLM Auto-Tuning Between Rounds ($r < \text{max\_rounds}$):**
   - If `val_metric < target_metric_threshold`:
     - The **host process** (which has network access, verified LLM credentials, and `llm_usage` metering bound to the conversation's `owner_id`) calls `AutoTunerAgent.diagnose_and_tune(env_code, round_result.sample_failure_trajectories, round_result.action_entropy)`.
     - The host validates the candidate `tuned_env_code` by running `bwrap ... --mode smoke`.
     - If `smoke` passes, `tuned_env_code` and updated hyperparameters (`ent_coef`, `learning_rate`) are staged into the next round's `bwrap` worker call.
     - If Round $r+1$ degrades `val_metric` or regresses `eval_corpus` relative to `best_val_metric`, the Verifier Gate rolls `env_module.py` and `model.zip` back to `best_val_env_module.py` and `best_val_model.zip` before proceeding!
4. **Final Held-Out Scorecard on `best_val_model.zip`:**
   - After the final round completes (or early-stops on target met), the host runs one final `bwrap` evaluation pass on the untouched `test` split using **`best_val_model.zip`** (closing the `progress.md` §11.4 single-shot selection residual simultaneously).

---

### Issue 3: Switching-Cost Moat Inversion & Missing "Golden Eval Corpus" Vault (Crossfire Claim 5)

#### 3.1 Crossfire Finding & Adversarial Counter-Evidence
- **Contradiction:** In `MASTER_SPEC_AND_BUILD_PLAN.md` §1.5, the pitch claimed that accumulated task configs, critique logs, and reward functions create a switching-cost moat. Crossfire refuted this:
  > *"The claim's lock-in mechanism depends on artifacts being non-portable, but the product's own transparency posture (exportable diffs, visible reward DSL, standard ONNX exports) makes the accumulated policy re-derivable in-context by a frontier model — so switching cost is lowest for exactly the users who have accumulated the most, inverting the retention claim... treat the resulting validated eval corpus, not exported configs, as the retention asset."*

#### 3.2 Current State in RLHarness `v5.4`
- `v5.4` stores `episodes` (rolling buffer of recent rollouts) and `scorecard.json` (summary statistics on the $15\%$ `test` split), but there is **no persistent, cumulative behavioral regression suite (`eval_corpus.json`)** that captures every human judgment, locked trajectory step, and critique assertion into an executable test suite for that domain.

#### 3.3 Elaborate Technical Remediation (`backend/rl_ide/autopilot/eval_corpus.py`)
We will build `EvalCorpusStore`—a first-class **Behavioral Regression Vault** per conversation/task that grows automatically as the user interacts with the studio:

1. **Data Schema (`GoldenEvalCase` & `EvalCorpusReport`):**
   ```python
   from pydantic import BaseModel, Field
   from typing import List, Optional, Dict, Any

   class GoldenEvalCase(BaseModel):
       case_id: str
       source: str = Field(description="'val_seed' | 'user_locked_step' | 'user_critique' | 'passive_log'")
       obs_vector: List[float]
       state_summary: Dict[str, Any] = Field(default_factory=dict)
       expected_actions: List[int] = Field(default_factory=list, description="If non-empty, policy greedy action must be in this set")
       forbidden_actions: List[int] = Field(default_factory=list, description="Policy greedy action must NOT be in this set")
       action_labels: List[str] = Field(default_factory=list)
       rationale: str = ""
       created_at: str = ""

   class EvalCorpusReport(BaseModel):
       total_cases: int
       passed_cases: int
       failed_cases: int
       pass_rate_pct: float
       by_source: Dict[str, Dict[str, int]]
       regressed_case_ids: List[str] = Field(default_factory=list)
       failed_details: List[Dict[str, Any]] = Field(default_factory=list)
   ```

2. **Three Automatic Harvesting Channels into `EvalCorpusStore`:**
   - **Channel A — Pre-Training Held-Out Seed Cases (`source="val_seed"`):** When a dataset or environment is initialized, automatically seed $20\text{--}50$ high-confidence disjoint cases from the `val` split (or environment teacher demonstrations) into `workspaces/<id>/eval_corpus.json`.
   - **Channel B — Interactive Play & Locked Steps (`source="user_locked_step"`):** Whenever a user plays in `/interact` or locks an episode/step (`POST /api/v4/conversations/{id}/corpus/lock` or via `/interact` with `{"lock_assertion": true, "expected_action": ...}` or `{"forbid_last_agent_action": true}`), persist that exact `(obs_vector, expected_actions, forbidden_actions)` into `eval_corpus.json`.
   - **Channel C — Plain-English Critiques (`source="user_critique"`):** When a user submits a critique in chat (e.g., *"Stop choosing Clearance Discount when days remaining > 7"* or *"Never Active Ping when threat is close"*), extract any matching failure states from the recent episode buffer where the agent took the criticized action and register them as `forbidden_actions` assertions in `eval_corpus.json`.

3. **Continuous Corpus Evaluation & UI Visibility:**
   - Every training round, critique burst, and ONNX export runs `eval_corpus.evaluate_policy(predict_fn)` and embeds the `EvalCorpusReport` inside `scorecard.json` (`scorecard["eval_corpus"]`) and `SessionState`.
   - The Frontend Header and Telemetry Panel display a prominent badge:
     **`🛡️ Golden Eval Corpus: 48/48 Verified Cases (100.0%)`**—making the accumulated test suite the centerpiece of customer retention.

---

### Issue 4: Cold-Start Labeling Friction & Single-Shot Checkpoint Selection (Crossfire Claim 2 & `progress.md` §11.4)

#### 4.1 Crossfire Finding & `progress.md` §11.4 Residual
- **Crossfire Claim 2 Counter-Evidence:**
  1. *"User must produce and verify a few hundred correctness examples before the first policy beats a frontier model; deadline-pressed developers skip this and keep the API."*
  2. *"Don't ship the RL-trained compact policy as the default path that users must hand-label their way into; instead make the harness passively harvest the critiques users already emit while they use a frontier model (accepts/rejects, edit diffs, review comments), and only train+downgrade to the local policy once ~300 such signals exist — with the 'fit' claim gated on a pre-registered rubric scored against a held-out critique set authored before training."*
- **`progress.md` §11.4 Residual:**
  > *"Selection is single-shot. Rounds are selected on `val`, but no checkpoint is retained per round, so the scorecard reports the final model rather than the best-on-validation one."*

#### 4.2 Current State in RLHarness `v5.4`
1. `feature_engineering.py` assumes an uploaded CSV has a single ground-truth `label_column`. It does **not** natively parse **Frontier Interaction / Critique Logs** where each row represents a past LLM call (`input_features...`, `llm_proposed_action`, `human_feedback: "accepted" | "rejected" | "edited"`, `corrected_action`, `critique_comment`).
2. `loop_controller.py` evaluates `test` at the end of training on whatever weights exist after the last round $R$, even if Round $1$ or Round $2$ had a higher `val` score before a later round overfit or drifted.

#### 4.3 Elaborate Technical Remediation (`profiler.py`, `feature_engineering.py`, & `loop_controller.py`)

1. **Best-on-Validation Checkpoint Retention (`loop_controller.py` & `sandbox_runner.py`):**
   - Maintain `best_val_score: float = -1.0` and `best_val_round: int = 0` across all rounds in a training job.
   - Whenever Round $r$'s validation score `val_score >= best_val_score`, save the model checkpoint to `workspaces/<id>/best_val_model.zip`.
   - At the end of the training loop (before computing the final `test` split `scorecard.json` and exporting `model.onnx`), explicitly reload `best_val_model.zip` so the customer always receives the highest-performing validation checkpoint, never a degraded late-round checkpoint.Record `selected_round: best_val_round` in `scorecard.json`.

2. **Passive Frontier Critique & Interaction Log Harvester (`ingest/profiler.py` & `synthesizer/feature_engineering.py`):**
   - Detect when an uploaded CSV/JSONL/SQL table is a **Frontier Interaction / Critique Log** by checking for dual columns:
     - **Proposed Action Column:** `llm_action`, `model_action`, `predicted_action`, `proposed_action`, `agent_action`, `candidate_action`.
     - **Human Feedback / Outcome Column:** `feedback`, `accepted`, `is_accepted`, `verdict`, `human_verdict`, `status` (with values like `accepted`/`approved`/`true`/`1` vs. `rejected`/`edited`/`overridden`/`false`/`0`), and optional `corrected_action` / `human_action` / `critique_text`.
   - When a Critique Log is detected:
     1. **Positive Training / BC Set:** Rows where `accepted == True` use `proposed_action` as the target label; rows where `accepted == False` and `corrected_action` is present use `corrected_action` as the target label.
     2. **Negative Constraint / Held-Out Critique Set:** Rows where `accepted == False` (the human rejected or critiqued the frontier model's choice) have their `(obs_vector, forbidden_actions=[proposed_action_idx], expected_actions=[corrected_action_idx] if present)` automatically pre-registered into `EvalCorpusStore` (`source="passive_log"`)!
     3. **Cold-Start Sample Volume Advisory:** If the uploaded critique log has fewer than $100$ usable rows, record a non-blocking `sample_size_warning` in `scorecard.json` and the intake summary noting how many additional logged decisions are recommended to reach statistical power.

---

### Issue 5: Missing Break-Even Volume & Payback Economics in `scorecard.json` (Crossfire Claim 1)

#### 5.1 Crossfire Finding & Adversarial Counter-Evidence
- **Contradiction:** Crossfire cited verified October 2026 LLM pricing benchmarks showing frontier API prices dropped $85\%$ (median flagship at $\$5.63 / 1\text{M}$ blended tokens; cheapest production-grade frontier model at $\$1.00 / \$5.00$ per $1\text{M}$ tokens).
- Crossfire concluded:
  > *"The 'train once vs. call repeatedly' comparison misstates both sides of the ledger — and the repeated-call baseline is a per-token meter whose price is falling, further compressing the amortization window the claim never states... compute break-even episode volume net of verifier maintenance and drift monitoring, and record the payback threshold before any training run."*

#### 5.2 Current State in RLHarness `v5.4`
- `v5.0` (§15.1) built `llm/metering.py` (`get_conversation_usage()`), which tracks exact token counts and USD spend per conversation.
- `v4.7` (§11) built `autopilot/scorecard.py`, which outputs held-out accuracy and baseline comparisons.
- **The Gap:** `scorecard.py` never queries `metering.py`, never measures ONNX wall-clock inference latency, and never computes the **Break-Even Call Volume (`calls_to_payback`)** against frontier API pricing.

#### 5.3 Elaborate Technical Remediation (`backend/rl_ide/autopilot/scorecard.py` & `exporter.py`)
We will add a dedicated `economics` block to `scorecard.json` (and display it in the Frontend Scorecard UI):

1. **Measure Real ONNX vs. Frontier Latency (`scorecard.py` / `exporter.py`):**
   - During `verify_behavioural_parity` (or scorecard generation), time $250$ sequential single-observation ONNX inference calls (`sess.run(None, {"input": obs})`) using `time.perf_counter()`.
   - Compute `onnx_p50_latency_ms` and `onnx_p99_latency_ms` (typically $0.05\text{--}0.40\text{ ms}$).
   - Compare against a configurable `frontier_reference_latency_ms` (default $850.0\text{ ms}$ for a network frontier API call) to report `latency_speedup_factor` $=\text{round}(850.0 / \max(0.01, \text{onnx\_p99\_latency\_ms}), 1)$.

2. **Compute Full-Ledger Break-Even Call Volume (`compute_payback_economics`):**
   - Query `conversation_store.get_usage_summary(conversation_id)` to obtain `actual_authoring_cost_usd` (the sum of all intake, synthesis, arena, and auto-tuning LLM calls for this conversation).
   - If the session ran in offline/deterministic mode where `actual_authoring_cost_usd == 0.0`, also compute `estimated_live_authoring_cost_usd` (defaulting to $\$0.015$ based on measured `deepseek-flash` runs of $\sim \$0.0073$ per turn in `progress.md` §15.4).
   - Include a maintenance reserve for periodic drift retraining: `drift_retrain_reserve_usd = max(actual_authoring_cost_usd, estimated_live_authoring_cost_usd) * 0.5`.
   - Total amortized investment:
     $$C_{\text{total}} = C_{\text{authoring}} + C_{\text{drift\_reserve}}$$
   - Using two reference frontier pricing tiers (configurable via environment variables):
     - **Budget Frontier Tier (e.g., GPT-6.1 Terra / DeepSeek V3 at $\$1.00/\$5.00$ per $1\text{M}$ tokens, $\sim 800$ tokens/decision $\approx \$0.0016$ per call):**
       $$\text{calls\_to\_payback\_budget\_frontier} = \left\lceil \frac{C_{\text{total}}}{0.0016} \right\rceil$$
     - **Flagship Frontier Tier (Median Flagship at $\$5.63 / 1\text{M}$ blended tokens, $\sim 800$ tokens/decision $\approx \$0.0045$ per call):**
       $$\text{calls\_to\_payback\_flagship\_frontier} = \left\lceil \frac{C_{\text{total}}}{0.0045} \right\rceil$$
   - Emit a clear `economic_verdict` string in `scorecard.json`:
     > *"Authoring + drift reserve: $0.022. Pays back after 5 decisions vs. Flagship LLM ($4.50/1k calls) or 14 decisions vs. Budget Frontier LLM ($1.60/1k calls), while delivering a 2,800x latency speedup (0.30ms vs 850ms) and 0 bytes leaving your boundary."*

---

## 5. Part 3: Step-by-Step Implementation Blueprint for DeepSeek (`v5.5`)

> **DIRECTIVE FOR DEEPSEEK:**
> Implement Phase P3 (`v5.5` — Crossfire Adversarial Audit Closure) across the backend and frontend. Maintain $100\%$ backward compatibility with all $163+$ existing tests. Follow the exact module specifications below.

### Step 1: Create `backend/rl_ide/autopilot/eval_corpus.py` (`EvalCorpusStore`)
- Implement `GoldenEvalCase`, `EvalCorpusReport`, and `EvalCorpusStore`.
- Store corpus files at `workspaces/<session_id>/eval_corpus.json`.
- Methods required:
  - `load(session_id) -> list[GoldenEvalCase]` and `save(session_id, cases)`.
  - `seed_from_validation_split(session_id, env, max_cases=30)`: extracts deterministic `(obs_vector, expected_actions=[true_action])` cases from the `val` split (for dataset tasks) or teacher demonstrations (for control tasks) without duplicating cases.
  - `add_case(session_id, obs_vector, expected_actions=None, forbidden_actions=None, source="user_critique", rationale="", state_summary=None) -> GoldenEvalCase`.
  - `harvest_from_critique(session_id, critique_text, episodes, action_labels) -> list[GoldenEvalCase]`: scans recent episodes for actions matching criticized labels or low-reward steps referenced by the critique and adds `forbidden_actions` assertions.
  - `evaluate_policy(session_id, predict_fn, baseline_passing_ids: set[str] | None = None) -> EvalCorpusReport`: runs `predict_fn(obs)` on every case in the store. A case passes if `pred_action in expected_actions` (when `expected_actions` is non-empty) AND `pred_action not in forbidden_actions`. Populates `regressed_case_ids` if any case in `baseline_passing_ids` failed.

### Step 2: Create `backend/rl_ide/autopilot/verifier_gate.py` (`VerifierGate` & `CritiqueTriage`)
- Implement `VerifierSnapshot`, `CritiqueTriageReport`, and `VerifierGate`:
  - `take_snapshot(session_id, env_code, model_path, val_metric, corpus_report) -> VerifierSnapshot`: saves `pre_gate_env.py` and `pre_gate_model.zip` in the session workspace.
  - `verify_and_triage(snapshot, post_val_metric, post_corpus_report, term_activation_stats, new_term_names, val_tolerance=1.5) -> tuple[bool, CritiqueTriageReport]`:
    - Checks `val_ok = (post_val_metric >= snapshot.val_metric - val_tolerance)` and `corpus_ok = (len(post_corpus_report.regressed_case_ids) == 0)`.
    - If `val_ok and corpus_ok`: returns `(True, CritiqueTriageReport(passed=True, status="retained", ...))`.
    - If false: restores `pre_gate_env.py` $\to$ `env_module.py` and `pre_gate_model.zip` $\to$ `model.zip`, and classifies failure into `"compiler_mismatch"` (new term activation rate $== 0.0$), `"cross_term_interference"` (new term activated $> 0$, but `val` dropped or golden cases regressed), or `"rl_variance"` (term activated, no golden contradiction, unconverged `val`), returning `(False, triage_report)`.
- Wire `VerifierGate` into `omni_agent.py` whenever a chat turn executes `REWARD_SHAPING` on a session that already has a trained model (`current_round >= 1`). Expose `critique_triage` in `StudioChatResponse` and `SessionState`.

### Step 3: Upgrade `sandbox_runner.py`, `worker.py`, and `loop_controller.py` (Host-Orchestrated Sandbox Auto-Tuning + `best_val_model.zip`)
- Add `--mode train_round` (or update `--mode train` with per-round checkpointing) so that:
  1. Whenever a round achieves a new best `val` metric without corpus regression, `best_val_model.zip` and `best_val_env.py` are persisted in the workspace.
  2. When a round plateaus below `target_metric_threshold` and `round_num < max_rounds`, the **host process** (outside `bwrap`) invokes `AutoTunerAgent` using the active LLM client, verifies the rewritten code in a `bwrap --mode smoke` worker, and runs the next round in `bwrap`.
  3. If the auto-tuned round degrades `val` or regresses `eval_corpus` compared to `best_val_metric`, the host rolls `env_module.py` and `model.zip` back to `best_val_env.py` and `best_val_model.zip` and logs a `verifier_gate_rollback` event.
  4. Before running final held-out `test` evaluation (`build_scorecard`) and ONNX export, load `best_val_model.zip` so the final reported scorecard and exported ONNX policy always correspond to the best validation checkpoint.

### Step 4: Passive Critique Log Harvesting in `ingest/profiler.py` & `synthesizer/feature_engineering.py`
- In `feature_engineering.py` / `profiler.py`, detect interaction/critique log schemas containing both a proposed/model action column (`llm_action`, `model_action`, `proposed_action`, `predicted_action`) and a feedback/outcome column (`accepted`, `feedback`, `verdict`, `human_verdict`, `corrected_action`, `human_action`):
  - Derive the ground-truth label for training from `proposed_action` when accepted (`True`/`1`/`"accepted"`/`"approved"`) or from `corrected_action`/`human_action` when overridden.
  - Extract rejected rows where `accepted` is `False`/`0`/`"rejected"` and provide a helper `extract_passive_critique_cases(df, plan) -> list[GoldenEvalCase]` that populates `forbidden_actions=[proposed_action_idx]` (and `expected_actions=[corrected_action_idx]` if present) into `EvalCorpusStore` (`source="passive_log"`) upon session creation.

### Step 5: Break-Even & Payback Economics in `autopilot/scorecard.py` & `export/exporter.py`
- In `scorecard.py`, add `compute_economics_summary(session_id, onnx_latency_stats=None) -> dict[str, Any]`:
  - Read `usage = get_conversation_store().get_usage_summary(session_id)`.
  - Compute `authoring_cost_usd`, `estimated_live_authoring_cost_usd`, `drift_retrain_reserve_usd`, `total_amortized_cost_usd`.
  - Compute `calls_to_payback_flagship_frontier` (at $\$0.0045/\text{call}$) and `calls_to_payback_budget_frontier` (at $\$0.0016/\text{call}$).
  - Measure ONNX single-step inference latency ($p_{50}$ and $p_{99}$ in ms) during scorecard / export parity verification, and compute `latency_speedup_factor` vs. $850\text{ ms}$ frontier latency.
  - Add `beats_best_baseline: bool` to `scorecard.json` (`True` if policy held-out metric $>$ best baseline metric).
  - Embed `eval_corpus` (`EvalCorpusReport.model_dump()`) and `economics` inside `scorecard.json`.

### Step 6: API & Frontend Surfacing (`server.py`, `TelemetryPanel.tsx`, `Header.tsx`)
- Add endpoints:
  - `GET /api/v4/conversations/{id}/corpus` — returns the session's `EvalCorpusReport` and list of `GoldenEvalCase` items.
  - `POST /api/v4/conversations/{id}/corpus/cases` — allows the user/UI to lock a step or add an expected/forbidden action assertion directly into `EvalCorpusStore`.
- Update Frontend (`TelemetryPanel.tsx` / `Header.tsx`):
  - Render the **Golden Eval Corpus** badge (`🛡️ Corpus: X/Y Verified`) and **Payback Economics** summary (`⚡ p99 Latency: ~0.2ms | Payback: N calls`) inside the Scorecard card.
  - If a chat critique is rolled back by the Verifier Gate, display the `critique_triage` reason (`compiler_mismatch`, `cross_term_interference`, or `rl_variance`) clearly in the timeline/chat.

---

## 6. Mandatory Verification Suite (`backend/tests/test_crossfire_closure.py`)

Create `backend/tests/test_crossfire_closure.py` with the following **10 behavioural tests** (all must run offline/fast under default `pytest` and pass $100\%$):

1. `test_eval_corpus_seeding_and_persistence`: Verifies `EvalCorpusStore` seeds disjoint cases from `val`, persists to `eval_corpus.json`, and evaluates a policy accurately.
2. `test_eval_corpus_harvests_user_critique_and_locked_steps`: Verifies `POST /conversations/{id}/corpus/cases` and critique harvesting append `forbidden_actions` / `expected_actions` cases and detect regressions when a policy violates them.
3. `test_verifier_gate_retains_improving_critique`: Verifies that a reward modification that maintains/improves `val` and passes $100\%$ of `eval_corpus` cases is retained (`passed=True`, `status="retained"`).
4. `test_verifier_gate_rolls_back_regressing_critique_and_triages_interference`: Simulates a poisonous reward edit that causes the policy to collapse to a single action (degrading `val` and regressing locked `eval_corpus` cases). Asserts `VerifierGate` automatically restores the pre-critique `env_module.py` and `model.zip` and classifies `triage.category == "cross_term_interference"`.
5. `test_verifier_gate_triages_dead_reward_term_as_compiler_mismatch`: Simulates a reward term whose condition never fires ($0\%$ activation across rollout steps) and fails to improve `val`. Asserts auto-rollback and `triage.category == "compiler_mismatch"`.
6. `test_host_orchestrated_sandbox_auto_tuning_runs_between_rounds`: Verifies that when a multi-round training run plateaus in Round 1 with `bwrap` isolation active, the host invokes `AutoTunerAgent` between Round 1 and Round 2 (no longer skipping with `auto_tune_skipped`) while keeping `train_round` and `smoke` execution inside `bwrap`.
7. `test_best_on_validation_checkpoint_selected_for_final_scorecard`: Runs a 3-round loop where Round 2 has higher `val` accuracy than Round 3. Asserts `best_val_model.zip` from Round 2 is restored and used for the final `test` split `scorecard.json` (`selected_round == 2`).
8. `test_passive_critique_log_ingestion_splits_accepted_and_rejected_rows`: Uploads a frontier interaction log CSV with columns `ticket_urgency, customer_tier, llm_action, accepted, corrected_action`. Asserts accepted/corrected rows train the policy while rejected `llm_action` rows are automatically registered as `forbidden_actions` in `EvalCorpusStore` (`source="passive_log"`).
9. `test_scorecard_economics_computes_payback_calls_and_onnx_latency`: Records mocked token usage in `llm_usage`, builds `scorecard.json` + ONNX export, and asserts `scorecard["economics"]` contains `authoring_cost_usd`, `calls_to_payback_flagship_frontier > 0`, `calls_to_payback_budget_frontier > 0`, `onnx_p99_latency_ms < 50.0`, and `latency_speedup_factor > 10.0`.
10. `test_end_to_end_crossfire_falsification_contract`: Runs the full pipeline on a dataset task: passive log upload $\to$ $70/15/15$ split $\to$ training $\to$ best-on-val checkpoint selection $\to$ scorecard with Wilson CI, baselines (`beats_best_baseline`), `eval_corpus` pass rate, and payback economics $\to$ ONNX behavioral parity.

---

## 7. Handoff & Completion Gate

Upon completing Steps 1–6:
1. Run the fast test suite (`pytest -m "not slow and not live_llm" -q`) and the full test suite (`pytest -q`).
2. Run the frontend typecheck and build (`npm run build`).
3. Append **§20. v5.5 — Phase P3: Crossfire Adversarial Audit Closure** to `progress.md` with measured outputs and test counts.