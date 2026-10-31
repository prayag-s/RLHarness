# Phase P3 (`v5.5`) — Refined Execution Plan

**Date:** October 31, 2026
**Baseline:** `v5.4` (main `a1c03f4`) — 219 backend tests defined; fast gate 155 passed / 2 skipped in 414 s at the start of this phase.
**Supersedes:** `crossfire_adversarial_audit_v5_5_remediation_specification.md` (DeepSeek draft) — kept for history; the eight corrections below are why this plan exists.
**Inputs:** `Crossfire_Decision_Memo.md`, `COMMERCIAL_READINESS_GAPS.md`, `progress.md` §9–§19.

---

## 0. Why this plan replaces the draft

The draft's **diagnosis** was accurate: I re-verified every one of its five gaps in the code. Its
**epistemology** was wrong: it graded a falsification audit with an instrumentation backlog. Eight
concrete corrections were applied before implementation.

| # | Draft said | Problem | Correction applied |
|---|---|---|---|
| 1 | "Claim 4 is 100 % resolved" | The claim was about build feasibility plus review/IT gates. `bwrap` closes the *runtime* half only; no named ML owner, no hostile-expression red team, no pilot IT approval exists in the repo. | Claim 4 recorded as **mitigated (runtime) / open (review + org)**. Hostile-expression suite added (§5.9). |
| 2 | Deliverables 1–5 close claims 1–5 | Instrumentation makes a claim *measurable*, not *true*. | Added **P3.0 pre-registered falsification protocol** that refuses to report closure on an unregistered rubric or a thin hold-out set. |
| 3 | Gate uses `val` for selection, gating **and** corpus seeding | The two gate conditions become correlated and `1.5 pp` on a 15 % split is ~1 row of noise. | Corpus seeds from the held-out split but the gate compares **paired** episode outcomes (exact sign test, min 20 pairs) instead of two point estimates. |
| 4 | `activation_rate == 0.0` → `compiler_mismatch` | Wrong in both directions: a penalty legitimately stops firing once the policy behaves, and a 1e-6 threshold flips on noise. | Ordered, deterministic **decision table** keyed on new-term activation, corpus regressions, paired val delta and entropy drift. |
| 5 | Single-observation corpus assertions | Stateful adapters carry hidden-ish state; a one-shot assertion can pass for the wrong reason. | Observation **prefixes** are stored for audit; the docstring and §6 state the limitation explicitly rather than implying full trajectory replay. |
| 6 | Gate covers reward shaping only | Chat also mutates environment mechanics and inference strategy — both retained on smoke-pass alone. | Gate applied to **all three mutation layers** (`reward`, `environment`, `inference`). |
| 7 | Economics: `get_usage_summary()`, fixed 850 ms, "$0.022 → pays back in 5 decisions" | Symbol does not exist (`metering.usage_report`), and the ledger omitted human labour — the dominant term. | Full **TCO ledger** with both bases (`token_only`, `with_human_review`), sensitivity, and stated caveats including the authoring data boundary. |
| 8 | Host auto-tune sends worker traces to the provider | Re-opens the data boundary and adds a prompt-injection path. | Redacted traces (raw obs → per-dimension ranges), untrusted-data framing, and `RL_IDE_ALLOW_HOST_LLM=0` for data-sovereign tenants. |

Also corrected: **the eval corpus is not a moat.** It is a quality asset and it is portable by
design. §1.5 of the master spec is superseded rather than re-housed (see `MASTER_SPEC_AND_BUILD_PLAN.md` §18).

---

## 1. Deliverables and acceptance criteria

### P3.0 — Pre-registered falsification protocol (`autopilot/crossfire_protocol.py`)
- Rubric registered **before** the first training step; SHA-256 over the rubric text.
- `record_training_start` stamps once and never overwrites.
- A verdict is refused (`closed: false`) when: no protocol, no training stamp, rubric hash mismatch,
  rubric registered after training started, or fewer hold-out judgements than `min_holdout` (default 100).
- **Acceptance:** `test_end_to_end_crossfire_falsification_contract`, `test_protocol_rejects_rubric_registered_after_training`.

### P3.1 — Golden eval corpus (`autopilot/eval_corpus.py`)
- `GoldenEvalCase` with `obs_prefix`, `expected_actions`, `forbidden_actions`, `source`, `waived`.
- Four growth channels: `val_seed`, `user_locked_step`, `user_critique`, `passive_log`.
- Content-addressed dedupe; corrupt rows never destroy the suite.
- **Acceptance:** `test_eval_corpus_seeding_and_persistence`, `test_eval_corpus_harvests_critique_and_locked_steps`.

### P3.2 — Independent verifier gate (`autopilot/verifier_gate.py`)
- Retention rule: keep iff **no previously passing case regressed** and held-out accuracy did not
  fall beyond tolerance *unless* a paired sign test cannot distinguish the drop from noise.
- Triage: deterministic ordered table → `compiler_mismatch` | `cross_term_interference` | `rl_variance` | `unclassified`, each with actionable advice.
- Applied in the training loop (mutation rounds), in chat mutations, and across all mutation kinds.
- **Acceptance:** four gate tests plus `test_triage_table_is_ordered_and_deterministic`, `test_verifier_gate_does_not_roll_back_on_noise`.

### P3.3 — Durable state and best-on-validation selection (`loop_controller.py`)
- `workspaces/<id>/session_state.json` so a restarted or per-round worker resumes round history.
- `best_val_model.zip` + `best_val_env_module.py` written when val improves; the final scorecard and
  the exported policy both use that checkpoint; `scorecard["selection"]` records `selected_round`.
- **Acceptance:** `test_best_on_validation_checkpoint_selection`, plus the existing scorecard suite.

### P3.4 — Per-round sandbox with host-side tuning (`sandbox_runner.run_rounds_in_sandbox`, worker `train_round`)
- One `bwrap` worker per round; the host tunes between rounds, so `auto_tune_skipped` is no longer the
  outcome whenever isolation is active.
- Convergence and target checks stay inside the worker; tuning decisions stay on the host.
- **Acceptance:** `test_host_orchestrated_rounds_tune_between_rounds` (asserts two `train_round` calls,
  exactly one host tune, and that `allow_host_llm=False` refuses tuning).

### P3.5 — Economics ledger (`autopilot/scorecard.py`)
- Measured token spend via `metering.usage_report`; human hours; retrain compute; drift review.
- Payback per tier on **two bases**, plus ±2× sensitivity and explicit caveats.
- **Acceptance:** `test_scorecard_economics_reports_both_bases_and_caveats`.

### P3.6 — Passive interaction-log harvest (`ingest/interaction_log.py`)
- Detected by action column + human verdict column; accepted/corrected rows become supervised rows.
- Rejections with a correction become hard `expected`+`forbidden` assertions; rejections without one
  become soft negatives and are reported as such.
- Cases are only created when the row matches a record the environment can encode.
- **Acceptance:** `test_passive_critique_log_ingestion_splits_accepted_and_rejected_rows`.

### P3.7 — API + UI surfacing
- `GET /api/v4/conversations/{id}/corpus`, `POST …/corpus/cases`, `POST …/corpus/harvest-passive`, `GET …/crossfire`.
- Telemetry panel: golden-corpus badge, selected checkpoint, Crossfire verdict, latency and payback.
- **Acceptance:** `npm run build` clean; endpoints covered by the API smoke tests already in the suite.

---

## 2. Claim status after this phase (the honest ledger)

| Claim | Before P3 | After P3 | Why it is not "closed" |
|---|---|---|---|
| 1 — market premise / payback | Open | **Instrumented** | The ledger now prices human maintenance and reports two bases. Whether any customer's volume pays that back is a business fact, not a code fact. |
| 2 — small policy beats frontier | Partly | **Testable** | Best-on-val selection + corpus + passive harvest remove the two hygiene defects (in-sample selection, cold-start labelling). The superiority claim still needs ≥100 frozen hold-out judgements per task. |
| 3 — critique → reward convergence | Open | **Bounded** | Every mutation is now judged by an oracle the mutation did not produce, and failures are triaged. It is bounded *to tasks that already have an oracle*; control tasks whose `is_success` came from the same generator remain circular, and the plan says so. |
| 4 — vibe-coded build | Contested | **Runtime mitigated / org open** | Sandbox + hostile suite are real. A named ML owner, IT approval and a production drift baseline are not code. |
| 5 — switching-cost moat | Contested | **Withdrawn** | Exportable corpus and diffs invert lock-in. The moat claim is removed rather than re-housed; retention now rests on integration and labelled history, stated plainly. |

---

## 3. Non-goals (unchanged)

- No marketplace, no model-hosting service, no frontier-model wrapper.
- No claim that a short burst "converges on the behaviour the user intended" without an independent verifier.
- No multi-customer adapter programme until a pilot's IT approval exists.
