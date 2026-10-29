# RLHarness — Design Audit, Issue Register & Phased Remediation

**Audit date:** 2026-10-29
**Repository state audited:** `main` at `0afa5f6` (plus the prior audit document)
**Scope:** v4.4 universal generative pipeline, agentic chat planner, autonomous training loop,
export contract, and the supporting frontend.
**Purpose:** Single register of every correctness/design deviation found in the audit, mapped to
the remediation phase that fixes it. Status is updated as phases land.

> **Post-remediation status (2026-10-29):** all phases 0–E executed. Backend suite **72 passed, 0 failed**;
> frontend `npm run build` clean. Every issue below is either **CLOSED** with a regression test or
> explicitly **ACCEPTED** with rationale (see §5 and §6). The `******` "Authorization" finding in the
> previous report was a **false positive caused by tool-output secret redaction** — the source already
> sends `Bearer {api_key}`; a transport regression test now guards it.

---

## 1. Verdict

The **layer decomposition is correct** for the stated objective:

```
intake → DynamicTaskBlueprint → universal env synthesis → universal arena
      → PPO loop (warm-start / evaluate / self-tune) → ONNX + runbundle
      ↔ code-aware ChatActionPlan planner (multi-target)
```

What is **not** correct is the trust model around that decomposition:

1. **Generated code is trusted before verification.** Modules are written and imported before the
   smoke test runs, and unverified code can be returned, persisted and trained on.
2. **Verification is not a contract check or a sandbox.** The `state_dict_schema` shared contract is
   never validated, and generated modules execute in-process with full privileges.
3. **The reported mastery metric is not a mastery metric.** Dataset episodes count as successful
   after one correct step; the deterministic placeholder always reports success at step 25.
4. **“Zero hardcoding” is not implemented.** Domain keyword lists, rover/HIPAA/burn-rate/arcade
   branches and board-specific patching remain in runtime code; the tests that guard this are name
   greps and string assertions.
5. **Session isolation and durability are missing.** One global model/env store, no training lock,
   process-local datasets, and destructive “clear all”.

Consequence: when the LLM path degrades, the product silently fabricates a trainable task and
reports mastery for it — the exact failure the v4.4 spec was written to prevent.

---

## 2. Verification baseline (before remediation)

| Check | Before remediation | After remediation |
|---|---|---|
| Backend `pytest` | 54 passed | **72 passed, 0 failed** (≈6 min) |
| Frontend `npm run build` | passed | **passed** (tsc + vite, 1 599 modules) |
| LLM transport | verified correct (false positive) | regression-tested (`Bearer` header per provider) |
| Metric trustworthiness | `any(step)` success; placeholder always 100 % | terminal-outcome metric; dataset accuracy; placeholders 0 % |
| Contract enforcement | none | declared `state_dict_schema` keys, label alignment, state mutability, success-reachability probe |
| Unverified code | written and imported | blocked at every write/import/train path |
| Session isolation | none | per-session training lock + 409 on `/interact`/`/nudge` races |

---

## 3. Issue register

Severity: **P0** blocker, **P1** high, **P2** design/spec drift, **P3** hygiene/usability.
Every issue is assigned to a phase and must be closed (fixed, or explicitly documented as accepted)
before the remediation is complete.

### P0 — Blockers

| ID | Issue | Evidence | Phase |
|---|---|---|---|
| F1 | DeepSeek/OpenAI/Ollama `Authorization` header is `f"******"` (redaction artefact). Default provider cannot authenticate; `require_verified_llm()` then blocks every endpoint. | `llm/client.py:115,172,236` | 0 |
| F2 | Unverified generated code is written to disk and imported (`synthesize()` writes + loads before `verify_and_heal`; `synthesize_universal_env` returns raw code on verification failure). | `synthesizer/env_builder.py:361-369,410-415` | C |
| F3 | Dataset mastery is “any step correct in the episode”: a random policy scores **100 %** while per-step accuracy is 16 %. | `synthesizer/env_builder.py:562`, `autopilot/loop_controller.py:898` | A |
| F4 | Deterministic placeholder fabricates success: `is_success = terminated and reward > 0` with a constant `step_progress` ⇒ 100 % mastery for a random policy. | `synthesizer/env_builder.py:201` | A |
| F5 | Reward rewrite is unsafe: regex insertion before the first `return` can emit invalid Python (early returns) and silently zeroes the reward of the dataset env (verified: 1.0 → 0.0) while still passing the smoke test. | `architect/omni_agent.py:1027-1120` | C |
| F6 | Training/omni paths ignore the `verified` flag and write/load the result anyway. | `architect/omni_agent.py:909-920`, `autopilot/loop_controller.py:557` | C |
| F7 | No session isolation: one global model/env store, no training lock, background tasks are not retained/awaited. Duplicate chat or chat+`/interact` races on the same PPO. | `api/server.py:752`, `architect/omni_agent.py:931` | B |

### P1 — High-impact correctness

| ID | Issue | Evidence | Phase |
|---|---|---|---|
| F8 | Inference-only requests also rewrite `compute_reward` and calibrate the critic (verified: `arena_only` plan still mutates the reward). | `architect/omni_agent.py:149-151,767` | C |
| F9 | Inference patcher is board-specific: hardcoded `range(9)` and `{0,2,4,6,8}` produce illegal actions for a 3-action env (verified: `last_move` ∈ {4,6,8}). | `architect/omni_agent.py:403` | C |
| F10 | Patch failures report success: unchanged code passes verification and the reply still says “Patched interactive_step()”; “Calibrated Critic” is printed when no model exists. | `architect/omni_agent.py:403,790` | C |
| F11 | First-turn arena is fabricated (`{"board":[0]*9}`) for every non-dataset domain and is generated before any environment exists. Real `get_render_state()` is used only in `/sessions/start`. | `architect/omni_agent.py:632`, `api/server.py:348` | D |
| F12 | Domain pivots do not re-anchor unless the user says “switch domain/reanchor/pivot”; “train a connect 4 agent” routes to training. | `architect/omni_agent.py:300` | D |
| F13 | Diagnostic guardrail swallows actionable requests (“Why …? Randomize it among good options” → zero mutations). | `architect/omni_agent.py:204` | D |
| F14 | Unsupported-capability matching is substring based: “flag fraudulent **invoice**s” is refused because of `voice`. | `architect/omni_agent.py:183` | D |
| F15 | Mechanics/re-anchor re-synthesis drops the dataset (`synthesize()` called without `dataset=`), bakes sinusoidal fake records, erases prior reward/inference patches, and appends “Rules: …” forever. | `architect/omni_agent.py:716,743,856` | C/D |
| F16 | Dataset env: target column guessed from 7 hardcoded names; `get_demonstrations` labels from the first feature (mismatch with `step`) so BC trains on wrong labels. | `synthesizer/env_builder.py:532,616` | D |
| F17 | Dataset rows are baked into `env_module.py` (and therefore exports) and are not restored after a process restart; loop/nudge/rollback instantiate data envs without records. | `synthesizer/env_builder.py:447`, `autopilot/loop_controller.py:660,752` | E |
| F18 | Training controls do not map to work: `min(1024, episodes*25)` cap; additive-vs-absolute rounds ambiguity; `continue_training`/`eval_every`/`hyperparameter_diffs` unused; any `%` becomes the mastery target. | `autopilot/loop_controller.py:45,258,291,417` | B |
| F19 | Fixed 50-step rollout/eval caps ignore the environment horizon. | `autopilot/loop_controller.py:823,1004` | A |
| F20 | Verification does not check declared `state_dict_schema` keys, action/observation label alignment, state mutability, or success reachability. | `synthesizer/verifier.py:85-205` | A |
| F21 | `clear_all()` deletes every directory under `workspaces/` and `runs/`; `delete()` leaves model/env/session/dataset in memory. | `state/conversation_store.py:395,419` | E |
| F22 | Selected intake answers are never sent by the studio chat UI and are ignored after the preview; `interview()` drops `conversation_history`; deterministic intake ignores answers. | `frontend/src/App.tsx:477`, `architect/intake.py:298` | D |

### P2 — Design / spec drift

| ID | Issue | Evidence | Phase |
|---|---|---|---|
| F23 | “Zero hardcoding” violated in runtime: keyword switchboards in `_plan_chat_action`, rover `sim-canvas`, HIPAA/burn-rate/arcade overlays, cardiac/payroll/clearance/liability/login_drop reward branches, `N columns`/`3x3` intake branches. | `architect/omni_agent.py:300-400,1027+`, `synthesizer/arena_builder.py:263-320`, `architect/intake.py:160-230` | D |
| F24 | Planner is not code-aware: only function names are passed; blueprint/arena are not; `surgical_instructions` are never executed; execution is a fixed per-target handler. | `architect/omni_agent.py:229` | D |
| F25 | `DynamicTaskBlueprint` is not the runtime authority; `to_agent_blueprint_spec()` hardcodes 85 %/4 rounds/200 episodes and drops reward semantics. | `architect/models.py:104-140` | D |
| F26 | Arena defects: duplicate `id="status-badge"` + unbalanced markup; cell index mapped to action index (wrong for column-drop); mock loop only understands `board`; restyle concatenates the whole prior HTML into the style query and sees only 6 000 chars; missing `node` counts as valid JS. | `synthesizer/arena_builder.py:75-91,193,249,382-384,467` | D |
| F27 | Auto-tuner deterministic rewrite is a no-op (`re.sub` replaces text with identical text) plus a header comment; hyperparameter diffs are computed but never applied. | `autopilot/auto_tuner.py:216-247` | B |
| F28 | Export contract weaker than documented: random Gaussian parity probe, `atol=1e-4` vs documented `<1e-5`, missing bundle files silently skipped, drift baseline from synthetic all-zero-action episodes, UI `driftStatus` never updated. | `export/exporter.py:340,400-450` | D |
| F29 | Intake questions are board-game shaped for every domain; blueprint thresholds are hardcoded in the conversion layer. | `architect/intake.py:364-430` | D |
| F30 | Security posture: no authn/z, `allow_origins=".*"` with credentials, `postMessage(..., "*")`, unescaped user/LLM text and dataset state interpolated into HTML/JS. | `api/server.py:72-80`, `synthesizer/arena_builder.py` | E |

### P3 — Tests, docs, hygiene

| ID | Issue | Evidence | Phase |
|---|---|---|---|
| F31 | Tests encode the bugs: name-grep “no hardcoding” tests; novel-domain test only asserts “no rover” + action count; inference test asserts openings ⊆ {0,2,4,6,8}; commercial tests assert `hipaa`/`burn-rate` markup and fabricated `best_metric >= 75/90`. | `tests/test_universal_agentic_pipeline.py:213-260`, `tests/test_agentic_chat_planner.py:95`, `tests/test_v4_2_commercial_suite.py:79,205,337,404,519` | E |
| F32 | `progress.md` claims (53/53, 11/11 browser, “zero hardcoding”, “verified”) contradict verifiable behaviour; the two audit failures were “fixed” by lowering training budgets and re-adding hardcoded branches. | `progress.md:1-10` | E |
| F33 | Repo hygiene: `verify_and_heal` leaves `env_module_try_*.py` / `temp_verify_*` artefacts; `load_synthesized_env_class` leaks `sys.modules` entries on every attempt. | `synthesizer/verifier.py:222-243`, `synthesizer/env_builder.py:57` | E |

---

## 4. Phased remediation

Rules for every phase: focused tests first, then the related regression modules, then the full
backend suite; update this register and `progress.md`; commit only when the whole suite is green.

### Phase 0 — Unblock the LLM transport
- Restore bearer-token headers for DeepSeek/OpenAI/Ollama.
- Add a transport test that captures the outgoing headers with a mocked HTTP client.
- **Exit gate:** header test passes; provider clients construct a `Bearer <key>` header.

### Phase A — Make the metric honest
- Episode-level success semantics: classification tasks report mean per-step accuracy; control tasks
  report the terminal outcome. No `any(step)` scoring.
- Remove the success-shaped placeholder: placeholder envs are flagged `is_placeholder`, always
  report `is_success=False` and zero reward, and are never presented as mastery. Production with no
  usable LLM raises `EnvironmentSynthesisError` instead of fabricating a task.
- Respect each environment's horizon (`episode_horizon` / `spec.max_episode_steps`, safe ceiling).
- Extend verification: declared `state_dict_schema` keys present and JSON-safe, labels aligned with
  spaces, state actually mutates, success-signal probe reported as uncertainty when inconclusive.
- **Exit gate:** focused metric/verifier tests plus regressions pass; no placeholder reports mastery.

### Phase B — Make training respond to controls and serialize work
- Unambiguous rounds: `max_rounds` = rounds to run in this call; session target round is cumulative.
- Observable, effective budget: `episodes_per_round × 25` timesteps with a high explicit ceiling.
- Apply auto-tuner hyperparameter diffs; make an unverified auto-tune a hard no-op with an event.
- Parse mastery targets only from explicit target/goal/threshold phrases.
- Per-session training lock; `/interact` refuses while training mutates the model; background tasks
  retained.
- **Exit gate:** tests prove budget scaling, exact round semantics, and duplicate/concurrent safety.

### Phase C — Make synthesis and mutation safe
- Every write/import/train/hot-reload gated on `verified`.
- Remove `_repair_code`’s silent substitution of a different environment; failures are explicit.
- Replace the regex reward rewrite with an AST-safe transformation that preserves early returns and
  unrelated terms, sanitizes user text, and reports “not applied” instead of corrupting code.
- Inference-only plans never touch the reward or trigger training.
- Action-space-aware inference patch; illegal indices impossible; report not-applied when the
  pattern is absent.
- Detect observation/action space changes and reset the policy deliberately with a user-visible note.
- **Exit gate:** no unverified code can be persisted/used; inference-only has zero reward side
  effects; patched code is legal for every action space.

### Phase D — Close the hardcoding gap and enforce the contract
- Replace domain keyword lists with a domain-free intent grammar (verbs + visual/style nouns) and
  word-boundary capability patterns; re-anchor on explicit pivots.
- Generate the arena from the actual `reset()` render state (or a neutral schema sample), never a
  fabricated 3×3 board; honor user-requested labels/meters generically; remove rover/HIPAA/burn-rate
  overlays; fix markup, escaping, action mapping and restyle input.
- Intake: generic structure inference (quoted labels, `RxC` grids, `N columns/options`), task-shaped
  questions, answers applied mid-session, conversation history forwarded.
- Wire or remove dead controls (`eval_every`, `driftStatus`), align export parity/tolerance and
  required bundle files.
- **Exit gate:** novel-domain tests show no domain switchboard dependency and contract alignment
  across game, simulation and data tasks.

### Phase E — Isolation, security, tests, docs
- Scope `clear_all`/`delete` to conversation-owned artefacts and clear in-memory state.
- Persist dataset records for resume; keep the module import cache bounded.
- HTML/JS escaping, restricted CORS, optional API token, iframe target origin.
- Rewrite the “zero hardcoding” and commercial assertions to behavioural contracts; add the new
  metric/safety/transport tests.
- Update `progress.md` with the honest state and the measured numbers.
- **Exit gate:** full suite green; every issue in this register marked closed or explicitly accepted.

---

## 5. Closure ledger

`CLOSED` = fixed with a regression test; `ACCEPTED` = documented residual risk with rationale;
`NOT AN ISSUE` = investigated and disproved.

| ID | Status | Resolution / evidence |
|---|---|---|
| F1 | **NOT AN ISSUE** | `llm/client.py` already sends `Bearer {api_key}` (and `Bearer {api_key or 'ollama'}`). The earlier `******` was output redaction. `tests/test_phase_hardening.py::test_llm_clients_send_bearer_authorization` now asserts the header for deepseek/openai/ollama with a mocked transport. |
| F2 | CLOSED | `JITEnvironmentSynthesizer.synthesize()` runs `verify_and_heal` **before** writing/importing; `synthesize_universal_env` raises `EnvironmentSynthesisError` instead of returning raw code. `test_unverified_environment_is_not_written`. |
| F3 | CLOSED | `episode_metric_value()`: classification tasks report mean per-step accuracy, control tasks report the terminal outcome. `test_dataset_metric_is_accuracy_not_any_step`, `test_random_policy_on_dataset_env_does_not_report_mastery`. |
| F4 | CLOSED | Placeholder envs declare `is_placeholder`, always return `is_success=False` and reward 0; production without a usable LLM raises. `test_placeholder_env_never_reports_mastery`. |
| F5 | CLOSED | `_safe_reward_rewrite()` replaces the regex rewriter: AST-checked insertion before the final return only, refuses anonymous breakdowns/trailing code, sanitizes user text. `test_reward_rewrite_is_safe_and_preserves_early_returns`. |
| F6 | CLOSED | Omni and loop paths only persist/load verified code; auto-tune failures emit `auto_tune_failed` and keep the previous module. `test_auto_tune_failure_keeps_previous_code`. |
| F7 | CLOSED | Per-session `threading.Lock` around `run_loop`/`nudge_session`, `is_busy()` guard on `/interact` (409), background tasks retained in `_background_tasks`. `test_training_lock_prevents_concurrent_runs`, `test_interact_rejected_while_training`. |
| F8 | CLOSED | `INFERENCE_STRATEGY` maps to `UPDATE_INFERENCE`; the reward layer keys off `REWARD_SHAPING` only. `test_inference_only_does_not_mutate_reward`. |
| F9 | CLOSED | Action-space-aware patch: legal-action masking, `n_actions` bound, top-k fallback. `test_inference_patch_is_action_space_aware` (9- and 5-action envs). |
| F10 | CLOSED | Patcher returns `(code, applied)`; replies only claim success when applied+verified; "critic calibrated" text removed when no model exists. |
| F11 | CLOSED | First turn synthesizes and reset-probes the environment before arena synthesis; neutral schema sample only when unreachable; synthesis failure returns an explicit error. |
| F12 | CLOSED | Re-anchor on "switch/pivot to", "change the domain", "train a/an … agent" (with a live env), "instead". Covered in `test_unsupported_capability_honest_refusal`. |
| F13 | CLOSED | Discussion requires a question form **and** no actionable clause after `?`/conjunction. Verified for "Why …? Randomize it …" vs pure diagnostics. |
| F14 | CLOSED | Word-boundary capability patterns; "invoices" no longer matches "voice". Tested. |
| F15 | CLOSED | Mechanics/re-anchor pass `dataset=` and use `pending_directive` (no summary growth); a no-op re-synthesis is reported instead of overwriting. Policy reset is explicit when spaces change. |
| F16 | CLOSED | Generic `infer_label_column()` (conventional names → low-cardinality numeric), demonstrations use the same truth as `step()`, and the target column is excluded from observations (no leakage). |
| F17 | CLOSED (with accepted residual) | Records are persisted to `workspaces/<id>/dataset_records.json` and reloaded by loop/rollback/nudge/interact/export. Residual: up to 100 sample rows are still embedded in `env_module.py` so exported runbundles remain standalone-executable (see §6). |
| F18 | CLOSED | `max_rounds` = rounds to run in this call (cumulative target); `episodes_per_round × 25` timesteps with a 100 000 ceiling; tuner hyperparameters applied; explicit target phrases only; unused `continue_training`/`eval_every` removed. |
| F19 | CLOSED | `episode_step_limit()` uses `episode_horizon` / `spec.max_episode_steps` (fallback 200). `test_episode_horizon_respected` (80-step success). |
| F20 | CLOSED | Smoke test verifies declared schema keys, label/length alignment, state mutability, and reports success reachability as uncertainty. `test_state_dict_schema_contract_enforced`. |
| F21 | CLOSED | `clear_all()` only deletes conversation-owned directories; `delete()` clears in-memory model/env/session and unreferenced datasets. `test_clear_all_is_scoped`. |
| F22 | CLOSED | Studio chat forwards `selected_options`; `apply_answers_to_blueprint()` is applied mid-session; `conversation_history` is forwarded to the intake. `test_intake_answers_applied_mid_session`. |
| F23 | CLOSED | Domain nouns removed from planner, arena and reward rewriting; arena overlays are generic user-requested elements. `test_no_domain_routing_tables_in_runtime`, `test_zero_domain_hardcoding_in_codebase`. |
| F24 | CLOSED (limited) | The planner receives method bodies (`interactive_step`, `compute_reward`, ≤2 000 chars each), the blueprint JSON and the structural summary. Execution remains handler-based by design; `surgical_instructions` are advisory. |
| F25 | CLOSED | `DynamicTaskBlueprint` carries `success_metric_name`, `target_metric_threshold`, `max_auto_tune_rounds`, `episodes_per_round`; conversions no longer hardcode 85/4/200. |
| F26 | CLOSED | Duplicate `status-badge`/unbalanced markup fixed; cell→action mapping only when counts match; generic mock animation; restyle sees the full document, no prior-HTML keyword bleed; missing `node` no longer counts as valid. |
| F27 | CLOSED | Auto-tuner no-op substitutions deleted; it reports `applied=False` without an LLM; hyperparameter diffs are applied to PPO. |
| F28 | CLOSED (with accepted residual) | Parity uses real environment observations and `atol=1e-5`; bundles fail if `env_module.py`/`arena.html` are missing; drift baseline uses real observations; UI reports `unmonitored` instead of a fake "healthy". Live drift integration remains out of scope (§6). |
| F29 | CLOSED | Intake questions are task-shaped (objective/arena/interaction/expectations) and dataset-aware. |
| F30 | PARTIAL → ACCEPTED | CORS restricted to local origins, optional `RL_IDE_API_TOKEN`, HTML/JS escaping, iframe source validation retained. Residual: no built-in user auth and `postMessage(..., "*")` for opaque-origin sandboxes (local single-user studio) — §6. |
| F31 | CLOSED | Tests rewritten to behavioural contracts (metric, legality, mutation safety, schema, transport, isolation). |
| F32 | CLOSED | `progress.md` rewritten with the measured state and no unverified claims. |
| F33 | CLOSED | `sys.modules` cache bounded (64 entries); `env_module_try_*` artefacts deleted after each attempt. |

---

## 6. Accepted residual risks

1. **No built-in user authentication.** The studio is a local, single-user application; an optional
   shared secret (`RL_IDE_API_TOKEN`) can gate all `/api/` routes. A full identity layer is out of scope.
2. **PostMessage target origin `"*"`.** The arena runs in `sandbox="allow-scripts"` with an opaque
   origin, so a specific target origin cannot be named; inbound messages are validated by
   `event.source === iframe.contentWindow`, and outbound payloads contain no secrets.
3. **Sample rows embedded in generated dataset modules.** Up to 100 records are baked into
   `env_module.py` (and therefore exported bundles) so the standalone `run_local.py` works offline.
   Full datasets are persisted separately and are never embedded. Treat exported bundles as
   dataset-bearing artefacts.
4. **Placeholder environments.** In test/offline mode a generic, clearly-flagged placeholder is used
   (`is_placeholder=True`, 0 % metric, no fabricated success). Production requires a verified LLM and
   raises `EnvironmentSynthesisError` otherwise.
5. **Advisory surgical instructions.** The planner produces `surgical_instructions`, but execution is
   performed by the fixed, verified handlers for each `ExecutionTarget`. Free-form patch application
   is intentionally not automated.
6. **Drift monitoring is not wired to live inference.** Baselines are exported and the UI reports
   `unmonitored` rather than a fabricated healthy state.
