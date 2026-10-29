# RLHarness Code Audit Findings & Remediation Plan

**Audit date:** 2026-10-29  
**Repository state audited:** `main` at `0afa5f6`, with pre-existing uncommitted changes in seven files.  
**Scope:** Compare the current implementation with the v4.4 universal synthesis / agentic planner objective and the preceding product specs.  
**Purpose:** Record correctness, design, and spec-conformance findings; define a phased fix, regression, commit, and resume workflow.

> **Update (2026-10-29):** remediation was executed as phases 0–E. See
> [DESIGN_AUDIT_AND_PHASED_REMEDIATION.md](./DESIGN_AUDIT_AND_PHASED_REMEDIATION.md) for the issue
> register (F1–F33), the closure ledger and the accepted residual risks. Backend suite: **72 passed,
> 0 failed**; frontend build clean.
>
> This document is a **restored snapshot** of the original audit (the working copy was lost while the
> remediation test suite was running). Its findings are preserved below and are all accounted for in
> the remediation document.

## Executive summary

The broad architecture is present: conversational intake, generated Gymnasium environments, arena HTML, a multi-target chat planner, autonomous PPO rounds, and ONNX/runbundle export. However, several core guarantees were not reliable:

1. **The reported mastery metric could be misleading or impossible to achieve.** Dataset episodes counted as successful after any one correct step; the schema-driven fallback fabricated success; fixed 50-step rollout caps could prevent longer tasks from reaching their goal.
2. **Training controls did not map consistently to actual work.** PPO training was capped at 1,024 timesteps per round for ordinary configured values, while user-configured rounds and episodes had inconsistent continuation semantics.
3. **Generated code was not consistently treated as untrusted until verification succeeded.** Some paths persisted or continued after a failed verification result.
4. **The “zero hardcoding” requirement was not met.** Domain-specific keyword routing, visual rendering, reward rewrites, and task-shape guesses remained in runtime code; the mechanical tests were too narrow to catch them.
5. **Session and interactive operations were not safely isolated.** Concurrent training and interaction could access shared model/session state; persisted datasets were process-local; some resume paths constructed data environments without their records.
6. **Arena generation and restyling could render the wrong state or degrade the interface.** Initial previews could be fabricated for a 3×3 board, and generic grid controls did not necessarily correspond to the environment’s action space.

These were design-level concerns, not just isolated test failures.

## Verification snapshot (at audit time)

- Frontend production build: `npm run build` **passed** (TypeScript and Vite).
- Backend tests: **52 passed, 2 failed**.
- Failing tests:
  - `tests/test_universal_agentic_pipeline.py::test_zero_hardcoding_novel_domain_synthesis`: generated arena contained the literal `sim-canvas` for a Connect-4 case.
  - `tests/test_v4_1.py::test_conversation_persistence_and_resumability`: interaction payload `{"action": "human_move", "cell": 4}` reached generated environment code as a non-numeric action and raised `ValueError`.
- Python syntax check of the seven modified Python files: **passed**.
- The audited dirty diff covered `backend/rl_ide/api/server.py`, `backend/rl_ide/architect/omni_agent.py`, `backend/rl_ide/llm/prompts.py`, `backend/rl_ide/synthesizer/arena_builder.py`, `backend/rl_ide/synthesizer/env_builder.py`, `frontend/src/App.tsx`, and `frontend/src/components/SandboxedArenaIframe.tsx`.

## Findings

### P0 — Objective and product-trust blockers

| ID | Finding | Evidence and consequence |
|---|---|---|
| A1 | **Dataset mastery was not classification accuracy.** | The dataset environment set `info["is_success"]` to `is_correct` on each record (`env_builder.py`). The evaluator defined episode success as `any(step.is_success)` (`loop_controller.py`). A 25-step episode was therefore successful after one correct classification. `accuracy_pct` was present but did not drive the metric. |
| A2 | **The schema-driven fallback could manufacture success.** | `_generate_schema_driven_env_code` used random state changes, a constant positive `step_progress` reward, and reported success when the 25th step terminated with positive reward. When generation fell back, the product could report mastery for a task the fallback never implemented. |
| A3 | **Fixed 50-step caps could make valid tasks unevaluable.** | PPO rollout collection and deterministic evaluation stopped at 50 steps, regardless of an environment’s own horizon. |
| A4 | **No synthesis-time success/reachability validation.** | `SandboxSmokeTester` validated spaces, basic stepping, JSON serialization, demonstrations and interaction, but did not check the declared render-state contract or whether success could occur. |
| A5 | **Training requests could race on shared model/session state.** | The chat endpoint created background `run_loop` tasks for each training request with no per-session lock. `/interact` also invoked inference on the shared model while training could mutate it. |
| A6 | **The initial arena used fabricated state rather than the environment contract.** | First-turn preview seeded `{"board": [0] * 9, ...}` for every non-dataset task, which could misrender unrelated domains. |

### P1 — Correctness and recovery defects

| ID | Finding | Evidence and consequence |
|---|---|---|
| B1 | **Deterministic reward rewrite could corrupt generated code.** | `_rewrite_reward_function` located the first `return` with a regex, inserted before it, and assumed a local `bd` variable. An early return could produce invalid indentation or unreachable/duplicate returns; the deterministic fallback also injected domain-specific reward terms. |
| B2 | **Deterministic auto-tuning usually did not change a universal environment’s behaviour.** | `AutoTunerAgent._rewrite_environment_code` applied literal regex substitutions (some replacing text with identical text); otherwise it only prepended a comment. |
| B3 | **Failed verification did not always stop mutation/training.** | The `verified` result was not used to prevent `healed_code` from being written; `AutoTunerAgent.tune` returned `healed_code` even when `verified=False`. |
| B4 | **Inference-only planner targets could also trigger reward rewriting.** | `classify_intents` mapped `INFERENCE_STRATEGY` to `UPDATE_AGENT_CHARACTERISTICS`, and the reward layer ran for that intent. |
| B5 | **Inference patcher was board-game-specific.** | `_patch_interactive_inference` assumed a flat `self.board`, empty cells equal to zero, nine strategic opening cells, and produced illegal action indices for other action spaces. |
| B6 | **A broad visual/play keyword branch could regenerate the environment.** | `is_play_feature` checked terms such as `"switch"`, `"both"` and `"symbol"`; a false positive could re-synthesize the environment and erase prior mechanics or inference patches. |
| B7 | **Environment changes could leave model and environment incompatible in live interaction.** | Observation/action compatibility was considered in `run_loop`, but not before `/interact` loaded or used the model. |
| B8 | **Arena controls were not guaranteed to match action semantics.** | Generic grid rendering mapped cell index to action index; for column-drop boards these indices are not legal actions. Mock animation flattened only some board shapes. |
| B9 | **Interaction normalization could prioritize a non-action string over a numeric cell.** | `InteractRequest.get_effective_payload()` aliased the first available action key without checking its type. |
| B10 | **Continuation and training parameter semantics were inconsistent.** | `run_loop` treated `max_rounds` as additive when it was less than or equal to completed rounds, but as absolute otherwise; `continue_training` was accepted but unused. |
| B11 | **Configured episodes did not scale training beyond a cap.** | PPO timesteps were computed as `min(1024, episodes_per_burst * 25)`, so 200 and 1 000 produced the same budget. |
| B12 | **Target percentage parsing could mistake ordinary percentages for mastery targets.** | `parse_nudge_meta_command` accepted any percentage in the message. |
| B13 | **Resume could pair stale environment code with newer weights.** | Conversation reconstruction preferred the database `env_code` and only read the workspace file when absent. |
| B14 | **Dataset-backed continuation was not durable across process restarts.** | Datasets lived in `ingested_datasets_store`; several loop/nudge/reload paths instantiated environments without `dataset_records`. |
| B15 | **Conversation clearing/deletion had overly broad and incomplete effects.** | `ConversationStore.clear_all()` removed every directory under the workspace root and `runs/`; in-memory stores were not cleared on deletion. |
| B16 | **Full episode traces were repeatedly serialized into SQLite.** | Episodes include full per-step traces and are persisted as JSON, causing write amplification. |

### P2 — Architecture/spec drift and misleading tests

| ID | Finding | Evidence and consequence |
|---|---|---|
| C1 | **“Zero hardcoding” was not implemented across the runtime.** | `_plan_chat_action` had a deterministic keyword fallback; intake contained special handling for 3×3 games and column counts; arena rendering included spatial/submarine/sonar/HIPAA/burn-rate/arcade branches; reward rewriting had domain-specific branches. |
| C2 | **The mechanical “zero-hardcoding” tests were too narrow.** | `test_no_keyword_routing_lists_in_omni_agent` checked a few variable names; `test_zero_domain_hardcoding_in_codebase` searched a fixed list of domain terms. |
| C3 | **Deterministic intake asked board-game questions for unrelated tasks.** | `_deterministic_intake` had board-oriented objective, interaction and 3×3 options. |
| C4 | **“Discussion”/unsupported responses depended on broad phrase matching.** | Substring lists could route ambiguous requests to mutation or refusal without a structured clarification step. |
| C5 | **Stale arena HTML made template heuristics sticky.** | `_generate_deterministic_arena_html` appended the entire prior HTML to `style_query`. |
| C6 | **Arena LLM update received a truncated document and had weak acceptance checks.** | Restyling sent only the first 6 000 characters, and syntactic validity is not proof of preserved interaction/telemetry. |
| C7 | **User-selected intake answers were not consistently used.** | The UI maintained answers, but the studio chat call did not send `selected_options`. |
| C8 | **Some documented tuning/monitoring features were not connected end-to-end.** | Auto-tuner hyperparameter differences were computed but not applied; `eval_every` and `continue_training` were unused; no generated environment implemented `get_critical_actions`; frontend `driftStatus` was never updated. |
| C9 | **ONNX export contract was weaker than documented.** | Parity compared random Gaussian observations with `atol=1e-4`, while docs claimed `<1e-5` and realistic observations. |

### P3 — Secondary robustness and usability

- Selecting a replay episode sent only `{episode_id, replayed}` as arena state, blanking the visual stage.
- Arena script validation treated a missing `node` executable as success.
- The arena template loads Tailwind from a remote CDN, so offline environments may render without styling.
- Task titles, action labels and state values were interpolated into HTML/JavaScript without consistent escaping.
- Some endpoint and task exceptions were logged or returned inconsistently; broad catches could convert failures into fallback behaviour that looked successful.
- UI polling had a silent error path, while background tasks and broadcast futures were not retained.

## Phased remediation (as executed)

- **Phase A — Make the metric honest.** Episode-level success semantics; dataset accuracy; remove success-shaped fallbacks; respect the environment horizon; extend verification to the declared contract; add focused tests.
- **Phase B — Make training respond to controls and serialize work.** Unambiguous round/budget semantics; per-session lock; correct continuation and target parsing; tests for budget scaling, round counts and duplicate/concurrent requests.
- **Phase C — Make synthesis and mutation safe.** Gate every write/import/train on verification; fail explicitly; AST-safe reward rewriting; inference-only changes without reward/training side effects; action-space-aware patching; deliberate policy reset on space changes.
- **Phase D — Close the hardcoding gap and enforce the design.** Domain-free planner grammar; arena from real render state; declarative restyle; task-shaped intake; behavioural “no hardcoding” tests; wire or remove dead controls; align export parity and required bundle files.
- **Phase E — Isolation, security, tests, docs.** Scoped cleanup; dataset durability; HTML escaping; restricted CORS/optional token; rewritten tests; honest `progress.md`.

**Status:** all phases complete. See the closure ledger in
[DESIGN_AUDIT_AND_PHASED_REMEDIATION.md](./DESIGN_AUDIT_AND_PHASED_REMEDIATION.md).

## Source documents consulted

- [MASTER_SPEC_AND_BUILD_PLAN.md](./MASTER_SPEC_AND_BUILD_PLAN.md)
- [NEXT_LEVEL_VIBE_MASTER_PLAN.md](./NEXT_LEVEL_VIBE_MASTER_PLAN.md)
- [V4_AUTONOMOUS_AGENT_SYNTHESIS_PLAN.md](./V4_AUTONOMOUS_AGENT_SYNTHESIS_PLAN.md)
- [V4_2_COMMERCIAL_E2E_AUDIT_AND_FIX_PLAN.md](./V4_2_COMMERCIAL_E2E_AUDIT_AND_FIX_PLAN.md)
- [agentic_chat_planner_multi_layer_executor_spec.md](./agentic_chat_planner_multi_layer_executor_spec.md)
- [universal_zero_hardcoding_agentic_synthesis_planner_spec.md](./universal_zero_hardcoding_agentic_synthesis_planner_spec.md)
