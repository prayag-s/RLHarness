# Trust Drawer — Spec and Implementation

**Status:** implemented (`cd6270f`). This document is the design record for the operator UI of the
Phase P3 / v5.5 controls (golden corpus, verifier gate, Crossfire protocol, cost ledger, passive
harvest).

> **Surface update (2026-11-01):** the drawer is now opened from the *evidence* entry in the studio
> thread (the held-out scorecard or the Crossfire verdict) instead of a header button. Its contents
> are unchanged — see [`UNIFIED_CHAT_UX_SPEC_AND_PLAN.md`](./UNIFIED_CHAT_UX_SPEC_AND_PLAN.md). It supersedes nothing; it closes the gap between *"the controls exist"* and *"a user can
operate them without curl"*.

Related: `PHASE_P3_V5_5_REFINED_EXECUTION_PLAN.md` (what the controls do),
`CROSSFIRE_CONTROLS_USER_GUIDE.md` §11 (how to use them), `progress.md` §20.6 (measured evidence).

---

## 1. Problem

The v5.5 work shipped five controls, all reachable only through HTTP:

- the golden eval corpus (`GET/POST …/corpus`),
- the Crossfire verdict (`GET …/crossfire`),
- passive harvest (`POST …/corpus/harvest-passive`),
- the economics ledger and gate decisions, embedded in the scorecard.

In the browser only two things were visible, both inside the scorecard card and both only *after* a
run finished:

| Visible | Hidden |
|---|---|
| `🛡️ Golden corpus 30/30 verified` | which cases, which ones failed, how to add one |
| `Crossfire: inconclusive · Only 0 frozen hold-out…` | the remaining reasons (tooltip only), hold-out progress, rubric hash, timestamps |

Observed consequence: a user concludes the controls "happen on their own", which is half true —
they *evaluate* on their own, but the two actions that require human judgement (locking a step,
retiring an assertion) had no surface at all.

## 2. Design constraints

1. **Read is separate from write.** Verdicts must never be editable into existence. The Crossfire
   and Gate tabs are read-only by construction; nothing in the UI can register a rubric or a
   baseline.
2. **No fabricated tests.** An assertion is only created from an observation the policy actually
   scored, through the environment's own encoder. The UI must not synthesise an observation vector
   from what it happens to have on screen.
3. **Say what is empty and why.** Every empty state names the next action that fills it
   (`corpus count 0` → "seeds at the next training round"; `skipped_no_env` → "retry after a
   training run").
4. **Nothing new is triggered implicitly.** Opening the drawer performs no mutation; the three
   writes are explicit button presses.
5. **Survive a restart.** History that explains a decision must outlive the process that made it.

## 3. Information architecture

Entry point: **Trust** button in the header, between the LLM badge and Policy Registry. It carries
the count of Crossfire blockers, so an unearned claim is visible without opening anything.

Drawer: right-hand slide-over, 30 rem, scoped to the active conversation
(`activeConversationId ?? activeSession.session_id`). Header shows the conversation id, a manual
refetch button, and close. Three tabs:

### 3.1 Crossfire (default)

| Element | Source | Notes |
|---|---|---|
| Verdict pill + `closed` flag | `closure.verdict`, `closure.closed` | One sentence per verdict explains what it means |
| `frozen hold-out n / 100` progress bar | `holdout_registered`, `min_holdout` | Amber until satisfied, then emerald |
| Scored line | `holdout_passed / holdout_total`, `holdout_pass_rate_pct`, `target_pass_rate_pct` | Only when a verdict was scored |
| **Blocking closure** | `closure.reasons[]` | Numbered list, not a tooltip. Replaced by an explicit sentence when there is nothing to list |
| Facts grid | `policy_metric`, `baseline_metric`, `beats_baseline`, `opened_at`, `training_started_at`, `rubric_hash` | Hash truncated, full value in `title` |
| What earns a win | rubric + `min_holdout` | Static explanation of the three requirements |

Nothing here is actionable by design: a verdict is a measurement, not a setting.

### 3.2 Corpus

| Element | Behaviour |
|---|---|
| Counters | scored / passed / failed / waived from the report |
| **Lock the step you just inspected** | Shows `step N · obs[d] · policy chose <action label>`; two buttons: *This is correct* (`expected_actions`) and *Never do this* (`forbidden_actions`), optional rationale |
| Import a frontier interaction log | File input → `POST /ingest/upload` → `POST …/corpus/harvest-passive`; reports usable rows, registered, soft negatives, unmatched, and the `skipped_no_env` caveat |
| Assertion rows | Status dot (passed / failed / waived), case id, source chip, `expect …` / `forbid …` with action labels, rationale; per-row **waive** with an inline reason field |
| Empty state | Explains seeding and that pre-feature conversations need one retrain/resume |

Status precedence: `waived` (from the case) wins over `failed` (id present in
`report.failing_case_ids`), which wins over `passed`.

### 3.3 Gate

| Element | Behaviour |
|---|---|
| Pending banner | `mutation_pending` — "A reward change is waiting: the next training round judges it" |
| Decision cards | Newest first: pass/rollback icon, round, change kind, origin (`training_round` / `chat_mutation`), triage category, reason, advice |
| Triage vocabulary | The four categories with one-line meanings |
| Empty state | "No gate decisions yet" — explains that the gate needs a policy to compare against |

## 4. Contracts

### 4.1 New endpoint

```
GET /api/v4/conversations/{conversation_id}/gates
{
  "conversation_id": str,
  "entries": [GateRecord],      # oldest first, capped at MAX_GATE_HISTORY = 20
  "last": GateDecision | null,  # the head of entries, denormalised for the fallback card
  "mutation_pending": str | null
}
```

Served from the live session when present, otherwise from `session_state.json`. A restarted server
must not lose the explanation of why a round was reverted, which is the entire point of keeping the
history.

### 4.2 `GateRecord`

| Field | Type | Meaning |
|---|---|---|
| `at` | ISO-8601 | When the decision was made |
| `round` | int | Round the decision belongs to (`0` when unknown) |
| `origin` | `training_round` \| `chat_mutation` | Which loop produced it |
| `kind` | `reward` \| `environment` \| `inference` \| null | What was changed |
| `passed` | bool | Kept (`true`) or rolled back (`false`) |
| `category` | triage category \| null | `compiler_mismatch`, `cross_term_interference`, `rl_variance`, `unclassified` |
| `reason` | str | Human sentence from the decision |
| `advice` | str | What the triage table says to do next |
| `evidence` | object | Term eligibility/activation, val delta, regressed ids, entropy delta |

Written by one helper, `record_gate_decision(session, decision, round_num=…, origin=…)`, called by
both the training loop and the chat mutation path — so the two cannot drift apart.

### 4.3 Closure payload enrichment

`closure_report()` and `evaluate()` now expose `min_holdout` and (before a verdict exists)
`holdout_registered`. Without them a client can only repeat the complaint
("requires 100") and cannot show progress.

### 4.4 Observation contract

`POST /sessions/{id}/interact` returns `obs_vector` — the vector the policy scored. It is captured
inside `worker.run_serve`'s `predict()` (see §6.1), not derived from `arena_state`, because arena
state is a rendering and the corpus asserts on the encoder's output.

## 5. Component map

| File | Role |
|---|---|
| `frontend/src/components/TrustPanel.tsx` | The drawer: three tabs, fetch orchestration, all three writes, empty and error states |
| `frontend/src/components/Header.tsx` | Trust entry point + blocker badge + verdict-coloured styling |
| `frontend/src/components/SandboxedArenaIframe.tsx` | Captures `obs_vector` / `action_probs` from each `ARENA_INTERACT` response and reports the step upward |
| `frontend/src/App.tsx` | Owns `isTrustOpen`, `lastStep`, `trustRefreshKey`; resets the step on new agent; bumps the key on `scorecard_ready` |
| `frontend/src/services/api.ts` | `fetchEvalCorpus`, `addCorpusCase`, `waiveCorpusCase`, `fetchGateHistory`, `fetchCrossfireClosure`, `harvestPassiveLog` |
| `frontend/src/types/index.ts` | `EvalCorpusCase`, `EvalCorpusReport`, `EvalCorpusResponse`, `CrossfireClosure`, `GateDecision`, `GateRecord`, `GateHistoryResponse`, `ObservedStep`, `PassiveHarvestResult` |
| `backend/rl_ide/autopilot/loop_controller.py` | `gate_history` field, `MAX_GATE_HISTORY`, `record_gate_decision()` |
| `backend/rl_ide/api/server.py` | `GET …/gates` |
| `backend/rl_ide/autopilot/crossfire_protocol.py` | `min_holdout` / `holdout_registered` in the payloads |
| `backend/rl_ide/security/worker.py` | Observation capture inside `predict()` |

Data flow for a lock: arena click → `ARENA_INTERACT` → `interactSession()` → response carries
`obs_vector` + `action_probs` → `onStepObserved` → App `lastStep` → drawer renders the card → button
→ `addCorpusCase()` → `POST …/corpus/cases` → refetch → new row + counters.

## 6. Defects found while implementing (and fixed)

### 6.1 The lock button had nothing to assert on

Environments whose `interactive_step()` manipulates its own board never call `step()` or `reset()`,
so `capture_observations()` recorded nothing and `/interact` returned **no `obs_vector`** — verified
with a live call on the tic-tac-toe conversation (`keys: action_probs, agent_action_label,
arena_state, drift, status_text`).

Fix: record the vector inside `predict(obs)`, the only place guaranteed to see what the policy
scored. Verified afterwards: `obs_vector` of 23 dimensions returned.

Side effect, worth stating: live drift scoring reads the same field, so those environments now
accumulate drift samples where they previously sat at `warming_up` forever.

### 6.2 "No pre-registration gaps" on a conversation with no protocol

With no protocol, `reasons` is empty, so the drawer printed a reassuring green line next to
`not preregistered`. Fixed with verdict-specific copy ("Nothing to block yet: the protocol is
opened automatically…").

### 6.3 The step counter read `step —`

The interact payload carries no `step` field; the arena state does. The UI now falls back to
`arena_state.move_count`.

### 6.4 Session history vanished on restart

`critique_gate` held one decision and only in memory. Hence `gate_history` + the persisted
fallback in `GET …/gates` (§4.1).

## 7. Verification

| Check | Result |
|---|---|
| Fast gate `pytest -m "not slow and not live_llm"` | **171 passed, 2 skipped, 60 deselected** (was 170 before the new gate-history test) |
| Targeted `test_crossfire_closure.py` + `test_scorecard.py` | 16 passed |
| `npm run build` (tsc + vite) | clean, 298 kB / 84.7 kB gzip |
| Crossfire tab, live conversation | `inconclusive`, `0 / 100`, one blocking reason, facts and rubric hash rendered |
| Lock a step from the arena | Case `case_…` created and visible; `waived: true, 'superseded'` after the waive flow |
| Harvest import through the file picker | `matched 220 usable rows · registered 0`, plus the `No environment loaded` caveat |
| Gate tab | Empty state with vocabulary; endpoint returns `entries: []` for a session written before the field existed |

## 8. Deliberate residuals

- **No rubric or hold-out editor.** A user cannot register their own acceptance rubric or freeze a
  chosen hold-out set from the UI; the protocol uses `DEFAULT_RUBRIC` and the corpus. Making the
  rubric user-editable is a product decision, not a missing binding.
- **No re-score button.** Crossfire is scored at scorecard finalisation. `GET …/crossfire` shows the
  last verdict; forcing a fresh evaluation is not exposed.
- **Harvest needs a warm process.** `skipped_no_env` is surfaced, not worked around: the encoder has
  to be in memory, so the action is "retry after training", not "retry harder".
- **Per-case predicted action is not shown**, only pass/fail plus `failed_details` on the report.
  Showing it means storing it; deferred until someone asks.
- **No keyboard affordances.** The drawer is pointer-driven; the app has no shortcut layer to hook
  into yet.
