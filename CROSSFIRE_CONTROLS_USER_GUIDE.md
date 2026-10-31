# Crossfire Controls — User Guide

A practical guide to the five controls added in Phase P3 v5.5. All examples were run against a
live server; the payloads and outputs below are real, not illustrative.

> Reading note: "const control" is covered here in both senses the new work supports —
> **control switches** (what the system will and will not allow, §3/§7) and **cost control**
> (what owning the policy costs, §5).
>
> **In a hurry?** Everything below is reachable from the **Trust** button in the app header — see
> §11. The curl examples stay because they are the only way to script or audit these controls.

Set a base URL once:

```bash
BASE=http://127.0.0.1:8000
CONV=conv_1793436658973_x9nc4    # your conversation id
```

---

## 1. What is new, in one line each

| # | Control | Question it answers | Where you see it |
|---|---------|--------------------|------------------|
| 1 | **Golden eval corpus** | "Does the policy still get right everything I already approved?" | Telemetry panel · `GET …/corpus` |
| 2 | **Verifier gate** | "Should this reward/env/strategy change be kept?" | Chat timeline + `scorecard.selection` |
| 3 | **Crossfire protocol** | "Am I allowed to call this claim proven?" | `GET …/crossfire` + scorecard badge |
| 4 | **Cost & latency ledger** | "At what point does owning this policy pay back?" | `scorecard.economics` + payback badge |
| 5 | **Passive harvest** | "Can I reuse my existing frontier logs as training data?" | `POST …/corpus/harvest-passive` |

---

## 2. Golden eval corpus — your regression vault

Every human judgement becomes an executable assertion. A retrain is measured against the whole
accumulated set, so fixing one behaviour while breaking an earlier one is now visible.

### 2.1 It seeds itself

On the first training round of a conversation the store seeds held-out validation rows into
`workspaces/<conversation>/eval_corpus.json` and emits:

```json
{"event": "eval_corpus_seeded", "round": 0, "cases": 30,
 "message": "Seeded 30 held-out behavioural cases into the eval corpus."}
```

Up to 30 held-out rows are used (`max_cases=30`): enough to catch a regression, small enough to
evaluate on every round.

Conversations trained **before** this feature have an empty corpus (verified: an existing
tic-tac-toe conversation reports `count: 0`). One retrain or resume seeds it; the corpus is never
retro-fabricated.

### 2.2 Read it

*In the UI: Trust → Corpus.*

```bash
curl -s "$BASE/api/v4/conversations/$CONV/corpus" | python3 -m json.tool
```

```json
{"count": 30,
 "report": {"total_cases": 30, "scored_cases": 30, "passed_cases": 30, "failed_cases": 0,
            "waived_cases": 0, "pass_rate_pct": 100.0,
            "by_source": {"val_seed": {"passed": 30, "failed": 0, "waived": 0}},
            "regressed_case_ids": [], "failing_case_ids": [], "failed_details": []},
 "cases": [{"case_id": "case_1a2b3c4d5e6f", "source": "val_seed", ...}]}
```

`failed_details` is the triage payload — it tells you *which* observation the policy now gets
wrong, what it predicted, what you approved, and the recorded rationale.

### 2.3 Add an assertion by hand

Lock any step you believe is non-negotiable:

```bash
curl -s -X POST "$BASE/api/v4/conversations/$CONV/corpus/cases" \
  -H 'Content-Type: application/json' \
  -d '{"obs_vector":[0,0,0,0,0,0,0,0,0],
       "expected_actions":[4],
       "rationale":"centre first move must stay blocked",
       "source":"user_locked_step"}'
```

```json
{"case":{"case_id":"case_8718d7911ddc","source":"user_locked_step",
         "obs_prefix":[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
         "expected_actions":[4],"forbidden_actions":[],
         "rationale":"centre first move must stay blocked","waived":false}}
```

Use `forbidden_actions` when you only know what must **not** happen (e.g. `[8]` for "never play
the corner here"). A case with neither list is rejected with HTTP 400 — an assertion that asserts
nothing is not a test.

`source` is free text. The values the system itself emits are `val_seed` (seeded held-out rows),
`user_locked_step` (this endpoint) and `passive_log` (harvested frontier rows); `user_critique` is
reserved for the critique-harvester, which is not yet wired to the chat flow.

### 2.4 Retire an assertion

When a later judgement supersedes an earlier one, waive it instead of deleting it (the history
stays auditable and the case stops being scored):

```bash
curl -s -X POST "$BASE/api/v4/conversations/$CONV/corpus/cases" \
  -H 'Content-Type: application/json' \
  -d '{"waive_case_id":"case_8718d7911ddc","waive_reason":"superseded by 3-move opening policy"}'
# {"waived":"case_8718d7911ddc"}
```

### 2.5 Where it gates work

The corpus is consulted at three points: every training round (retention), every chat-authored
code change (mutation gate), and scorecard finalisation (reporting). A regression anywhere blocks
the keep decision.

**Scope, stated honestly:** an assertion names the observation at the moment of judgement, not a
full trajectory replay. That is sound for these adapters because agent state (cooldown, rolling
alert rate, queue load) lives inside the observation vector. The observation prefix is stored for
audit, and an environment with hidden state outside the vector would make the corpus a lower bound.

---

## 3. The verifier gate — what is now refused

*In the UI: Trust → Gate shows every verdict, newest first, with the triage category and advice.*

A change is kept only if **zero corpus cases regress** and held-out accuracy either stays within
1.5 pp or the drop is statistically indistinguishable from noise (paired sign test, p > 0.05,
≥ 20 paired episodes).

Otherwise the change is reverted and you get a deterministic reason — from a fixed rule table,
not a model's opinion:

| Category | Meaning | What to do |
|----------|---------|-----------|
| `compiler_mismatch` | The new term never fired on any visited state — its condition does not match reality | Show the compiled expression next to the observed variable ranges; widen the threshold or fix the field name |
| `cross_term_interference` | The term over-dominated and broke previously approved cases | Halve the new term's weight, re-apply |
| `rl_variance` | Signal was fine; entropy rose or the burst budget ran out | Rerun with a larger step budget, do not edit the term |
| `unclassified` | Nothing above matched | Inspect `evidence` in the triage report |

The gate runs on all three layers: **environment mechanics**, **reward shaping**, and
**inference strategy**. Concretely, asking the architect to change any of them no longer just
edits the file — it is scored against the corpus and the current split first, and on failure the
previous module is restored and the triage reason is posted to the chat timeline.

Read the last decision from the scorecard:

```bash
curl -s "$BASE/api/v4/sessions/<session_id>" | python3 -c \
 "import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get('scorecard',{}).get('selection'),indent=1))"
```

---

## 4. Crossfire protocol — when you may claim a win

*In the UI: Trust → Crossfire (the default tab) lists every blocking reason and the hold-out
progress bar, so you do not have to hover a badge.*

The protocol is opened **before** the first training step and stores:

- the acceptance rubric and its SHA-256 (registered before training, so it cannot be rewritten after seeing results),
- a training-start timestamp (pre-registration proof),
- frozen hold-out case ids (minimum 100).

```bash
curl -s "$BASE/api/v4/conversations/$CONV/crossfire" | python3 -m json.tool
```

```json
{"conversation_id": "conv_1793436658973_x9nc4",
 "closure": {"closed": false, "verdict": "inconclusive",
             "holdout_total": 0, "holdout_passed": 0, "holdout_pass_rate_pct": 0.0,
             "target_pass_rate_pct": 60.0,
             "policy_metric": 88.0, "baseline_metric": 35.0, "beats_baseline": true,
             "rubric_hash": "ac76e79a…f943c5bc",
             "opened_at": "2026-10-31T08:57:56Z",
             "training_started_at": "2026-10-31T08:57:56Z",
             "reasons": ["Only 0 frozen hold-out judgements exist; the protocol requires 100."]}}
```

Verdicts:

| Verdict | Meaning |
|---------|---------|
| `win` | Pre-registration sound, hold-out pass rate ≥ target, **and** beats a measured baseline |
| `loss` | Pre-registration sound, pass rate below target — a real result, reported as such |
| `inconclusive` | Something in the protocol is unsound (rubric edited late, no training stamp, too few frozen judgements, no baseline) |
| `not_preregistered` | No protocol exists for this conversation |

The gate refuses closure on an unsound protocol by construction: `closed` is `false` whenever any
pre-registration gap exists. Notice the example above — 88% against a 35% baseline is still
**inconclusive** because the frozen hold-out is empty. That is the point.

**To earn a `win` you must:** hold ≥ 100 corpus judgements and have a measured baseline in the
scorecard. Val-seeded cases count automatically; anything else you add — locked steps, harvested
frontier rows — counts too, because the hold-out set is frozen from the full corpus at scoring
time. Below 100, `reasons` says so and the verdict stays `inconclusive`.

---

## 5. Cost control — the ledger and the payback point

*In the UI: right-hand Telemetry panel, scorecard card (`⚡ p99 latency` and the payback lines).*

`scorecard.economics` prices two things separately: the **token bill** and the **human bill**. A
token-only comparison flatters the product (~500k decisions to pay back), so both bases are
reported and the human-inclusive one is the honest headline.

Measured on the existing tic-tac-toe session:

| Field | Value |
|-------|-------|
| Authoring tokens | $0.00 |
| Authoring human time | 0.1 h → $12.00 (at $120/h loaded) |
| Retrain compute / yr | $0.24 |
| Drift review / yr | $2,190.00 (1.5 h × 12 cycles) |
| **First-year total** | **$2,202.24** |
| Payback, budget frontier, token-only | 1,393,825 decisions (5,575 days) |
| Payback, budget frontier, **with human review** | **4,772 decisions (19.1 days)** |
| Latency p99 end-to-end | 0.308 ms ONNX (+ pipeline) vs 850 ms frontier reference |

Levers you control, in order of effect:

1. **`drift_review_hours_per_cycle`** — dominates the first-year total. If your team reviews drift
   quarterly instead of monthly, the annual bill drops ~4×.
2. **`decisions_per_day`** — sets the payback horizon; 250/day is the default assumption.
3. **`frontier_tiers`** — replace the built-in `$1.00/$5.00` and `$5.63` blended prices with your
   real invoice.
4. **`RL_IDE_LLM_PRICES`** — the price table metering uses for the authoring-token line.
5. **`RL_IDE_OWNER_BUDGET_USD`** — refuses new authoring calls once measured spend reaches the cap.

Recompute with your own numbers:

```bash
cd backend && ../.venv/bin/python -c "
from rl_ide.autopilot.scorecard import compute_economics_summary
import json; print(json.dumps(compute_economics_summary('$CONV',
    decisions_per_day=1000, drift_review_hours_per_cycle=0.5), indent=1))"
```

With 1,000 decisions/day and quarterly drift review the same session pays back in **1,651
decisions (1.65 days)** instead of 19 — that is the whole spread of the assumption, on identical
measured inputs.

Visible in the UI: the Telemetry panel shows payback in decisions next to the corpus and Crossfire
badges, with the token-only figure in the tooltip so the two bases cannot be confused.

---

## 6. Passive harvest — free labels from your frontier logs

If you already run a frontier model in production, you already own labelled examples: every
accept/reject/edit is a human judgement about one decision.

### 6.1 Required shape

A tabular file (`.csv`/`.xlsx`) with feature columns plus any of:

| Role | Accepted column names |
|------|----------------------|
| proposed action | `llm_action`, `model_action`, `predicted_action`, `proposed_action`, `agent_action`, `candidate_action` |
| feedback | `accepted`, `is_accepted`, `feedback`, `verdict`, `human_verdict`, `status` |
| correction | `corrected_action`, `human_action`, `edited_action`, `final_action` |
| rationale | `critique_text`, `critique`, `comment`, `review_note`, `notes` |

Accept values: `accepted/approve/approved/true/1/yes/y/kept/correct`. Reject values:
`rejected/reject/false/0/no/n/edited/overridden/wrong`.

### 6.2 Upload, then harvest

*In the UI: Trust → Corpus → Import a frontier interaction log (same pipeline, no curl).*

```bash
DATASET=$(curl -s -X POST "$BASE/api/v4/ingest/upload" -F "file=@frontier_log.csv" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['dataset_id'])")

curl -s -X POST "$BASE/api/v4/conversations/$CONV/corpus/harvest-passive" \
  -H 'Content-Type: application/json' \
  -d "{\"dataset_id\":\"$DATASET\",\"min_usable_rows\":100,\"max_cases\":50}" | python3 -m json.tool
```

```json
{"detected": true,
 "detection": {"proposed_column": "llm_action", "feedback_column": "accepted",
               "corrected_column": "corrected_action", "rationale_column": "critique",
               "total_rows": 240, "usable_rows": 240,
               "accepted_rows": 176, "rejected_rows": 64, "corrected_rows": 48,
               "feature_columns": ["f0", "…", "f8"], "sample_size_warning": null},
 "harvested": {"registered": 60, "soft_negatives": 16, "unmatched_rows": 4,
               "skipped_no_env": false}}
```

### 6.3 The conservative split (deliberate)

- **Hard assertion** — rejection *with* a corrected action. The human stated what they wanted, so
  the case carries both `expected_actions=[corrected]` and `forbidden_actions=[proposed]`.
- **Soft negative** — rejection alone. The case carries only `forbidden_actions=[proposed]`: it
  claims that one action was wrong, never that some other action was right. Counted separately in
  `soft_negatives` (and included in `registered`) so you can see how much of your assertion set
  rests on bare rejections.
- **Unmatched** — the row cannot be tied back to a record in the conversation's ingested dataset
  (matching is by feature values), so no observation exists to assert about. An assertion about an
  observation the policy never sees would be a fabricated test, so the row is skipped, not invented.

Only rejected rows become assertions — a "the model was right" case would just re-test training
data. The accepted/corrected pairs are extracted by `supervised_rows()` in the same module for
supervised induction; that path is not wired into the training loop yet, so today accepted rows
influence training only when you also pass the same file as the conversation's dataset.

### 6.4 One caveat, measured

Harvest needs the conversation's environment **loaded in the running server process**:

```json
{"harvested": {"registered": 0, "soft_negatives": 0, "unmatched_rows": 0, "skipped_no_env": true}}
```

`skipped_no_env: true` means no environment was in memory (e.g. server restarted since training).
It writes nothing and is safe to retry. Run the harvest in the same server session as a
train/resume.

---

## 7. Control switches

| Switch | Values | Effect |
|--------|--------|--------|
| `RL_IDE_ALLOW_HOST_LLM` | `1` (default) / `0` | `0` forbids the host LLM between sandbox rounds. Per-round training continues sandbox-only — the data-sovereign mode |
| `RL_IDE_REQUIRE_ISOLATION` | `1` in the shipped Dockerfile | Refuses to run if `bwrap` is unavailable |
| `RL_IDE_SANDBOX` | set by the launcher | Worker-side marker: tells the round loop it is inside the sandbox, where the LLM tuner cannot run, so tuning waits for the host. Don't set it by hand |
| `RL_IDE_LLM_PRICES` | JSON | Token price table for metering + §5 authoring line |
| `RL_IDE_OWNER_BUDGET_USD` | number | Pre-flight refusal once the owner's measured spend reaches the cap (raise it or start a new billing period) |
| `RL_IDE_REQUIRE_AUTH` / `RL_IDE_API_TOKEN` / `RL_IDE_ADMIN_TOKEN` | — | Tenant auth |
| `skip_auto_tune` | worker flag | Skip in-worker tuning. Sandbox mode already implies it (no network), so set it only when calling the controller directly |
| `finalize_scorecard` | `False` for non-final rounds | Scorecard computed once, not once per round |

Training now runs **round by round inside the sandbox** with tuning decisions made on the host
between rounds, so a single `RL_IDE_ALLOW_HOST_LLM=0` covers the whole run.

---

## 8. What lands on disk, per conversation

```
workspaces/<conversation>/
├── eval_corpus.json          # the regression vault (cases + waivers)
├── crossfire_protocol.json   # rubric hash, timestamps, frozen hold-out ids, verdict
├── scorecard.json            # + selection, eval_corpus, crossfire, economics blocks
├── best_val_env_module.py    # best-on-validation checkpoint (reloaded at the end)
├── best_val_model.zip
└── session_state.json        # durable: best_val_metric, mutation_pending, gate outcome
```

---

## 9. Ten-minute walkthrough

1. `GET /capabilities` — confirm LLM verified and isolation available.
2. Train or resume the conversation once → corpus seeds, protocol opens, first ledger appears.
3. `GET …/corpus` — read the assertions you now owe the policy.
4. Ask the architect for a reward change ("penalise stalling") → watch the gate either keep it or
   return a triage reason in the chat timeline.
5. `GET …/corpus` again — `regressed_case_ids` should be empty; if not, that is exactly what the
   gate protected you from.
6. `GET …/crossfire` — read `reasons`; each entry is a concrete thing to fix before any claim.
7. `POST …/corpus/harvest-passive` with your production log → rejected-with-correction rows become
   assertions for free.
8. Open the Telemetry panel — corpus ratio, Crossfire verdict, and payback in decisions.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `count: 0` on an old conversation | Trained before the corpus existed | Retrain or `POST …/training/resume`; seeding runs at the next round 0 |
| Verdict `inconclusive` with high accuracy | Fewer than 100 corpus judgements, or no baseline | Grow the corpus past 100 (locked steps and harvested rows count); check `baseline_metric` is set |
| Verdict `not_preregistered` | Session has no protocol file | Will be created at the next training start |
| `skipped_no_env: true` | Env not loaded in this server process | Harvest in the same process as a train/resume |
| `400 A case needs an observation and at least one expectation` | Empty `expected_actions` and `forbidden_actions` | Add one of the two |
| Gate keeps reverting your change | `compiler_mismatch` | Read the compiled expression and the observed ranges from `evidence`, widen the threshold |

---

## 11. Using everything from the UI (no curl)

The **Trust** button sits in the header, between the LLM badge and Policy Registry. It opens a
drawer scoped to the active conversation, with three tabs. The button carries a count of Crossfire
blockers, so an unearned claim is visible before you open anything.

| Tab | What it shows | What you can do |
|-----|---------------|-----------------|
| **Crossfire** (default) | Verdict pill (`win` / `loss` / `inconclusive` / `not preregistered`), a progress bar `frozen hold-out x / 100`, every blocking reason as a numbered list, policy vs baseline, rubric hash, and the pre-registration timestamps | Read only — a verdict cannot be edited into existence |
| **Corpus** | Scored/passed/failed/waived counters, then one row per assertion with its status dot, source, expectation and rationale | **Lock the step you just inspected**, **waive** a superseded case, **import a frontier log** |
| **Gate** | Every verifier-gate decision, newest first: round, change kind, pass/rollback, triage category and the advice | Read only; a pending change is flagged while it waits for the next round |

### 11.1 Locking a step (the important one)

1. Switch the arena to **🎮 Play / Interact** and make a move.
2. Open **Trust → Corpus**. The card reads e.g. `step 3 · obs[23] · policy chose 4 · Place at center`.
3. Add a reason (optional) and press **This is correct** (locks the action as expected) or
   **Never do this** (locks it as forbidden).
4. The assertion appears in the list immediately and now gates every retrain and every
   chat-authored change.

The observation comes from the policy's own encoder for that step, so the assertion is testable by
construction. If the card instead says *"Play the arena in Interactive mode, then reopen this
drawer"*, no interactive step has been recorded yet in this session.

### 11.2 Importing a frontier log

**Trust → Corpus → Import a frontier interaction log** → choose the CSV. The panel reports how
many rows matched, how many assertions were registered, how many were soft negatives, and how many
rows could not be tied to a record (skipped rather than invented). If it says *"No environment
loaded in the server"*, retry after a training run in the same server session — the harvest needs
the conversation's encoder in memory.

### 11.3 What is still automatic (by design)

Corpus seeding, the Crossfire protocol opening, gate decisions, best-on-validation selection and
the economics ledger all happen without a click. The drawer is where you *read and judge* them,
not where you trigger them.

The design record for this UI — constraints, tab-by-tab behaviour, backend contracts and the
residuals left open on purpose — is `TRUST_DRAWER_SPEC_AND_IMPLEMENTATION.md`.
