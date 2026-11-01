# Journey Coach — Spec and Implementation

**Status:** implemented. This is the design record for the layer that catches a shortcut before it
costs a run, and explains the evidence after one finishes.

> **Surface update (2026-11-01):** the findings now appear as `coach` entries inside the single
> studio thread, and the separate Coach drawer and header badge are gone. The rules, stages,
> severities and actions below are unchanged — see
> [`UNIFIED_CHAT_UX_SPEC_AND_PLAN.md`](./UNIFIED_CHAT_UX_SPEC_AND_PLAN.md) for the surface change.

Related: `docs/USER_GUIDE.md` §24 (the operator tour: checkpoints, severities, a worked example and
the switches), `TRUST_DRAWER_SPEC_AND_IMPLEMENTATION.md` (the controls the coach explains),
`progress.md` §21.

---

## 1. Problem

The studio already judges **outcomes**:

| Control | Question it answers |
|---|---|
| Verifier gate | Was this mutation good enough to keep? |
| Crossfire | Is the claim "this works" earned? |
| Scorecard | What did the held-out evidence say? |
| Auto-tuner | The run plateaued — now what? |

Nothing judged **intent before execution** or **comprehension after it**. A user could write "make a
trading bot", attach a table with no label column, set a 100% target with 50 episodes, and every
existing control would behave correctly while the user learned nothing. The field guide describes
the right path, but a guide is only read by people who already suspect they need it.

The missing layer is not another verifier. It is an advisor:

> Verifier: *was the change good?* · Crossfire: *is the claim earned?*
> **Coach: is this the right thing to attempt, and do you understand what you got?**

## 2. Design constraints

1. **Facts from code, words from the model.** Every number in a finding is computed by
   `coach/facts.py` before any model call. The judge receives only those facts and may not add a
   number. This is the same rule as "no fabricated tests", applied to advice.
2. **Rules are free and always on.** Deterministic checks run on every checkpoint without a
   provider, a network call, or a token.
3. **The judge is narrow, cached and optional.** It runs only on objective, data, preview and
   results; once per content hash; capped per conversation; and fails open.
4. **A model may never block a run.** The judge's `blocker` severity is downgraded to `warning`.
   Structural blockers are facts, not opinions.
5. **Guide, but do not trap.** In hard mode a blocker stops the expensive action (build, train,
   resume) and returns the exact fix. The intake chat stays advisory — it is the repair surface.
6. **Every finding ends in an action.** `apply_params`, `use_objective`, `send_message`,
   `open_guide`, `run_episodes`. A message without an action is a lecture.
7. **Quiet by default.** At most three findings per card, ranked by severity; dismissals persist by
   evidence fingerprint; a fixed finding is resolved, not left open.

## 3. Checkpoints

| Stage | Deterministic facts | Judge | Example finding |
|---|---|---|---|
| **Objective** | Word count; explicit actions; observation features; `extract_hard_constraints`; prohibition cue; reward penalties; metric cue; multiple jobs | Completeness, internal consistency, a rewritten objective | *"You said 'never approve a risky invoice', but it became neither a constraint nor a penalty."* |
| **Data** | Target detection; class distribution; minority ratio; missing ratio; duplicates; ID-like columns; perfect single-feature separation; row count | Does this data answer the stated objective? | *"The minority class is 3.0%. Accuracy will look excellent while the costly class is missed."* |
| **Parameters** | Target 0–100; 100% trap; Wilson interval width; majority baseline; uniform chance; projected episodes; reward penalties vs stated prohibition | — (structural, rule-only) | *"At 10 episodes the interval is ±22.1 points. About 200 episodes are needed for ±5."* |
| **Preview** | Arena length; live-update hooks; metric signal; controls | Could a human judge success from this screen? | *"The score you will trust is 'Audit Accuracy', but the arena never displays it."* |
| **Results** | Held-out status; CI width; best baseline; uplift; corpus regressions; Crossfire reasons; illegal actions; plateau | Plain-language "what happened → what it means → do this next" | *"80.0% with a 70–90 interval against a 79.0% majority baseline proves nothing yet."* |
| **Export** | Scorecard presence; Crossfire closure; deployment agreement; ONNX parity | — | *"Action agreement is 96.0%. The bundle is not the policy the scorecard describes."* |

## 4. Architecture

```
backend/rl_ide/coach/
  models.py   # CoachStage, Severity, GuidanceAction, GuidanceItem, CoachReport, CoachBlocked
  facts.py    # pure fact extraction; the only source of numbers
  rules.py    # pure rule catalogue; no network, no mutable state
  judge.py    # LLM prioritisation, strict JSON, number allow-list
  store.py    # findings, judge cache, per-conversation budget
  service.py  # gather → rules → judge → dedupe → persist → report
```

Flow:

```
facts (pure) → rules (free) ─┐
                             ├─ dedupe (rules win) → persist → CoachReport
judge (cached, capped) ──────┘
```

The service is the only place that decides to spend a model call. A judge exception returns
`llm_used=False` and a reason; rules still produce findings.

## 5. Contract

```python
class GuidanceItem(BaseModel):
    rule_id: str                      # "params.target_100"
    stage: CoachStage
    severity: Severity                # blocker | warning | tip
    title: str
    why: str
    evidence: dict[str, Any]          # deterministic facts only
    actions: list[GuidanceAction]
    source: Literal["rule", "llm"]
    status: Literal["open", "acknowledged", "resolved"]

class CoachReport(BaseModel):
    stage: CoachStage | None          # None = multi-stage preflight
    status: Literal["ok", "attention", "blocked"]
    items: list[GuidanceItem]
    llm_used: bool
    llm_skipped_reason: str | None
```

`GuidanceItem.fingerprint()` hashes `(rule_id, stage, evidence)`. Re-evaluating updates a row instead
of duplicating it; an acknowledged warning stays acknowledged until the evidence changes.

## 6. Enforcement

Blockers are enforced at the points where money and time are spent, never at the conversation that
fixes them:

| Endpoint | Behaviour |
|---|---|
| `POST /architect/chat` | Advisory. Returns `guidance` with the fix; the conversation continues. |
| `POST /sessions/start` | `evaluate_preflight(objective + data + parameters)`; a blocker returns HTTP 409 with `guidance`. Synthesis does not start. |
| Studio chat, training intent | Evaluated inside `StudioOmniAgent` before the training branch. A blocker sets `status="blocked"`, removes the training action, and returns `guidance`. No job is enqueued. |
| `POST …/training/resume` | Evaluated before the job row is created; a blocker returns 409. |
| `POST …/export` | Evaluated and attached to the response. Export already succeeded; findings explain what the bundle cannot claim. |
| `POST /ingest/upload`, `POST /ingest/sql` | Rule-only data findings are attached to the response. Ingestion itself succeeds. |

Mode switch:

- `RL_IDE_COACH_BLOCKERS=hard` (default) — blockers raise `CoachBlocked`.
- `RL_IDE_COACH_BLOCKERS=advisory` — blockers are reported but never raised.

A blocker cannot be acknowledged in hard mode (`409`). It must be fixed. Warnings and tips can be
acknowledged.

## 7. Judge safety

- **Input:** a facts dict, plus the original objective at the objective stage.
- **Output:** at most three findings, each with exactly one allowed action.
- **Number allow-list:** every numeric token in a title or reason must already appear in the facts
  (small counts 0–5 are allowed for ordinals). A hallucinated statistic drops the finding.
- **Severity:** `blocker` is downgraded to `warning`; unknown values become `tip`.
- **Dedup:** an LLM finding whose title overlaps a rule finding (Jaccard ≥ 0.5) is dropped. Rules win.
- **Cache:** keyed by `(stage, facts, prompt)`. Re-opening a panel costs nothing.
- **Budget:** `RL_IDE_COACH_MAX_LLM_CALLS` (default 12) per conversation. Exhaustion returns rules only.
- **Failure:** any exception returns rules only with `llm_skipped_reason`.

Switch: `RL_IDE_COACH_LLM=0` disables the judge entirely.

## 8. UI

- **Inline `CoachCard`** at the decision point:
  - intake and Studio chat → left panel, above the chat input;
  - results → inside the held-out scorecard card;
  - data upload → returned with the dataset summary.
- **Header `Coach` badge** with blocker count (rose) or warning count (amber). Opens `CoachPanel`,
  a right-hand drawer grouped by stage, with refresh and acknowledgement.
- **`open_guide`** deep-links into the field guide at a section number (`5.1`, `9.3`, `19.4`).
- **`apply_params` / `run_episodes`** patch the training controls in place.
- **`send_message` / `use_objective`** send the fix through the Studio chat, so the existing
  re-anchor and gate paths apply.

## 9. Cost

| Item | Cost |
|---|---|
| All rules, every checkpoint | zero |
| Judge at objective | one call, cached by `(prompt, facts)` |
| Judge at data | one call, only when a dataset is present |
| Judge at preview | one call per arena change |
| Judge at results | one call per scorecard change |
| Re-opening panels | zero (cache hit) |

At DeepSeek pricing a call is roughly 700 input / 250 output tokens — well under $0.001. The
conversation budget is visible in the Coach panel footer and already metered by `llm/metering.py`.

## 10. Verification

- `backend/tests/test_journey_coach.py` — 29 tests covering: each rule family, hard vs advisory
  enforcement, blocker acknowledgement refusal, persistence across a restart, acknowledged-warning
  resolution when evidence changes, judge number rejection, blocker downgrade, fail-open, cache
  reuse, rule-over-LLM dedup, and the three HTTP endpoints.
- Frontend: `npm run build` (TypeScript + Vite) passes.

## 11. Deliberate residuals

- **No learned threshold tuning.** Rule thresholds (5% minority, 30% missing, 10% duplicates, 20
  episodes, 10-point CI) are opinions encoded once. They are visible in `rules.py`, not tuned per
  tenant.
- **The judge does not see code.** It reviews the objective, facts and arena HTML, not the
  synthesized `env_module.py`. The smoke tester and verifier gate already own correctness.
- **Results guidance is triggered by the scorecard event.** A run with no scorecard gets the
  "no held-out evidence" tip, not an inferred narrative.
- **Hard mode is process-wide.** A per-conversation override would be a settings surface the
  product does not have yet; the environment variable is the documented switch.
- **No automatic fix application.** Every fix is a button. The coach never edits a reward or a
  parameter on its own.
