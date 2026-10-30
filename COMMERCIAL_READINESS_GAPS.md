# RLHarness — Commercial Readiness Gap Analysis

**Analysis date:** 2026-10-30
**Repository state analysed:** `main` at `5e9ab72`, working tree clean
**Scope:** what stands between the current working studio and software a customer would pay for.
**Not in scope:** correctness of the existing design (covered by
[DESIGN_AUDIT_AND_PHASED_REMEDIATION.md](./DESIGN_AUDIT_AND_PHASED_REMEDIATION.md), issues F1–F39)
and the test history (covered by [progress.md](./progress.md), §10).

**Method:** every claim below was read out of the source or measured on the live instance. Numbers in
§3 were produced by running the commands shown next to them, not estimated.

---

## 1. Verdict

The engineering that exists is real. Specifically, these are genuine assets and should not be
regressed:

- Deterministic, dataset-grounded feature derivation
  ([feature_engineering.py](./backend/rl_ide/synthesizer/feature_engineering.py)) — observations come
  from real columns, encodings are process-stable, labels map to actions by name rather than order.
- LLM-free verification of supervised environments
  ([verifier.py](./backend/rl_ide/synthesizer/verifier.py)) — the supervised path proves itself on
  attempt 0 with a stub LLM client, so a broken module cannot hide behind an LLM rewrite.
- Hard failure on dead observations, unmappable labels, and unreachable success criteria.
- Prompt-declared action lists are treated as a contract
  ([intake.py](./backend/rl_ide/architect/intake.py)).
- ONNX export with parity < 1e-6 and a standalone policy daemon
  ([exporter.py](./backend/rl_ide/export/exporter.py)).

What is missing is not more intelligence in the pipeline. It is the **trust layer** — proof that the
policy generalises, isolation between customers, and the operational plumbing that lets the thing run
for more than one person at a time.

The blunt summary: today this is a **single-tenant research studio that produces an impressive demo
and an unverifiable number**. A buyer's three questions — *"does it work on data you didn't train on?"*,
*"where does my data go?"*, *"who else can see it?"* — have no good answer in the product.

---

## 2. Gap register

Severity is commercial, not technical: **P0** = a paying customer would refuse to sign, **P1** = you
cannot operate or bill the service, **P2** = it limits depth of value or slows sales.

### P0 — blocks a sale

#### G1. The headline metric is in-sample and partly oracle-assisted
- **Evidence:** `records_from_frame` serialises `df.head(120)` — the first 120 rows of the uploaded
  file, in file order, with no shuffle and no split
  ([feature_engineering.py](./backend/rl_ide/synthesizer/feature_engineering.py) L190–L193).
  The same records are handed to the training env and to the 50-episode deterministic evaluation
  ([loop_controller.py](./backend/rl_ide/autopilot/loop_controller.py) L537, L981). For LLM-authored
  control tasks the evaluation loop first masks logits to `get_critical_actions()` — an oracle hint
  ([loop_controller.py](./backend/rl_ide/autopilot/loop_controller.py) L1024–L1040). There is no
  baseline comparison, no confidence interval, and no per-segment breakdown.
- **Why it blocks revenue:** the "87.0 % / 95.4 % / 81.2 %" run in `progress.md` §10 is a
  training-set number with a favourable action mask. Any competent buyer or data scientist will ask
  for a held-out figure, and when it is not available the whole tool reads as a demo.
- **Fix:** deterministic 70/15/15 split by row hash (stable across runs); train on 70 %, select on
  15 %, report on 15 %; evaluate with `action_masks=None` unless the mask is part of the deployed
  policy; report human-label agreement (the CSV's own label) as the baseline the policy must beat;
  add a Wilson 95 % CI to every success-rate figure; emit a per-class / per-segment table.
- **Effort:** ~1.5 days including tests.

#### G2. LLM-generated Python executes inside the API process
- **Evidence:** `load_synthesized_env_class` uses `importlib.util.spec_from_file_location` +
  `spec.loader.exec_module`
  ([env_builder.py](./backend/rl_ide/synthesizer/env_builder.py) L75–L92). The only defence is a
  static AST denylist of module names and call names
  ([verifier.py](./backend/rl_ide/synthesizer/verifier.py) L32–L41). The exported policy daemon binds
  `0.0.0.0:9000` with no authentication
  ([exporter.py](./backend/rl_ide/export/exporter.py) L220–L226).
- **Why it blocks revenue:** hosted multi-tenancy means executing model-authored code next to other
  customers' data with the API process's privileges. A denylist is not a boundary; `open` is blocked
  but `numpy.fromfile`, `pandas.read_csv` and object-graph traversal are not. Security review of a
  hosted deployment will stop at this line.
- **Fix:** run the env in a subprocess (`multiprocessing` spawn) or container with a wall-clock
  timeout, no network, a read-only root, and only a whitelisted stdlib/numpy surface; communicate over
  pipes. For the exported daemon, bind `127.0.0.1` by default, require a token, and document the flag
  that opens it up.
- **Effort:** ~2–3 days for the subprocess sandbox; ~2 hours for the daemon defaults.

#### G3. Customer rows are embedded in the workspace and the exported bundle
- **Evidence:** `records_json = json.dumps(records_from_frame(usable), default=str)` is written into
  the generated module
  ([env_builder.py](./backend/rl_ide/synthesizer/env_builder.py) L643–L664), which is copied into every
  `.runbundle`. Already logged as accepted residual F17 in the audit document, with the rationale
  "bundles remain standalone-executable".
- **Why it blocks revenue:** under any DPA or security questionnaire, shipping 120 raw customer rows
  in an artefact that gets emailed around is a finding. It is defensible for a local studio; it is not
  defensible for a paid deployment.
- **Fix:** export two bundle modes — `--with-data` (explicit opt-in, clearly labelled) and the default
  `--schema-only` which ships the fitted feature bounds and label map plus a `data/README` telling the
  operator to mount their own CSV. Keep the workspace copy (it is already git-ignored) but strip it
  from the download path.
- **Effort:** ~0.5 day.

#### G4. No identity, no tenancy
- **Evidence:** one optional shared secret gates `/api/` only — the middleware checks
  `request.url.path.startswith("/api/")`, so `/ws/v4/{session_id}` accepts any client that knows a
  conversation id ([server.py](./backend/rl_ide/api/server.py) L108–L126); `list_conversations()`
  selects every row
  with no owner column
  ([conversation_store.py](./backend/rl_ide/state/conversation_store.py) L178–L190); uploads live in a
  process-global dict, so they disappear on restart
  ([server.py](./backend/rl_ide/api/server.py) L86).
- **Why it blocks revenue:** two customers on one instance see each other's tasks, datasets and
  models. Even single-tenant, "your datasets vanish when we deploy" is a support ticket that ends a
  pilot.
- **Fix:** `owner_id` column on `conversations` plus a `datasets` table keyed by
  `(owner_id, dataset_id)`, persisted rather than held in a dict; scope every query by the caller;
  keep `RL_IDE_API_TOKEN` working as the single-user fallback so local usage does not regress.
- **Effort:** ~2 days.

### P1 — cannot operate or bill it

#### G5. Synthesis is a blocking HTTP POST with no job identity
- **Evidence:** `POST /api/v4/conversations/{id}/chat` awaits
  `asyncio.to_thread(global_omni_agent.process_message, …)` and only afterwards may spawn background
  training ([server.py](./backend/rl_ide/api/server.py) L723–L754). Measured latency is 2–3 minutes per
  environment (`progress.md` §10.4) and 111–155 seconds for the arena alone (commit `5e9ab72`); the
  client timeout had to be raised to 240 s to stop silent fallbacks.
- **Why it blocks revenue:** a three-minute spinner with no progress, no cancel and no recovery is the
  single most common reason a trial user bounces. It also makes the API unusable for automation.
- **Fix:** `POST /chat` returns `202 {job_id, stage}` immediately; synthesis runs as a tracked job;
  stage events (`intake`, `env_synthesis`, `verification`, `arena`, `ready`) stream over the existing
  `/ws/v4/{session_id}`; add `POST /jobs/{job_id}/cancel` and refetch-on-reconnect semantics.
- **Effort:** ~2 days.

#### G6. Single process, in-memory runtime state
- **Evidence:** models, env instances and locks live in `AutonomousLoopController` dicts
  ([loop_controller.py](./backend/rl_ide/autopilot/loop_controller.py) L147–L152, L286–L296); the store
  is SQLite behind a single file ([conversation_store.py](./backend/rl_ide/state/conversation_store.py) L44).
- **Why it blocks revenue:** you cannot run two Uvicorn workers, cannot restart during a training run
  without losing it, and cannot scale past one machine.
- **Fix:** move job/run state into the database, make training a worker consuming a queue (even a
  SQLite/Postgres table is enough), and let the API be stateless. This is the prerequisite for G5 as
  well.
- **Effort:** ~3 days; can be folded into G5.

#### G7. No usage metering or cost control
- **Evidence:** `llm/client.py` records nothing about tokens, calls or spend per conversation. There is
  no quota, no budget, no per-tenant accounting.
- **Why it blocks revenue:** you cannot invoice, cannot cap a runaway customer, cannot tell an
  enterprise buyer what a run costs, and cannot detect that synthesis on a large file is burning money.
- **Fix:** capture `usage` from each provider response, persist `(conversation_id, provider, model,
  input_tokens, output_tokens, cost_estimate, latency_ms)`, expose it per conversation and in
  aggregate; add an optional per-tenant budget that fails the job cleanly.
- **Effort:** ~1 day.

#### G8. No CI, no container, no pinned backend dependencies
- **Evidence:** no `.github/` directory, no `Dockerfile`, no backend lockfile; `backend/pyproject.toml`
  lists unpinned names (`fastapi`, `uvicorn[standard]`, …) with only `>=` on Pydantic. The suite takes
  ≈11 minutes and is run by hand.
- **Why it blocks revenue:** there is no reproducible artefact to hand a customer or a security team,
  and no gate preventing a regression from shipping.
- **Fix:** container image with pinned versions and a lockfile; CI running `pytest` (split the slow
  synthesis tests behind a marker so the fast suite stays under ~2 minutes), `tsc`/`vite build`, and a
  smoke test that boots the API.
- **Effort:** ~1 day.

#### G9. Unbounded local-state growth, no retention
- **Evidence (measured):**
  - `backend/workspaces/` holds **609** `temp_verify_*` directories that nothing ever deletes
    (`verifier.py` L333 creates them; no `rmtree` counterpart exists).
  - `backend/runs/conversations.db` is **220 774 400 bytes** while the `conversations` table holds
    **0** rows and `freelist_count` is **53 897 of 53 900** pages — the file is 99.99 % reclaimed-but-
    never-vacuumed space.
  - Deleting a conversation removes the workspace but never compacts the database.
- **Why it blocks revenue:** disk grows monotonically per customer and there is no "delete my data"
  story that a DPO would accept.
- **Fix:** `tempfile.TemporaryDirectory` for verification workspaces; `PRAGMA auto_vacuum=INCREMENTAL`
  plus a periodic `vacuum` after deletes; a retention policy (configurable, default 30 days) for
  bundles and episode rows; a `DELETE /datasets/{id}` that actually removes bytes.
- **Effort:** ~1 day.

### P2 — depth and sales friction

#### G10. Tasks are contextual bandits, not sequential policies
- **Evidence:** dataset environments run 25-step episodes over independent records with immediate
  reward ([env_builder.py](./backend/rl_ide/synthesizer/env_builder.py) L176, L221, L693, L800).
- **Why it matters:** routing and triage are one-shot decisions and the product fits them well. The
  higher-value problems — renewal plays over a quarter, claim journeys, multi-touch support — need
  state that persists across steps. Today those are modelled as 25 independent guesses.
- **Fix:** let the blueprint declare a `transition` (row → next row) and a delayed reward, and add a
  multi-step dataset template alongside the existing one-shot template.
- **Effort:** ~3–4 days. Deliberately last.

#### G11. Drift detection is built but never wired
- **Evidence:** `DriftDetector` is imported only by the exported bundle
  ([exporter.py](./backend/rl_ide/export/exporter.py) L124–L127); neither `server.py` nor
  `omni_agent.py` references it. The header therefore reports `unmonitored`, which is honest but
  incomplete.
- **Fix:** score each live `/interact` observation against the stored baseline, broadcast a
  `drift_warning` event when the quantile boundary is crossed, and show the status in the header.
- **Effort:** ~1 day.

#### G12. LLM-authored control tasks are not reliably learnable
- **Evidence:** the same inventory prompt produced 100 %, 62 %, 18 %, 6 % and 100 % success across
  runs (`progress.md` §10.4). Detection exists (unreachable KPI is surfaced), correction does not.
- **Fix:** when `_probe_success` cannot reach the target, ask the LLM to relax the KPI and re-verify
  automatically, rather than only warning. Cap at two rewrites, then fall back to the deterministic
  template.
- **Effort:** ~1–2 days.

#### G13. Onboarding requires a verified key before anything works
- **Evidence:** every substantive route calls `require_verified_llm()` and returns 403 otherwise
  ([server.py](./backend/rl_ide/api/server.py) L97–L106). Only one provider family is configured in
  practice (`deepseek-flash`).
- **Why it matters:** fine for BYO-key developers, friction for a pilot, and a procurement question for
  enterprises that require Azure OpenAI or a self-hosted model.
- **Fix:** allow exploration (upload + profiling + preview) without a key, keep synthesis gated; add
  the enterprise provider targets.
- **Effort:** ~0.5 day for the gate, provider work depends on the target.

#### G14. Only the first 120 rows of a file are ever used
- **Evidence:** `records_from_frame(df, limit=MAX_ROWS)` with `MAX_ROWS = 120` and `df.head(limit)`
  ([feature_engineering.py](./backend/rl_ide/synthesizer/feature_engineering.py) L25, L190–L193). A
  600-row upload discards 80 % of the data, always from the tail.
- **Why it matters:** with G1 this compounds — the policy may train on 120 rows and report on the same
  120. Tail rows are also systematically different in export-ordered files.
- **Fix:** shuffle with a fixed seed before sampling, raise the cap for tabular data (the env is a
  lookup over records; 5 000 rows is cheap), and record the sampled-row count in the metric metadata
  so the number is never presented as "your whole dataset".
- **Effort:** ~0.5 day; folds into G1.

#### G15. The arena is a 30–155 s LLM call on the critical path
- **Evidence:** commit `5e9ab72` raised the arena timeout to 240 s because live synthesis needs
  111–155 s and the previous 35 s ceiling silently degraded to a generic fallback.
- **Fix:** declare the deterministic arena first (instant), then upgrade to the bespoke LLM arena when
  it arrives; never block "Lock Arena" on it.
- **Effort:** ~1 day; mostly frontend sequencing.

#### G16. No evaluation of the *deployed* policy, only of the trained one
- **Evidence:** metrics come from `evaluate_policy_deterministic` over the env instance
  ([loop_controller.py](./backend/rl_ide/autopilot/loop_controller.py) L981+). The ONNX artefact is
  checked for numeric parity (≤ 5.1e-07) but never for behavioural equivalence over a full episode.
- **Fix:** run one held-out episode batch through the exported ONNX session and assert action agreement
  with the SB3 policy; include the result in the bundle's `baseline.json`.
- **Effort:** ~0.5 day.

---

## 3. Measured facts

| Measurement | Value | How it was obtained |
|---|---|---|
| Conversations table rows | 0 | `select count(*) from conversations` |
| `conversations.db` size | 220 774 400 B | `ls -la backend/runs/conversations.db` |
| Free pages in that file | 53 897 / 53 900 | `pragma page_count` / `pragma freelist_count` |
| Stale `temp_verify_*` workspaces | 609 | `ls -d backend/workspaces/temp_verify_* \| wc -l` |
| `backend/workspaces` size | 20 MB | `du -sh` |
| `backend/runs` size | 213 MB | `du -sh` |
| Rows used from a 600-row upload | 120 | `MAX_ROWS`, `df.head(limit)` |
| Dataset episode horizon | 25 steps | `self.episode_horizon` |
| Synthesis latency | 2–3 min/env | `progress.md` §10.4 |
| Arena synthesis latency | 111–155 s | commit `5e9ab72` |
| Backend suite runtime | ≈11 min, 90 tests | `pytest -q` |
| CI configuration files | 0 | `ls .github` → not found |
| Backend dependency lockfile | none | no `uv.lock`/`poetry.lock`; unpinned `pyproject.toml` |

---

## 4. Shortest credible path to a paying pilot

Three phases, each independently shippable and each ending in a green suite plus a commit.

**Phase P0-A — "honest number" (≈2.5 days)**
G1 + G14 + G16.
Acceptance: a run produces a `scorecard.json` containing held-out success rate, human-label baseline,
Wilson CI, per-segment breakdown, sampled-row counts, and ONNX-vs-SB3 action agreement; the browser
suite asserts the held-out number, not the training number; a test proves the split is stable across
processes and that evaluation runs without oracle masking.

**Phase P0-B — "safe to host" (≈3 days)**
G2 + G3 + G4 + G9.
Acceptance: env modules execute in a sandboxed subprocess with a timeout and no network; default
bundles contain no raw rows; every conversation and dataset is owner-scoped and survives a restart;
verification directories are created under `TemporaryDirectory` and the DB is compacted after deletes.

**Phase P1-C — "operable" (≈3 days)**
G5 + G6 + G7 + G8.
Acceptance: `/chat` returns a job id in < 1 s; progress streams over the websocket; cancel works;
restarting the API mid-synthesis resumes or fails cleanly with a retry hint; usage and cost are
recorded per conversation; CI runs the fast suite on every push and a container image builds.

Only after these three does G10 (sequential tasks) make sense — it is a value upgrade, not a
trust-building one.

---

## 5. What I would deliberately not do yet

- **Fine-tuning or training an LLM policy.** The differentiator here is a small, auditable,
  exportable decision policy with a measured scorecard. Adding an LLM-policy mode before the scorecard
  exists would make the product harder to evaluate, not more valuable.
- **Multi-agent orchestration features.** Nothing in the current buyer conversation asks for them.
- **Hosting customer policies for them.** Export to ONNX + a policy daemon is already the right shape;
  keep deployment inside the customer's environment until G2 lands.

---

## 6. Relationship to the existing documents

This file does not replace the audit register; it promotes three items that were previously
**ACCEPTED** as local-only residual risk into commercial blockers, and adds the operational layer that
the audit never covered:

| This document | Prior reference | Change of status |
|---|---|---|
| G2 | F30 residual (no built-in auth, `postMessage "*"`) | accepted → **blocking** for hosted use |
| G3 | F17 residual (sample rows in exported bundles) | accepted → **blocking** under a DPA |
| G11 | F28 residual (drift "unmonitored") | accepted → **P2 product gap** |
| G1, G4, G5, G6, G7, G9 | not previously registered | new |
