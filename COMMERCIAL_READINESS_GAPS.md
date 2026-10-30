# RLHarness — Commercial Readiness Gap Analysis

**Analysis date:** 2026-10-30
**Repository state analysed:** `main` at `5e9ab72`, working tree clean
**Scope:** what stands between the current working studio and software a customer would pay for.
**Not in scope:** correctness of the existing design (covered by
[DESIGN_AUDIT_AND_PHASED_REMEDIATION.md](./DESIGN_AUDIT_AND_PHASED_REMEDIATION.md), issues F1–F39)
and the test history (covered by [progress.md](./progress.md), §10).

**Method:** every claim below was read out of the source or measured on the live instance. Numbers in
§3 were produced by running the commands shown next to them, not estimated.

> **Remediation status (2026-10-30):** Phase **P0-A** is implemented and committed — **G1, G14 and
> G16 are CLOSED** (see each entry, and `progress.md` §11). Phase **P0-B.1** is implemented and
> committed — **G3, G4 (scoping) and G9 are CLOSED**, with real authentication still outstanding
> under G4 (`progress.md` §12). Phase **P0-B.2** is implemented and committed — **G2 is CLOSED**:
> training, verification, export and interactive play all execute generated code only inside a
> `bwrap` worker (`progress.md` §13). Phase **P1-C** is complete: **G5, G6, G7 and G8 are CLOSED**
> (`progress.md` §15.1–§15.3).

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
  ([loop_controller.py](./backend/rl_ide/autopilot/loop_controller.py) L537, L981).
- **Evidence — three different policies, all called "the policy":**

| Stage | Action masking | Where |
|---|---|---|
| Training | none — plain SB3 `model.learn` | `loop_controller.py` L522 |
| Critic calibration | `get_valid_actions()` | `loop_controller.py` L938–L941 |
| Deterministic evaluation | `get_critical_actions()` **first**, else `get_valid_actions()` | `loop_controller.py` L1024–L1040 |
| ONNX export / daemon | none — the graph takes `obs` only | `exporter.py` L109 |

`get_critical_actions()` is a tactical oracle ("block the immediate threat"), so the evaluation number
is produced by a policy the customer will never receive. The two functions are not even part of a
validated contract: the env specification only *suggests* them to the LLM
([env_builder.py](./backend/rl_ide/synthesizer/env_builder.py) L373), so whether an environment
implements them, and with what semantics, varies per synthesis. There is also no baseline comparison,
no confidence interval, and no per-segment breakdown.

- **Why it blocks revenue:** the "87.0 % / 95.4 % / 81.2 %" run in `progress.md` §10 is a
  training-set number produced by an oracle-assisted policy that is neither the trained one nor the
  exported one. Any competent buyer or data scientist will ask for a held-out figure, and when it is
  not available the whole tool reads as a demo.
- **Fix:**
  1. **Split** deterministically 70/15/15 by row hash (stable across processes, independent of file
     order); train on 70 %, select on 15 %, report on 15 %.
  2. **Masking doctrine.** Remove `get_critical_actions()` from evaluation entirely — it is an
     oracle, not semantics. Keep `get_valid_actions()` **only if all three hold**: it is a pure
     legality predicate (environment physics, not a hint), the exported ONNX daemon applies the
     identical mask at inference, and training sees the same mask. Today none of the three hold, so
     the default must be: evaluate unmasked, and report the illegal-action rate as a separate
     number. Anything else is a different policy from the one being shipped.
  3. **Baseline and error bars.** Report human-label agreement (the CSV's own label) as the bar the
     policy must clear, add a Wilson 95 % CI to every success-rate figure, and emit a per-class /
     per-segment table.
- **Effort:** ~1.5 days including tests.
- **Status: CLOSED in Phase P0-A.** Content-hash 70/15/15 split (`zlib.crc32`, stable across
  processes); rounds score on `val`, the scorecard scores `test` once; both masks are gone from
  evaluation and illegal moves are reported; baselines are uniform (1/k), majority class and
  logistic regression, with a Wilson 95 % interval. On the 600-row support-ticket fixture the honest
  number is **85.4 % [83.3–87.2]** versus the previously reported 90.6 %, and the strongest baseline
  (logistic regression) reaches 84.5 % — the RL policy is only 0.9 pts ahead, and the product now
  says so.

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
- **Evidence — `multiprocessing` alone is not a sandbox (measured):** a `spawn`-context worker runs
  with the API's UID and filesystem, so it read the tenant database
  (`read_ok`, uid 1000) exactly like the parent. The same read inside `bwrap` failed with
  `FileNotFoundError`, and outbound sockets failed with `Network is unreachable`.
- **Status: CLOSED (Phase P0-B.2) for training, verification, export and `/interact`.**
  Generated modules are imported and exercised only inside a `bwrap` worker with a private case
  directory, no network, no provider keys and a wall-clock timeout. Proven by tests that make the
  module hostile at import time: reading `runs/conversations.db` fails with `FileNotFoundError`,
  opening a socket fails with "Network is unreachable", and no `*KEY*`/`*TOKEN*` variable reaches the
  worker. `unshare -rn` is deliberately **not** accepted as a substitute (it leaves the filesystem
  readable). Training emits `auto_tune_skipped` inside the sandbox because the LLM tuner has no
  network there. Interactive play uses a persistent `serve` worker that keeps environment state
  across requests; the API no longer instantiates the module for `/interact` or
  `GET /conversations/{id}`. **Residual:** without `bwrap` there is no isolation, so a hosted tenant
  still gets 501 on `/interact` (the local studio keeps the legacy in-process path). That is a host
  requirement, not a gap in the code path.
- **Fix:** keep the pipe-based worker design (hard wall-clock timeout, crash isolation, no shared
  memory with the API), and launch **that worker under `bwrap`**:
  `bwrap --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
  --ro-bind <case-dir> /app --dev /dev --proc /proc --tmpfs /tmp \
  --unshare-net --unshare-pid --die-with-parent <python> -m worker`.
  Bind only the interpreter, the wheels, and the current case directory; no `/home`, no `/etc`, no
  network, read-only root. `bwrap` is present at `/usr/bin/bwrap` on this host and the probe passed,
  so it is testable today. Fall back to `unshare -rn` where `bwrap` is unavailable; if neither
  exists, **fail closed in multi-tenant mode** rather than silently running generated code
  in-process (single-user local mode may keep the current behaviour behind an explicit flag).
  For the exported daemon: bind `127.0.0.1` by default, require a token, and document the flag that
  opens it up.
- **Effort:** ~2–3 days for the sandboxed worker plus the fallback matrix; ~2 hours for the daemon
  defaults.

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
- **Status: CLOSED in Phase P0-B.1.** Bundles are schema-only by default: the embedded `_records`
  line is replaced with `_records = []`, `data/README.md` documents mounting `records.json`, and
  `scorecard.json → deployment.contains_training_data` records whether data was included.
  `POST /export {"include_data": true}` restores the self-contained bundle for operators who need
  it. A regression test packs the same environment both ways and asserts the customer value appears
  only in the opt-in bundle.

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
- **Status: CLOSED for scoping in Phase P0-B.1 (authentication still open).** `conversations.owner_id`
  with a migration, owner-scoped list/get/delete, and a durable `DatasetStore` (`runs/datasets.db` +
  gzipped rows) that fixes uploads vanishing on restart. Cross-owner reads and deletes return 404.
  **Residual:** the owner is read from a trusted `x-owner-id` header — that is tenancy plumbing, not
  identity. A real authentication layer is still required before hosting.

### P1 — cannot operate or bill it

#### G5. Synthesis is a blocking HTTP POST with no job identity
- **Evidence:** `POST /api/v4/conversations/{id}/chat` awaits
  `asyncio.to_thread(global_omni_agent.process_message, …)` and only afterwards may spawn background
  training ([server.py](./backend/rl_ide/api/server.py) L723–L754). Measured latency is 2–3 minutes per
  environment (`progress.md` §10.4) and 111–155 seconds for the arena alone (commit `5e9ab72`); the
  client timeout had to be raised to 240 s to stop silent fallbacks.
- **Why it blocks revenue:** a three-minute spinner with no progress, no cancel and no recovery is the
  single most common reason a trial user bounces. It also makes the API unusable for automation.
- **Status: CLOSED (Phase P1-C.2).** `POST /chat` with `Prefer: respond-async` returns
  `202 {job_id, stage}` in ~60 ms; `intake → env_synthesis → verification → arena → ready` are
  persisted and published as `job_stage` events on `/ws/v4/{session_id}`; `GET /jobs/{id}` polls,
  `POST /jobs/{id}/cancel` cancels at a stage boundary, and a restart fails the job with a retry hint
  (`progress.md` §15.2). The synchronous form is preserved for simple clients.
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
- **Status: CLOSED (Phase P1-C.2) for job and queue state.** Jobs live in a `jobs` table with atomic
  claims; training is enqueued and run by a queue consumer, not by a coroutine in the request
  process; both stores run SQLite in WAL mode so processes can share the file. **Residual:** SB3 models,
  environment instances and the loop's training lock are still per-process, so several Uvicorn workers
  with shared training state and resuming an interrupted round are not supported yet — an interrupted
  job fails with a retry hint (`progress.md` §15.2).
- **Fix:** move job/run state into the database, make training a worker consuming a queue (even a
  SQLite/Postgres table is enough), and let the API be stateless. This is the prerequisite for G5 as
  well.
- **Effort:** ~3 days; can be folded into G5.

#### G7. No usage metering or cost control
- **Evidence:** `llm/client.py` records nothing about tokens, calls or spend per conversation. There is
  no quota, no budget, no per-tenant accounting.
- **Why it blocks revenue:** you cannot invoice, cannot cap a runaway customer, cannot tell an
  enterprise buyer what a run costs, and cannot detect that synthesis on a large file is burning money.
- **Status: CLOSED (Phase P1-C.1).** Every provider call records `(conversation_id, owner_id,
  provider, model, input_tokens, output_tokens, cost_usd, latency_ms)` in the `llm_usage` table,
  `GET /api/v4/usage` exposes totals, a per-model split and the remaining budget, and
  `RL_IDE_OWNER_BUDGET_USD` fails the call *before* it is made (`progress.md` §15.1).
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
- **Status: CLOSED (Phase P1-C.3).** `requirements.lock` / `requirements-dev.lock` pin the resolved
  set (a clean venv built from the lock passes the fast suite); `Dockerfile` installs that set with
  bubblewrap, serves the built SPA and sets `RL_IDE_REQUIRE_ISOLATION=1`, which now fails startup on a
  host without a sandbox; `.github/workflows/ci.yml` runs the fast suite (`-m "not slow and not
  live_llm"`, 67 passed in 1:08), boots the API, builds the frontend and builds+boots the image; the
  full suite runs nightly (`progress.md` §15.3).
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
- **Status: CLOSED in Phase P0-B.1 (retention policy still open).** Verification runs in a
  `TemporaryDirectory` that is always cleaned; deleting a conversation removes its workspace, runs
  directory, exported bundles and dataset rows, then `VACUUM`s (the DB here was 220 MB / 0 rows /
  53 897 free pages before this change). `DELETE /api/v4/datasets/{id}` removes the payload file and
  compacts. 609 stale `temp_verify_*` directories were cleaned from the checkout. **Residual:** no
  scheduled retention window yet — deletion is explicit, not time-based.

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
- **Status: CLOSED in Phase P2-A.** `SandboxSmokeTester.relax_unreachable_success` rewrites the
  *success criteria* (a relaxed threshold cannot help a signal that never fires), re-runs the full
  smoke test, and accepts the result only if success becomes observable; capped at two rewrites, and
  the verified module is kept when the provider cannot fix it. The probe now separates "reachable in
  principle" (the environment's own `get_critical_actions` hint) from "reachable by a policy that can
  be learned" (random/heuristic play), so an oracle-only task is reported as a low achievable rate
  instead of looking fine. The turn tells the user when criteria were relaxed, because a silently
  relaxed KPI is a different task.

#### G13. Onboarding requires a verified key before anything works
- **Evidence:** every substantive route calls `require_verified_llm()` and returns 403 otherwise
  ([server.py](./backend/rl_ide/api/server.py) L97–L106). Only one provider family is configured in
  practice (`deepseek-flash`).
- **Why it matters:** fine for BYO-key developers, friction for a pilot, and a procurement question for
  enterprises that require Azure OpenAI or a self-hosted model.
- **Fix:** allow exploration (upload + profiling + preview) without a key, keep synthesis gated; add
  the enterprise provider targets.
- **Effort:** ~0.5 day for the gate, provider work depends on the target.
- **Status: CLOSED in Phase P2-A (gate only).** The startup lockout is gone: the studio opens in
  explore mode, uploads and existing agents work, and only the four model-backed actions
  (`synthesize_environment`, `architect_chat`, `train`, `restyle_arena`) refuse. The refusal carries
  an `X-RL-Reason: llm-key-required` header so the UI opens the setup dialog instead of
  string-matching prose, and `GET /api/v4/capabilities` publishes exactly what needs a model. The
  enterprise provider targets (Azure OpenAI, self-hosted) remain open.

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
- **Status: CLOSED in Phase P0-A.** Sampling is shuffled with a fixed seed (`random_state=1337`);
  training and evaluation already used the full runtime record set, and the scorecard now records
  `rows.total` (usable rows) next to `rows.embedded_in_module` (the 120-row standalone sample) and
  the exact per-split counts.

#### G15. The arena is a 30–155 s LLM call on the critical path
- **Evidence:** commit `5e9ab72` raised the arena timeout to 240 s because live synthesis needs
  111–155 s and the previous 35 s ceiling silently degraded to a generic fallback. When that fallback
  did appear (the Submarine run), it was indistinguishable from a broken generator — the UI showed
  generic telemetry cards with no indication that a bespoke canvas was still coming.
- **Fix:** declare the deterministic arena first (instant), never block "Lock Arena" on the LLM, then
  hot-swap when the bespoke arena arrives. The swap mechanism already exists — `arena_updated` events
  replace the iframe HTML ([App.tsx](./frontend/src/App.tsx) L699–L701) and the `mock_preview` mode
  plumbing is in place
  ([SandboxedArenaIframe.tsx](./frontend/src/components/SandboxedArenaIframe.tsx) L67, L82) — so what
  is missing is the *pending* state, not the swap:
  - a persistent status pill in the arena header while the job is in flight, e.g.
    `🎨 Synthesizing bespoke domain canvas (~90 s) — showing live telemetry preview`, driven by the
    `arena` / stage event from G5 rather than a fixed timer;
  - the pill must change to a clear failure state (with a retry affordance) if synthesis errors,
    instead of silently leaving the generic arena on screen;
  - the generic arena must be visibly labelled as a preview, so it can never be mistaken for the
    final artefact in a screenshot or a demo.
- **Effort:** ~1 day; mostly frontend sequencing plus the stage event from G5.
- **Status: CLOSED in Phase P2-A.** The arena is declared from the blueprint *before* environment
  synthesis and replaced in place when the bespoke canvas arrives, so nothing waits on the provider.
  The toolbar carries a live pill driven by the real job stage, a failed generation turns into an
  explicit error with a **Retry** action, and the generic arena is labelled as a preview. Measured on
  a real turn: `202` in 0.08 s, preview visible at 135 s (the moment the environment existed — it
  used to be the *final* state at ~260 s), bespoke canvas at 261 s, and a lock set during generation
  survived the completion save.

#### G16. No evaluation of the *deployed* policy, only of the trained one
- **Evidence:** metrics come from `evaluate_policy_deterministic` over the env instance
  ([loop_controller.py](./backend/rl_ide/autopilot/loop_controller.py) L981+). The ONNX artefact is
  checked for numeric parity (≤ 5.1e-07) but never for behavioural equivalence over a full episode.
- **Fix:** run one held-out episode batch through the exported ONNX session and assert action agreement
  with the SB3 policy; include the result in the bundle's `baseline.json`.
- **Effort:** ~0.5 day.
- **Status: CLOSED in Phase P0-A.** `PolicyExporter.verify_behavioural_parity` replays identical
  seeded episodes through SB3 and ONNX and reports action agreement plus both success rates; the
  result is written to the bundle as `scorecard.json → deployment`, next to the numeric parity
  figure. A regression test asserts 100 % agreement for a freshly exported policy.

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
| `multiprocessing` spawn child reads `runs/conversations.db` | **yes** (uid 1000, same as server) | spawn probe |
| Same read inside `bwrap` (ro `/usr`,`/lib`,`/lib64`, no `/home`) | `FileNotFoundError` | bwrap probe |
| Outbound socket inside `bwrap --unshare-net` | `Network is unreachable` | bwrap probe |
| Unprivileged user namespaces | enabled (`unprivileged_userns_clone=1`, `max_user_namespaces=26377`) | `/proc/sys/...` |
| Action masking by stage | train **none** / eval **critical→valid** / export **none** | read + grep |
| Synthesis latency | 2–3 min/env | `progress.md` §10.4 |
| Arena synthesis latency | 111–155 s | commit `5e9ab72` |
| Backend suite runtime | ≈11 min, 90 tests | `pytest -q` |
| CI configuration files | 0 | `ls .github` → not found |
| Backend dependency lockfile | none | no `uv.lock`/`poetry.lock`; unpinned `pyproject.toml` |

---

## 4. Shortest credible path to a paying pilot

Three phases, each independently shippable and each ending in a green suite plus a commit.

**Phase P0-A — "honest number" (≈2.5 days) — DONE (2026-10-30)**
G1 + G14 + G16.
Acceptance: a run produces a `scorecard.json` containing held-out success rate, human-label baseline,
Wilson CI, per-segment breakdown, sampled-row counts, and ONNX-vs-SB3 action agreement; the browser
suite asserts the held-out number, not the training number; a test proves the split is stable across
processes, that evaluation never consults `get_critical_actions()`, and that the exported policy is
scored under the same masking regime it will run under (today: none, with the illegal-action rate
reported separately).

**Phase P0-B — "safe to host" (≈3 days) — DONE (B.1 2026-10-30, B.2 2026-10-30)**
G2 + G3 + G4 + G9.
Acceptance: env modules execute in a `bwrap` (or `unshare -rn`) worker with a wall-clock timeout, no
network, a read-only root and only the case directory visible; the multi-tenant path fails closed when
isolation is unavailable; default bundles contain no raw rows; every conversation and dataset is
owner-scoped and survives a restart; verification directories are created under `TemporaryDirectory`
and the DB is compacted after deletes.

**Phase P1-C — "operable" (≈3 days)**
G5 + G6 + G7 + G8.
Acceptance: `/chat` returns a job id in < 1 s; progress streams over the websocket; cancel works;
restarting the API mid-synthesis resumes or fails cleanly with a retry hint; usage and cost are
recorded per conversation; CI runs the fast suite on every push and a container image builds.

**Phase P2-A — "remove sales friction" (≈2.5 days) — DONE (2026-10-30)**
G12 + G13 + G15, plus a write-ahead-log leak found while doing them.
Acceptance: a studio with no provider still opens, uploads, lists agents and shows an arena; only
the four model-backed actions refuse, in a machine-readable way; the arena is visible and lockable
before the bespoke canvas exists and is labelled honestly if that canvas fails; an environment whose
success criteria can never fire is repaired and re-verified rather than trained against.
All met, with a live measurement in `progress.md` §16.

Only after these does G10 (sequential tasks) make sense — it is a value upgrade, not a
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

---

## 7. External review (2026-10-30) — refinements folded into G1, G2, G15

A frontier-model review of the first draft raised three refinements. All three were checked against
the source and accepted; two changed the prescribed fix, one turned out to need less work than the
reviewer assumed.

| Refinement | Verdict | Effect on this document |
|---|---|---|
| R1 — separate `get_valid_actions()` (rules) from `get_critical_actions()` (oracle) | **Accepted, strengthened** | G1 evidence and fix rewritten |
| R2 — `multiprocessing` is not a security boundary; use `bwrap`/`unshare` | **Accepted, verified empirically** | G2 fix rewritten, §3 rows added |
| R3 — make the instant-preview → bespoke-arena swap explicit in the UI | **Accepted, scope reduced** | G15 expanded |

**R1 — masking doctrine (accepted, and stronger than proposed).**
The review is right that legality masking is legitimate where it models physics, and that
`get_critical_actions()` is an oracle. Verified in code: evaluation prefers `critical_actions` and only
falls back to `valid_actions` (`loop_controller.py` L1024–L1026), while the exported ONNX graph takes
`obs` alone with no mask at all (`exporter.py` L109) and training runs plain `model.learn`
(`loop_controller.py` L522) with no mask either. So the proposed condition — *allow legality masking
only if the bundle applies it too* — is currently **false in every respect**, which makes the correct
first move unambiguous: strip both masks from evaluation, report the illegal-action rate, and only
reintroduce `get_valid_actions()` together with a masked export path and matching training. The deeper
finding is train/serve skew, not just eval inflation: three stages, three policies
(§3), one reported number.

**R2 — sandbox (accepted, measured).**
Confirmed rather than assumed: a `spawn` child has the server's UID and filesystem and read the tenant
SQLite database (probe in §3). Under `bwrap` with only the interpreter, libraries and the case
directory bound read-only, the same read fails with `FileNotFoundError` and sockets fail with
`Network is unreachable`. Both probes ran on this host, so the fix is testable now. Two operational
caveats recorded with the fix: `bwrap` needs unprivileged user namespaces (present here: 26 377
available), and where namespaces are disabled — some CI and hardened container runtimes — the system
must **fail closed** in multi-tenant mode rather than silently reverting to in-process execution.

**R3 — arena swap (accepted, but smaller than it looks).**
The behaviour the review describes already half exists: `arena_updated` events replace the iframe
HTML (`App.tsx` L699–L701) and `mock_preview` mode is wired
(`SandboxedArenaIframe.tsx` L67, L82). What is missing is the *pending* affordance and an honest
failure state — so G15 is scoped to a stage-event-driven status pill, not to new transport. One
correction to the proposed wording: the pill must be driven by real job events rather than a fixed
"~90 s" estimate, since the measured range is 111–155 s and a countdown that overruns reads as another
silent failure.
