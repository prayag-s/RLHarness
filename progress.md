# RL-Native Agent IDE — Progress & Session Handoff Log

**Project:** RL-Native Agent IDE (v3.0 Vibe Studio & Universal Ingestion)  
**Last Updated:** 2026-10-27  
**Current Status:** Phase 0A-0E, Phase 1, Phase 2 & Phase 3A Completed (100% Tests Passing: 28/28; Frontend Build Clean 0 Errors)  


---

## 1. Environment & Setup

- **Python Version:** 3.12.3
- **Virtual Environment:** `/home/devuser/projects/RLHarness/.venv`
- **Package Manager:** `uv` (`/snap/bin/uv`)
- **Key Installed Dependencies:**
  - `fastapi`, `uvicorn[standard]`
  - `pydantic>=2.0` (2.14.1)
  - `pyyaml`, `numpy`, `pandas`
  - `gymnasium` (1.4.0), `stable-baselines3` (2.9.2), `torch` (2.14.1)
  - `onnx` (1.23.2), `onnxruntime` (1.31.1), `onnxscript` (0.7.2), `scikit-learn` (1.9.1)
  - `httpx`, `pytest` (9.1.1), `pytest-asyncio`

### Quick Commands
```bash
# Activate virtual environment
source .venv/bin/activate

# Run Phase 0D tests
CUDA_VISIBLE_DEVICES="" PYTHONPATH=backend pytest backend/tests/test_phase_0d.py -v -s

# Run full backend test suite (0A + 0B + 0C + 0D)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=backend pytest backend/tests/ -v
```

---

## 2. Completed Work: Phase 0A (Proof of Loop Foundation)

- **Adapter Protocol (`backend/rl_ide/contracts/adapter.py`):** `ObservationSpec`, `ActionSpec`, `StepResult`, `EpisodeLog`, `@runtime_checkable` `DomainAdapter`.
- **AST Sandbox & DSL (`backend/rl_ide/reward/dsl.py`, `evaluator.py`):** Whitelist node enforcement, rejection of `Call`, `Import`, `Lambda`, dunders, static schema validation against `event_schema()`, pre-compiled AST bytecode evaluator (<2ms execution).
- **Reference Adapter 1 (`backend/rl_ide/adapters/stream_adapter.py`):** `CSVStreamAdapter` stateful MDP tracking cooldown and alert budget, plus `generate_synthetic_stream`.

---

## 3. Completed Work: Phase 0B (Authoring & State Persistence)

- **BYOK LLM Layer (`backend/rl_ide/compiler/llm_client.py`):** OpenAI REST, Anthropic Messages API, local Ollama/vLLM, and deterministic offline `MockRulesLLMClient`.
- **Compiler Models (`backend/rl_ide/compiler/models.py`):** Pydantic v2 `TaskConfig`, `TermDiff`, `ConflictWarning`, `Stage1Preview`, `Stage2Preview`, `RewardDiffProposal`.
- **NL Compiler (`backend/rl_ide/compiler/nl_to_config.py`):** Automatic AST static validation and up to 2 repair retries.
- **State Store (`backend/rl_ide/state/store.py`):** SQLite (`tasks.db`) and immutable filesystem snapshot bundles (`runs/<task_id>/v<N>/`) with full rollback restoring DSL and checkpoint weights.

---

## 4. Completed Work: Phase 0C (Training Runner & Critique Diff Engine)

- **Gymnasium Wrapper (`backend/rl_ide/trainer/gym_env.py`):** `AdapterGymEnv` mapping observation/action spaces, live reward evaluation, dynamic hot-swapping via `set_evaluator()`.
- **Episode Buffer (`backend/rl_ide/trainer/buffer.py`):** 200-capacity FIFO circular buffer with retroactive batch re-scoring.
- **Training Runner (`backend/rl_ide/trainer/runner.py`):** Warm-start behavioral cloning pre-pass (~50ms CPU), streaming PPO micro-bursts, diff snapshot persistence, and 15-epoch Critic Warmup ($V_\phi$) on re-scored discounted returns.
- **Critique→Diff Engine (`backend/rl_ide/compiler/critique_engine.py`):** Two-Stage Preview engine with sub-20ms counterfactual re-scoring (Stage 1) and in-memory policy cloning speculative micro-updates (Stage 2) preserving 100% of locked golden episodes.

---

## 5. Completed Work: Phase 0D (Export, Drift & FastAPI Server)

### 5.1 Unsupervised Drift Detector
- **File:** `backend/rl_ide/drift/monitor.py`
- `DriftDetector` & `DriftBaseline`:
  1. **Input Feature PSI:** 10-bucket quantile boundaries fitted on baseline training episodes. Computes $\text{PSI} = \sum (P_i - Q_i) \ln(P_i / Q_i)$ on live inference batches (flags drift if $\max(\text{PSI}) > 0.25$).
  2. **Policy Action-Probability Entropy Shift:** Calculates Shannon entropy $H(\pi(\cdot|s)) = -\sum p(a|s) \log p(a|s)$ and flags drift if rolling entropy shifts by $> 2.5\sigma$.
  3. **Action-Rate Distribution Shift:** Tracks rolling frequency of discrete actions against baseline binomial standard error and flags drift if Z-score $> 3.0$.

### 5.2 ONNX Policy Exporter & .runbundle Packaging
- **File:** `backend/rl_ide/export/exporter.py`
- `PolicyExporter`:
  - Wraps SB3 Actor network and exports to `model.onnx` (`opset_version=17`).
  - Strict numerical parity verification with `onnxruntime.InferenceSession` across 32 test observations (`np.allclose(atol=1e-5)`). Parity verified: **5.96e-08**.
  - Packages self-contained `.runbundle` directory and `.zip` archive containing:
    - `model.onnx`
    - `task_config.yaml`
    - `reward_dsl.yaml`
    - `drift_baseline.json`
    - `run_local.py`: Standalone, zero-LLM, zero-SB3 Python CLI supporting `--sample` and custom CSV/NPY input files. Verified working via subprocess execution.

### 5.3 FastAPI REST + WebSocket Server
- **File:** `backend/rl_ide/api/server.py`
- Full API implementation:
  - `GET /api/adapters`: lists registered adapters, observation/action specs, and event schemas.
  - `POST /api/tasks/compile`: translates user prompt into `TaskConfig`, warm-starts policy, saves v1.
  - `POST /api/train/burst`: triggers PPO training burst, streaming episode metrics.
  - `POST /api/episodes/{ep_id}/lock`: locks/unlocks golden episodes.
  - `POST /api/critique/preview`: evaluates critique, generating YAML diff, Stage 1 counterfactual delta, and Stage 2 action flip preview.
  - `POST /api/critique/apply`: applies confirmed diff, saves snapshot, executes Critic Warmup.
  - `POST /api/tasks/rollback`: rolls back to version `vN`.
  - `POST /api/export`: exports `.runbundle` and `.zip` with numerical parity check.
  - `WS /ws/stream`: real-time WebSocket telemetry broadcasting episode frames and rewards.

---

## 6. Verification Gate Results (Phase 0A through 0D)

### Full Regression Suite: 18 / 18 Tests Passing (100% Green)

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.7.0 -- /home/devuser/projects/RLHarness/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/devuser/projects/RLHarness/backend
configfile: pyproject.toml
plugins: asyncio-1.4.1, anyio-4.15.1

backend/tests/test_phase_0a.py::test_synthetic_stream_generation PASSED  [  5%]
backend/tests/test_phase_0a.py::test_1_valid_yaml_dsl_evaluation PASSED  [ 11%]
backend/tests/test_phase_0a.py::test_2_undefined_variable_raises_validation_error PASSED [ 16%]
backend/tests/test_phase_0a.py::test_3_malicious_code_injection_raises_security_error PASSED [ 22%]
backend/tests/test_phase_0a.py::test_4_adapter_episode_stepping_and_fast_rescoring PASSED [ 27%]
backend/tests/test_phase_0b.py::test_1_compile_english_prose_to_valid_config PASSED [ 33%]
backend/tests/test_phase_0b.py::test_2_automatic_validator_retry_on_invalid_field PASSED [ 38%]
backend/tests/test_phase_0b.py::test_3_saving_sequential_versions_and_rollback PASSED [ 44%]
backend/tests/test_phase_0b.py::test_byok_llm_factory_fallback PASSED    [ 50%]
backend/tests/test_phase_0c.py::test_1_adapter_gym_env_and_hotswap PASSED [ 55%]
backend/tests/test_phase_0c.py::test_2_episode_buffer_operations PASSED  [ 61%]
backend/tests/test_phase_0c.py::test_3_warmstart_behavioral_cloning PASSED [ 66%]
backend/tests/test_phase_0c.py::test_4_critique_diff_engine_and_two_stage_preview PASSED [ 72%]
backend/tests/test_phase_0c.py::test_5_full_critique_alignment_loop PASSED [ 77%]
backend/tests/test_phase_0d.py::test_1_drift_detector_signals PASSED     [ 83%]
backend/tests/test_phase_0d.py::test_2_onnx_export_and_numerical_parity PASSED [ 88%]
backend/tests/test_phase_0d.py::test_3_runbundle_packaging_and_standalone_run_local PASSED [ 94%]
backend/tests/test_phase_0d.py::test_4_fastapi_rest_endpoints_and_websocket PASSED [100%]

======================= 18 passed, 7 warnings in 48.12s ========================
```

### Key Performance Metrics:
- ONNX vs PyTorch Numerical Parity Max Difference: **5.960e-08** (Gate: **< 1e-5**)
- Standalone `run_local.py` execution: **Verified Exit Code 0**
- Sub-10ms episode re-scoring: **1.957 ms**
- False positive reduction on critique: **96.0%**

---

## 7. Completed Work: Phase 0E (Three-Panel IDE Frontend)

### 7.1 Architecture & Stack
- **Framework:** React 18 + TypeScript + Vite + Tailwind CSS + `lucide-react`.
- **Layout:** Single-screen workbench with 3 responsive panels (28% Left | 44% Center | 28% Right).
- **Communication:** REST service client (`frontend/src/services/api.ts`) connecting to FastAPI endpoints + native WebSocket connection (`/ws/stream`) for streaming episode frames, reward curves, and per-term breakdowns.

### 7.2 Implemented Components
1. **Header Bar (`frontend/src/components/Header.tsx`):**
   - Brand banner (`RL-Native Agent IDE v2.0 Hardened`).
   - Adapter dropdown switcher (`CSVStreamAdapter` / `TextClassifyAdapter`).
   - Active Version badge with instant rollback dropdown (`v1`, `v2`, ...).
   - Real-time Drift Status indicator (`Drift: Healthy` / `Drift: Alert`).
   - BYOK Settings modal launcher.
   - Primary Export CTA: `[Export Owned Policy (ONNX)]`.

2. **Left Panel — Task & Reward DSL (`frontend/src/components/LeftPanel.tsx`):**
   - Natural language task description textarea with pre-populated baseline domain instructions.
   - `[Compile & Warm-Start]` button with loading spinners and status indicators.
   - **Active Terms View:** Visual card list of all compiled DSL terms showing `name`, `when` AST conditions, and signed `value` weights.
   - **YAML Inspector:** Syntax-styled raw DSL inspector with 1-click clipboard copy.
   - **Version History Timeline:** Chronological log of versions, timestamp, applied critique prose, and 1-click rollback buttons.

3. **Center Panel — Live Agent (`frontend/src/components/CenterPanel.tsx`):**
   - Playback control bar: `[Run 15-Ep Burst]`, `[Pause / Resume]`, `[Lock Episode ★]`.
   - Buffer telemetry badge (total buffer capacity and count of golden locked episodes).
   - **Domain Visualizer:** Responsive SVG rendering the 100-step time-series trajectory:
     - Price trajectory path (sky blue).
     - 90th-percentile rolling volatility threshold (amber dashed).
     - Agent action flags: True Positives (emerald pulse) vs False Alarms (rose circle).
     - Interactive step hover crosshair showing step index, price, action taken, and reward.
   - **Clickable Episode History Strip:** Scrollable horizontal chip list displaying episode ID, total score, and golden star lock status.
   - **Live Reward Curve:** Sparkline tracking total return across all episodes with min/avg/max statistics.
   - **Per-Term Reward Breakdown:** Horizontal proportional bar chart showing positive/negative contributions of every active DSL term for the inspected episode.

4. **Right Panel — Critique & Diff Workbench (`frontend/src/components/RightPanel.tsx`):**
   - Selected Episode Telemetry card (Reward, Anomaly alert count, Avg price, Peak volatility, Representative action).
   - English critique input box with 3 quick-critique demo chips.
   - `[Preview Critique (Two-Stage)]` trigger.
   - **Interactive Diff Card:**
     - Term diff badges (`+ Added`, `~ Modified`, `- Removed`) with 1-line rationales.
     - Conflict Warning notification box.
     - **Stage 1 Counterfactual Score Preview Delta:** Compares before vs after score on target episode with net $\Delta$ score.
     - **Stage 2 Forked Policy Flip Rate Preview:** Shows target action flip status, batch flip rate, and golden locked episodes preservation confirmation (`Preserved 100% of locked golden episodes`).
     - `[Apply Diff & Run Burst]` and `[Discard]` buttons.

5. **Modals:**
   - **Export Modal (`frontend/src/components/ExportModal.tsx`):** Displays ONNX numerical parity verification (`Max diff: 5.96e-08 ✅`), bundle file tree (`model.onnx`, `task_config.yaml`, `reward_dsl.yaml`, `drift_baseline.json`, `run_local.py`), and copyable standalone CLI command (`python run_local.py --sample`).
   - **Settings Modal (`frontend/src/components/SettingsModal.tsx`):** BYOK configuration for OpenAI, Anthropic, Ollama / vLLM, and model selection with local persistence.

---

## 8. Verification Gate Results (Phase 0E)

### Frontend Production Build: 100% Clean (0 Errors)
```text
> rl-ide-frontend@0.1.0 build
> tsc && vite build

vite v6.4.5 building for production...
✓ 1595 modules transformed.
dist/index.html                   0.56 kB │ gzip:  0.36 kB
dist/assets/index-Uxj3abLK.css   24.77 kB │ gzip:  5.19 kB
dist/assets/index-pp2IJfuF.js   212.24 kB │ gzip: 62.21 kB
✓ built in 16.20s
```

### Full Backend Regression Suite: 18 / 18 Tests Passing (100% Green)
```text
======================= 18 passed, 7 warnings in 47.54s ========================
```

---

## 9. Completed Work: Phase 1 (TextClassifyAdapter & Cross-Domain Conformance)

### 9.1 Second Reference Adapter: `TextClassifyAdapter`
- **File:** `backend/rl_ide/adapters/text_classify_adapter.py`
- **Domain:** Natural Language support ticket & email triage routing.
- **Actions (4 Discrete):** `["urgent", "normal", "spam", "escalate"]`.
- **Synthetic Generator:** `generate_synthetic_tickets(n_records=500, seed=42)` creating realistic tickets across categories (VIP outages, billing inquiries, phishing/scam emails, technical outages) with:
  - `ticket_id`, `subject`, `body`, `sender_tier` (`vip`, `free`, `internal`), `sender_is_vip`, `sender_is_internal`, `word_count`, `urgency_keyword_count`, `angry_tone_score` (0.0 - 1.0), `has_attachment`, `queue_load`, `true_label`.
- **Observation Pipeline:**
  - Fast deterministic 64-dim `HashingVectorizer(n_features=64, alternate_sign=False, norm="l2")` on text.
  - 7 normalized metadata features: `[sender_is_vip, sender_is_internal, word_count_norm, urgency_keyword_norm, angry_tone_score, has_attachment, queue_load_norm]`.
  - Total observation dimension: **71 dimensions**.
  - Fast CPU execution (<10 microseconds per forward pass, PPO trains in < 3 seconds).
- **Event Schema for DSL:** Exposes `obs.sender_is_vip`, `obs.sender_is_internal`, `obs.sender_tier`, `obs.word_count`, `obs.urgency_keyword_count`, `obs.angry_tone_score`, `obs.has_attachment`, `obs.queue_load`, `action.label`, `action.action_idx`, `state.queue_load`, `state.processed_count`, `truth.has_label`, `truth.label`, `truth.is_vip`, `truth.true_anomaly`.
- **Registration:** Added to `ADAPTER_REGISTRY` in `backend/rl_ide/compiler/nl_to_config.py` and supported in `MockRulesLLMClient`.

### 9.2 Frontend Text Episode Renderer
- **File:** `frontend/src/components/CenterPanel.tsx`
- Dynamically activates when `TextClassifyAdapter` is selected or when ticket step metadata is detected.
- **Ticket Viewer Card:**
  - Ticket ID with sender tier badges (`VIP Customer` in gold with Crown, `Internal Service` in blue, `Free Tier` in slate).
  - Step selector controls (`Ticket Step 3 of 25`).
  - Email Subject and styled body preview box.
  - Extracted NLP signals strip: Urgency keywords count, Angry tone score meter, Queue backlog count, Ground Truth label.
  - Agent Routing Decision banner: Action pill (`URGENT`, `ESCALATE`, `SPAM`, `NORMAL`), Ground truth comparison badge (`Correct Routing ✅` vs `Misclassified ⚠️`), and step reward.

### 9.3 Cross-Domain Conformance Suite
- **File:** `backend/tests/test_adapter_conformance.py`
- Parameterized test executing the identical end-to-end RL pipeline across both adapters:
  `[CSVStreamAdapter, TextClassifyAdapter]`
- Verifies:
  1. Protocol conformance (`isinstance(adapter, DomainAdapter)`).
  2. `NLToConfigCompiler` AST validation and compilation into `TaskConfig`.
  3. `TrainingRunner.warmstart_policy` behavioral cloning pre-pass.
  4. `TrainingRunner.run_burst` streaming PPO episodes into `EpisodeBuffer`.
  5. `CritiqueEngine.propose_diff` generating valid diffs, Stage 1 counterfactual score delta, and Stage 2 forked behavior action flips.
  6. `TrainingRunner.apply_reward_diff` hot-swapping evaluator and executing Critic Warmup.
  7. `PolicyExporter.export_onnx` with strict numerical parity (< 1e-5) against `onnxruntime`.
  8. `PolicyExporter.create_runbundle` creating standalone `.runbundle` with `run_local.py`.
  9. `DriftDetector` fitting quantile PSI baseline and evaluating batch drift.

---

## 10. Verification Gate Results (Phase 1)

### Full Backend Regression Suite: 20 / 20 Tests Passing (100% Green)
```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.7.0 -- /home/devuser/projects/RLHarness/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/devuser/projects/RLHarness/backend
configfile: pyproject.toml
plugins: asyncio-1.4.1, anyio-4.15.1

backend/tests/test_adapter_conformance.py::test_full_cross_domain_conformance_pipeline[CSVStreamAdapter] PASSED [  5%]
backend/tests/test_adapter_conformance.py::test_full_cross_domain_conformance_pipeline[TextClassifyAdapter] PASSED [ 10%]
backend/tests/test_phase_0a.py::test_synthetic_stream_generation PASSED  [ 15%]
backend/tests/test_phase_0a.py::test_1_valid_yaml_dsl_evaluation PASSED  [ 20%]
backend/tests/test_phase_0a.py::test_2_undefined_variable_raises_validation_error PASSED [ 25%]
backend/tests/test_phase_0a.py::test_3_malicious_code_injection_raises_security_error PASSED [ 30%]
backend/tests/test_phase_0a.py::test_4_adapter_episode_stepping_and_fast_rescoring PASSED [ 35%]
backend/tests/test_phase_0b.py::test_1_compile_english_prose_to_valid_config PASSED [ 40%]
backend/tests/test_phase_0b.py::test_2_automatic_validator_retry_on_invalid_field PASSED [ 45%]
backend/tests/test_phase_0b.py::test_3_saving_sequential_versions_and_rollback PASSED [ 50%]
backend/tests/test_phase_0b.py::test_byok_llm_factory_fallback PASSED    [ 55%]
backend/tests/test_phase_0c.py::test_1_adapter_gym_env_and_hotswap PASSED [ 60%]
backend/tests/test_phase_0c.py::test_2_episode_buffer_operations PASSED  [ 65%]
backend/tests/test_phase_0c.py::test_3_warmstart_behavioral_cloning PASSED [ 70%]
backend/tests/test_phase_0c.py::test_4_critique_diff_engine_and_two_stage_preview PASSED [ 75%]
backend/tests/test_phase_0c.py::test_5_full_critique_alignment_loop PASSED [ 80%]
backend/tests/test_phase_0d.py::test_1_drift_detector_signals PASSED     [ 85%]
backend/tests/test_phase_0d.py::test_2_onnx_export_and_numerical_parity PASSED [ 90%]
backend/tests/test_phase_0d.py::test_3_runbundle_packaging_and_standalone_run_local PASSED [ 95%]
backend/tests/test_phase_0d.py::test_4_fastapi_rest_endpoints_and_websocket PASSED [100%]

======================= 20 passed, 15 warnings in 14.25s =======================
```

### Frontend Production Build: 100% Clean (0 Errors)
```text
> rl-ide-frontend@0.1.0 build
> tsc && vite build

vite v6.4.5 building for production...
✓ 1595 modules transformed.
dist/index.html                   0.56 kB │ gzip:  0.36 kB
dist/assets/index-Bcde3mL1.css   25.58 kB │ gzip:  5.31 kB
dist/assets/index-DQp1q17f.js   222.71 kB │ gzip: 64.36 kB
✓ built in 18.02s
```

---

## 11. Completed Work: Phase 2 (Productization, RunBundle Sharing & Hardening)

### 11.1 RunBundle Registry & Sharing Engine
- **File:** `backend/rl_ide/export/registry.py`
- Implements `RunBundleRegistry`, `BundleMetadata`, and `BundleDetails`:
  - **Auto-Discovery:** Automatically scans search directories (`exported_bundles/`, `runs/`) for `.runbundle` directories and `.zip` / `.runbundle.zip` archives.
  - **Metadata Extraction:** Parses `task_config.yaml` / `.json`, `reward_dsl.yaml`, `drift_baseline.json`, and `model.onnx` to extract task name, ID, adapter type, term counts, feature counts, quantile bucket counts, ONNX size bytes, and creation timestamps.
  - **Detailed Inspection:** `get_bundle(bundle_id) -> BundleDetails` returns full parsed task configuration, reward DSL, drift baseline parameters (entropy mean/std), and checks whether the bundle includes the standalone daemon.
  - **Safe ZIP & Directory Import:** `import_bundle(archive_or_dir_path, target_dir)` unpacks archives, verifies structural completeness (`model.onnx`, `task_config.yaml`, `reward_dsl.yaml`, `drift_baseline.json`, `run_local.py`), and returns the installed path and `TaskConfig`.
  - **Bundle Deletion:** `delete_bundle(bundle_id) -> bool` safely removes bundle directories and archives.

### 11.2 Standalone Zero-SB3 HTTP Microservice Daemon (`--serve --port 9000`)
- **File:** `backend/rl_ide/export/exporter.py`
- Upgraded `STANDALONE_RUN_LOCAL_TEMPLATE` inside every `.runbundle` package:
  - **Standard Library Only:** Built with pure Python `http.server.HTTPServer` and `BaseHTTPRequestHandler`, requiring strictly zero dependencies beyond `numpy` and `onnxruntime`. No FastAPI, PyTorch, SB3, or LLMs needed at inference.
  - **Daemon Mode:** `python run_local.py --serve --port 9000 --host 0.0.0.0` launches the lightweight policy microservice.
  - **Endpoints:**
    - `GET /health` & `GET /`: returns status, loaded model name, feature count, and domain action labels.
    - `POST /predict`: accepts single observation `{"observation": [...]}` or batches `{"observations": [[...], ...]}`. Computes model forward pass, predicted action indices and friendly labels, action probabilities, and live unsupervised drift detection (Input PSI, Policy Entropy Z-score, Drift status `HEALTHY` or `DRIFT_DETECTED`).
    - `POST /shutdown`: enables clean programmatic shutdown in automated deployment environments.
  - **Backward Compatibility:** Retains offline CLI modes `python run_local.py --sample` and `--input <file>` with identical summary outputs.

### 11.3 Multi-Provider BYOK Hardening & Validation
- **Files:** `backend/rl_ide/compiler/llm_client.py` & `backend/rl_ide/api/server.py`
- Implemented `validate_connection(config: LLMConfig) -> tuple[bool, str]`:
  - Validates connectivity and API keys across OpenAI (tests `/v1/models`), Anthropic (validates format and credentials), local Ollama / vLLM (checks `/v1/models` and `/api/tags` with 2.0s timeout), and offline Mock client.
- Added API endpoints in `server.py`:
  - `POST /api/settings/validate`: validates BYOK configuration before compilation.
  - Updated `POST /api/tasks/compile` and `POST /api/critique/preview`: dynamically routes to user's configured BYOK client (`LLMConfig`) when provided.
  - Added `GET /api/bundles`, `GET /api/bundles/{bundle_id}`, `POST /api/bundles/import`, `DELETE /api/bundles/{bundle_id}`.

### 11.4 Frontend Productization, Registry & Serving UI
- **Files:**
  - `frontend/src/components/RegistryModal.tsx`: Policy Registry Hub displaying all cataloged RunBundles, adapter tags, ONNX sizes, creation dates, inline DSL terms and drift baseline inspection, 1-click loading into workbench, and clipboard commands for the standalone daemon.
  - `frontend/src/components/SettingsModal.tsx`: Added `[Test Connection]` button with live visual validation against `/api/settings/validate`, showing clear green confirmation or actionable error alerts.
  - `frontend/src/components/ExportModal.tsx`: Added dedicated Production HTTP Daemon section with copyable `python run_local.py --serve --port 9000` command.
  - `frontend/src/components/Header.tsx`: Added `[Registry]` CTA button with real-time bundle count pill.
  - `frontend/src/App.tsx`: Wired bundle loading into workbench and connected BYOK configuration to task compilation and critique preview calls.

---

## 12. Verification Gate Results (Phase 2)

### Full Backend Regression Suite: 25 / 25 Tests Passing (100% Green)
```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.7.0 -- /home/devuser/projects/RLHarness/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/devuser/projects/RLHarness/backend
configfile: pyproject.toml
plugins: asyncio-1.4.1, anyio-4.15.1
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function

backend/tests/test_adapter_conformance.py::test_full_cross_domain_conformance_pipeline[CSVStreamAdapter] PASSED [  4%]
backend/tests/test_adapter_conformance.py::test_full_cross_domain_conformance_pipeline[TextClassifyAdapter] PASSED [  8%]
backend/tests/test_phase_0a.py::test_synthetic_stream_generation PASSED  [ 12%]
backend/tests/test_phase_0a.py::test_1_valid_yaml_dsl_evaluation PASSED  [ 16%]
backend/tests/test_phase_0a.py::test_2_undefined_variable_raises_validation_error PASSED [ 20%]
backend/tests/test_phase_0a.py::test_3_malicious_code_injection_raises_security_error PASSED [ 24%]
backend/tests/test_phase_0a.py::test_4_adapter_episode_stepping_and_fast_rescoring PASSED [ 28%]
backend/tests/test_phase_0b.py::test_1_compile_english_prose_to_valid_config PASSED [ 32%]
backend/tests/test_phase_0b.py::test_2_automatic_validator_retry_on_invalid_field PASSED [ 36%]
backend/tests/test_phase_0b.py::test_3_saving_sequential_versions_and_rollback PASSED [ 40%]
backend/tests/test_phase_0b.py::test_byok_llm_factory_fallback PASSED    [ 44%]
backend/tests/test_phase_0c.py::test_1_adapter_gym_env_and_hotswap PASSED [ 48%]
backend/tests/test_phase_0c.py::test_2_episode_buffer_operations PASSED  [ 52%]
backend/tests/test_phase_0c.py::test_3_warmstart_behavioral_cloning PASSED [ 56%]
backend/tests/test_phase_0c.py::test_4_critique_diff_engine_and_two_stage_preview PASSED [ 60%]
backend/tests/test_phase_0c.py::test_5_full_critique_alignment_loop PASSED [ 64%]
backend/tests/test_phase_0d.py::test_1_drift_detector_signals PASSED     [ 68%]
backend/tests/test_phase_0d.py::test_2_onnx_export_and_numerical_parity PASSED [ 72%]
backend/tests/test_phase_0d.py::test_3_runbundle_packaging_and_standalone_run_local PASSED [ 76%]
backend/tests/test_phase_0d.py::test_4_fastapi_rest_endpoints_and_websocket PASSED [ 80%]
backend/tests/test_phase_2.py::test_1_runbundle_registry_catalog_and_inspection PASSED [ 84%]
backend/tests/test_phase_2.py::test_2_runbundle_zip_import_and_roundtrip PASSED [ 88%]
backend/tests/test_phase_2.py::test_3_standalone_run_local_daemon_http_serving PASSED [ 92%]
backend/tests/test_phase_2.py::test_4_byok_settings_validation_and_routing PASSED [ 96%]
backend/tests/test_phase_2.py::test_5_fastapi_bundle_registry_and_settings_endpoints PASSED [100%]

================== 25 passed, 23 warnings in 85.63s (0:01:25) ==================
```

### Frontend Production Build: 100% Clean (0 Errors)
```text
> rl-ide-frontend@0.1.0 build
> tsc && vite build

vite v6.4.5 building for production...
✓ 1596 modules transformed.
dist/index.html                   0.56 kB │ gzip:  0.36 kB
dist/assets/index-CJsI4LuI.css   26.80 kB │ gzip:  5.54 kB
dist/assets/index-8pbjSn4v.js   241.77 kB │ gzip: 68.19 kB
✓ built in 15.02s
```

---

## 13. Completed Work: Phase 3A (Universal File & Live SQL Ingestor + AutoDataAdapter)

### 13.1 Live SQL Database Connector & Read-Only Safety Guardrails
- **File:** `backend/rl_ide/ingest/sql_connector.py`
- Implemented `SQLConnector`, `validate_read_only_sql`, `test_sql_connection`, `list_sql_tables`, and `load_sql_dataset`:
  - Supports SQLAlchemy connection URIs (`sqlite:///...`, `postgresql://...`, `mysql+pymysql://...`).
  - **Strict Read-Only Enforcement (`validate_read_only_sql`):** Blocks `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, `GRANT`, `REVOKE`, `CREATE`, `REPLACE`, `EXEC`, multi-statement attacks, and comment-based evasions.
  - Automatic `LIMIT 2000` pagination wrapper for arbitrary queries or table scans.

### 13.2 Universal Schema Profiler, Vectorizer & Document Rule Extraction
- **File:** `backend/rl_ide/ingest/profiler.py`
- **Schema Profiler (`SchemaProfiler`):** Inspects incoming dataFrames and categorizes columns into `numeric_continuous`, `numeric_discrete`, `categorical_low_card`, `text_freeform`, `boolean`, `timestamp`, or `target_candidate`. Sanitizes column names into safe AST variables (`obs.<sanitized_name>`).
- **AutoFeatureVectorizer (`AutoFeatureVectorizer`):** Produces deterministic `float32` observation vectors with standard-scaled continuous features, one-hot encoded categoricals, booleans, and 32-dim `HashingVectorizer` embeddings for freeform text.
- **Document Rule Extractor (`DocumentRuleExtractor`):** Analyzes policy text from `.pdf`, `.docx`, `.txt`, `.md` and extracts candidate `RewardTerm` AST rules (e.g. high-dollar thresholds, missing PO penalties, critical risk escalations).

### 13.3 Universal File Loader
- **File:** `backend/rl_ide/ingest/loader.py`
- Supports `.csv`, `.tsv`, `.xlsx` (via `openpyxl`), `.json`, `.jsonl`, `.sqlite`/`.db`, `.pdf` (via `pypdf`), `.docx` (via `python-docx`), `.txt`, and `.md`.
- Normalizes structured tables or document chunks into `IngestedDataset` with attached `DatasetProfile` and extracted candidate rules.
- Extracts document NLP features (`obs.page_number`, `obs.word_count`, `obs.has_currency_amount`, `obs.max_dollar_value`, `obs.has_date`, `obs.risk_keyword_count`, `obs.sentiment_score`).

### 13.4 AutoDataAdapter
- **File:** `backend/rl_ide/adapters/auto_data_adapter.py`
- Implements full `@runtime_checkable` `DomainAdapter` protocol.
- Dynamically derives observation vectors, discrete action labels, event schema, step execution, and polymorphic UI rendering.
- Emits `ui_mode: "tabular_inspector"` for SQL/CSV/Excel or `ui_mode: "document_reader"` for PDF/DOCX chunk records.
- Registered in `ADAPTER_REGISTRY` in `nl_to_config.py` with mock client support in `llm_client.py`.

---

## 14. Verification Gate Results (Phase 3A)

### Full Backend Regression Suite: 28 / 28 Tests Passing (100% Green)
```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.7.0 -- /home/devuser/projects/RLHarness/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/devuser/projects/RLHarness/backend
configfile: pyproject.toml
plugins: asyncio-1.4.1, anyio-4.15.1
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function

backend/tests/test_adapter_conformance.py::test_full_cross_domain_conformance_pipeline[CSVStreamAdapter] PASSED [  3%]
backend/tests/test_adapter_conformance.py::test_full_cross_domain_conformance_pipeline[TextClassifyAdapter] PASSED [  7%]
backend/tests/test_phase_0a.py::test_synthetic_stream_generation PASSED  [ 10%]
backend/tests/test_phase_0a.py::test_1_valid_yaml_dsl_evaluation PASSED  [ 14%]
backend/tests/test_phase_0a.py::test_2_undefined_variable_raises_validation_error PASSED [ 17%]
backend/tests/test_phase_0a.py::test_3_malicious_code_injection_raises_security_error PASSED [ 21%]
backend/tests/test_phase_0a.py::test_4_adapter_episode_stepping_and_fast_rescoring PASSED [ 25%]
backend/tests/test_phase_0b.py::test_1_compile_english_prose_to_valid_config PASSED [ 28%]
backend/tests/test_phase_0b.py::test_2_automatic_validator_retry_on_invalid_field PASSED [ 32%]
backend/tests/test_phase_0b.py::test_3_saving_sequential_versions_and_rollback PASSED [ 35%]
backend/tests/test_phase_0b.py::test_byok_llm_factory_fallback PASSED    [ 39%]
backend/tests/test_phase_0c.py::test_1_adapter_gym_env_and_hotswap PASSED [ 42%]
backend/tests/test_phase_0c.py::test_2_episode_buffer_operations PASSED  [ 46%]
backend/tests/test_phase_0c.py::test_3_warmstart_behavioral_cloning PASSED [ 50%]
backend/tests/test_phase_0c.py::test_4_critique_diff_engine_and_two_stage_preview PASSED [ 53%]
backend/tests/test_phase_0c.py::test_5_full_critique_alignment_loop PASSED [ 57%]
backend/tests/test_phase_0d.py::test_1_drift_detector_signals PASSED     [ 60%]
backend/tests/test_phase_0d.py::test_2_onnx_export_and_numerical_parity PASSED [ 64%]
backend/tests/test_phase_0d.py::test_3_runbundle_packaging_and_standalone_run_local PASSED [ 67%]
backend/tests/test_phase_0d.py::test_4_fastapi_rest_endpoints_and_websocket PASSED [ 71%]
backend/tests/test_phase_2.py::test_1_runbundle_registry_catalog_and_inspection PASSED [ 75%]
backend/tests/test_phase_2.py::test_2_runbundle_zip_import_and_roundtrip PASSED [ 78%]
backend/tests/test_phase_2.py::test_3_standalone_run_local_daemon_http_serving PASSED [ 82%]
backend/tests/test_phase_2.py::test_4_byok_settings_validation_and_routing PASSED [ 85%]
backend/tests/test_phase_2.py::test_5_fastapi_bundle_registry_and_settings_endpoints PASSED [ 89%]
backend/tests/test_phase_3a.py::test_1_ingest_csv_xlsx_and_train_export PASSED [ 92%]
backend/tests/test_phase_3a.py::test_2_sqlite_sqlalchemy_readonly_guardrails_and_query_burst PASSED [ 96%]
backend/tests/test_phase_3a.py::test_3_document_pdf_docx_rule_extraction_and_chunk_stepping PASSED [100%]

======================= 28 passed, 27 warnings in 17.98s =======================
```

### Phase 3A Verification Suite:
- `test_phase_3a.py::test_1_ingest_csv_xlsx_and_train_export`: **PASSED** (CSV & XLSX ingestion, AST compilation, warmstart, burst, ONNX parity < 1e-5, RunBundle export)
- `test_phase_3a.py::test_2_sqlite_sqlalchemy_readonly_guardrails_and_query_burst`: **PASSED** (SQLite DB connection, read-only SQL validation guardrail blocking DROP/DELETE/UPDATE/INSERT, query loading, burst)
- `test_phase_3a.py::test_3_document_pdf_docx_rule_extraction_and_chunk_stepping`: **PASSED** (PDF & DOCX rule extraction, NLP feature extraction, document chunk stepping, `ui_mode: "document_reader"`)

### Frontend Production Build: 100% Clean (0 Errors)
```text
> rl-ide-frontend@0.1.0 build
> tsc && vite build

vite v6.4.5 building for production...
✓ 1596 modules transformed.
dist/index.html                   0.56 kB │ gzip:  0.37 kB
dist/assets/index-CJsI4LuI.css   26.80 kB │ gzip:  5.54 kB
dist/assets/index-Bf12CDTc.js   242.58 kB │ gzip: 68.34 kB
✓ built in 4.11s
```




