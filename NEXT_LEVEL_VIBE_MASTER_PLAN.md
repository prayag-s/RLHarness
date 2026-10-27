# RL-Native Agent IDE (v3.0) — "Vibe Training", Universal Data/SQL/Doc Ingestion, Robotics & Dynamic Simulation Synthesis

**Prerequisite State:** Phases 0A–0E, Phase 1, and Phase 2 are 100% complete (`25/25` backend tests passing, frontend build clean).
**Objective of v3.0:** Transform the technical RL workbench into a **Zero-Jargon "Vibe Training" Studio** where any non-technical user can:
1. **Describe any game, robot control task, or business simulation from scratch** (e.g., *"Train an agent to play Tic-Tac-Toe"*, *"Train a 2D robot rover/arm to reach targets while avoiding obstacles and saving battery"*, *"Optimize warehouse inventory restocking"*) and watch the IDE automatically synthesize the environment, interactive visual canvas, and physics/opponent.
2. **Ingest any file format or live SQL database** (`CSV`, `TSV`, `XLSX`, `JSON/JSONL`, `SQLite`, live `PostgreSQL` / `MySQL` connection strings + SQL queries, `PDF`, `DOCX`, `TXT/MD`) and have the IDE automatically profile schemas, extract business rules from documents, and synthesize a runnable `DomainAdapter`[cite: 2, 3] with zero manual configuration.
3. **Train and align agents with zero RL jargon** ("Vibe Studio" mode by default, with a `{ } Developer IDE` toggle preserving full access to the v2.0 YAML DSL, AST inspector, and Stage 1/2 telemetry[cite: 2, 3]).

---

## 1. Architectural Upgrades Overview

~~~text
┌────────────────────────────────────────────────────────────────────────────┐
│                     VIBE STUDIO FRONTEND (Default View)                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Magic Prompt Bar + Universal File Dropzone + Live SQL DB Modal       │  │
│  │ 1-Click Demos: [🎮 Tic-Tac-Toe] [🤖 Robot Rover] [📄 PDF+SQL Audit]  │  │
│  └──────────────────────────────────┬───────────────────────────────────┘  │
│  ┌─────────────────────┬────────────▼─────────────┬─────────────────────┐  │
│  │ PLAIN-ENGLISH RULES │  POLYMORPHIC ARENA       │ VIBE COACH CHAT     │  │
│  │ (No YAML/AST shown) │  - 3x3 Game Grid (Games) │ - "Play vs Agent"   │  │
│  │ ✅ Avoid obstacles  │  - 2D Robot Canvas (Sim) │ - Click-to-Set Goal │  │
│  │ ✅ Smooth trajectory│  - Document Card (PDF)   │ - Thumbs Up / Down  │  │
│  │ [Skill: 96% Master] │  - SQL/CSV Record Card   │ [Get My Agent App]  │  │
│  └─────────────────────┴──────────────────────────┴─────────────────────┘  │
│                 (Toggle `{ } Dev Mode` -> Reveals v2.0 3-Panel IDE)        │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │ REST + WebSocket
┌─────────────────────────────────────▼──────────────────────────────────────┐
│                   UNIVERSAL SYNTHESIS LAYER (v3.0 Backend)                 │
│  ┌─────────────────────────┐  ┌────────────────────────┐  ┌─────────────┐  │
│  │ 1. Universal Ingestor   │  │ 2. Intent & Env Router │  │ 3. Vibe     │  │
│  │ - CSV / Excel / JSONL   │  │ - Data / SQL Task      │  │ Translator  │  │
│  │ - Live Postgres/MySQL   │  │ - Game / Board Task    │  │ (AST <->    │  │
│  │ - SQLite + SQL Queries  │  │ - Robotics Control Sim │  │  Plain Eng) │  │
│  │ - PDF / DOCX / Markdown │  │ - Hybrid Doc+Data Task │  │             │  │
│  └────────────┬────────────┘  └───────────┬────────────┘  └──────┬──────┘  │
│               ▼                           ▼                      ▼         │
│  ┌────────────────────────────────────┐ ┌────────────────────────────────┐ │
│  │ AutoDataAdapter (Dynamic Schema)   │ │ DynamicSimAdapter (Games/Robot)│ │
│  │ - Auto-vectorizes mixed SQL/CSV    │ │ - TicTacToe & Board Engines    │ │
│  │ - Extracts rules from PDF/DOCX SOPs│ │ - 2D Robot Kinematics & Obstacl│ │
│  │ - Emits `tabular` or `doc_reader`  │ │ - Interactive Human/Target API │ │
│  └─────────────────┬──────────────────┘ └───────────────┬────────────────┘ │
└────────────────────┼────────────────────────────────────┼──────────────────┘
                     ▼                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│        EXISTING v2.0 HARDENED CORE (DomainAdapter -> SB3 PPO -> ONNX)      │
└────────────────────────────────────────────────────────────────────────────┘
~~~

---

## 2. Core Technical Specifications for v3.0 Modules

### 2.1 Universal Data & Live SQL Ingestion Engine (`backend/rl_ide/ingest/`)

Must accept uploaded files OR live SQL database connection strings and normalize them into an `IngestedDataset` artifact:

1. **Tabular & Live SQL Database Connectors (`tabular_loader.py` & `sql_connector.py`):**
   - **File Formats:** `.csv`, `.tsv`, `.xlsx` (via `openpyxl`), `.json`, `.jsonl`, `.sqlite` / `.db`.
   - **Live SQL Database Connector (`SQLConnector` via `sqlalchemy`):**
     - Supports `postgresql://...`, `mysql+pymysql://...`, and `sqlite:///...` connection strings.
     - **Security Guardrail:** Enforces **Read-Only Query Validation** (rejects `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`; permits only `SELECT` queries or table name inspection with automatic `LIMIT 5000` pagination).
     - Provides `test_sql_connection(uri)`, `list_sql_tables(uri)`, and `ingest_sql_query(uri, query_or_table)`.
   - **Auto-Profiler (`schema_profiler.py`):**
     - Inspects every column and classifies it as: `numeric_continuous`, `numeric_discrete`, `categorical_low_card` ($\le 20$ unique values), `text_freeform` (avg length $> 30$ chars), `timestamp`, or `target_label_candidate` (`label`, `target`, `status`, `outcome`, `action`, `class`, `is_*`).
     - Automatically builds a fast, deterministic `AutoFeatureVectorizer` combining normalized numerical features, one-hot/ordinal encodings for categoricals, and `HashingVectorizer(n_features=32)`[cite: 3] for any freeform text columns.
     - Automatically populates `event_schema()`[cite: 2, 3] with clean, human-readable `obs.<column_name>` variables so the `RewardEvaluator`[cite: 2, 3] can reference any SQL or CSV column directly!

2. **Unstructured Document Connectors (`document_loader.py`):**
   - **Formats:** `.pdf` (via `pypdf`), `.docx` (via `python-docx`), `.txt`, `.md`.
   - **Dual-Mode Document Intelligence:**
     - **Mode A — Knowledge / Policy Extraction (`role="knowledge_rules"`):** When a user uploads a policy PDF/DOCX alongside SQL/CSV data (or says *"Audit these SQL transactions using our SOP.pdf"*), extract text chunks and pass them to `DocumentRuleExtractor`, which synthesizes plain-English rules and initial `RewardDSL` terms[cite: 2, 3] from the document.
     - **Mode B — Document Episode Stream (`role="episode_records"`):** When a user uploads PDFs/DOCX files to classify, route, or audit (e.g., resumes, contracts, invoices), split the document(s) into logical records (by page or section) where each chunk is an episode step with extracted NLP metadata (`obs.word_count`, `obs.page_number`, `obs.has_currency_amount`, `obs.max_dollar_value`, `obs.has_date`, `obs.risk_keyword_count`, `obs.sentiment_score`).

### 2.2 Dynamic Game, Robotics & Simulation Synthesizer (`backend/rl_ide/adapters/dynamic_sim_adapter.py`)

When a user asks to train an agent on a game, robot task, or interactive logic problem without uploading a dataset:

1. **Cartridge 1 — `TicTacToeSimEngine` (Turn-Based Board Games):**
   - 3x3 board (`-1` = Opponent `O`, `0` = Empty, `+1` = Agent `X`), actions `0..8`.
   - Exposed tactical signals in `event_schema()`[cite: 2, 3]: `obs.center_open`, `obs.corners_open_count`, `obs.is_legal_move`, `obs.took_winning_move`, `obs.blocked_opponent_win`, `obs.created_fork`, `obs.picked_center`, `obs.picked_corner`, `state.winner`, `state.game_over`, `action.cell_idx`, `action.is_center`, `action.is_corner`.
   - Built-in configurable sparring opponent (`random`, `tactical_blocker`, `minimax`).
   - Emits `ui_mode: "board_grid"`.

2. **Cartridge 2 — `RobotKinematicsSimEngine` (2D Robot Rover & Multi-Joint Arm Trainer):**
   - **Use Case:** *"Train a robot rover / robotic arm to reach target coordinates while avoiding obstacles, minimizing energy/jerk, and docking smoothly."*
   - **Physics & State Space:** Continuous 2D workspace ($[0, 100] \times [0, 100]$) with agent position $(x, y)$, velocity $(v_x, v_y)$, heading/joint angles $(\theta_1, \theta_2)$, target goal $(g_x, g_y)$, dynamic obstacle coordinates + lidar distance rays (`obs.min_obstacle_dist`, `obs.dist_to_target`, `obs.distance_reduced`, `obs.velocity_magnitude`, `obs.jerk_penalty`, `obs.battery_remaining`, `obs.collided`, `obs.reached_target`, `obs.in_docking_zone`).
   - **Discrete Control Actions (9 Thrust/Torque vectors):** `["thrust_up", "thrust_down", "thrust_left", "thrust_right", "thrust_up_right", "thrust_up_left", "thrust_down_right", "thrust_down_left", "brake_hold"]`.
   - **Interactive Target/Obstacle Manipulation:** Allows the user in the frontend canvas to **click anywhere to move the target goal or drop a new obstacle in real time** and watch the trained robot policy dynamically steer around it!
   - Emits `ui_mode: "robot_canvas"` (renders robot trajectory trail, lidar rays, obstacles, target beacon, battery gauge, and thruster vectors).

3. **Cartridge 3 — `SmartInventorySimEngine` (Supply Chain, Pricing & Resource Control):**
   - Simulates daily demand surges, supplier lead times, holding costs, and stockouts (`obs.stock_level`, `obs.demand_forecast`, `obs.is_stockout`, `obs.holding_cost`, `obs.profit_margin`, `action.order_tier`).
   - Emits `ui_mode: "sim_dashboard"`.

4. **Cartridge 4 — `CustomDeclarativeSimEngine` (Universal LLM-Synthesized State Machine):**
   - For any novel user simulation prompt, the LLM emits a declarative JSON `SimSpec` defining state variables, bounds, discrete actions, transition equations (evaluated safely via our existing AST sandbox[cite: 2, 3]), and termination conditions.

### 2.3 Polymorphic UI Renderer Contract (`render_frame` v3.0)

Every `DomainAdapter`[cite: 2, 3] emits a `ui_mode` field inside `render_frame()` so the Frontend Center Panel automatically morphs to match the task:
- `ui_mode: "board_grid"`: Interactive 3x3 Tic-Tac-Toe board with `X`/`O`, action probability heatmap on empty squares, and a **"🎮 Play Against Your Agent"** mode.
- `ui_mode: "robot_canvas"`: Interactive 2D Robotics Canvas showing the robot/arm, lidar proximity ring, obstacles, target goal, velocity vector, and **"🎯 Click Canvas to Move Target / Add Obstacle"** live test mode.
- `ui_mode: "document_reader"`: Clean PDF/DOCX page viewer with highlighted extracted signals, policy compliance rules, and agent decision stamp.
- `ui_mode: "tabular_inspector"`: Clean SQL/CSV/Excel record inspector showing row fields, values, anomaly/routing badge, and top contributing features.
- `ui_mode: "time_series"` & `"ticket_card"`: Preserves existing v2.0 visualizers[cite: 3].

### 2.4 The "Vibe Mode" Translation Layer (`backend/rl_ide/vibe/translator.py`)

Hides all RL complexity behind conversational English:
- **Plain-English Rule Cards (`PlainRuleCard`):** Translates every `RewardTerm` in `RewardDSL`[cite: 2, 3] into:
  - `title_plain`: e.g., *"Keep safe clearance from obstacles"* or *"Take the center square when open"*
  - `description_plain`: e.g., *"Penalizes the robot heavily if lidar obstacle distance drops below 8 units"*
  - `importance_label`: `"Must Do (+10)" | "High Priority (+5)" | "Bonus (+2)" | "Avoid (-5)" | "Strict Ban (-10)"`
- **One-Click Autopilot Loop (`POST /api/vibe/create` & `POST /api/vibe/coach`):**
  - `/api/vibe/create` automatically ingests files/SQL OR selects the simulation cartridge, compiles `TaskConfig`[cite: 2, 3], runs Warm-Start Behavioral Cloning + an initial 25-episode PPO burst[cite: 3], and returns a ready-to-use agent in `<3 seconds`.
  - `/api/vibe/coach` takes plain-English feedback (or 👍/👎 clicks), runs Stage 1 & Stage 2 verification[cite: 3] behind the scenes, applies Critic Warmup[cite: 3] + a 20-episode burst, and replies with a 1-sentence coaching summary: *"Got it! Added rule 'Brake smoothly near the target'. Target docking rate jumped from 62% → 95%."*

---

## 3. Sequential Implementation Phases for Antigravity

> **DIRECTIVE FOR ANTIGRAVITY:**
> Execute strictly ONE phase at a time (`Phase 3A` -> `Phase 3B` -> `Phase 3C` -> `Phase 3D`).
> Maintain 100% backward compatibility with all 25 existing tests (`test_phase_0a.py` through `test_phase_2.py`)[cite: 3].
> Run `pytest` at the end of every phase and confirm 100% pass before proceeding.

---

### 🚀 PHASE 3A: Universal File & Live SQL Ingestor (CSV, XLSX, JSONL, Postgres/MySQL/SQLite, PDF, DOCX) + `AutoDataAdapter`

**Objective:** Allow users to upload any structured/unstructured file OR connect a live SQL database (`PostgreSQL`, `MySQL`, `SQLite`) and automatically synthesize a compliant `DomainAdapter`[cite: 2, 3] + extracted document rules.

**Tasks for Antigravity:**
1. **Install Dependencies:**
   Add `pypdf`, `python-docx`, `openpyxl`, `sqlalchemy`, `pymysql`, `psycopg2-binary`, and `python-multipart` to `backend/pyproject.toml` via `uv pip install`.
2. **Build `backend/rl_ide/ingest/sql_connector.py`, `profiler.py`, & `loader.py`:**
   - **Live SQL Connector:** Support SQLAlchemy connection URIs (`sqlite:///...`, `postgresql://...`, `mysql+pymysql://...`). Implement `validate_read_only_sql(query)` (strictly block `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `GRANT`), `test_connection(uri)`, `list_tables(uri)`, and `load_sql_dataset(uri, query_or_table, limit=2000)`.
   - **File Loaders:** Support `.csv`, `.tsv`, `.xlsx`, `.json`, `.jsonl`, `.sqlite`/`.db`, `.pdf`, `.docx`, `.txt`, and `.md`.
   - **Schema Profiler (`SchemaProfiler`):** Auto-detect column types (`numeric`, `categorical`, `text`, `boolean`, `target_candidate`), compute normalization stats, and build `AutoFeatureVectorizer` outputting fixed-size `float32` observation vectors.
   - **Document Rule Extractor (`DocumentRuleExtractor`):** Parse uploaded `.pdf`, `.docx`, or `.md` policy documents and extract candidate business rules into plain-English rule cards and `RewardTerm` AST templates.
3. **Build `backend/rl_ide/adapters/auto_data_adapter.py` (`AutoDataAdapter`):**
   - Implements the full `DomainAdapter` protocol[cite: 2, 3] dynamically from any `IngestedDataset` (SQL query result, CSV/Excel table, or PDF/DOCX chunk stream).
   - Populates `event_schema()`[cite: 2, 3] dynamically with sanitized column names (`obs.<col_name>`) and document features (`obs.word_count`, `obs.page_num`, `obs.max_dollar_value`, `obs.risk_keyword_count`).
   - Emits `ui_mode: "tabular_inspector"` (for SQL/CSV/Excel) or `ui_mode: "document_reader"` (for PDF/DOCX chunks) in `render_frame()`.
4. **Write Verification Suite (`backend/tests/test_phase_3a.py`):**
   - Test 1: Ingest custom `.csv` and `.xlsx` files into `AutoDataAdapter`, compile `RewardDSL`[cite: 2, 3] referencing custom columns, and run warm-start + PPO burst + ONNX export[cite: 3].
   - Test 2: Connect to a SQLite/SQLAlchemy database URI, verify `validate_read_only_sql` blocks `DROP TABLE`, ingest via a `SELECT ... WHERE` query into `AutoDataAdapter`, and verify training burst.
   - Test 3: Programmatically generate a sample `.pdf` and `.docx` containing SOP rules and invoice records; verify `DocumentRuleExtractor` extracts reward rules and `AutoDataAdapter` steps through document chunks.

**Completion Gate 3A:** Run `pytest backend/tests/` — all 25 existing tests[cite: 3] + new `test_phase_3a.py` tests must pass 100%.

---

### 🚀 PHASE 3B: Dynamic Game, Robotics & Simulation Synthesizer (`DynamicSimAdapter`: Tic-Tac-Toe, 2D Robot Kinematics, Smart Inventory)

**Objective:** Allow users to train agents on games, robotics control, or custom simulations from a text prompt alone, and interact with them live.

**Tasks for Antigravity:**
1. **Build `backend/rl_ide/adapters/dynamic_sim_adapter.py`:**
   - Implement `DynamicSimAdapter(DomainAdapter)` with 4 simulation engines:
     - **Engine 1: `TicTacToeSimEngine` (`mode="tictactoe"`):** 3x3 board, tactical features (`obs.center_open`, `obs.corners_open_count`, `obs.is_legal_move`, `obs.took_winning_move`, `obs.blocked_opponent_win`, `obs.picked_center`, `obs.picked_corner`, `state.winner`), configurable opponent (`random`, `tactical`, `minimax`), and `ui_mode: "board_grid"`.
     - **Engine 2: `RobotKinematicsSimEngine` (`mode="robot_rover"`):** 2D continuous arena ($100 \times 100$) with robot $(x, y, v_x, v_y)$, target $(g_x, g_y)$, circular obstacles, lidar distance rays, battery budget, and 9 discrete thruster/brake actions. Exposes `obs.dist_to_target`, `obs.distance_reduced`, `obs.min_obstacle_dist`, `obs.collided`, `obs.reached_target`, `obs.jerk_magnitude`, `obs.battery_remaining`, `state.targets_reached` in `event_schema()`[cite: 2, 3]. Emits `ui_mode: "robot_canvas"`.
     - **Engine 3: `SmartInventorySimEngine` (`mode="inventory_sim"`):** Demand surges, restocking lead times, holding costs vs stockout penalties. Emits `ui_mode: "sim_dashboard"`.
     - **Engine 4: `CustomDeclarativeSimEngine` (`mode="custom_sim"`):** Evaluates custom state transition specs safely via the AST sandbox[cite: 2, 3].
2. **Add Interactive Live Testing API (`play_interactive_step`):**
   - For `tictactoe`: accepts current `board_state` and `human_cell_idx`, executes human move, runs trained policy inference to pick agent counter-move, returns updated board + action probability distribution across all 9 cells.
   - For `robot_rover`: accepts `robot_state`, `target_xy`, `obstacles_list`, and steps the trained robot policy forward by $K$ frames so the user can move the target goal or drop obstacles with their mouse and watch the robot navigate live!
3. **Update `NLToConfigCompiler` & `MockRulesLLMClient` (`backend/rl_ide/compiler/`):**
   - Auto-detect intents for `"tic-tac-toe"`, `"robot"` / `"rover"` / `"arm"` / `"obstacle"`, and `"inventory"` / `"pricing"`, routing to `DynamicSimAdapter` with tailored initial `RewardDSL` rules[cite: 2, 3].
4. **Write Verification Suite (`backend/tests/test_phase_3b.py`):**
   - Test 1 (Tic-Tac-Toe): Compile prompt *"Train an agent to play tic-tac-toe, take center when open, and block opponent wins"*, run warm-start + PPO burst, verify $>85\%$ non-loss rate and `0%` illegal moves, and test `play_interactive_step`.
   - Test 2 (Robotics): Compile prompt *"Train a robot rover to reach the target quickly while avoiding obstacles and minimizing jerk"*, run warm-start + PPO burst, verify collision rate drops significantly after critique *"Stay further away from obstacles"*, and verify ONNX export[cite: 3].
   - Test 3 (Inventory Sim): Verify compilation, stepping, and reward evaluation on `inventory_sim`.

**Completion Gate 3B:** Run `pytest backend/tests/` — all prior tests + `test_phase_3b.py` must pass 100%.

---

### 🚀 PHASE 3C: "Vibe Mode" Autopilot Orchestrator & FastAPI Endpoints

**Objective:** Create the conversational "Vibe API" that abstracts away all RL internals while keeping v2.0 endpoints intact.

**Tasks for Antigravity:**
1. **Build `backend/rl_ide/vibe/translator.py` & `orchestrator.py`:**
   - `dsl_to_plain_rules(dsl: RewardDSL) -> list[PlainRuleCard]`: converts AST expressions into plain-English titles, descriptions, and friendly importance pills (`Must Do`, `High Priority`, `Bonus`, `Avoid`, `Strict Ban`).
   - `compute_skill_summary(episodes: list[EpisodeLog], adapter_name: str) -> SkillSummary`: computes 0–100% **Agent Mastery Score**, status badge, win/success rate, and plain-English behavioral insights.
2. **Add Vibe, File Upload & SQL Endpoints in `backend/rl_ide/api/server.py`:**
   - `POST /api/ingest/upload`: multipart file upload for `.csv`, `.xlsx`, `.json`, `.jsonl`, `.sqlite`, `.pdf`, `.docx`, `.txt`, `.md`. Returns dataset ID, auto-profiled schema, sample records, and extracted document rules.
   - `POST /api/ingest/sql`: accepts `connection_uri` (`sqlite:///...`, `postgresql://...`, `mysql+pymysql://...`) and `query_or_table`. Validates read-only safety, ingests rows into an `IngestedDataset`, and returns dataset ID + schema preview.
   - `POST /api/vibe/create`: one-click autopilot creation! Accepts `prompt`, optional `dataset_id`, and `llm_config`[cite: 3]. Automatically selects or synthesizes the adapter, compiles initial rules (incorporating any uploaded PDF/DOCX rules), runs Warm-Start + initial 25-episode PPO burst[cite: 3], and returns `VibeSessionResponse`.
   - `POST /api/vibe/coach`: conversational coaching endpoint! Accepts `critique_text` (or episode 👍/👎), runs Stage 1/2 verification[cite: 3], applies the reward diff + Critic Warmup[cite: 3] + 20-episode burst, and returns `coach_reply`, updated `PlainRuleCard` list, and before/after Mastery Score.
   - `POST /api/sim/interact`: interactive endpoint for playing Tic-Tac-Toe moves against the agent or updating target/obstacle coordinates on the 2D Robot Canvas.
3. **Write Verification Suite (`backend/tests/test_phase_3c.py`):**
   - Test `/api/ingest/upload` (CSV, PDF, DOCX) and `/api/ingest/sql` (SQLite URI + SELECT query).
   - Test `/api/vibe/create` -> `/api/vibe/coach` -> `/api/sim/interact` across both Tic-Tac-Toe and Robot Rover tasks.

**Completion Gate 3C:** Run `pytest backend/tests/` — confirm 100% pass across all test suites.

---

### 🚀 PHASE 3D: The "Vibe Studio" Frontend (Tic-Tac-Toe Board, 2D Robot Canvas, SQL/Doc Ingest Modal & Dev Mode Switch)

**Objective:** Upgrade `frontend/` so the default view is the effortless **✨ Vibe Studio**, with a top-bar toggle to switch to the **🛠️ Developer IDE** (existing v2.0 workbench[cite: 3]) at any time.

**Tasks for Antigravity:**
1. **Header Bar Mode Switch (`frontend/src/components/Header.tsx`):**
   - Add a prominent segmented switch: **`[ ✨ Vibe Studio ]`** (Default) | **`[ 🛠️ Pro RL Workbench ]`** (reveals existing v2.0 panels[cite: 3]).
2. **Hero Onboarding & Universal Ingest Bar (`frontend/src/components/VibeCreatorBar.tsx`):**
   - Conversational prompt bar: *"Describe what you want your agent to learn, or drop in your data/docs..."*
   - **One-Click Vibe Demo Pills:**
     - 🎮 **Tic-Tac-Toe Master** (*"Train an unbeatable Tic-Tac-Toe agent and play against it live"*)
     - 🤖 **2D Robot Rover Trainer** (*"Train a robot rover to navigate around obstacles to a target beacon smoothly"*)
     - 📄 **PDF Policy + SQL/CSV Auditor** (*"Extract rules from an SOP PDF and audit financial records"*)
     - 📬 **VIP Support Ticket Router** (*"Triage customer emails by urgency and VIP status"*)
     - 📈 **Flash-Crash Stream Monitor** (*"Detect high-volatility market anomalies without false alarms"*)
   - **Universal Data & Live SQL Dropzone:**
     - Drag-and-drop `.csv`, `.xlsx`, `.sqlite`, `.pdf`, `.docx`, `.json`.
     - **`[ 🗄️ Connect SQL DB ]` Button:** Opens a clean modal to enter a `PostgreSQL`, `MySQL`, or `SQLite` connection URI + `SELECT` query, previewing detected columns and sample rows instantly.
3. **Vibe Left Panel — Plain-English Behavior Rules (`frontend/src/components/VibeRulesPanel.tsx`):**
   - Circular **Agent Mastery Meter (`96% Mastered`)** with win/success rate badge.
   - Displays active rules as zero-jargon **Plain-English Rule Cards** (`Must Do`, `High Priority`, `Bonus`, `Avoid`, `Strict Ban`) with 1-click priority boost/reduce buttons.
4. **Vibe Center Panel — The Polymorphic Arena (`frontend/src/components/PolymorphicArena.tsx`):**
   - Dynamically renders the visual environment matching `ui_mode`:
     - **`board_grid` (Tic-Tac-Toe):** Interactive **3x3 Tic-Tac-Toe Grid** with animated `X` and `O` pieces, per-square confidence percentage overlay, and a **`[ 🎮 Play Against Your Agent ]`** mode where the user clicks squares to challenge their own agent!
     - **`robot_canvas` (2D Robotics Trainer):** Interactive **500x350 SVG/Canvas Robot Arena** showing the robot rover, lidar proximity rays, circular obstacles, glowing target goal, trajectory path, battery bar, and a **`[ 🎯 Interactive Mode: Click to Move Target / Add Obstacle ]`** toggle so users can test their robot policy live!
     - **`document_reader` (PDF / DOCX):** Styled document page card showing extracted rules/entities and agent classification stamp.
     - **`tabular_inspector` (SQL / CSV / Excel):** Clean data record card displaying row columns, values, and agent decision badge.
     - **`time_series` & `ticket_card`:** Existing v2.0 visualizers[cite: 3].
   - Includes simple 👍 **"Good Behavior (Lock)"** and 👎 **"Fix This"** buttons on every episode card.
5. **Vibe Right Panel — Conversational AI Coach (`frontend/src/components/VibeCoachPanel.tsx`):**
   - Chat interface where the user types plain-English coaching (e.g., *"Take the center square first"*, *"Keep a wider safety margin around obstacles"*, or *"Flag any invoice over $5,000 missing a PO"*).
   - Displays conversational coach replies with before/after Mastery Score deltas and a 1-click `[Undo]` button.

**Completion Gate 3D:**
- Run `npm run build` (`tsc && vite build`) in `frontend/` and verify 0 errors.
- Run full backend `pytest` suite (`test_phase_0a` through `test_phase_3c`) and verify 100% green.
- Update `progress.md`[cite: 3] with Phase 3A–3D completion details and test logs.