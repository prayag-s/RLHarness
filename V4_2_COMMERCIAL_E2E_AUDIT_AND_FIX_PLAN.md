# RL-Native Agent IDE (v4.2) — 10-Domain Commercial Usability Audit & Auto-Remediation Plan

**Target Environment:** `/home/devuser/projects/RLHarness` (Backend: FastAPI on `:8000`, Frontend: React/Vite on `:5173`)[cite: 3, 4]
**Objective for Antigravity:**
Execute an end-to-end browser and API usability audit across **10 high-value commercial use cases**. For every scenario:
1. **Test** the complete conversational, visual, training, interactive, and export lifecycle in the live application (`http://localhost:5173`).
2. **Detect** any bugs, UI glitches, React console warnings, iframe `postMessage` failures, convergence stalls, or state persistence drops.
3. **Fix** all discovered issues immediately in `backend/` or `frontend/`, re-run the failing scenario until it passes 100%, and log every fix in `progress.md`[cite: 3, 4].

---

## 1. Global Verification Checklist (Must Pass on Every Use Case)

Before marking any of the 10 use cases as `PASSED`, verify all **9 Core Usability Invariants**:

1. **Zero Browser Console Errors/Warnings:** Monitor `page.on("console")` and `page.on("pageerror")`. There must be **0** duplicate React key warnings (e.g., `eval_5`), **0** unhandled promise rejections, and **0** CORS/WebSocket errors.
2. **Instant Mock-Animated Arena Preview (`arena_locked == False`):** Submitting the initial prompt immediately renders `arena.html` inside the center `<iframe sandbox="allow-scripts">` running its self-contained mock animation loop (`startMockAnimationLoop`) before RL training begins.
3. **Multi-Question Selection & Explicit Training Controls:** When clarifying questions appear (`q_objective`, `q_arena_ui`, `q_interaction`), verify the user can select option pills across **all** questions simultaneously (without premature submission) AND edit `Rounds`, `Episodes / Round`, and `Target Goal %` before clicking `[🚀 Confirm Selections & Build Agent]`.
4. **Live Arena Restyling (Pre-Training & Post-Training):** Asking in chat to change the Arena look (e.g., *"Add a telemetry header and neon highlights"*) hot-swaps `arena.html` inside the iframe without wiping the blueprint or trained model weights.
5. **Lock Arena & Fast Convergence:** Clicking `[🔒 Lock Arena & Start Training]` transitions the iframe from Mock Preview to Live Training Stream, runs Warm-Start + PPO bursts with valid-action masking, and achieves the target mastery threshold[cite: 3, 4].
6. **Post-Training Conversational Flexibility (Never Locked):** After training completes, verify the user can continue chatting in the **same conversation** to:
   - Modify agent characteristics/reward rules (`compute_reward()`).
   - Train additional rounds (`"Train for 2 more rounds with 150 episodes per round"`) resuming from the existing checkpoint (`Round N -> N+2`).
   - Restyle the Arena UI again.
7. **Bi-Directional Live Human Interaction (`[🎮 Play / Interact Live]`):** Switching to Interactive Mode and clicking controls inside the `arena.html` iframe (`frame_locator("iframe")`) emits `ARENA_INTERACT`, invokes `/interact`, and updates the iframe via `ARENA_UPDATE` in `<100ms`.
8. **Conversation Persistence & Resumable Switching:** Switching to another conversation in the Left Sidebar and switching back restores the full chat history, custom `arena.html`, episode replay strip, telemetry curves, and live interactive model weights.
9. **Owned Local Artifact Export (`.runbundle`):** Clicking `[⬇️ Export Owned Policy (.onnx)]` verifies PyTorch-to-ONNX parity (`max_diff < 1e-5`) and packages a runnable `run_local.py` bundle[cite: 3, 4].

---

## 2. Automated Test Fixture Preparation

Before running the 10 browser scenarios, generate deterministic test fixtures in `backend/tests/fixtures/commercial_suite/` so file uploads and SQL connections can be tested reproducibly:
- `clinical_sop.pdf` & `patient_claims.csv` (Use Case #1)[cite: 4]
- `wire_transfers.sqlite` with table `wire_transfers` (Use Case #2)[cite: 4]
- `k8s_telemetry.csv` (Use Case #3)[cite: 3, 4]
- `sku_flash_sale.xlsx` (Use Case #4)[cite: 4]
- `vendor_nda_clauses.docx` (Use Case #5)[cite: 4]
- `travel_policy.pdf` & `employee_expenses.csv` (Use Case #7)[cite: 4]
- `saas_customer_health.csv` (Use Case #9)[cite: 4]

---

## 3. The 10 Commercial Use Cases to Execute & Verify

### Use Case 1: Healthcare — HIPAA Prior-Authorization & Clinical Triage Router
- **Why Customers Pay:** Hospitals and insurance TPAs cannot send patient PHI outside their boundary to cloud LLM APIs[cite: 1, 2].
- **Data Input:** Upload `clinical_sop.pdf` + `patient_claims.csv`[cite: 4].
- **Initial Prompt:**
  `"Train an agent to triage incoming patient prior-authorization requests into 'Auto-Approve', 'Request Lab Docs', 'Urgent Physician Review', or 'Deny' using our clinical SOP PDF and claims CSV."`
- **Browser Test Steps:**
  1. Upload both files and submit the prompt.
  2. Verify Mock-Animated Arena shows a clinical patient chart card.
  3. Chat restyle: `"Highlight abnormal vital signs in red and add a HIPAA Boundary Verified badge."`
  4. Select clarifying options across all questions, set `Rounds=2, Episodes/Round=100, Target=85%`, and lock/train.
  5. Post-training chat tweak: `"Never deny a claim if cardiac symptoms are present—always route to Urgent Physician Review, and train 1 more round."` Verify Round increments from 2 to 3 and preserves weights.

---

### Use Case 2: FinTech — Real-Time Wire & ACH Fraud Interceptor
- **Why Customers Pay:** Payment rails require `<5ms` deterministic local ONNX scoring with zero per-call API fees[cite: 1, 2].
- **Data Input:** Connect via `[🗄️ Connect SQL DB]` to `sqlite:///.../wire_transfers.sqlite` with query `SELECT * FROM wire_transfers`[cite: 4].
- **Initial Prompt:**
  `"Train a real-time wire fraud interceptor with actions: 'Clear Instant', 'Step-Up 2FA Challenge', or 'Freeze & Alert Fraud Desk'."`
- **Browser Test Steps:**
  1. Verify SQL modal blocks destructive queries (`DROP TABLE`) and loads `wire_transfers`[cite: 4].
  2. Lock Arena and train initial rounds.
  3. Post-training chat tweak: `"It is freezing too many verified corporate payroll wires; reduce false positives on verified accounts and train 2 more rounds."`
  4. Switch to `[🎮 Play / Interact Live]` inside the Arena iframe and click/trigger a sample wire transfer test to verify instant fraud scoring.

---

### Use Case 3: DevOps / FinOps — Kubernetes Cluster Auto-Scaler & SLA Protector
- **Why Customers Pay:** Calling frontier LLMs on high-frequency server telemetry streams every 5 seconds is prohibitively expensive[cite: 1, 2].
- **Data Input:** Upload `k8s_telemetry.csv` (`cpu_pct`, `mem_pct`, `queue_depth`, `spot_price`, `latency_ms`)[cite: 3, 4].
- **Initial Prompt:**
  `"Train a Kubernetes auto-scaler agent with actions: 'Scale Out +2 Nodes', 'Scale Out +1 Node', 'Hold Steady', or 'Scale Down -1 Node'. Minimize hourly cloud cost while keeping SLA latency violations at zero."`
- **Browser Test Steps:**
  1. Verify Mock-Animated Arena displays server cluster nodes and load gauges.
  2. Chat restyle: `"Add a live $/hour cloud burn-rate meter to the arena."`
  3. Lock Arena, train to target, and test Interactive Mode by triggering a simulated traffic spike in the iframe.

---

### Use Case 4: E-Commerce & Retail — Flash-Sale Dynamic Pricing & Inventory Clearance
- **Why Customers Pay:** Directly increases gross margin across thousands of recurring SKU price updates[cite: 1, 2].
- **Data Input:** Upload `sku_flash_sale.xlsx` (or run as dynamic inventory simulation)[cite: 4].
- **Initial Prompt:**
  `"Train a dynamic pricing agent for a 14-day flash sale with 500 units of stock. Actions: 'Raise Price +10%', 'Keep Base Price', 'Discount 15%', or 'Clearance 30% Off' to maximize total revenue before day 14."`
- **Browser Test Steps:**
  1. Verify Mock-Animated Arena renders inventory depletion and revenue curves.
  2. Lock Arena and train.
  3. Post-training chat modification: `"Forbid using 'Clearance 30% Off' when more than 7 days remain in the sale, and continue training for 2 rounds."`

---

### Use Case 5: LegalTech — NDA & Vendor Contract Clause Risk Auditor
- **Why Customers Pay:** Law firms and procurement teams cannot expose confidential pre-signature contracts to external AI providers[cite: 1, 2].
- **Data Input:** Upload `vendor_nda_clauses.docx`[cite: 4].
- **Initial Prompt:**
  `"Audit each contract clause in this DOCX file and classify into: 'Standard Acceptable', 'Flag Unlimited Liability', 'Flag IP Ownership Risk', or 'Reject Non-Compete'."`
- **Browser Test Steps:**
  1. Verify `docx` ingestion extracts clause chunks and NLP metadata (`word_count`, `risk_keyword_count`, `max_dollar_value`)[cite: 4].
  2. Verify Arena renders a Document Clause Inspector card[cite: 4].
  3. Post-training diagnostic chat: `"Why did it flag clause #3, and can we make it stricter on liability caps over $100,000?"` Verify conversational answer and reward update.

---

### Use Case 6: CleanTech & Edge IoT — Solar + Commercial Battery Tariff Arbitrage
- **Why Customers Pay:** Runs offline on local edge gateways (`run_local.py`) with zero cloud dependency[cite: 1, 2, 3, 4].
- **Data Input:** Pure Simulation (Synthesized Continuous State Environment).
- **Initial Prompt:**
  `"Train a smart solar and battery controller observing solar_kw, building_load_kw, battery_soc_pct, and grid_price_kwh. Actions: 'Charge Battery', 'Discharge to Building', 'Sell Back to Grid', or 'Idle Hold'."`
- **Browser Test Steps:**
  1. Verify Mock-Animated Arena renders solar/battery/grid energy flow animation.
  2. Chat restyle before locking: `"Show animated power-flow arrows between Solar, Battery, Building, and Grid."`
  3. Lock Arena, train, and verify `[⬇️ Export Owned Policy (.onnx)]` produces a valid `.runbundle` that runs `python run_local.py --sample` cleanly[cite: 3, 4].

---

### Use Case 7: Corporate Finance — Automated Travel & Expense (T&E) Receipt Auditor
- **Why Customers Pay:** Automates repetitive back-office expense auditing against internal company policy PDFs[cite: 2, 4].
- **Data Input:** Upload `travel_policy.pdf` + `employee_expenses.csv`[cite: 4].
- **Initial Prompt:**
  `"Audit employee expense rows against our travel policy PDF. Actions: 'Auto-Reimburse', 'Request Itemized Receipt', or 'Flag Policy Violation'."`
- **Browser Test Steps:**
  1. Verify `DocumentRuleExtractor` extracts meal/hotel cap rules from `travel_policy.pdf`[cite: 4].
  2. Lock Arena, train, and test Interactive Mode by inspecting/submitting a high-dollar unreceipted dinner row inside the Arena iframe.

---

### Use Case 8: Robotics & Logistics — 2D Autonomous Mobile Robot (AMR) Warehouse Navigator
- **Why Customers Pay:** Fast local reward prototyping and ONNX policy export for onboard robot controllers[cite: 2, 4].
- **Data Input:** Pure Simulation (2D Kinematics & Lidar Obstacle Arena)[cite: 4].
- **Initial Prompt:**
  `"Train a 2D warehouse robot rover to navigate to target loading docks while avoiding obstacles, minimizing jerky turns, and conserving battery."`
- **Browser Test Steps:**
  1. Verify 2D Canvas Arena animates the robot, lidar rays, obstacles, and target beacon in Mock Preview[cite: 4].
  2. Lock Arena and train.
  3. Post-training chat tweak: `"Increase obstacle safety clearance distance and train 2 more rounds."`[cite: 4]
  4. Switch to `[🎮 Play / Interact Live]` and click on the 2D canvas inside the iframe to move the target beacon and verify live trajectory updates[cite: 4].

---

### Use Case 9: SaaS & Telecom — VIP Customer Churn & Retention Offer Optimizer
- **Why Customers Pay:** Protects net revenue retention by preventing unnecessary discount giveaways to healthy accounts.
- **Data Input:** Upload `saas_customer_health.csv` (`mrr`, `login_drop_pct`, `open_tickets`, `days_to_renewal`, `is_vip`)[cite: 4].
- **Initial Prompt:**
  `"Train a retention agent with actions: 'No Intervention', 'Automated Check-In Email', 'Offer 15% Discount', or 'Assign Dedicated CSM' to maximize net retained MRR."`
- **Browser Test Steps:**
  1. Lock Arena and train initial rounds.
  2. Post-training chat modification: `"Penalize offering the 15% discount when login_drop_pct is below 20%—save discounts only for at-risk VIP accounts, and train 2 more rounds."`

---

### Use Case 10: Gaming & Strategy — Interactive Tactical Grid / Connect-Four Bot
- **Why Customers Pay:** Game studios bundle lightweight client-side ONNX opponent bots with zero server hosting cost[cite: 1, 2].
- **Data Input:** Pure Simulation (Turn-Based Board Game against Randomized Multi-Strategy Opponent Pool)[cite: 4].
- **Initial Prompt:**
  `"Train an agent to play Tic-Tac-Toe / Tactical Grid against a randomized opponent pool. Prioritize center control, block opponent threats immediately, and create winning forks."`
- **Browser Test Steps:**
  1. Verify Mock-Animated Arena animates moves on the grid before locking.
  2. Chat restyle: `"Give the board a sleek neon arcade theme and display per-cell win probability percentages."`
  3. Lock Arena and train to `>= 90%` non-loss rate with `0%` illegal moves.
  4. Switch to `[🎮 Play / Interact Live]` and play 3 live moves against the agent inside the iframe to verify it blocks threats and responds in `<50ms`[cite: 4].
  5. **Cross-Conversation Switching Audit:** Open the Left Conversation Sidebar, click through Use Cases #1, #5, #8, and #10, and verify each conversation restores its exact chat thread, custom `arena.html`, episode replay strip, and interactive model state with zero console errors!

---

## 4. Remediation & Reporting Protocol

1. **Automated E2E Script (`backend/tests/test_v4_2_commercial_suite.py` + Browser Verification):**
   - Create an automated test suite that runs all 10 commercial scenarios through the v4.2 pipeline and verifies the live frontend UI in a headless browser (or Playwright/browser tool).
2. **Fix-on-Failure Loop:**
   - Whenever any of the 10 scenarios or 9 Global Invariants fails, **stop, diagnose the root cause in `backend/` or `frontend/`, apply the code fix, and re-run** until 100% of checks pass.
3. **Update `progress.md`:**
   - Document all 10 commercial use cases tested, every bug discovered and fixed during the audit, final convergence metrics per domain, and full `pytest` + `npm run build` output[cite: 3, 4].