# The RL Harness Field Guide

**A plain-English handbook for building, judging, and shipping agents — without a machine-learning background.**

This book is for the person who opens the studio, sees "PPO", "held-out scorecard", "Crossfire" and "$0.51 spent", and quietly wonders which button is safe to press. Every screen and control described here exists in the app. Every number is either shown in the product or comes from the code that computes it.

---

## How to read this book

There are three ways in, depending on how much time you have.

| You want to… | Read |
|---|---|
| Get one working agent today | **Part II — Your first agent in 15 minutes** (chapters 4–8) |
| Understand what you are looking at | **Part I** (chapters 1–3) then **Part IV** (chapters 13–16) |
| Stop spending too much | **Chapter 12** (models) and **Chapter 18** (cost playbook) |
| Fix an agent that plateaued | **Chapter 19** (quality playbook) |
| Find out why a build or a training run is blocked | **Chapter 24 — The Journey Coach** |
| Look up a word | **Chapter 22 — Glossary** |
| Keep a single page by your desk | **Chapter 23 — Cheat sheet** |

**A note on confidence.** This tool is honest about uncertainty, and so is this guide. Where the product says "inconclusive", this book will not pretend it means "good". Where a number is an estimate, it is labelled an estimate.

---

# Part I — Orientation

## 1. What this tool actually does

Most "AI agent" tools are a chat window with tools attached. This one is different: **it builds a small, self-contained decision-maker and then lets you own it.**

Think of it as an agent factory with four stations:

1. **Design** — you describe the job in English. A large language model (the "architect") turns that description into a runnable world and a scoring rule.
2. **Train** — a reinforcement-learning algorithm practises inside that world, thousands of times, and gets better.
3. **Judge** — independent checks ask whether the trained behaviour is genuinely better, or merely better at the scoring rule. This is the part most tools skip.
4. **Ship** — the finished policy is exported as a standalone ONNX file that runs without any LLM, with no per-call token bill.

The single most important consequence:

> **The LLM is the builder, not the agent.**
> You pay for the LLM while the agent is being designed and edited. Once trained, the agent is a tiny local model. Running it costs nothing in tokens.

That one sentence explains most of the cost behaviour you will see.

### What "agent" means here

An agent is a **policy**: a function that looks at the current situation (the *observation*) and chooses one of the allowed moves (the *action*). In this product a policy is usually a small neural network trained with PPO (Proximal Policy Optimization — a standard reinforcement-learning algorithm), sometimes combined with a hand-checked rule that guarantees a hard constraint.

It is **not** a chat persona and **not** a prompt. It does not call an LLM when it decides.

---

## 2. The five words you must know

Everything else in this app is built from these five ideas. Learn them once and the UI stops being mysterious.

| Word | Plain meaning | Where you meet it |
|---|---|---|
| **Objective** | The job, in English: what the agent should do, in what situation, to what standard | The thread, in the centre |
| **Environment** (also called **Arena**) | The simulated world the agent practises in, including its rules and what counts as a valid move | The arena on the right, `env_module.py` |
| **Policy** | The agent's decision-making brain | The trained model; **Export Owned Policy** |
| **Reward** | The score the training algorithm is trying to maximise; it is a *means*, not the goal | `compute_reward()` inside the environment |
| **Episode** | One complete attempt, from start to finish (one game, one delivery, one claim) | The episode counter and **Replay Episodes** |

Three supporting words:

| Word | Plain meaning |
|---|---|
| **Observation** | What the agent is allowed to see at a moment in time |
| **Action** | What the agent may do at that moment; the set of actions is the *action space* |
| **Round** | One training burst plus one evaluation. Many episodes happen inside one round. |

### The distinction that matters most: reward vs objective

The **objective** is what *you* want. The **reward** is what the *training algorithm* is told to maximise. They are usually not the same sentence.

If you say "win at tic-tac-toe", the reward might be `+1 for a win, 0 for a draw, -1 for a loss`. An agent optimising that reward might discover that *never losing* is easier than *winning*, and play for draws. That is not a bug in the training; it is a gap between objective and reward. The app gives you a way to close that gap with English critiques (chapter 10), and a way to catch the damage with the verifier gate (chapter 15).

---

## 3. The loop at a glance

This is the whole product in eight steps. Each step is a chapter later.

```
 1. Describe the objective  ──►  2. Architect drafts the Arena  ──►  3. You preview & approve
                                                                          │
 8. Export / own the policy  ◄──  7. Read the evidence  ◄──  6. Play & probe  ◄──  5. Train rounds
                                                                          │
                                                          4. Lock the Arena
```

- **Steps 1–4** happen once per agent, in minutes.
- **Step 5** is a loop: train → evaluate → maybe auto-tune → repeat until the target is met or the round budget runs out.
- **Step 6** is how you build intuition and feed the drift monitor.
- **Step 7** is where you decide whether to trust the result.
- **Step 8** is a file you can run anywhere.

### Where everything happens

The studio has one conversation and one clock.

| Surface | What it is | What you do there |
|---|---|---|
| **Thread** (centre) | Every event, in order: your instruction, the studio's plan, arena changes, coach findings, each training round, every tuning decision | Read what happened; give the next instruction |
| **Scoreboard** (left) | The numbers only: target vs current, held-out scorecard, Crossfire verdict, drift, rounds, spend | Glance; click a round to jump to it in the thread |
| **Arena** (right) | The world the agent plays in | Preview, play, inspect — it never narrates |

Only one party has the turn at a time. While the studio or the trainer is working, the input is
**queued** rather than injected, and **Stop** hands the turn back to you. Nothing is ever written
into a turn that someone else is still writing.

Everything you can read, you can copy. Text in the thread, the scoreboard and this guide selects
normally, and every thread entry carries a **copy** button that puts that message (title and body)
on the clipboard — handy for pasting a round summary into a ticket or a prompt.

At every arrow in that diagram the **Journey Coach** checks the thing you are about to rely on and tells you when it will not do what you think. It is advisory while you are still deciding, and it stops an expensive build or training run on a structural blocker. Chapter 24 is the full tour; the rest of this book is what it is checking against.

---

# Part II — Your first agent in 15 minutes

## 4. Before you press anything

### 4.1 Start the app

From the repository root:

```bash
./run.sh dev      # hot-reload development mode
# or
./run.sh prod     # build once, serve one origin
```

Defaults: backend on `http://127.0.0.1:8000`, frontend on `http://127.0.0.1:5173` in dev. In `prod` the backend serves the built UI on port 8000.

Each run is a clean restart: whatever instance is already running from a previous `dev` or `prod` launch (even on another port) is stopped first, so you can re-run the same command instead of hunting for a process. A port held by an unrelated program is reported rather than taken over.

You need **Python 3.11+**, **Node.js 20+**, and the Python dependencies installed (`backend/requirements.lock`). `run.sh` checks for these and tells you exactly what is missing.

### 4.2 The first screen: the model gate

On first launch you will see **"LLM Synthesis & Auto-Tuner Settings"**. This is not a paywall; it is a capability gate. The model is needed to *design* an agent, not to run one.

You have three choices:

| Choice | What it means | When to use it |
|---|---|---|
| **Explore first** | Open the studio without a key. Existing agents, uploads, playback and exports work. Building a **new** agent opens this dialog again. | You are evaluating the tool or opening a saved agent |
| **A cloud provider + API key** | The key is tested live, then saved in your browser and used by the backend for this session | Normal use |
| **Ollama (Local / vLLM)** | A model running on your own machine. No key required; nothing leaves the boundary. | Private data, offline work, zero marginal cost |

The app supports DeepSeek, Anthropic, OpenAI, Azure OpenAI, a generic OpenAI-compatible endpoint, and Ollama. Press **Test Connection** before saving; the app will not pretend a dead key works.

**If your deployment is hosted**, a **Tenant token** field appears at the bottom. Paste the `rlt_…` token your administrator issued. Leave it blank for the single-user local studio.

### 4.3 Where your work lives

Each agent is a **conversation**, stored under `workspaces/<conversation-id>/`. You will find:

| File | What it is |
|---|---|
| `arena.html` | The visual arena you approved |
| `env_module.py` | The runnable environment, including `compute_reward()` |
| `session_state.json` | Rounds, metrics, settings, gate history |
| `scorecard.json` | The final evidence package |
| `drift_baseline.json` | What "normal" looked like during training |
| `eval_corpus.json` | The golden set of approved behaviours (appears once seeded) |
| `best_val_model.zip` | The best validation checkpoint |

The **Agents** button in the header lists them. **+ New Agent Synthesis** starts a fresh one. Deleting a conversation deletes its workspace, its run history **and any exported bundles that came from it**, so copy out anything you need to keep first.

---

## 5. Write an objective that works

The architect is capable, but it is not a mind reader. A vague prompt produces a generic arena and a reward that optimises the wrong thing. A good objective is one paragraph, written like a brief to a competent colleague.

### 5.1 The six-part formula

Answer these six questions, in this order:

1. **Who / what** is deciding?
2. **In what situation** does it decide?
3. **What may it choose between?**
4. **What does success look like**, in measurable terms?
5. **What must it never do?** (hard constraints)
6. **What must the human see** while it works? (arena, controls, explanation)

You do not need to label the parts. You do need to cover them. The Coach checks this list before the blueprint locks (chapter 24); a missing part appears as a finding instead of a silent default.

### 5.2 Good and bad objectives

| Bad | Why it fails | Better |
|---|---|---|
| "Make a trading bot" | No market, no actions, no risk limit, no success measure | "An agent watches a simulated order book and chooses buy, sell, or hold each minute. Success is positive return over 30 simulated days with maximum drawdown below 5%. It must never hold more than 20% of the portfolio in one asset. Show a candlestick arena with a live P&L readout." |
| "Detect fraud" | No data, no decision, no cost of mistakes | "Given one invoice row, decide approve, review, or block. Success is catching at least 90% of fraudulent invoices while keeping false blocks under 2%. Attach `invoices.csv`. Show the row, the decision, and the top three reasons." |
| "Play tic-tac-toe well" | "Well" is ambiguous; draws may satisfy it | "An agent plays tic-tac-toe as X or O and must never lose to a human, including a perfect player. It should win when the human errs. Let the human choose who moves first. Show a chalkboard arena." |
| "Optimise the warehouse" | No layout, no moves, no metric | "An agent routes pickers in a 20×20 warehouse grid from a start cell to a target shelf. Success is average steps within 10% of the shortest path across 500 layouts. Never enter blocked cells. Show the grid and the current route." |

### 5.3 Rules of thumb

- **One agent, one job.** If your sentence contains "and also", you probably want two agents.
- **Name the cost of mistakes.** "Never lose" and "keep false blocks under 2%" change the reward far more than "be accurate".
- **Ask for the arena you need to judge it.** If you cannot tell from the screen whether it did well, the arena is under-specified.
- **State whether a rule can guarantee it.** Some jobs have a provable answer (tic-tac-toe cannot be lost from a draw-safe strategy); say so if you want a guard (chapter 17).
- **Do not paste raw data into the chat.** Attach it with **Universal Data Intake** (chapter 11).

### 5.4 If you already have data

Mention it in the objective and attach it:

> "Given one row of `claims.csv`, decide pay, investigate, or deny. Success is ≥ 85% agreement with the recorded outcome and zero denies above $10,000 without investigation."

The architect will use the dataset's columns to define observations and actions. Chapter 11 covers when this is the right move — and when it is not.

---

## 6. Preview, refine, and lock the arena

After you send the objective, the thread shows the studio's progress entries (collapsed into one row, expandable) while the arena is generated. The arena on the right then shows a **✨ Mock Preview** with a banner asking you to approve it.

### 6.1 The three centre tabs

| Tab | What it shows | Use it for |
|---|---|---|
| **✨ Mock Preview** | A live animation of the arena using the environment's real state | Judging the look and the information shown |
| **📺 Live Stream** | The training stream as it happens | Watching rounds and episodes progress |
| **🎮 Play / Interact** | You play against / operate the current policy | Building intuition; feeding drift samples |

### 6.2 Change it before you lock it

The fastest way to refine the arena is plain English in **Studio Chat**:

- "Make the board larger and use a chalkboard look."
- "Show the reward breakdown under the score."
- "Add a 'who moves first' control for the human."
- "Remove the cartoon graphics; keep it minimal."

Visual changes hot-reload the arena **without touching the trained weights** — the app will tell you when that happens. You can do this at any time, before or after training.

### 6.3 Lock Arena & Start Training

When the preview shows what you need, click **🔒 Lock Arena & Start Training** inside the preview banner.

Locking does three things:

1. Freezes the environment contract used for this training run.
2. Seeds the **golden eval corpus** from the held-out validation split (up to 30 cases) on the first round.
3. Opens the **Crossfire protocol** — a pre-registered rubric, written *before* the result is known.

That last point matters: a claim scored against a rubric invented afterwards is not evidence. The app deliberately records the rubric first.

> **If you lock the wrong thing**, do not panic. You can ask for environment changes in chat; the verifier gate will judge them. But it is cheaper and cleaner to get the arena right first.

---

## 7. Train, watch, and know when to stop

Click the **Params** pill (it reads like `Params: 5 Rounds • 200 eps • 90%`) to open the training controls.

### 7.1 What the three numbers mean

| Control | Range | Plain meaning |
|---|---|---|
| **Rounds** | 1–100 | How many train–evaluate cycles to run |
| **Episodes / Round** | ≥ 1 | How much practice happens in each cycle |
| **Target Goal (%)** | 50–100 | The success rate at which training stops early |

Before the run starts, the Coach checks these three numbers against the objective and the data (chapter 24). A zero budget is a blocker; a 100% target or a ten-episode evaluation is a warning with a better preset attached.

A round is not "one episode". Inside a round the app runs a PPO training burst sized from the episode budget, then scores the policy on **50 deterministic evaluation episodes**, then decides what to do next.

### 7.2 What happens inside a round

1. **Warm start (round 1 only).** The policy is pre-trained on heuristic demonstrations so it does not start from pure noise.
2. **PPO burst.** The policy practises. Episode budget maps to timesteps, with a hard ceiling; the UI reports the real episode count, not a rounded claim.
3. **Evaluation.** 50 deterministic episodes on the selection split.
4. **Target check.** If the metric reaches your target, training stops with **target achieved**.
5. **Plateau check.** If the policy is stuck — below 25% after round 2, or improving by less than 3 points for two consecutive rounds — the auto-tuner may rewrite the reward function. This is one of the few moments training calls the LLM.
6. **Verifier gate.** Any reward, environment, or inference change is judged against the held-out data and the golden corpus. If it damages approved behaviour, it is rolled back.
7. **Next round.** If neither target nor budget ends the run, continue.

Training itself runs locally. **Episodes do not cost tokens.** You pay the provider when the architect writes code and when the auto-tuner intervenes on a plateau.

### 7.3 Reading the live screen

- The **round bars** (R1, R2, R3…) show each round's evaluated success rate. A rising staircase is healthy; a flat line is a plateau; a falling line after a change is a warning.
- The **plan card** shows `rounds × episodes = total episodes`, the target, and how far through the current round you are.
- **＋ Train +1 Episode** adds a single extra episode beyond the plan. It is useful for a final nudge, not for serious improvement.
- The policy badge (for example `verified_minimax_guard`) tells you whether a verified rule is participating in decisions.
- **SAFE MODE: ON** means hard constraints are being enforced.

### 7.4 When to stop

Stop when **all** of these are true:

- The held-out scorecard (chapter 13) clears your target, not just the live round metric.
- The scorecard says **beats every baseline**.
- The golden corpus line shows `n/n verified` with no regressions.
- You have played at least 20 interactive steps and the agent behaved sanely.

Do **not** stop merely because the target badge fired. The target is computed on the selection split; the scorecard is the honest number.

---

## 8. Play, inspect, export

### 8.1 🎮 Play / Interact

Switch to the Play tab. For a game arena you will see **New Game** and **Who moves first? Human / Agent**. For a decision arena you will see the current record and the legal actions.

Use this tab to:

- **Probe edge cases** the training set may have missed.
- **Warm up drift monitoring.** The header shows `warming up` with a sample count; drift needs live decisions before it can compare them to the training baseline. Twenty or so varied plays is enough to start.
- **Build your own confidence.** If you cannot explain why the agent made a choice, the observation or the arena is probably under-specified.

### 8.2 🔍 Inspect Code

Opens the generated environment. The two things worth reading:

- **`reset()` / `step()`** — the rules of the world.
- **`compute_reward()`** — the actual incentive. Every term here is what the agent is being paid to do.

You do not need to be a Python expert to spot the classic problem: a reward term that pays for something you do not want. If you see `+0.25 for centre` in a game where centre is not always best, you have found a reward/objective gap.

The **Code Inspector** is read-only by design. Change behaviour through chat, so the verifier gate can judge the change.

### 8.3 Replay Episodes

The **Replay Episodes** strip lists recorded episodes (for example `R7 • #r7_eval_all_31`). Click one to step through what the policy saw and chose. This is the fastest way to answer "why did it do that?" without guessing.

### 8.4 Export Owned Policy (.onnx)

The green **Export Owned Policy (.onnx)** button in the header becomes available once a policy exists.

What export does:

1. Converts the trained actor to **ONNX**, a portable inference format.
2. Runs a **numerical parity check** against the original policy (tolerance `1e-5`). If the exported model disagrees with the trained one, export fails rather than shipping a silent lie.
3. Packages a **`.runbundle`**: model, action labels, baseline data, feature pipeline, and a small inference server.
4. Registers the bundle so **Policy Registry** can show it.

The bundle includes a CLI for single decisions and a lightweight HTTP daemon:

```bash
python run_local.py                          # run on synthetic sample observations
python run_local.py --input observations.csv # score your own rows
python run_local.py --serve --port 9000      # start the inference daemon
```

When the objective declares a hard constraint, the bundle also ships `safety_rules.json` and the
supported runtime **refuses to run blind**: supply `--record-json records.json` (one raw record per
input row) so the mask can be applied. A constrained bundle without raw records exits with an error
instead of emitting an unconstrained decision.

Use export when the evidence is good. An exported policy is a commitment; it is also the point at which the token bill stops.

### 8.5 Policy Registry

**Policy Registry** lists packaged bundles. From there you can copy the exact run command and inspect the bundle metadata. The count in the header is how many bundles exist, not how many conversations.

---

# Part III — The control panel

## 9. Rounds, episodes, and targets in practice

The three training numbers interact. More episodes per round gives a better gradient estimate; more rounds gives more chances to recover; a higher target gives a longer run.

### 9.1 Presets that work

| Goal | Rounds | Episodes / Round | Target | Why |
|---|---|---|---|---|
| **Smoke test** — "does this world even work?" | 1 | 50 | 70% | Cheapest way to find a broken environment |
| **First real attempt** | 3 | 200 | 85% | Good default for games and small control tasks |
| **Serious run** | 5–8 | 500 | 90% | Enough practice for non-trivial policies |
| **High-stakes run** | 10–15 | 1,000 | 95% | Diminishing returns below this are common; evidence matters more |
| **Deterministic task with a provable rule** | 3 | 200 | 100% | Only sensible when a guard can guarantee correctness |

### 9.2 What more of each actually buys you

| Increase | Buys you | Costs you |
|---|---|---|
| **More episodes per round** | Smoother learning, less noise per round | CPU time per round; more chance of overfitting a bad reward |
| **More rounds** | Chances to recover from a plateau or a rolled-back change | Wall-clock time; more auto-tune LLM calls if it plateaus repeatedly |
| **Higher target** | Stops earlier only when genuinely better | Longer runs; at 98–100% you are often chasing noise |

### 9.3 The 100% trap

Success rates on stochastic tasks have confidence intervals. A scorecard that reads `88% [76–94]` does not prove the policy cannot reach 95%; it proves you have not measured it precisely enough. Raising the target to 100% often just burns rounds. Raise **episodes** if you need a tighter estimate.

---

## 10. Talking to the agent (the thread is a control surface)

The thread is the only place work is narrated, and only one party may write in it at a time:

| Turn state | Who may act | What you see |
|---|---|---|
| **Your turn** | You | The input is live; **Send** starts a turn |
| **Studio has the turn** | Studio or Trainer | Live entries; **Stop** is available; anything you type is queued for the next turn |
| **A decision is waiting** | You | The last entry asks a question or reports a blocker; answer below |

A turn always ends with one terminal entry — a question, a blocker, a result, or an answer — and only
then does the input come back. **Esc** stops a running turn.

The studio always answers in English, even if you write in another language. Code, identifiers,
column names and file contents are never translated.

The instruction you send is not a general chatbot message. It routes your instruction to one or more execution layers: environment code, reward code, inference strategy, arena UI, or training parameters. The architect classifies your intent before acting.

### 10.1 What you can ask for

| Intent | Example | Usually retrains? |
|---|---|---|
| **Change the world** | "Make the grid 30×30 and add moving obstacles." | Yes — the environment changed |
| **Change the incentives** | "Penalise illegal moves much more heavily." | Yes — the reward changed |
| **Change the strategy** | "Prefer the centre on the first move." | Sometimes — inference change |
| **Change the look** | "Use a dark board and bigger pieces." | No — weights are preserved |
| **Train more** | "Train 5 more rounds with 500 episodes per round." | It is training |
| **Explain** | "Why did it move to the corner in episode 31?" | No |
| **Set parameters** | "Target 92% and 8 rounds." | Applies to the next run |

The quick chips do the common versions of these:

- **🔄 Train 5 More Rounds (500 eps/round)**
- **🧠 Make Agent More Aggressive**
- **🛡️ Penalize Mistakes More Heavily**
- **🎨 Redesign Arena Look**

### 10.2 How to phrase a change

Use this sentence shape:

> **Change X to Y, because Z. Do not change W. I will judge it by V.**

Example:

> "Make illegal moves cost −5 instead of −1, because the agent is learning to test walls. Do not change the win bonus. I will judge it by the held-out success rate and the corpus staying at 100%."

Why this works: it names the mechanism (X), the target value (Y), the reason (Z), the blast radius (W), and the acceptance criterion (V). The verifier gate can only protect behaviour you have told it matters.

### 10.3 One change at a time

If you ask for "faster, safer, and prettier" in one message, the architect may batch several code changes. When the result regresses, you will not know which change caused it. Batch **cosmetic** changes freely; separate **behavioural** changes.

### 10.4 What the gate does with your change

Any change to the reward, the environment, or the inference strategy is provisional until the next training round judges it:

- It is compared on the **same deterministic episodes** as before.
- It must not regress cases in the **golden corpus**.
- If it fails, it is rolled back, and the **Gate** tab tells you why in plain language.

A rollback is not a rejection of you. It is the system refusing to keep a change whose only evidence is the reward it just rewrote.

### 10.5 Things not to say

| Don't say | Say instead |
|---|---|
| "Make it smarter." | "Raise the held-out success rate above 90% without regressing the corpus." |
| "Fix everything." | One named behaviour, with an example. |
| "Use GPT-4 to decide each move." | That is a different architecture; this product trains a local policy. |
| "Set the reward to always be positive." | You have just removed the signal; say what should be rewarded. |

---

## 11. Datasets: when to attach one

Attach a dataset when the objective depends on **patterns in your own records** — fraud, triage, pricing, routing, quality inspection. Do not attach one when the world is fully described by rules — games, simulators, physical control.

### 11.1 What you can attach

The **Universal Data Intake** panel accepts:

`CSV` · `TSV` · `XLSX` · `JSON` · `JSONL` · `PDF` · `DOCX` · `TXT` · `MD` · `SQLite`

And **SQL DB** connects to a live database with a connection URI plus a read-only `SELECT`:

```sql
SELECT id, amount, risk_score, department
FROM invoices
WHERE status = 'pending'
LIMIT 1000
```

The query is the boundary. Load a bounded, purposeful slice — not the whole warehouse.

### 11.2 What the app does with it

1. **Profiles** the columns (types, ranges, missingness, cardinality).
2. **Splits** rows into train / validation / test where the task supports it.
3. Uses the split to define observations and, where a labelled outcome exists, the success metric.
4. Seeds the golden corpus from held-out rows.

The scorecard will show split counts (`train … · val … · test …`) and whether the score is **held out** or **in-sample**.

### 11.3 Data hygiene that changes results

- **One row = one decision.** If a row contains a whole history, flatten it or the agent will learn leakage.
- **Label the outcome column clearly.** "status" is ambiguous; "was_fraud" is not.
- **Keep the label out of the observations.** If the target is visible at decision time, the score will be perfect and useless.
- **200+ rows minimum** for a meaningful split. Below that, expect wide confidence intervals.
- **Encode time.** If rows span time, split by time, not randomly, or the model will "predict" the past from the future.
- **Remove identifiers and personal data** unless the task genuinely needs them.

The Coach applies these checks when you attach a dataset (chapter 24) and repeats the objective-aware ones before training. A missing label or a single-class target is a blocker; imbalance, missingness and leakage are warnings.

### 11.4 The leakage test

Ask: *could a careful human have known this at decision time?* If not, the column must not be an observation. This single check prevents most disappointing deployments.

---

## 12. Choosing the model (and what it costs)

### 12.1 The model builds; the policy runs

| Phase | Who does the work | Do you pay tokens? |
|---|---|---|
| Reading your objective, writing the arena and reward | LLM | **Yes** |
| Training rounds | Local PPO | No |
| Plateau auto-tuning (occasional) | LLM | **Yes** |
| Chat edits | LLM | **Yes** |
| Playing, replaying, exporting | Local policy | No |
| Running the exported ONNX bundle | Local policy | No |

### 12.2 Provider comparison

| Provider | Good at | Watch out for | Relative cost |
|---|---|---|---|
| **DeepSeek** | Code synthesis, long contexts, strong price/performance | Needs a key; data leaves your boundary | **Lowest cloud** |
| **Anthropic (Claude)** | Careful code and explanation | Highest per-token cost here | High |
| **OpenAI (GPT-4o)** | Broad code competence | Cost adds up on long prompts | Medium–high |
| **Azure OpenAI** | Enterprise controls, private networking | Deployment name required | Same as OpenAI |
| **OpenAI-compatible / self-hosted** | Your own gateway or vLLM | You own availability and quality | Infrastructure |
| **Ollama (local)** | Privacy, offline, zero marginal cost | Weaker synthesis, slower, larger machine | **Free at the margin** |

### 12.3 The actual price table used by the app

Prices are USD per **1 million tokens**, applied to every metered call:

| Model | Input | Output | Cached input |
|---|---:|---:|---:|
| `deepseek-flash` (peak) | $0.30 | $1.20 | $0.006 |
| `deepseek-flash` (off-peak) | $0.15 | $0.60 | $0.003 |
| `deepseek-chat` | $0.27 | $1.10 | $0.027 |
| `deepseek-reasoner` | $0.55 | $2.19 | $0.055 |
| `gpt-4o` | $2.50 | $10.00 | $1.25 |
| `gpt-4o-mini` | $0.15 | $0.60 | $0.075 |
| `claude-3-5-sonnet` | $3.00 | $15.00 | $0.30 |
| Ollama / local | $0.00 | $0.00 | $0.00 |

These are coarse list prices for planning, not an invoice. A real contract can override them:

```bash
export RL_IDE_LLM_PRICES='{"deepseek/deepseek-flash": [0.30, 1.20, 0.006]}'
```

**Reasoning tokens are billed as output tokens.** DeepSeek enables thinking by default at high
effort, so a one-word answer can still be billed for a dozen reasoning tokens. Every call therefore
declares a *role* with its own budget (`backend/rl_ide/llm/roles.py`):

| Role | Thinking | Output ceiling |
|---|---|---|
| `chat_planner`, `coach_judge`, `diagnostic_chat`, `intake_blueprint`, `arena_ui` | off | 700–1,800 |
| `auto_tuner`, `reward_rewrite` | off | 900–1,200 |
| `verifier_repair` | on, low effort | 6,000 |
| `env_synthesizer` | on, low effort | 8,000 |

A thinking role's ceiling covers reasoning *and* the answer. If a response is cut off at the ceiling
the call fails with an explicit error instead of returning a truncated module: the tokens are
already paid for, and a half-written environment only costs more to verify.

The usage report shows `totals`, an `efficiency` block (`reasoning_tokens`, `cached_input_tokens`,
`cache_hit_rate`) and a `by_role` breakdown, so a surprising bill can be traced to the call site
that caused it. Off-peak DeepSeek pricing (outside 01:00–04:00 and 06:00–10:00 UTC on weekdays) is
applied automatically.

### 12.4 Which model should you pick?

| Situation | Choose |
|---|---|
| **Default** | `deepseek-flash` — best cost/quality for code synthesis |
| Complex environment logic, repeated failures, subtle rules | `deepseek-reasoner` — about 2× the price, slower, better at hard reasoning |
| You already have an OpenAI or Anthropic contract | The model you already pay for; keep an eye on long prompts |
| Regulated or private data | Ollama / self-hosted. Nothing leaves the boundary |
| Cosmetic arena changes | The cheapest model you have. Do not use a reasoner for colours |

Three decisions, in order of effect: keep the default `deepseek-flash` (it is already the cheapest
capable model here), leave thinking off for everything that only classifies or styles, and let the
arena be rendered by code from a spec rather than written as a full HTML document. Raise effort or
switch models only when the environment itself is the blocker.

### 12.5 What the header is telling you

The pill shows the active provider and **Verified**. If verification lapses, new synthesis stops and the setup dialog returns. Existing agents still run.

The dollar figure next to it (`$0.51 spent`) is metered provider spend for the active conversation — or for the whole workspace before you select one — with the call count in the tooltip. It is not training compute and not your subscription.

---

# Part IV — Reading the evidence

## 13. The held-out scorecard

The **Held-Out Scorecard** in the scoreboard on the left is the most important number in the product. It is the closest thing to "how will this behave on data it has not seen?"

### 13.1 Field by field

| Field | What it means | What good looks like |
|---|---|---|
| **held out · test** badge | Scored on data excluded from training | Prefer this over **in-sample** every time |
| **Metric name** | What is being counted (success rate, agreement, etc.) | It should match your objective |
| **Value + `[low–high]`** | Point estimate and 95% confidence interval | A narrow interval; the low end should still clear your bar |
| **Baselines** | Simple comparators (uniform random, random policy, majority class) | You must beat them, not just score well |
| **Beats every baseline by …** | Uplift over the best comparator | Positive and meaningful, not +0.4 points |
| **ONNX behaviour: … action agreement** | Whether the exported model chooses the same action as the trained one | ≥ 99% |
| **train / val / test counts** | How the data was split | Non-zero test rows |
| **🛡️ Golden corpus: n/n verified** | Whether previously approved behaviours still hold | All verified, zero regressed |
| **Selected checkpoint** | Which round was kept when validation chose an earlier one | Best validation, not necessarily the last |
| **Crossfire** | The pre-registered verdict | `win` when you have enough evidence |
| **⚡ p99 latency** | End-to-end decision latency | Lower is better; compare with the frontier path |
| **🛡️ Constraints** | How many raw policy actions a declared hard constraint had to override | Zero violations, runtime-enforced |
| **Payback rows** | Decisions needed to repay authoring cost | Read with chapter 16 |

### 13.2 In-sample is a warning, not a score

For control tasks with no dataset, the app may report `held out · all` or `in-sample`. That is honest: there is no unseen split, so the number describes the same dynamics the policy trained on. Treat it as a sanity check and lean on the golden corpus and interactive play.

### 13.3 A worked judgement

> Policy: **88.0%** `[76.2–94.4]` · 50 episodes
> Baselines: uniform random **11.1%**, random policy **35.0%**
> Beats every baseline by **+53.0 pts**
> Golden corpus: **30/30 verified**
> Crossfire: **inconclusive** — 0 frozen hold-out judgements, protocol requires 100

Reading:

- The point estimate is strong and clearly beats both baselines.
- The interval is wide because 50 episodes is a small sample. The true value could be ~76%.
- The corpus is intact, so nothing approved was broken.
- Crossfire is **not** a failure; it says the evidence standard for a formal claim has not been met yet. More frozen judgements (or a dataset with a held-out split) close it.

**Decision:** ship a pilot, keep the corpus, and do not claim "unbeatable" in marketing. Raise evaluation size before making a strong claim.

The Coach produces this reading automatically when the scorecard changes: it names the interval problem, the baseline comparison and the next action, and links to the playbook section (chapter 24). It does not replace your judgement about whether to ship.

---

## 14. Drift

Drift is the gap between the world the policy trained in and the world it is meeting now.

The header shows one of:

| Status | Meaning | What to do |
|---|---|---|
| **warming up** (with a count) | Not enough live decisions to compare yet | Play 20+ varied steps |
| **healthy / green** | Live inputs and actions look like training | Nothing |
| **amber / drifted** | Inputs, entropy, or action mix shifted | Inspect the drifted features, then retrain or widen the training data |
| **Drift not monitored** | No baseline exists for this conversation | Finish a training run; the baseline is written from it |

Drift is **not** accuracy. A drifted agent may still be right today. Drift says the assumptions changed, which is your early warning.

---

## 15. The Trust drawer: Crossfire, Corpus, Gate

Click **Trust** in the header, or open the **Held-out scorecard** entry in the thread and click **Review evidence**. Both lead to the same drawer. Three tabs. The Coach's finding appears in the same thread: Crossfire answers "may I claim it?", while the Coach answers "did I understand it?" (chapter 24).

### 15.1 Crossfire — "am I allowed to call this proven?"

Crossfire is a **pre-registered protocol**. Before training, it records a rubric and a target. After training, it scores the policy against that rubric and against a baseline. The verdict is one of:

| Verdict | Plain meaning | What to do |
|---|---|---|
| **win** | Pre-registered, enough frozen judgements, beat the baseline | You may make the claim, with the stated scope |
| **loss** | Pre-registered and honestly below target | A real result. Fix the policy or the target, then re-run |
| **inconclusive** | Protocol is sound; evidence is insufficient | Not a verdict on the policy. Add judged cases or a held-out dataset |
| **not preregistered** | There is no rubric recorded before the result | Do not claim anything; re-run from a locked arena |

**Why it exists:** it is trivially easy to look at a good number and invent a story that explains it. Crossfire makes the story come first. When it says `inconclusive`, believe it — the app is protecting you from overclaiming.

The badge count in the drawer is the number of open reasons/blockers, not "errors".

### 15.2 Corpus — your regression vault

Every approved human judgement becomes an executable assertion. On retraining, the policy is measured against the whole accumulated set.

- **Seeded automatically** from held-out validation rows (up to 30) on the first round.
- **Reviewed by source:** `val_seed` (automatic), `user_locked_step` (you locked it), `passive_log` (harvested from real usage).
- **Failures are triaged:** the drawer tells you which observation changed, what the policy chose, and what you approved.
- **Waive, don't delete.** When a later judgement supersedes an earlier one, waive the old case with a reason. The history stays auditable.

The corpus is consulted on every training round, every chat-authored code change, and scorecard finalisation. A regression anywhere blocks the keep decision.

### 15.3 Gate — why a change was kept or rolled back

Every post-training mutation gets a decision. The **Gate** tab lists them oldest first, with a reason and a triage category:

| Category | Meaning | What to do |
|---|---|---|
| `compiler_mismatch` | The new reward term never actually fired | Check the code path; the term may be unreachable |
| `cross_term_interference` | The new term over-dominated an approved case | Reduce its weight or scope it |
| `rl_variance` | Signal is fine; the run was too short or exploration too high | More episodes or rounds; retry before rewriting |
| `unclassified` | The evidence does not fit a known pattern | Inspect the failing episodes manually |

If a change is waiting, the tab says so. The next training round judges it. Do not stack another behavioural change on top.

---

## 16. Economics: latency and payback

The economics block answers a procurement question: **at what volume does owning this policy beat calling a frontier model?**

### 16.1 What the numbers mean

| Field | Plain meaning |
|---|---|
| **Authoring ledger** | Token cost + estimated human hours + first-year cost |
| **Payback tiers** | For a given frontier price, how many decisions until the local policy repays its authoring cost |
| **Token-only basis** | Compares just the token bill |
| **With-human-review basis** | Also accounts for the human review each path needs — usually the dominant term |
| **p99 latency** | 99th-percentile decision time, end to end |
| **Caveats** | What the estimate deliberately excludes. Read them |

### 16.2 How to read it without fooling yourself

- The **token-only** payback can look absurd (millions of decisions) because frontier token costs per decision are tiny. That is real.
- The **with-human-review** payback is usually the meaningful one, because review time dominates. In the live example, a flagship frontier path repays in roughly **4,700 decisions / ~19 days** at 250 decisions per day.
- Latency is a **single decision vs a single provider round-trip**. It excludes data arrival and human review on both sides.
- The authoring human-hours figure is **estimated from provider call count** unless you pass measured turns. Treat it as a placeholder, not payroll.

The honest conclusion is usually: local ownership wins on **latency and per-decision cost**, frontier ownership wins on **flexibility and zero authoring effort**. This drawer lets you see the crossover instead of arguing about it.

---

# Part V — Getting the best out of it

## 17. Choosing a policy strategy

There are three shapes of solution. Choosing the right one up front saves the most time of anything in this book.

### 17.1 Learned policy

**What:** the trained neural network alone.
**Best for:** judgement under uncertainty — fraud risk, triage, pricing, control.
**Strength:** handles patterns no rule captures.
**Weakness:** no guarantee. It can be confidently wrong.

### 17.2 Verified guard

**What:** a checked rule that enforces a hard constraint, with the learned policy shaping everything else (the badge might read `verified_minimax_guard`).
**Best for:** games with a provable non-loss strategy, safety envelopes, regulatory limits.
**Strength:** the constraint holds regardless of training noise.
**Weakness:** a badly written guard can make the agent rigid or predictable.

You do not have to build the guard by hand. When the objective contains a checkable rule — for
example "must never Auto-Approve an invoice whose vendor_risk_score is 85 or higher" — the app
extracts it into a declarative constraint, applies it during training, play and scoring, and ships it
with the bundle. The scorecard's **Constraints** line reports how many raw policy actions the mask
had to override.

### 17.3 Hybrid (the usual best answer)

**What:** guard for the invariant, learned policy for the style.
**Best for:** "never lose, but win when you can"; "never exceed the limit, but optimise throughput".
**How to ask:** put the invariant in the objective as a *must never*, and the preference as a *should*.

> "The agent must never lose at tic-tac-toe, including against perfect play. Subject to that, it should win whenever the opponent errs."

### 17.4 Decision table

| Your objective contains… | Choose |
|---|---|
| "must never", "must always", a legal or safety limit | Guard (or hybrid) |
| "at least X% agreement", "rank", "predict" | Learned policy |
| A short list of deterministic rules that fully decide the answer | You may not need RL at all — say so in chat; a rule is cheaper and more auditable |
| Both a hard limit and a quality goal | Hybrid |

**Do not** use a learned policy for something a five-line rule does perfectly. The app will happily train it, and you will own a more expensive, less explainable rule.

---

## 18. The cost playbook

### 18.1 What costs money

| Activity | Metered? |
|---|---|
| Sending an objective / chat message | **Yes** |
| Arena generation and code synthesis | **Yes** |
| Plateau auto-tuning | **Yes** (only on plateau) |
| Training episodes | No |
| Evaluation episodes | No |
| Play / replay / drift | No |
| Export / registry / ONNX inference | No |

In a live example in this app, about **120 provider calls cost roughly $0.50** on DeepSeek. Most of that is input tokens: synthesis prompts carry the current code, the dataset profile and the conversation. Output tokens are comparatively small.

### 18.2 Ten levers, strongest first

1. **Use `deepseek-chat` for iteration.** Switch to a reasoner only when the environment is the blocker. Automatic reward tuning now runs only after a detected plateau, and its change is judged by the next round's verifier gate instead of being applied ungated.
2. **Get the objective right the first time.** Every re-synthesis is a full prompt.
3. **One behavioural change per message.** Batching unrelated changes causes regressions, and regressions cause re-runs.
4. **Do cosmetic changes freely but cheaply.** Looks do not need a reasoner.
5. **Let the plateau tuner be the only auto-tune.** It fires only when the policy is stuck; do not force it with tiny episode budgets.
6. **Set a realistic target.** A 100% target on a stochastic task adds rounds with no evidence gain.
7. **Use the golden corpus.** Catching a regression early is cheaper than discovering it in production.
8. **Prefer local models for private or high-volume authoring.** Marginal token cost becomes zero.
9. **Export when done.** A shipped bundle has no token bill.
10. **Cap the budget** (below).

### 18.3 Budget cap

The backend honours a per-tenant spend cap:

```bash
export RL_IDE_OWNER_BUDGET_USD=25
```

When the cap is reached, the provider call is refused **before** it is made. This is a guardrail for experiments and demos, not a billing system.

### 18.4 A cost worksheet

For a new agent, budget roughly:

| Item | Typical |
|---|---|
| Initial synthesis (objective → arena) | 1 large call |
| 2–5 refinement turns | small calls |
| Plateau auto-tunes | 0–2 calls depending on rounds |
| Final explanation / summary | 1 call |

If you find yourself above ~25 calls for one agent, something is looping. Check the thread for repeated synthesis entries, and simplify the objective.

---

## 19. The quality playbook

### 19.1 "It plateaus below target"

1. Check the **scorecard**, not the round metric. If the held-out number is close, you may already be done.
2. Raise **episodes per round** before raising rounds. Noise is often the problem.
3. Ask the agent to **explain the failure episodes** ("what do the failing episodes have in common?").
4. If a specific behaviour is missing, name it: "It never blocks a two-in-a-row threat; make blocking dominate."
5. Add a **corpus case** for the behaviour you care about, so a future change cannot silently drop it.
6. Only then rewrite the reward.

### 19.2 "A change made it worse"

- Open **Trust → Gate**. The triage category tells you which of four things happened.
- `rl_variance`: retry with more episodes before changing anything.
- `cross_term_interference`: reduce the new term's weight.
- `compiler_mismatch`: the term did not fire; ask for the exact code path.
- Then re-run one round and compare the **same** deterministic episodes.

### 19.3 "It scores well but behaves badly"

This is reward hacking. The scorecard is measuring the reward, not your intent.

- Inspect `compute_reward()` for a term you did not intend.
- Add a **forbidden-action** corpus case (what must never happen).
- Restate the objective as a constraint, not a bonus: "must never" rather than "prefer not to".

### 19.4 "The confidence interval is too wide"

The interval is driven by evaluation sample size. Increase episodes, or attach a dataset with a larger held-out split. Do not narrow it by re-running until you like the number.

### 19.5 "Crossfire says inconclusive"

That is the protocol telling you the evidence bar is not met. For control tasks with no dataset, the app needs **100 frozen hold-out judgements** before it will close. Add judged cases through the corpus, or move to a dataset-backed task where held-out rows exist. An inconclusive verdict is not permission to claim a win.

### 19.6 "It behaves differently after export"

Check **ONNX behaviour: action agreement** on the scorecard. Export fails on a parity mismatch, so a large behaviour change usually means the feature pipeline around the model differs, not the model. The export notes say whether action masking was applied during evaluation.

### 19.7 "I do not trust the arena"

You are probably right. A misleading arena is worse than no arena: it makes a bad policy look good. Ask for the information you need on screen — the decision, the alternatives, and the reason — before locking.

---

## 20. Recipes

Copy these objectives. Adjust the numbers, keep the structure.

### 20.1 Never-lose tic-tac-toe (hybrid guard)

**Objective**

> "An agent plays tic-tac-toe as X or O and must never lose to a human, including perfect play. Subject to that, win whenever the opponent errs. The human chooses who moves first. Show a chalkboard arena with the reward breakdown and a New Game control."

**Params:** 5 rounds × 200 episodes × 90%.
**Evidence to require:** corpus 30/30, beats random baseline by a wide margin, interactive play ≥ 20 games with zero losses.
**What "done" sounds like:** `verified_minimax_guard`, `SAFE MODE ON`, Crossfire may still say inconclusive until 100 frozen judgements — do not overclaim.

### 20.2 Grid navigation (control)

> "An agent moves a rover on a 20×20 grid from start to a target shelf using up/down/left/right. Success is average steps within 10% of the shortest path over 500 random layouts. It must never enter a blocked cell. Show the grid, the route, and the step count."

**Params:** 5 rounds × 500 episodes × 90%.
**Watch:** illegal-action percentage on the scorecard should be 0.

### 20.3 Invoice fraud triage (dataset)

Attach `invoices.csv` with columns including `amount`, `risk_score`, `department`, `was_fraud`.

> "Given one invoice row, decide approve, review, or block. Success is catching at least 90% of fraudulent invoices while keeping false blocks under 2%. Never block an invoice above $10,000 without review. Show the row, the decision, and the top three reasons."

**Params:** 8 rounds × 500 episodes × 90%.
**Evidence to require:** held-out badge, beats majority-class baseline, corpus 100%, action agreement ≥ 99%.
**Watch:** leakage (no `was_fraud` in observations), class imbalance, and a wide interval if fraud cases are few.

### 20.4 Support ticket router (dataset)

> "Given one support ticket, route it to billing, technical, account, or refund. Success is ≥ 85% agreement with the recorded department. Refunds above $500 must route to refund and never to billing. Show the ticket, the route, and the confidence."

**Params:** 5 rounds × 500 episodes × 90%.
**Watch:** rare classes; check per-class failures in the corpus, not just overall accuracy.

### 20.5 Document-driven process (PDF/DOCX)

> "Read a supplier contract and decide whether it needs legal review. Success is ≥ 90% agreement with the recorded review outcome. Never auto-approve a contract containing an unlimited liability clause. Show the extracted clause and the decision."

**Params:** 8 rounds × 500 episodes × 90%.
**Watch:** extraction quality. If the PDF text layer is bad, no amount of RL fixes it — clean the document first.

---

# Part VI — Reference

## 21. Troubleshooting

| Symptom | What it means | Do this |
|---|---|---|
| "A verified model is required" | No active, tested provider | Open the model pill → **Test Connection** |
| Training button does nothing | Arena not locked, or no policy yet | Click **Lock Arena & Start Training** |
| Rounds finish below target | Budget exhausted before the goal | Add rounds/episodes, or lower an unrealistic target |
| Scorecard says **in-sample** | No held-out split exists for this task | Treat as a sanity check; add a dataset or rely on the corpus |
| **Crossfire: inconclusive** | Evidence bar not met, not a policy failure | Add frozen judgements or a held-out dataset |
| Gate rolled a change back | The change damaged approved behaviour | Read the triage category; adjust weight or scope |
| **warming up** in the header | Drift has too few live samples | Play 20+ varied steps |
| Corpus shows `regressed` | A previously approved case now fails | Inspect the case; fix or waive with a reason |
| Export greyed out | No trained policy yet | Train at least one round |
| Export fails parity | ONNX disagrees with the trained policy | Do not ship; report the failure |
| Arena looks generic | The synthesised arena is still pending | Read the thread; wait or ask for a change |
| Spend climbing with no progress | Repeated synthesis loop | Simplify the objective; switch to a cheaper model; cap budget |
| **Coach: blocked before training** | A structural finding: zero budget, empty or one-class data, no label for a supervised metric, no action space | Use the fix on the coach entry in the thread, then send the request again. A blocker cannot be acknowledged away |
| **Coach: warning will not go away** | You changed nothing, so the evidence is unchanged | Fix the artefact, or acknowledge the warning; it stays quiet until the evidence changes |
| **No coach entry yet** | No checkpoint has been evaluated for this conversation yet | Send an objective, attach data, or finish a round; the first finding appears with intake |

---

## 22. Glossary

**Action** — one choice the agent may make at a step.

**Action space** — the full set of legal choices.

**Agent** — in this product, a trained policy plus its environment, corpus and evidence. A "conversation" in the UI.

**Arena** — the visual and interactive representation of the environment.

**Baseline** — a simple comparator (random, majority class). A policy that does not beat baselines has not demonstrated value.

**BYOK** — bring your own key. You supply the provider credentials; the app does not bundle a model.

**Checkpoint** — a saved snapshot of the policy at a round. The scorecard may select an earlier, better-validation checkpoint.

**Confidence interval (95%)** — the range in which the true value is likely to sit. Wide intervals mean "not enough evidence", not "bad policy".

**Corpus (golden eval corpus)** — a growing set of approved behaviours that every change must preserve.

**Crossfire** — the pre-registered protocol that decides whether a performance claim is allowed.

**Drift** — a change in live inputs or actions relative to the training baseline.

**Episode** — one complete attempt at the task.

**Environment** — the rules and dynamics of the world, implemented in `env_module.py`.

**Evaluation split** — data reserved for scoring. `test` is held out; `val` guides selection; `train` teaches.

**Gate (verifier gate)** — the judge that keeps or rolls back reward/environment/inference changes.

**Held out** — scored on data excluded from training. The honest number.

**In-sample** — scored on data the policy trained on. Optimistic; use with care.

**JIT synthesis** — generating the environment and reward code on demand from your description.

**Observation** — what the agent can see at a step.

**ONNX** — a portable model format. The exported policy runs without Python training code or an LLM.

**Payback** — how many decisions it takes for local ownership to repay its authoring cost.

**Plateau** — a stretch with too little improvement; the trigger for auto-tuning.

**Policy** — the agent's decision function.

**PPO** — Proximal Policy Optimization, the reinforcement-learning algorithm used for training.

**Reward** — the score the training algorithm maximises. A means, not the objective.

**Round** — one training burst plus one evaluation.

**RunBundle** — the self-contained export: model, labels, baseline, feature pipeline, inference server.

**Safe mode** — hard constraints being enforced by a verified guard.

**Scorecard** — the evidence package: metric, interval, baselines, corpus result, Crossfire verdict, economics.

**Target** — the success rate at which training stops early.

**Tenant token** — the `rlt_…` credential for hosted, multi-user deployments.

**Validation split** — data used to choose between checkpoints; not the final honesty check.

**Verifier** — the independent check that generated code runs and behaves.

---

## 23. Cheat sheet

**First agent**

1. Set a provider, **Test Connection**, save.
2. Write a six-part objective.
3. Preview the arena; refine in chat; **Lock Arena & Start Training**.
4. Params: **3 rounds · 200 eps · 85%** for a first attempt.
5. Play 20+ steps; inspect a replay.
6. Read the scorecard; require **beats all baselines** and **corpus n/n**.
7. Export only when the evidence is good.

**Choose params**

| Goal | Rounds | Episodes | Target |
|---|---:|---:|---:|
| Smoke test | 1 | 50 | 70% |
| First attempt | 3 | 200 | 85% |
| Serious | 5–8 | 500 | 90% |
| High-stakes | 10–15 | 1,000 | 95% |

**Choose a model**

- Default: `deepseek-chat`
- Hard reasoning: `deepseek-reasoner` (~2×)
- Private: Ollama
- Cosmetic edits: cheapest available

**Phrase a change**

> Change **X** to **Y**, because **Z**. Do not change **W**. I will judge it by **V**.

**Trust checklist**

- [ ] Held-out, not in-sample
- [ ] Narrow-enough confidence interval
- [ ] Beats every baseline
- [ ] Corpus: all verified, zero regressions
- [ ] ONNX action agreement ≥ 99%
- [ ] Played 20+ steps without surprises
- [ ] Crossfire verdict understood — `inconclusive` is not `win`

**Coach checklist**

- [ ] No coach blocker in the thread before you lock the arena
- [ ] Objective has a measurable success metric
- [ ] Every "never" rule has a threshold or a penalty behind it
- [ ] Dataset has a label column and a clearly named outcome
- [ ] Target is not 100% unless a guard can guarantee it
- [ ] At least 200 episodes per round for a first attempt
- [ ] After the run: read the interval and the next action the card offers
- [ ] Acknowledged warnings were read, not dismissed

---

# Part VII — Staying on the path

## 24. The Journey Coach: the checklist the app runs for you

The rest of this book is advice. The Coach is that advice turned into checks that run at the moment
you are about to spend time or money. It is not a second chatbot and it is not a score. It is a short
list of "this will not do what you think" findings, and each finding carries the button that fixes it.

Coach findings appear as entries in the thread, authored by **Coach**, at the moment they are
produced:

| Where | What it shows |
|---|---|
| **In the turn that caused it** | Objective, data, budget, arena or result findings, with the button that fixes each one |
| **As the turn's last entry** | A blocker, which stops the run before it is paid for |

There is no separate coach panel and no badge to hunt for: the finding sits in the conversation next
to the decision it belongs to, and it appears exactly once. Acknowledging a warning marks it quiet
until the evidence actually changes.

### 24.1 The six checkpoints

| Stage | When it runs | A finding you may see |
|---|---|---|
| **Objective** | When the architect drafts a blueprint | "No measurable success criterion was stated" |
| **Data** | When you attach a dataset | "The minority class is 3.0%; accuracy will look excellent while the costly class is missed" |
| **Parameters** | Before a training run | "At 10 episodes the interval is ±22.1 points; about 200 are needed for ±5" |
| **Preview** | When an arena is generated | "The arena never displays the metric you will be asked to trust" |
| **Results** | When a scorecard is produced | "80.0% with a 70–90 interval against a 79.0% majority baseline proves nothing yet" |
| **Export** | When you package a bundle | "The scorecard describes a policy that is not the one in the bundle" |

### 24.2 Blocker, warning, tip

| Severity | What it means | What happens |
|---|---|---|
| **Blocker** | The run cannot answer the question you asked | In **hard** mode, build, training or resume is stopped; the card names the exact fix |
| **Warning** | The run will probably waste time or mislead you | You may proceed, or acknowledge it |
| **Tip** | A better next step exists | Nothing is blocked |

Blockers are structural, not stylistic: zero rounds or zero episodes, an empty dataset, a label
column with only one class, a supervised metric with no label column, or no action space at all.
A warning is the more common case: a 100% target, a wide interval, an ID used as a feature, a
"never" rule that became a soft preference.

In the default **hard** mode a blocker **cannot be acknowledged away**; it must be fixed. In
advisory mode it behaves like a warning: reported, acknowledgeable, and the run proceeds. A warning
or tip can be acknowledged: it goes quiet, and stays quiet until the evidence changes (you change
the target, the data, or the objective). That is deliberate — the same warning should not nag on
every render.

### 24.3 What a blocker actually stops

In the default **hard** mode, a blocker stops the three actions that cost real time:

1. **Lock Arena & Start Training** (and synthesis before it).
2. A **training request** sent through Studio Chat.
3. **Resume training**.

It does **not** stop the conversation. The intake chat is the repair surface: answer the question,
name the label column, set a reachable target, then send the request again. The app returns the
finding as a `409` with the same card the UI renders; a chat turn that would have trained returns
`status: "blocked"` with the same card. Either way, an API client is not left guessing.

If you are deliberately running an experiment that trips a rule, the deployment switch is
`RL_IDE_COACH_BLOCKERS=advisory`. In advisory mode blockers are reported and never stop a run.
That switch is for operators, not a per-conversation setting.

### 24.4 The findings you will meet most often

| Finding | What to do |
|---|---|
| The objective is too short to constrain the build | Use the six-part formula in §5.1, or send the rewritten objective the Coach offers |
| No measurable success criterion | Add a number: "≥ 90% agreement", "within 10% of the shortest path", "false blocks under 2%" |
| A "never" rule was stated but not encoded | Give it a threshold ("never approve when `risk_score` ≥ 85") or add a penalty term; §17 covers guards |
| The data has no label column | Name the column that holds the recorded outcome, or switch to a task that does not need one |
| The classes are severely imbalanced | Optimise recall on the costly class at a fixed false-positive rate, not plain accuracy |
| A row identifier is being used as a feature | Exclude it; an ID lets the policy memorise the split |
| A single feature separates the label perfectly | Remove or lag it; this is the target leaking into a feature (§11.4) |
| A 100% target | Lower it to 90% and raise episodes; §9.3 explains why |
| Too few episodes to trust the score | Apply the recommended episode count, or run the action the card offers |
| The target is already met by doing nothing | Raise the target above the majority-class or random baseline |
| The confidence interval is too wide | Run the episode count in the action; do not re-run until you like the number (§19.4) |
| The policy is no better than the baseline | Read §19.3; the reward is measuring something other than your intent |
| Previously approved behaviour regressed | Open Trust → Gate; the triage category names the cause |
| The result cannot be called proven yet | Open Trust → Crossfire; `inconclusive` is not a failure, it is an evidence gap (§15.1) |
| The arena does not show the metric | Ask for the readout before you lock; a misleading arena is worse than none (§19.7) |

### 24.5 What the model does — and does not do

This is the part that keeps the Coach honest:

- **Every number comes from code.** The rules read the objective, the dataset profile, the training
  parameters, the arena and the scorecard. The model receives those facts and may only prioritise
  them and write the sentence.
- **It cannot invent a number.** A finding that mentions a value absent from the evidence is
  discarded before it reaches you.
- **It can never return a blocker.** Structural blockers are facts. A model must not be the thing
  that stops your run.
- **It is cached.** Re-opening a card, or re-evaluating the same artefact, costs no tokens. The
  judge is keyed by the stage, the facts and the objective.
- **It is capped.** Twelve judge calls per conversation by default. When the cap is reached, the
  rules keep working and the model is skipped.
- **It fails open.** If the provider is down, the rules still run and the card says the model was
  skipped. A coach failure never breaks the studio.

Judge calls are billed like any other model call, so they show up in the spend figure on the
scoreboard and in the normal usage ledger (§16).

### 24.6 A worked example

> Objective: "Build a fraud detector."
> Data: `invoices.csv`, 3% fraudulent, no column named as the outcome.

1. **Objective:** the Coach warns that the objective has no measurable success criterion.
2. **Data:** it reports a blocker — the metric is accuracy-like but no label column exists — and a
   warning that the classes are severely imbalanced.
3. You name `was_fraud` as the label and restate success as "catch at least 90% of fraud while
   keeping false blocks under 2%".
4. **Parameters:** you set 1 round, 50 episodes, 100%. The Coach warns about both the 100% target
   and the interval width, and offers **Apply recommended** (3 rounds · 200 episodes · 90%).
5. You train. The scorecard reads `82.0% [74.0–89.0]`, tied with the majority baseline.
6. **Results:** the Coach says the interval is too wide and the policy is no better than doing
   nothing, with **Run 300 episodes** and a link to §19.3.

That is the same loop the book describes. The difference is that the app tells you which step you
are on instead of leaving you to infer it from a chart.

### 24.7 What it will not do

- It does not edit your reward, environment, or parameters on its own. Every fix is a button.
- It does not prove anything. Crossfire owns the claim.
- It does not replace the scorecard, the corpus, or your judgement.
- It does not silently waive a blocker, and it does not hide a warning you have not addressed.
- It does not read `env_module.py`. Code correctness belongs to the sandbox smoke test and the
  verifier gate.

### 24.8 Switches

| Switch | Default | Effect |
|---|---|---|
| `RL_IDE_COACH_BLOCKERS` | `hard` | `advisory` reports blockers without stopping a run |
| `RL_IDE_COACH_LLM` | `1` | `0` disables the model and keeps the rules |
| `RL_IDE_COACH_MAX_LLM_CALLS` | `12` | Per-conversation judge-call cap |
| `RL_IDE_LLM_THINKING` | unset | `0` disables reasoning tokens for every role |
| `RL_IDE_LLM_MAX_TOKENS` | unset | Hard output-token ceiling applied on top of every role budget |
| `RL_IDE_VERIFY_MAX_REPAIRS` | `2` | Self-healing rewrites allowed while verifying a new environment |
| `RL_IDE_TUNE_MAX_REPAIRS` | `1` | Self-healing rewrites allowed while verifying an auto-tuned reward |
| `RL_IDE_ARENA_MODE` | `spec` | `spec` (model styles, code renders), `bespoke` (model writes the HTML), `deterministic` (no provider call) |
| `RL_IDE_PLANNER_LLM` | `auto` | `auto` asks the model only for directives the local grammar cannot route; `always`, `off` |

---

## 25. Limits, safety, and honest answers

**What this tool is good at.** Well-scoped decisions with a measurable outcome: games, control, triage, routing, risk scoring.

**What it cannot do.** It cannot invent ground truth. If your labels are wrong, the agent learns the wrong thing. If the objective is vague, the reward will be too. If the arena hides the information needed, the policy cannot use it.

**Privacy.** With a cloud provider, prompts leave the boundary during **authoring** and plateau tuning. Only inference is local. For sensitive data, configure Ollama or a self-hosted endpoint.

**Safety.** A trained policy is not a licence. For regulated decisions, keep a human in the loop, keep the corpus, and treat the exported bundle as software that needs monitoring — because it is.

**Claims.** Crossfire exists because it is easy to overclaim. Use its vocabulary: `win` means you may claim, within scope. `inconclusive` means keep gathering evidence. `loss` is a result.

**The Coach.** It is a checklist, not a guarantee. It catches the mistakes that are visible in the artefacts — a missing metric, an unlabelled dataset, an unencoded "never", an over-wide interval — and it stops a run when one of them makes the run meaningless. It cannot tell you whether the job is worth doing, and it does not inspect the generated code. Use it to avoid wasting a run, then use the scorecard and Crossfire to decide whether to ship.

**Maintenance.** Drift is your early warning. Review it, retrain on schedule, and add corpus cases whenever a human corrects the agent.

---

*End of the field guide. If a screen in the product disagrees with this book, the product is right — and the book should be updated.*
