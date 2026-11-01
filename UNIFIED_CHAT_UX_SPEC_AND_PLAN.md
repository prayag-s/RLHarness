# One Chat, One Clock — Unified Sequential Studio UX

**Status:** AGREED — decisions locked, implementation in progress.
**Date:** 2026-11-01
**Scope:** frontend information architecture, turn protocol, transcript durability. No change to training, reward tuning, coach rules, Crossfire, or export logic.
**Delivery:** all phases P0–P5 in one pass.

---

## 0. The ask

In the user's words:

> UX experience is very disjointed and not coherent. Some actions are in chat in left, some findings are in a panel in left, coach is top right. It does not seem to be turn by turn — sometimes actions are occurring and the user can interject in between. It's not like Codex or Claude Code, where only one party is allowed to give input while the other waits.
>
> One single chat where everything happens: user instruction, app output, coach outputs, training outcome, tuning — all in the left-hand chat, in sequential order. Success rate and other metrics (held-out scorecards, core metrics) continue to be shown in the left pane.

Three requirements fall out of this, and everything below serves them:

1. **One thread.** There is exactly one place where things happen: a single, ordered conversation.
2. **One turn at a time.** Exactly one party may act; the other waits. No interjection, no concurrent appends.
3. **Metrics are a dashboard, not a conversation.** Live numbers stay visible and quiet; they never compete with the thread.

Reference behaviour: Codex CLI / Claude Code — you type, the agent works and streams, the turn ends, you get the prompt back.

---

## 1. What the app does today (audited)

### 1.1 Current shell

| Region | Component | Width | Contents |
|---|---|---|---|
| Top | `Header` ([Header.tsx](frontend/src/components/Header.tsx)) | full | model status, Guide, **Coach**, **Trust**, Registry, Export, Conversations, spend |
| Left | `ArchitectPanel` ([ArchitectPanel.tsx:320](frontend/src/components/ArchitectPanel.tsx#L320)) | `w-96` | data intake, chat feed, clarifying questions, activity stream, quick chips, params, coach card, composer |
| Center | `SandboxedArenaIframe` ([App.tsx:1390](frontend/src/App.tsx#L1390)) | flexible | live arena, play/interact, inspect code |
| Right | `TelemetryPanel` ([TelemetryPanel.tsx:64](frontend/src/components/TelemetryPanel.tsx#L64)) | `w-80` | target vs current, held-out scorecard, **coach card**, round history, sparkline |
| Overlays | `CoachPanel` ([App.tsx:1482](frontend/src/App.tsx#L1482)), `TrustPanel` ([App.tsx:1471](frontend/src/App.tsx#L1471)), `ConversationSidebar`, `GuideModal`, `SettingsModal`, `RegistryModal`, `LLMVerificationGateModal` | — | seven overlays |

### 1.2 Concrete findings

**F1 — Coach speaks from three places at once.**
Coach findings render inside the left composer area ([ArchitectPanel.tsx:637](frontend/src/components/ArchitectPanel.tsx#L637)), inside the right telemetry panel ([TelemetryPanel.tsx:285](frontend/src/components/TelemetryPanel.tsx#L285)), and in a full drawer opened from the top-right header ([Header.tsx:234](frontend/src/components/Header.tsx#L234) → [App.tsx:1482](frontend/src/App.tsx#L1482)). Three surfaces, three loading paths, no single authority.

**F2 — Metrics live in the right pane, not the left.**
Target/current metric, held-out scorecard bands, Crossfire verdict, drift, round history, and sparkline are all inside `TelemetryPanel`, the right-most column. What the user calls "the left pane" currently holds the chat.

**F3 — There is no thread; there is a client-side event list.**
`ActivityTimelineItem` ([types/index.ts:560](frontend/src/types/index.ts#L560)) has `type: synthesis | arena | round | auto_tune | mastery | nudge | error | info`, `title`, `description`, `status`, `timestamp`. It has **no actor**, **no turn id**, **no sequence number**, and **no correlation to the request that produced it**. It is React state in `App.tsx`, not a record.

**F4 — Reload loses most of the conversation, and mislabels the rest.**
Restore maps every non-user entry to the title "Architect Evaluation" ([App.tsx:470-484](frontend/src/App.tsx#L470)). The backend only stores `role`/`content`/`timestamp` (plus `questions` and `user_answers`) — see [server.py:519](backend/rl_ide/api/server.py#L519) and `append_chat_message` in [conversation_store.py:528](backend/rl_ide/state/conversation_store.py#L528). Rounds, auto-tune diagnoses, arena reloads, scorecards, and coach findings are **not persisted at all**, so a reload silently rewrites history. `user_answers` entries are rendered as assistant text.

**F5 — Nothing owns the turn.**
The composer is gated only by `isAskingArchitect` ([ArchitectPanel.tsx:669](frontend/src/components/ArchitectPanel.tsx#L669), [:674](frontend/src/components/ArchitectPanel.tsx#L674)). The quick chips are gated by the same flag ([:541](frontend/src/components/ArchitectPanel.tsx#L541)). `isTraining` is passed into the panel but **never used to gate input**. So while a training run is streaming rounds over WebSocket, the user can submit another studio turn, click a chip, or open the coach drawer and apply a recommendation — and the WebSocket handler keeps appending to the same list ([App.tsx:1013-1180](frontend/src/App.tsx#L1013)). That is the interleaving the user is describing.

**F6 — Two doors into the same room.**
The same composer routes to `onSendStudioMessage` ([App.tsx:699](frontend/src/App.tsx#L699)) when the prop exists, otherwise to `onNudge` / `onChatWithArchitect` ([App.tsx:637](frontend/src/App.tsx#L637)) — see the branch at [ArchitectPanel.tsx:149-200](frontend/src/components/ArchitectPanel.tsx#L149). Identical UI text reaches different backends depending on wiring, which is invisible to the user.

**F7 — The mid-flight nudge is unreachable in the live configuration.**
`handleNudge` ([App.tsx:1183](frontend/src/App.tsx#L1183)) is only called from the fallback branch *after* `onSendStudioMessage` returns early. In the running app the nudge path is dead, so "mid-flight guidance" actually becomes a new studio turn.

**F8 — The header is an action soup.**
Seven unrelated controls sit in one row, none of them part of the flow. Actions that change state (Coach, Trust, Export, Registry) look identical to navigation (Guide, Conversations).

**F9 — No relationship between chat entries and the things they talk about.**
An entry saying "Round 3 evaluated 78%" cannot take you to the round, the arena state, or the scorecard. There is no anchor.

### 1.3 What it feels like to use

- "Where did that happen?" — the same event class appears in two panes.
- "Did I just interrupt it?" — you can, silently, and the display will interleave.
- "Which coach is the real one?" — three copies, independent freshness.
- "What did I actually do?" — reload rewrites the transcript.

---

## 2. Target model

### 2.1 One thread, one clock

There is a single ordered transcript: the **Studio Thread**. Everything that changes the state of the work is an entry in it. Entries are append-only and monotonically sequenced. Nothing renders workflow state outside it.

### 2.2 Actors

| Actor | Speaks as | Examples |
|---|---|---|
| **You** | user entry | instructions, answers to questions, approvals |
| **Studio** | assistant entry | plan, clarification questions, synthesis result, arena changes, chat answers |
| **Trainer** | assistant entry | round started, round evaluated, auto-tune diagnosis, budget exhausted, target achieved |
| **Coach** | assistant entry (distinct styling) | warnings, blockers, "what this result means" |
| **Gatekeeper** | system entry | Crossfire verdict, drift warning, trust/gate events |
| **System** | system entry | errors, restores, model gate |

The UI must never blur these. An entry always shows who said it.

### 2.3 Turn protocol

A turn is a unit of work with exactly one owner.

```
              ┌──────────── you type ─────────────┐
  IDLE ──────▶│  RUNNING (owner: Studio/Trainer)  │──────▶ DECISION_REQUIRED ──▶ IDLE
   ▲          └───────────────────────────────────┘              │
   │                        │                                    │
   │                     Stop (you)                               │
   └────────────────────────┴────────────────────────────────────┘
```

| State | Who may act | Composer | What the thread does |
|---|---|---|---|
| `idle` | **You** | enabled, focused | shows the last terminal entry |
| `running` | Studio or Trainer only | locked ("Studio is working · Round 3 of 5"), **Stop** available | streams entries in order |
| `decision_required` | **You** | unlocked, with the decision card attached | last entry is a card: question / blocker / result |
| `blocked` | **You** | unlocked; action chip on the blocker | last entry is the Coach blocker |

Rules:

1. **A turn ends with exactly one terminal entry** — a question card, a blocker card, a result card, or a plain answer. Only then does the composer unlock.
2. **No mid-turn injection.** If you type while `running`, the text is preserved and sent as the *next* turn. An explicit **Interrupt** cancels the current turn first, so even an interruption is a clean turn boundary.
3. **Training is one long turn owned by the Trainer.** Rounds, tuning, and scorecards are entries *inside* it. You can Stop; you cannot talk over it.
4. **Coach never speaks out of turn.** Its findings belong to the turn they were produced by, and a blocker becomes that turn's terminal entry.
5. **Server enforces it too.** One in-flight turn per conversation; a second request returns `409` with the active `turn_id`. A client-side lock is not a guarantee.

### 2.4 Thread entry schema (canonical)

```ts
type ThreadEntry = {
  seq: number                 // monotonic within conversation
  turn_id: string             // groups the turn
  actor: 'user' | 'studio' | 'trainer' | 'coach' | 'gatekeeper' | 'system'
  kind: 'instruction' | 'question' | 'answer' | 'plan' | 'stage'
      | 'arena' | 'round' | 'tune' | 'scorecard' | 'verdict'
      | 'finding' | 'result' | 'error'
  status: 'pending' | 'running' | 'success' | 'warning' | 'blocked'
  title: string
  body: string
  at: string                  // ISO timestamp
  anchor?: { round?: number; metric?: number; entry_id?: string }
  actions?: { label: string; kind: string; payload?: unknown }[]
}
```

`ActivityTimelineItem` is replaced by this. The existing `type` taxonomy becomes `kind + actor`, which is what makes correct labelling possible (see F4).

### 2.5 What is deliberately *not* in the thread

- Live numbers (success rate, held-out bands, drift, spend).
- Arena pixels.
- Settings, registry contents, guide prose.

These are surfaces, not events. They live beside the thread and never inside it.

---

## 3. Layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Header: identity · model · connection · conversation switcher    (no actions) │
├───────────────┬──────────────────────────────────────┬───────────────────────┤
│ SCOREBOARD    │ STUDIO THREAD                        │ ARENA                 │
│ (read-only)   │ (the only chat)                      │ (visual, silent)      │
│               │                                      │                       │
│ Target  90%   │ You    Train a Tic-Tac-Toe agent     │  ┌─────────────────┐  │
│ Current 78%   │ Studio ● Asking 3 pillars…           │  │                 │  │
│ ▓▓▓▓▓▓░░░     │ Studio ▸ Blueprint locked            │  │   live arena    │  │
│               │ Studio ▸ Arena preview ready         │  │                 │  │
│ Held-out      │ Trainer ▸ Round 1 started            │  └─────────────────┘  │
│  win 0.82     │ Trainer ▸ Round 1 evaluated 78%      │                       │
│  safety 0.91  │ Coach  ⚠ Reward penalises stalling   │  Play · Inspect       │
│  crossfire ✔  │ Trainer ▸ Auto-tuned (see diff)      │                       │
│               │ Studio ▸ Target achieved at 91%      │                       │
│ Drift  stable │ ─────────────────────────────────    │                       │
│ Spend $0.42   │ [ Decision: accept / refine / export ]│                      │
│               │ ┌──────────────────────────────────┐ │                       │
│               │ │ Ask, instruct, or answer…    ▶   │ │                       │
│               │ └──────────────────────────────────┘ │                       │
└───────────────┴──────────────────────────────────────┴───────────────────────┘
```

### 3.1 Left rail — Scoreboard (`ScoreboardRail`)

Width ~320px, collapsible, read-only. Contents in priority order:

1. Target vs current, with progress bar.
2. Held-out scorecard: bands, Crossfire verdict, drift status.
3. Rounds sparkline.
4. Budget and spend.
5. Dataset + session status.

Rules: **no buttons that start work**, no coach cards, no chat. Clicking a number jumps the thread to the entry that produced it (anchor scroll + highlight).

### 3.2 Center — Studio Thread (`ThreadView`)

The only place the work is narrated. Groups entries by turn with a subtle turn separator carrying the turn's outcome. Long turns (training) collapse into a summary row that expands: `Round 3 · evaluated 78% · auto-tuned 2 terms`.

### 3.3 Right — Arena (`ArenaStage`)

Kept, resizable, collapsible, and quiet: it renders state, it does not narrate. Arena-affecting events get a "see arena" affordance that highlights the corresponding thread entry.

### 3.4 Header and overlays

- Header keeps: model status, connection state, conversation switcher, spend, and the three
  **read-only** entry points that were always there — **Guide**, **Trust** and **Policy Registry**.
  It carries no control that starts or mutates work; Export and the training budget stay in the
  rail and the composer.
- `CoachPanel` → **deleted.** Coach is an actor in the thread.
- `TrustPanel` → evidence viewer, opened *from* a verdict entry. Same content, no separate door.
- `ConversationSidebar` → thread switcher listing the same conversations; opening one replays its thread.
- `RegistryModal`, `GuideModal`, `SettingsModal` → unchanged, but only reachable from non-workflow places (footer of the rail, header overflow).

---

## 4. Where each current surface goes

| Today | Tomorrow | Why |
|---|---|---|
| Left chat feed ([ArchitectPanel](frontend/src/components/ArchitectPanel.tsx)) | Center thread | one conversation, primary |
| Quick chips under the composer | Inline `actions` on the terminal entry, plus 2–3 contextual chips in the composer | actions belong to the decision they respond to |
| Params panel (rounds/eps/target) | Composer drawer + proposal entry from Studio | changing a budget is a turn |
| Right metric panel ([TelemetryPanel](frontend/src/components/TelemetryPanel.tsx)) | Left scoreboard rail + Trainer entries in the thread | numbers pinned, events narrated |
| Coach card in composer | Coach entry in thread | single authority |
| Coach card in telemetry | Coach entry in thread | single authority |
| Coach drawer (`CoachPanel`) | deleted | single authority |
| Trust drawer | evidence opened from the verdict entry | one door per fact |
| Header Export / Registry | terminal "result" entry actions; registry stays a library view | workflow belongs to the flow |
| Activity-stream "Job stage: …" spam | one collapsing "Studio is working" entry per turn | stops the feed from becoming a log |

---

## 5. Data and API changes

**5.1 Durable thread.**
Add `thread_entries` table (or extend `chat_history` rows with `actor`, `kind`, `status`, `turn_id`, `seq`). Backward-compatible read: existing `chat_history` rows map to `actor=user|studio`, `kind=instruction|answer`.

**5.2 Turn state persisted.**
`turn_state` (`idle | running | decision_required | blocked`) + `active_turn_id` on the conversation, so a reload mid-run reattaches to the WebSocket and shows "Trainer is working", instead of an empty prompt.

**5.3 WebSocket events carry identity.**
Every event includes `turn_id`, `seq`, `actor`. The client appends; it never reorders or invents.

**5.4 One in-flight turn enforced server-side.**
Second concurrent request → `409` + active `turn_id`. This is the real lock; the UI lock is cosmetic.

**5.5 Restore = replay.**
Loading a conversation replays `thread_entries` verbatim. No re-labelling, no dropping rounds, no "Architect Evaluation".

---

## 6. Frontend structure

New:

- `ScoreboardRail` — metrics only.
- `ThreadView` + `ThreadEntry` variants (`UserEntry`, `StudioEntry`, `TrainerRoundEntry`, `CoachFindingEntry`, `VerdictEntry`, `DecisionCard`, `ErrorEntry`).
- `TurnComposer` — lock states, queue, Stop/Interrupt.
- `useTurnMachine` — single source of truth for `turnState`, `turnId`, `seq`.
- `ArenaStage` — the current iframe, made collapsible.

Absorbed/deleted: `Timeline` state in `App.tsx`, `CoachPanel`, coach rendering in `ArchitectPanel` and `TelemetryPanel`, the fallback router in `handleUnifiedSend`.

`App.tsx` currently holds ~1,500 lines of orchestration; the turn machine and thread store move out of it.

---

## 7. Phased plan

| Phase | Work | Exit criteria |
|---|---|---|
| **P0** | Thread entry schema + turn state machine + server-side single-in-flight guard | Scripted run shows zero interleaved appends; 409 on concurrent turn; schema round-trips |
| **P1** | Layout: `ScoreboardRail` left, `ThreadView` center, `ArenaStage` right; header stripped of workflow actions | All current metrics visible in the rail; no regression in arena interaction |
| **P2** | Turn lock: composer states, queue, Stop/Interrupt, chips folded into decision cards | Composer disabled while running; queued text survives; Stop returns the turn |
| **P3** | Coach + Trust folded into the thread; `CoachPanel` deleted | Each finding renders exactly once; verdict opens evidence from its entry |
| **P4** | Durability: persist entries + turn state; restore replays the thread | Kill the tab mid-training; reload shows the full thread including rounds |
| **P5** | Polish: turn separators, collapsing long turns, anchors from rail → entry, keyboard (Enter send, Esc stop), a11y labels | Manual script passes; guide updated |

P0–P2 are the heart of the complaint; P3–P5 make it coherent and durable. Recommended: ship P0–P2 as one milestone.

---

## 8. Acceptance criteria (testable)

1. Exactly one party may append during a turn — asserted by an automated transcript test with a training run in flight.
2. The composer is disabled while `running`; text typed then is preserved and sent as the next turn.
3. `Stop` mid-training ends the turn; the next entry is a user entry.
4. Reload during training restores target, rounds, tuning, scorecard, coach findings, and verdict, in original order, with original actors.
5. No metric is rendered inside the thread; no conversational entry is rendered in the rail.
6. A coach finding with a given `fingerprint` renders exactly once.
7. No entry titled "Architect Evaluation" unless it was authored by the Studio as an evaluation.
8. Header contains no control that starts or mutates work (read-only views such as Guide, Trust and
   Registry are explicitly allowed).
9. Clicking a rail metric scrolls to the entry that produced it.
10. `handleUnifiedSend`'s two-backend fallback is gone: one composer, one backend contract.
11. Every surface is selectable and copyable: no page-level `select-none`, each thread entry has a
    copy button, and the arena's resize drag never leaves the document-wide selection lock on.

---

## 9. Risks and trade-offs

| Risk | Mitigation |
|---|---|
| Losing glanceable arena if the right pane must shrink | Keep the arena, resizable + collapsible; it never narrates |
| Users who *want* to interject mid-run | `Interrupt` is one click and creates a real turn boundary — the requirement is "one voice at a time", not "no control" |
| Long training turns flood the thread | Collapse a turn to one summary row; expand on demand |
| Backend migration on existing databases | Additive columns/table + read-compat mapping; no destructive migration |
| Big diff in `App.tsx` | P0 introduces the store + machine behind the existing UI, so the layout swap in P1–P2 is mechanical |

---

## 10. Decisions (locked 2026-11-01)

| # | Decision | Locked answer |
|---|---|---|
| D1 | Arena placement | **Right column, resizable and collapsible** |
| D2 | Typing during a turn | **Preserved and sent as the next turn** (no injection) |
| D3 | Metrics rail | **Left** — Scoreboard rail left, thread centre, arena right |
| D4 | Coach drawer | **Deleted entirely**; Coach is an actor in the thread |
| D5 | Scope | **All phases P0–P5 in one pass** |

Note on D2: because a turn is atomic, a queued message is visibly held ("queued for after this turn") and is never silently merged into the running turn. `Interrupt` remains available as a deliberate, turn-bounded act.

Note on D5: all phases land together, but the plan's phase boundaries stay as the implementation order and as the review checkpoints within the single change.

---

## 11. Implementation record

Delivered as one change (D5: all phases).

**Backend**

| File | Change |
|---|---|
| [`backend/rl_ide/state/thread_store.py`](backend/rl_ide/state/thread_store.py) | New. `thread_entries` + `thread_turns`: gap-free per-conversation sequence, `begin_turn` as an atomic compare-and-set, `handover_turn`, `close_turn`, `reconcile_interrupted`, legacy chat-history migration. |
| [`backend/rl_ide/state/thread_events.py`](backend/rl_ide/state/thread_events.py) | New. Pure mapping from studio/training events and studio responses to entries, plus `settle_state`. |
| [`backend/rl_ide/api/server.py`](backend/rl_ide/api/server.py) | Every emitted event is recorded against the owning turn; chat, intake, nudge and training all claim a turn and hand it back; a queued training job inherits the chat turn as **trainer**; new `GET /conversations/{id}/thread` and `POST /conversations/{id}/turn/stop`; conversation load returns `thread` + `turn`; deleting a conversation deletes its thread. |
| [`backend/tests/test_unified_thread.py`](backend/tests/test_unified_thread.py), [`backend/tests/test_unified_thread_api.py`](backend/tests/test_unified_thread_api.py) | 18 tests: sequencing, one-owner turns, stale-turn recovery, legacy migration, 409 on a concurrent turn, stop, replay, delete. |

**Frontend**

| File | Change |
|---|---|
| [`frontend/src/components/ScoreboardRail.tsx`](frontend/src/components/ScoreboardRail.tsx) | New. Left rail: target/current, held-out scorecard, drift, rounds, spend. Read-only; clicking a round jumps to it in the thread. |
| [`frontend/src/components/ThreadView.tsx`](frontend/src/components/ThreadView.tsx) | New. The one transcript, grouped by turn, with per-actor rendering, collapsed stage runs, inline coach cards, and round anchors. |
| [`frontend/src/components/TurnComposer.tsx`](frontend/src/components/TurnComposer.tsx) | New. Turn-aware composer: locked while another party works, **Queue** instead of inject, **Stop** / Esc to hand the turn back. |
| [`frontend/src/components/ArenaStage.tsx`](frontend/src/components/ArenaStage.tsx) | New. Resizable, collapsible, maximisable stage; renders only, never narrates. |
| [`frontend/src/App.tsx`](frontend/src/App.tsx) | Rebuilt around `threadEntries` + `turn`; polls only while a turn is running; optimistic user entry; queue flush; restore replays the durable thread. |
| [`frontend/src/components/Header.tsx`](frontend/src/components/Header.tsx) | Identity, model status, agents, spend, plus the read-only Guide / Trust / Policy Registry entry points. The Coach button is gone with the drawer; Export moved to the rail. |
| `ArchitectPanel.tsx`, `TelemetryPanel.tsx`, `CoachPanel.tsx` | Deleted; their responsibilities are absorbed above. |

**Behaviour notes**

- A training burst runs in one sandbox process and cannot be cut mid-round. **Stop** therefore
  requests cancellation, records "the current round will finish; no further rounds will start" in
  the thread, and hands the prompt back instead of leaving the user locked with a spinner.
- Late events that arrive after a turn was stopped are attributed to the turn that produced them,
  never to a new pseudo-turn.
- A cancelled turn is not reported as a failure.

**Verification**

- `pytest backend/tests` — full suite.
- `npm run build` (`tsc` + Vite) — clean.

---

## 12. Explicit non-goals

- No change to training, PPO, reward tuning, auto-tuner heuristics, coach rules, Crossfire, gates, or export formats.
- No change to the arena renderer or the interactive play loop.
- No new LLM calls; this is information architecture and turn ownership, not new intelligence.
