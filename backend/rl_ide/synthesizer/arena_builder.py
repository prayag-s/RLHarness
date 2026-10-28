"""Dynamic JIT Arena Synthesizer for RL-Native Agent IDE v4.0.

Generates self-contained, interactive HTML5/Canvas/Tailwind arena visualizations
(arena.html) rendered in a sandboxed iframe with postMessage bi-directional bridges.
"""

from __future__ import annotations
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.architect.intake import is_tictactoe_spec
from rl_ide.llm.client import BaseLLMClient, create_llm_client, FallbackCodeSynthesizer, get_active_llm_client


ARENA_BUILDER_SYSTEM_PROMPT = """You are a Creative Frontend & Simulation UI Engineer.
Given an `AgentBlueprintSpec` and the exact JSON structure returned by `SynthesizedGymEnv.render_arena_state()`, generate a complete, self-contained, single-file `arena.html` document.

REQUIREMENTS:
1. Use dark-mode modern aesthetics (`bg-slate-950 text-slate-100`, Tailwind CSS via CDN or embedded CSS, clean SVG/HTML5 Canvas animations).
2. Listen for `window.addEventListener("message", (event) => ...)` handling `event.data.type === "ARENA_UPDATE"`.
3. Render the domain state visually and unmistakably based on `event.data.arena_state` (e.g., an interactive game board, a 2D physics/robotics radar canvas, a visual shift calendar, a live financial/sensor chart, or an interactive card table).
4. Render live telemetry pills at the top (Episode ID, Step, Total Reward, and a horizontal bar strip for `reward_breakdown`).
5. Include an interactive mode: when the user clicks interactive elements in the arena (or clicks a "Reset / New Interactive Turn" button), emit `window.parent.postMessage({ type: "ARENA_INTERACT", payload: {...} }, "*")` so the user can play against or test the trained agent directly inside the arena!
6. Handle missing/initial state gracefully with a sleek "Waiting for first episode stream..." skeleton state.
"""


class JITArenaSynthesizer:
    """Synthesizes self-contained interactive arena.html visualizations."""

    def __init__(self, llm_client: BaseLLMClient | None = None):
        self._llm_client = llm_client

    @property
    def client(self) -> BaseLLMClient:
        if self._llm_client:
            return self._llm_client
        return get_active_llm_client()

    def synthesize(
        self,
        blueprint: AgentBlueprintSpec,
        sample_arena_state: dict[str, Any] | None = None,
        session_id: str | None = None,
        workspace_dir: str | Path | None = None,
    ) -> tuple[str, Path]:
        """Generates and writes arena.html to the workspace directory."""
        sid = session_id or f"session_{uuid.uuid4().hex[:8]}"
        out_dir = Path(workspace_dir) if workspace_dir else Path("workspaces") / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / "arena.html"

        html_str = self.generate_arena_html(blueprint, sample_arena_state)
        file_path.write_text(html_str, encoding="utf-8")
        return html_str, file_path

    def generate_arena_html(
        self,
        blueprint: AgentBlueprintSpec,
        sample_arena_state: dict[str, Any] | None = None,
    ) -> str:
        """Synthesizes HTML code using LLM or deterministic template."""
        if not isinstance(self.client, FallbackCodeSynthesizer):
            try:
                html = self._call_llm_arena_gen(blueprint, sample_arena_state)
                if html:
                    return html
            except Exception as exc:
                raise RuntimeError(f"LLM Arena HTML synthesis failed: {exc}")

        return self._generate_deterministic_arena_html(blueprint, sample_arena_state)

    def _call_llm_arena_gen(
        self,
        blueprint: AgentBlueprintSpec,
        sample_arena_state: dict[str, Any] | None,
    ) -> str | None:
        prompt = (
            f"Generate the self-contained `arena.html` for:\n"
            f"Blueprint: {blueprint.model_dump_json(indent=2)}\n"
            f"Sample Arena State: {json.dumps(sample_arena_state or {}, default=str)}\n\n"
            f"Return ONLY valid HTML inside a ```html ... ``` code block."
        )
        raw = self.client.generate(prompt, system_prompt=ARENA_BUILDER_SYSTEM_PROMPT)
        match = re.search(r"```(?:html)?\s*([\s\S]*?)\s*```", raw)
        return match.group(1).strip() if match else raw.strip()

    def _generate_deterministic_arena_html(
        self,
        blueprint: AgentBlueprintSpec,
        sample_arena_state: dict[str, Any] | None,
    ) -> str:
        """Constructs a responsive, feature-complete arena.html with full postMessage support."""
        task_title = blueprint.task_title
        is_tictactoe = is_tictactoe_spec(blueprint)
        is_dataset = (
            blueprint.data_source_type == "uploaded_dataset"
            or (sample_arena_state and "record_index" in sample_arena_state)
        )

        action_labels_json = json.dumps(blueprint.action_labels)
        target_threshold = blueprint.target_metric_threshold

        # Domain-specific interactive stage and rendering JS
        if is_tictactoe:
            domain_html = """
            <!-- Tic-Tac-Toe 3x3 Grid -->
            <div class="flex flex-col items-center justify-center py-4">
              <div class="text-sm font-medium text-slate-400 mb-3" id="game-status-label">Turn: Agent (X)</div>
              <div class="grid grid-cols-3 gap-3 bg-slate-900/80 p-4 rounded-2xl border border-slate-800 shadow-2xl">
                <button onclick="cellClick(0)" id="cell-0" class="w-20 h-20 bg-slate-800 hover:bg-slate-700/80 rounded-xl text-3xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(1)" id="cell-1" class="w-20 h-20 bg-slate-800 hover:bg-slate-700/80 rounded-xl text-3xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(2)" id="cell-2" class="w-20 h-20 bg-slate-800 hover:bg-slate-700/80 rounded-xl text-3xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(3)" id="cell-3" class="w-20 h-20 bg-slate-800 hover:bg-slate-700/80 rounded-xl text-3xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(4)" id="cell-4" class="w-20 h-20 bg-slate-800 hover:bg-slate-700/80 rounded-xl text-3xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(5)" id="cell-5" class="w-20 h-20 bg-slate-800 hover:bg-slate-700/80 rounded-xl text-3xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(6)" id="cell-6" class="w-20 h-20 bg-slate-800 hover:bg-slate-700/80 rounded-xl text-3xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(7)" id="cell-7" class="w-20 h-20 bg-slate-800 hover:bg-slate-700/80 rounded-xl text-3xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(8)" id="cell-8" class="w-20 h-20 bg-slate-800 hover:bg-slate-700/80 rounded-xl text-3xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
              </div>
              <div class="mt-4 flex gap-3">
                <button onclick="resetBoard()" class="px-4 py-1.5 bg-indigo-600/80 hover:bg-indigo-500 rounded-lg text-xs font-semibold text-white transition-all shadow">
                  🔄 Reset / Play as O
                </button>
              </div>
            </div>
            """
            domain_js = """
            function renderDomainState(state) {
              if (!state || !state.board) return;
              for (let i = 0; i < 9; i++) {
                const cell = document.getElementById('cell-' + i);
                if (!cell) continue;
                const val = state.board[i];
                if (val === 1) {
                  cell.innerText = 'X';
                  cell.className = 'w-20 h-20 bg-cyan-950/60 border border-cyan-500/50 rounded-xl text-3xl font-black text-cyan-400 flex items-center justify-center shadow-lg shadow-cyan-500/20';
                } else if (val === -1) {
                  cell.innerText = 'O';
                  cell.className = 'w-20 h-20 bg-rose-950/60 border border-rose-500/50 rounded-xl text-3xl font-black text-rose-400 flex items-center justify-center shadow-lg shadow-rose-500/20';
                } else {
                  cell.innerText = '';
                  cell.className = 'w-20 h-20 bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700/50 rounded-xl text-3xl font-black flex items-center justify-center cursor-pointer';
                }
              }
              const status = document.getElementById('game-status-label');
              if (status && state.game_status) {
                status.innerText = 'Status: ' + state.game_status + (state.winner !== null ? ' (Winner: ' + state.winner + ')' : '');
              }
            }

            function cellClick(idx) {
              sendInteraction({ user_action: idx });
            }

            function resetBoard() {
              sendInteraction({ reset: true });
            }
            """
        elif is_dataset:
            domain_html = """
            <!-- Data Inspector Card -->
            <div class="flex flex-col items-center justify-center py-4 w-full max-w-lg mx-auto">
              <div class="w-full bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-2xl">
                <div class="flex justify-between items-center mb-4 pb-2 border-b border-slate-800">
                  <div class="text-xs uppercase tracking-wider text-slate-400 font-bold">Record Telemetry</div>
                  <div id="data-idx-badge" class="px-2.5 py-1 bg-cyan-950 text-cyan-400 border border-cyan-800/60 rounded-md text-xs font-mono">Row #0</div>
                </div>
                <div id="data-preview-table" class="space-y-2 mb-4 text-xs font-mono">
                  <div class="text-slate-500 italic">Awaiting dataset stream...</div>
                </div>
                <div class="flex justify-between items-center pt-3 border-t border-slate-800/80">
                  <span class="text-xs text-slate-400">Cumulative Accuracy:</span>
                  <span id="data-accuracy-val" class="text-sm font-bold text-emerald-400 font-mono">100.0%</span>
                </div>
              </div>
              <div class="mt-4 flex gap-3">
                <button onclick="nextRecord()" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-xs font-semibold rounded-lg text-slate-200 border border-slate-700 transition">
                  Next Record ⏩
                </button>
              </div>
            </div>
            """
            domain_js = """
            function renderDomainState(state) {
              if (!state) return;
              const idxBadge = document.getElementById('data-idx-badge');
              if (idxBadge && state.record_index !== undefined) {
                idxBadge.innerText = 'Row #' + state.record_index + ' / ' + (state.total_records || 100);
              }
              const accVal = document.getElementById('data-accuracy-val');
              if (accVal && state.accuracy_pct !== undefined) {
                accVal.innerText = state.accuracy_pct + '%';
              }
              const preview = document.getElementById('data-preview-table');
              if (preview && state.current_record_preview) {
                let html = '';
                for (const [k, v] of Object.entries(state.current_record_preview)) {
                  html += `<div class="flex justify-between py-0.5 border-b border-slate-800/40">
                    <span class="text-slate-400 font-sans">${k}:</span>
                    <span class="text-slate-200">${v}</span>
                  </div>`;
                }
                preview.innerHTML = html;
              }
            }

            function nextRecord() {
              sendInteraction({ user_action: 1 });
            }
            """
        else:
            domain_html = """
            <!-- 2D Canvas Physics / Navigation -->
            <div class="flex flex-col items-center justify-center py-4">
              <canvas id="sim-canvas" width="400" height="300" class="bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl cursor-crosshair"></canvas>
              <div class="text-xs text-slate-500 mt-2">Click anywhere on canvas to deploy interactive waypoint target</div>
            </div>
            """
            domain_js = """
            const canvas = document.getElementById('sim-canvas');
            const ctx = canvas ? canvas.getContext('2d') : null;

            if (canvas) {
              canvas.addEventListener('click', (e) => {
                const rect = canvas.getBoundingClientRect();
                const x = ((e.clientX - rect.left) / canvas.width) * 16 - 8;
                const y = ((e.clientY - rect.top) / canvas.height) * 16 - 8;
                sendInteraction({ target_pos: [x, y] });
              });
            }

            function renderDomainState(state) {
              if (!ctx || !state) return;
              ctx.clearRect(0, 0, canvas.width, canvas.height);

              // Grid lines
              ctx.strokeStyle = '#1e293b';
              ctx.lineWidth = 1;
              for (let i = 0; i < canvas.width; i += 40) {
                ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height); ctx.stroke();
              }
              for (let j = 0; j < canvas.height; j += 40) {
                ctx.beginPath(); ctx.moveTo(0, j); ctx.lineTo(canvas.width, j); ctx.stroke();
              }

              // Transform coordinates
              function toScreen(x, y) {
                return [
                  ((x + 8) / 16) * canvas.width,
                  ((y + 8) / 16) * canvas.height
                ];
              }

              // Draw Target
              if (state.target_pos) {
                const [tx, ty] = toScreen(state.target_pos[0], state.target_pos[1]);
                ctx.fillStyle = '#10b981';
                ctx.beginPath();
                ctx.arc(tx, ty, 8, 0, Math.PI * 2);
                ctx.fill();
              }

              // Draw Agent
              if (state.agent_pos) {
                const [ax, ay] = toScreen(state.agent_pos[0], state.agent_pos[1]);
                ctx.fillStyle = '#38bdf8';
                ctx.beginPath();
                ctx.arc(ax, ay, 10, 0, Math.PI * 2);
                ctx.fill();
              }
            }
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{task_title} — Visual Arena</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body {{
      background-color: #020617;
      color: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 0;
      overflow-x: hidden;
    }}
  </style>
</head>
<body class="p-4 flex flex-col justify-between min-h-screen">
  <!-- Top Telemetry Bar -->
  <header class="flex flex-wrap items-center justify-between pb-3 border-b border-slate-800 gap-2">
    <div class="flex items-center gap-2">
      <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
      <h1 class="text-sm font-bold text-slate-100">{task_title}</h1>
      <span class="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700 font-mono">v4.0 JIT Arena</span>
    </div>
    <div class="flex items-center gap-3 font-mono text-xs">
      <div class="bg-slate-900 border border-slate-800 px-3 py-1 rounded-lg">
        <span class="text-slate-500">EPISODE:</span>
        <span id="telemetry-episode" class="font-bold text-cyan-400 ml-1">#1</span>
      </div>
      <div class="bg-slate-900 border border-slate-800 px-3 py-1 rounded-lg">
        <span class="text-slate-500">STEP:</span>
        <span id="telemetry-step" class="font-bold text-indigo-400 ml-1">0</span>
      </div>
      <div class="bg-slate-900 border border-slate-800 px-3 py-1 rounded-lg">
        <span class="text-slate-500">TOTAL REWARD:</span>
        <span id="telemetry-reward" class="font-bold text-amber-400 ml-1">0.0</span>
      </div>
      <div class="bg-slate-900 border border-slate-800 px-3 py-1 rounded-lg">
        <span class="text-slate-500">TARGET:</span>
        <span class="font-bold text-emerald-400 ml-1">{target_threshold}%</span>
      </div>
    </div>
  </header>

  <!-- Live Reward Breakdown Bar Strip -->
  <section class="py-2">
    <div class="text-[10px] font-semibold uppercase tracking-wider text-slate-500 mb-1">Live Reward Breakdown</div>
    <div id="reward-breakdown-bar" class="flex flex-wrap gap-2 text-xs font-mono">
      <span class="text-slate-600 italic">Waiting for first step telemetry...</span>
    </div>
  </section>

  <!-- Main Center Domain Stage -->
  <main id="domain-arena-stage" class="flex-1 flex flex-col justify-center">
    {domain_html}
  </main>

  <!-- Bottom Action Probability Strip -->
  <footer class="pt-3 border-t border-slate-800">
    <div class="flex justify-between items-center text-[10px] uppercase tracking-wider text-slate-500 mb-1">
      <span>Action Probabilities (π)</span>
      <span id="selected-action-tag" class="text-cyan-400 font-bold font-mono"></span>
    </div>
    <div id="action-probs-bar" class="grid grid-cols-3 sm:grid-cols-5 md:grid-cols-9 gap-1.5 text-[10px] font-mono"></div>
  </footer>

  <!-- postMessage Communication Protocol -->
  <script>
    const ACTION_LABELS = {action_labels_json};

    // Initialize action bars
    const probsBar = document.getElementById('action-probs-bar');
    if (probsBar && ACTION_LABELS.length > 0) {{
      probsBar.innerHTML = ACTION_LABELS.map((lbl, idx) => `
        <div class="bg-slate-900 border border-slate-800 p-1.5 rounded flex flex-col gap-1">
          <div class="truncate text-slate-400" title="${{lbl}}">${{lbl}}</div>
          <div class="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div id="prob-bar-${{idx}}" class="bg-cyan-500 h-full transition-all duration-150" style="width: 0%"></div>
          </div>
          <div id="prob-val-${{idx}}" class="text-[9px] text-right text-slate-500">0%</div>
        </div>
      `).join('');
    }}

    // Dispatch interaction to parent IDE
    function sendInteraction(payload) {{
      if (window.parent) {{
        window.parent.postMessage({{
          type: "ARENA_INTERACT",
          payload: payload
        }}, "*");
      }}
    }}

    // Domain Rendering JS
    {domain_js}

    // Listen for ARENA_UPDATE from Parent IDE
    window.addEventListener("message", function(event) {{
      if (!event.data) return;
      if (event.data.type === "ARENA_UPDATE") {{
        const data = event.data.payload || event.data;

        // 1. Update Telemetry
        if (data.episode_id !== undefined) {{
          const ep = document.getElementById('telemetry-episode');
          if (ep) ep.innerText = '#' + data.episode_id;
        }}
        if (data.step !== undefined) {{
          const st = document.getElementById('telemetry-step');
          if (st) st.innerText = data.step;
        }}
        if (data.total_reward !== undefined) {{
          const rw = document.getElementById('telemetry-reward');
          if (rw) rw.innerText = Number(data.total_reward).toFixed(2);
        }}

        // 2. Update Reward Breakdown Strip
        if (data.reward_breakdown) {{
          const rb = document.getElementById('reward-breakdown-bar');
          if (rb) {{
            const terms = Object.entries(data.reward_breakdown);
            if (terms.length > 0) {{
              rb.innerHTML = terms.map(([k, v]) => {{
                const color = v >= 0 ? 'text-emerald-400 bg-emerald-950/60 border-emerald-800/60' : 'text-rose-400 bg-rose-950/60 border-rose-800/60';
                return `<span class="px-2 py-0.5 rounded border text-[11px] ${{color}}">${{k}}: ${{Number(v).toFixed(2)}}</span>`;
              }}).join('');
            }}
          }}
        }}

        // 3. Update Action Probabilities
        if (data.action_probs && Array.isArray(data.action_probs)) {{
          data.action_probs.forEach((p, idx) => {{
            const bar = document.getElementById('prob-bar-' + idx);
            const val = document.getElementById('prob-val-' + idx);
            const pct = Math.round(p * 100);
            if (bar) bar.style.width = pct + '%';
            if (val) val.innerText = pct + '%';
          }});
        }}

        // 4. Update Domain Arena State
        if (data.arena_state) {{
          renderDomainState(data.arena_state);
        }}
      }}
    }});
  </script>
</body>
</html>
"""
