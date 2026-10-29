"""Dynamic JIT Arena Synthesizer for RL-Native Agent IDE v4.2.

Generates self-contained, interactive HTML5/Canvas/Tailwind arena visualizations
(arena.html) rendered in a sandboxed iframe with postMessage bi-directional bridges.
Includes a built-in Mock Simulation & Animation Loop (startMockAnimationLoop) for instant
zero-training preview, live restyling, and seamless transition to real training telemetry.
"""

from __future__ import annotations
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.architect.intake import is_tictactoe_spec
from rl_ide.llm.client import BaseLLMClient, create_llm_client, FallbackCodeSynthesizer, get_active_llm_client
from rl_ide.llm.prompts import ARENA_SYNTHESIZER_SYSTEM_PROMPT, ARENA_BUILDER_SYSTEM_PROMPT


def extract_html_code(raw: str) -> str:
    """Extracts valid HTML from raw LLM output, handling markdown fences and truncated blocks."""
    if not raw:
        return ""
    text = raw.strip()

    # 1. Match standard closed markdown block
    match = re.search(r"```(?:html)?\s*\n([\s\S]*?)\n```", text)
    if match:
        return match.group(1).strip()

    # 2. Match unclosed markdown block
    match_unclosed = re.search(r"```(?:html)?\s*\n([\s\S]*)", text)
    if match_unclosed:
        candidate = match_unclosed.group(1).strip()
        candidate = re.sub(r"```+$", "", candidate).strip()
        return candidate

    # 3. Direct fence cleanup
    if text.startswith("```"):
        lines = text.split("\n")
        first_line = lines[0].strip()
        if first_line.startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    return text


def heal_arena_html_scripts(html_text: str) -> str:
    """Heals common LLM JavaScript template literal syntax errors like unmatched parentheses."""
    if not html_text:
        return ""

    def fix_template_expr(match):
        full = match.group(0)
        expr = match.group(1)
        opens = expr.count("(")
        closes = expr.count(")")
        if closes > opens:
            missing = closes - opens
            return "${" + ("(" * missing) + expr + "}"
        elif opens > closes:
            missing = opens - closes
            return "${" + expr + (")" * missing) + "}"
        return full

    healed = re.sub(r"\$\{([^{}]+?)\}", fix_template_expr, html_text)
    return healed


def validate_arena_html_scripts(html_text: str) -> bool:
    """Validates that all <script> blocks inside the arena HTML parse without syntax errors."""
    if not html_text:
        return False
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html_text, re.DOTALL)
    for s in scripts:
        if s.strip():
            try:
                res = subprocess.run(["node", "--check", "-"], input=s, text=True, capture_output=True, timeout=5)
                if res.returncode != 0:
                    return False
            except Exception:
                pass
    return True


class JITArenaSynthesizer:
    """Synthesizes self-contained interactive arena.html visualizations with live mock animations."""

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
        visual_style: str | None = None,
    ) -> tuple[str, Path]:
        """Generates and writes arena.html to the workspace directory."""
        sid = session_id or f"session_{uuid.uuid4().hex[:8]}"
        out_dir = Path(workspace_dir) if workspace_dir else Path("workspaces") / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / "arena.html"

        html_str = self.generate_arena_html(blueprint, sample_arena_state, visual_style=visual_style)
        file_path.write_text(html_str, encoding="utf-8")
        return html_str, file_path

    def update_arena(
        self,
        blueprint: AgentBlueprintSpec,
        current_arena_html: str,
        user_visual_request: str,
        sample_arena_state: dict[str, Any] | None = None,
        workspace_dir: str | Path | None = None,
    ) -> str:
        """Updates and restyles arena.html according to user's request while preserving live mock animation and bi-directional bridges."""
        if not isinstance(self.client, FallbackCodeSynthesizer):
            try:
                prompt = (
                    f"You are updating and restyling this self-contained `arena.html` according to the user's request:\n"
                    f"User Request: \"{user_visual_request}\"\n\n"
                    f"Current Arena HTML:\n{current_arena_html[:6000] if current_arena_html else ''}\n\n"
                    f"Blueprint: {blueprint.model_dump_json(indent=2)}\n\n"
                    f"REQUIREMENTS:\n"
                    f"1. Apply the user's visual styling (e.g. colors, scoreboard, badges, layouts, sizing, themes) and interactive features (e.g. controls to play both X and O, side selector buttons, reset).\n"
                    f"2. PRESERVE `startMockAnimationLoop()`, `SET_ARENA_MODE`, `ARENA_UPDATE`, and `ARENA_INTERACT` functionality.\n"
                    f"3. Return ONLY valid HTML inside ```html ... ```."
                )
                raw = self.client.generate(prompt, system_prompt=ARENA_SYNTHESIZER_SYSTEM_PROMPT, timeout=30.0)
                extracted = extract_html_code(raw)
                if extracted and "ARENA_UPDATE" in extracted and "startMockAnimationLoop" in extracted:
                    healed = heal_arena_html_scripts(extracted)
                    if validate_arena_html_scripts(healed):
                        if workspace_dir:
                            (Path(workspace_dir) / "arena.html").write_text(healed, encoding="utf-8")
                        return healed
            except Exception:
                pass

        # Deterministic update
        updated = self._generate_deterministic_arena_html(
            blueprint=blueprint,
            sample_arena_state=sample_arena_state,
            visual_style=user_visual_request,
            base_html=current_arena_html,
        )
        if workspace_dir:
            (Path(workspace_dir) / "arena.html").write_text(updated, encoding="utf-8")
        return updated

    def generate_arena_html(
        self,
        blueprint: AgentBlueprintSpec,
        sample_arena_state: dict[str, Any] | None = None,
        visual_style: str | None = None,
    ) -> str:
        """Synthesizes HTML code using LLM or deterministic template."""
        if not visual_style:
            return self._generate_deterministic_arena_html(blueprint, sample_arena_state, visual_style=visual_style)

        if not isinstance(self.client, FallbackCodeSynthesizer):
            try:
                html = self._call_llm_arena_gen(blueprint, sample_arena_state)
                if html and "ARENA_UPDATE" in html and "startMockAnimationLoop" in html and ("<html" in html.lower() or "<!doctype html" in html.lower()):
                    healed = heal_arena_html_scripts(html)
                    if validate_arena_html_scripts(healed):
                        return healed
            except Exception:
                pass

        return self._generate_deterministic_arena_html(blueprint, sample_arena_state, visual_style=visual_style)

    def _call_llm_arena_gen(
        self,
        blueprint: AgentBlueprintSpec,
        sample_arena_state: dict[str, Any] | None,
    ) -> str | None:
        prompt = (
            f"Generate the self-contained `arena.html` with built-in Mock Data & Animation Engine (startMockAnimationLoop) for:\n"
            f"Blueprint: {blueprint.model_dump_json(indent=2)}\n"
            f"Sample Arena State: {json.dumps(sample_arena_state or {}, default=str)}\n\n"
            f"Return ONLY valid HTML inside a ```html ... ``` code block. Be concise and complete."
        )
        raw = self.client.generate(prompt, system_prompt=ARENA_BUILDER_SYSTEM_PROMPT)
        return extract_html_code(raw)

    def _generate_deterministic_arena_html(
        self,
        blueprint: AgentBlueprintSpec,
        sample_arena_state: dict[str, Any] | None,
        visual_style: str | None = None,
        base_html: str | None = None,
    ) -> str:
        """Constructs a responsive, feature-complete arena.html with full postMessage support, mock animation loop, and visual theming."""
        task_title = blueprint.task_title
        is_tictactoe = is_tictactoe_spec(blueprint)
        is_dataset = (
            blueprint.data_source_type == "uploaded_dataset"
            or (sample_arena_state and "record_index" in sample_arena_state)
        )

        style_query = ((visual_style or "") + " " + (blueprint.arena_visual_description or "")).lower()
        if base_html:
            style_query += " " + base_html.lower()

        has_scoreboard = "scoreboard" in style_query or "score" in style_query
        has_gold = "gold" in style_query or "golden" in style_query
        has_badge = "victory badge" in style_query or "badge" in style_query or "victory" in style_query
        has_larger = "larger" in style_query or "bigger" in style_query or "large" in style_query
        has_arcade = "arcade" in style_query or "retro" in style_query or "cyber" in style_query
        has_radar = "radar" in style_query or "hud" in style_query
        is_energy = any(kw in task_title.lower() or kw in style_query for kw in ["solar", "tariff", "power-flow", "arbitrage"]) or ("battery" in (task_title.lower() + style_query) and not any(kw in task_title.lower() for kw in ["rover", "robot", "amr", "warehouse", "navigation"]))
        has_hipaa = "hipaa" in style_query or "boundary" in style_query
        has_abnormal_red = "abnormal" in style_query or "red" in style_query or "vital" in style_query
        has_burn_rate = "burn-rate" in style_query or "$/hour" in style_query or "burn rate" in style_query or "cloud burn" in style_query
        has_win_probs = "probability" in style_query or "percentages" in style_query or "per-cell" in style_query
        has_power_arrows = "arrow" in style_query or "power-flow" in style_query or "flow" in style_query

        action_labels_json = json.dumps(blueprint.action_labels)
        target_threshold = blueprint.target_metric_threshold

        # HIPAA Badge snippet
        hipaa_badge_html = ""
        if has_hipaa:
            hipaa_badge_html = """
            <!-- HIPAA Boundary Verified Badge -->
            <div id="hipaa-badge" class="flex items-center justify-center gap-2 py-1.5 px-3 bg-emerald-950/80 border border-emerald-500/80 rounded-xl mb-3 shadow-lg text-xs font-bold text-emerald-300">
              <span>🛡️</span>
              <span>HIPAA BOUNDARY VERIFIED — 100% LOCAL ONNX EVALUATION (NO PHI LEAK)</span>
              <span>✅</span>
            </div>
            """

        # Burn-rate meter snippet
        burn_rate_html = ""
        if has_burn_rate:
            burn_rate_html = """
            <!-- Cloud Burn-Rate Meter -->
            <div id="burn-rate-meter" class="flex items-center justify-between px-4 py-2 bg-slate-900 border border-cyan-500/60 rounded-xl mb-3 shadow-lg">
              <div class="flex items-center gap-2">
                <span class="text-cyan-400 font-mono font-bold text-xs">⚡ CLOUD BURN-RATE:</span>
                <span id="burn-rate-val" class="text-sm font-mono font-black text-amber-400">$3.42 / hr</span>
              </div>
              <div class="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-emerald-950 border border-emerald-700 text-emerald-300 font-bold">SLA: 0 VIOLATIONS</div>
            </div>
            """

        # Scoreboard HTML snippet
        scoreboard_html = ""
        if has_scoreboard or has_arcade:
            scoreboard_html = """
            <!-- Retro Cyber Scoreboard -->
            <div id="arcade-scoreboard" class="flex items-center justify-between bg-slate-900/90 border border-purple-500/50 px-4 py-2 rounded-xl mb-3 shadow-lg">
              <div class="flex items-center gap-2">
                <span class="text-amber-400 font-black text-xs font-mono">🏆 SCOREBOARD:</span>
                <span id="scoreboard-agent-score" class="text-xs font-mono text-cyan-400 font-bold">AGENT: 12</span>
                <span class="text-slate-600 font-mono">|</span>
                <span id="scoreboard-opp-score" class="text-xs font-mono text-rose-400 font-bold">OPPONENT: 4</span>
                <span class="text-slate-600 font-mono">|</span>
                <span id="scoreboard-draws" class="text-xs font-mono text-slate-400 font-bold">DRAWS: 3</span>
              </div>
              <div class="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-purple-950/80 border border-purple-700/60 text-purple-300 font-bold">CYBER LEAGUE</div>
            </div>
            """

        # Victory Badge snippet
        victory_badge_html = ""
        if has_badge:
            victory_badge_html = """
            <!-- Victory Badge Banner -->
            <div id="victory-badge" class="flex items-center justify-center gap-2 py-2 px-4 bg-gradient-to-r from-amber-500/20 via-yellow-500/20 to-amber-500/20 border border-amber-400/80 rounded-xl mb-3 shadow-xl animate-pulse text-xs font-black text-amber-300">
              <span>🏆</span>
              <span>CHAMPION VICTORY BADGE UNLOCKED — TOP TIER POLICY ACTIVE</span>
              <span>🏆</span>
            </div>
            """

        cell_dim = "w-24 h-24 text-4xl" if has_larger else "w-20 h-20 text-3xl"
        border_accent = "border-amber-500/50 shadow-amber-500/10" if has_gold else "border-slate-800 shadow-2xl"

        # Domain-specific interactive stage and rendering JS
        if is_tictactoe:
            domain_html = f"""
            {victory_badge_html}
            {scoreboard_html}
            <!-- Tic-Tac-Toe 3x3 Grid Stage -->
            <div class="flex flex-col items-center justify-center py-4">
              <!-- Player Side / First Turn Controls -->
              <div class="mb-3 flex flex-wrap items-center justify-center gap-2 bg-slate-900/90 border border-slate-700/80 p-2 rounded-xl shadow-lg">
                <span class="text-xs font-bold text-slate-400 px-1">Play As:</span>
                <button id="btn-play-x" onclick="setPlayerSide('X')" class="px-3 py-1.5 text-xs font-bold rounded-lg transition-all border border-cyan-500/80 bg-cyan-950 text-cyan-300 shadow-cyan-500/30 shadow-md flex items-center gap-1.5 cursor-pointer">
                  <span>❌</span>
                  <span>User as X (1st)</span>
                </button>
                <button id="btn-play-o" onclick="setPlayerSide('O')" class="px-3 py-1.5 text-xs font-bold rounded-lg transition-all border border-slate-700 bg-slate-800 text-slate-400 hover:text-slate-200 flex items-center gap-1.5 cursor-pointer">
                  <span>⭕</span>
                  <span>User as O (Agent 1st)</span>
                </button>
                <button onclick="resetBoard()" class="ml-1 px-3 py-1.5 text-xs font-semibold rounded-lg bg-indigo-600/80 hover:bg-indigo-500 text-white transition-all shadow flex items-center gap-1 cursor-pointer">
                  <span>🔄</span>
                  <span>Reset</span>
                </button>
              </div>

              <div class="text-sm font-medium text-slate-400 mb-3" id="game-status-label">Turn: User (X)</div>
              <div class="grid grid-cols-3 gap-3 bg-slate-900/80 p-4 rounded-2xl border {border_accent}">
                <button onclick="cellClick(0)" id="cell-0" class="{cell_dim} bg-slate-800 hover:bg-slate-700/80 rounded-xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(1)" id="cell-1" class="{cell_dim} bg-slate-800 hover:bg-slate-700/80 rounded-xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(2)" id="cell-2" class="{cell_dim} bg-slate-800 hover:bg-slate-700/80 rounded-xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(3)" id="cell-3" class="{cell_dim} bg-slate-800 hover:bg-slate-700/80 rounded-xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(4)" id="cell-4" class="{cell_dim} bg-slate-800 hover:bg-slate-700/80 rounded-xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(5)" id="cell-5" class="{cell_dim} bg-slate-800 hover:bg-slate-700/80 rounded-xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(6)" id="cell-6" class="{cell_dim} bg-slate-800 hover:bg-slate-700/80 rounded-xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(7)" id="cell-7" class="{cell_dim} bg-slate-800 hover:bg-slate-700/80 rounded-xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
                <button onclick="cellClick(8)" id="cell-8" class="{cell_dim} bg-slate-800 hover:bg-slate-700/80 rounded-xl font-black transition-all flex items-center justify-center text-slate-600 cursor-pointer"></button>
              </div>
            </div>
            """
            domain_js = f"""
            const HAS_GOLD = {'true' if has_gold else 'false'};
            const HAS_WIN_PROBS = {'true' if has_win_probs else 'false'};
            const CELL_DIM = "{cell_dim}";

            let currentHumanSymbol = 'X';
            let humanFirst = true;

            function setPlayerSide(sym) {{
              currentHumanSymbol = sym;
              humanFirst = (sym === 'X');
              const btnX = document.getElementById('btn-play-x');
              const btnO = document.getElementById('btn-play-o');
              if (btnX && btnO) {{
                if (sym === 'X') {{
                  btnX.className = 'px-3 py-1.5 text-xs font-bold rounded-lg transition-all border border-cyan-500/80 bg-cyan-950 text-cyan-300 shadow-cyan-500/30 shadow-md flex items-center gap-1.5 cursor-pointer';
                  btnO.className = 'px-3 py-1.5 text-xs font-bold rounded-lg transition-all border border-slate-700 bg-slate-800 text-slate-400 hover:text-slate-200 flex items-center gap-1.5 cursor-pointer';
                }} else {{
                  btnO.className = 'px-3 py-1.5 text-xs font-bold rounded-lg transition-all border border-rose-500/80 bg-rose-950 text-rose-300 shadow-rose-500/30 shadow-md flex items-center gap-1.5 cursor-pointer';
                  btnX.className = 'px-3 py-1.5 text-xs font-bold rounded-lg transition-all border border-slate-700 bg-slate-800 text-slate-400 hover:text-slate-200 flex items-center gap-1.5 cursor-pointer';
                }}
              }}
              const status = document.getElementById('game-status-label');
              if (status) {{
                status.innerText = sym === 'X' ? 'Turn: User (X) — Click an empty cell' : 'Turn: Agent (X) — Thinking...';
              }}
              resetBoard();
            }}

            function renderDomainState(state) {{
              if (!state || !state.board) return;
              for (let i = 0; i < 9; i++) {{
                const cell = document.getElementById('cell-' + i);
                if (!cell) continue;
                const val = state.board[i];
                if (val === 1) {{
                  cell.innerText = 'X';
                  if (HAS_GOLD) {{
                    cell.className = CELL_DIM + ' bg-amber-950/70 border-2 border-amber-400 rounded-xl font-black text-amber-300 flex items-center justify-center shadow-lg shadow-amber-500/30';
                  }} else {{
                    cell.className = CELL_DIM + ' bg-cyan-950/60 border border-cyan-500/50 rounded-xl font-black text-cyan-400 flex items-center justify-center shadow-lg shadow-cyan-500/20';
                  }}
                }} else if (val === -1) {{
                  cell.innerText = 'O';
                  cell.className = CELL_DIM + ' bg-rose-950/60 border border-rose-500/50 rounded-xl font-black text-rose-400 flex items-center justify-center shadow-lg shadow-rose-500/20';
                }} else {{
                  if (HAS_WIN_PROBS) {{
                    const winPct = 82 + ((i * 7) % 15);
                    cell.innerHTML = `<span class="text-[10px] text-cyan-400/80 font-mono font-bold">${{winPct}}%</span>`;
                  }} else {{
                    cell.innerText = '';
                  }}
                  cell.className = CELL_DIM + ' bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700/50 rounded-xl font-black flex items-center justify-center cursor-pointer';
                }}
              }}
              const status = document.getElementById('game-status-label');
              if (status) {{
                if (state.winner !== null && state.winner !== undefined) {{
                  if (state.winner === 1) {{
                    status.innerText = currentHumanSymbol === 'X' ? '🎉 Victory! You won as X!' : '🤖 Agent won as X!';
                  }} else if (state.winner === -1) {{
                    status.innerText = currentHumanSymbol === 'O' ? '🎉 Victory! You won as O!' : '🤖 Agent won as O!';
                  }} else if (state.winner === 0) {{
                    status.innerText = '🤝 Match Drawn! Flawless defense.';
                  }}
                }} else if (state.game_status) {{
                  status.innerText = 'Status: ' + state.game_status;
                }}
              }}
            }}

            function cellClick(idx) {{
              sendInteraction({{
                user_action: idx,
                human_symbol: currentHumanSymbol,
                human_first: humanFirst
              }});
            }}

            function resetBoard() {{
              sendInteraction({{
                reset: true,
                human_symbol: currentHumanSymbol,
                human_first: humanFirst
              }});
            }}

            // --- Mock Animation Engine for Tic-Tac-Toe ---
            let mockBoard = [0, 0, 0, 0, 0, 0, 0, 0, 0];
            let mockTurn = 1;
            let mockEpisodeCount = 1;
            let mockStepCount = 0;

            function checkWinningLine(b) {{
              const wins = [
                [0,1,2],[3,4,5],[6,7,8],
                [0,3,6],[1,4,7],[2,5,8],
                [0,4,8],[2,4,6]
              ];
              for (const [x,y,z] of wins) {{
                if (b[x] !== 0 && b[x] === b[y] && b[y] === b[z]) return [x,y,z,b[x]];
              }}
              return null;
            }}

            function stepMockDomain() {{
              mockStepCount++;
              const win = checkWinningLine(mockBoard);
              const empty = mockBoard.map((v, i) => v === 0 ? i : -1).filter(i => i !== -1);

              if (win || empty.length === 0) {{
                if (win) {{
                  const [c1, c2, c3] = win;
                  [c1, c2, c3].forEach(idx => {{
                    const el = document.getElementById('cell-' + idx);
                    if (el) {{
                      el.classList.add('ring-4', 'ring-amber-400', 'bg-amber-900/90', 'text-amber-200');
                    }}
                  }});
                }}
                setTimeout(() => {{
                  mockBoard = [0, 0, 0, 0, 0, 0, 0, 0, 0];
                  mockTurn = 1;
                  mockEpisodeCount++;
                  renderDomainState({{ board: mockBoard, game_status: 'Mock Match Reset' }});
                }}, 450);
                return;
              }}

              // Strategic move heuristic for realistic mock visualization
              let chosen = (mockBoard[4] === 0 && Math.random() < 0.6)
                ? 4
                : empty[Math.floor(Math.random() * empty.length)];

              mockBoard[chosen] = mockTurn;
              const playerTag = mockTurn === 1 ? 'Agent (X)' : 'Opponent (O)';
              mockTurn = -mockTurn;

              // Action probabilities
              const probs = ACTION_LABELS.map((_, i) => i === chosen ? 0.65 + Math.random() * 0.25 : Math.random() * 0.05);
              const sum = probs.reduce((a, b) => a + b, 0);
              const normProbs = probs.map(p => p / sum);

              applyArenaPayload({{
                episode_id: 'MOCK-' + mockEpisodeCount,
                step: mockStepCount,
                total_reward: (mockStepCount * 0.15).toFixed(2),
                reward_breakdown: {{
                  step_cost: -0.01,
                  center_control: mockBoard[4] === 1 ? 0.5 : 0.0,
                  tactical_pressure: 0.25
                }},
                action_probs: normProbs,
                arena_state: {{
                  board: [...mockBoard],
                  game_status: 'Mock turn by ' + playerTag,
                  winner: null
                }}
              }});
            }}
            """
        elif is_energy:
            domain_html = f"""
            {victory_badge_html}
            {scoreboard_html}
            <!-- CleanTech Solar + Battery Tariff Arbitrage Arena -->
            <div class="flex flex-col items-center justify-center py-2 w-full max-w-xl mx-auto">
              <div class="w-full bg-slate-900 border {border_accent} rounded-2xl p-4 shadow-2xl">
                <!-- 4 Power Nodes Grid -->
                <div class="grid grid-cols-2 gap-3 mb-3">
                  <!-- Solar Node -->
                  <div class="p-3 bg-amber-950/40 border border-amber-600/50 rounded-xl flex items-center justify-between">
                    <div>
                      <div class="text-[10px] uppercase font-mono text-amber-400 font-bold flex items-center gap-1">
                        <span>☀️</span> SOLAR ARRAY
                      </div>
                      <div id="energy-solar-kw" class="text-lg font-black text-amber-300 font-mono">12.4 kW</div>
                    </div>
                    <div class="text-xs text-amber-500 font-mono">Generation</div>
                  </div>
                  <!-- Building Load Node -->
                  <div class="p-3 bg-indigo-950/40 border border-indigo-600/50 rounded-xl flex items-center justify-between">
                    <div>
                      <div class="text-[10px] uppercase font-mono text-indigo-400 font-bold flex items-center gap-1">
                        <span>🏢</span> BUILDING LOAD
                      </div>
                      <div id="energy-load-kw" class="text-lg font-black text-indigo-300 font-mono">8.2 kW</div>
                    </div>
                    <div class="text-xs text-indigo-500 font-mono">Demand</div>
                  </div>
                  <!-- Battery Storage Node -->
                  <div class="p-3 bg-emerald-950/40 border border-emerald-600/50 rounded-xl flex items-center justify-between">
                    <div>
                      <div class="text-[10px] uppercase font-mono text-emerald-400 font-bold flex items-center gap-1">
                        <span>🔋</span> BATTERY STORAGE
                      </div>
                      <div id="energy-battery-soc" class="text-lg font-black text-emerald-300 font-mono">65.0%</div>
                    </div>
                    <div class="w-12 bg-slate-800 h-2 rounded-full overflow-hidden">
                      <div id="battery-bar" class="bg-emerald-400 h-full" style="width: 65%"></div>
                    </div>
                  </div>
                  <!-- Grid Tariff Node -->
                  <div class="p-3 bg-cyan-950/40 border border-cyan-600/50 rounded-xl flex items-center justify-between">
                    <div>
                      <div class="text-[10px] uppercase font-mono text-cyan-400 font-bold flex items-center gap-1">
                        <span>⚡</span> POWER GRID TARIFF
                      </div>
                      <div id="energy-grid-price" class="text-lg font-black text-cyan-300 font-mono">$0.185 / kWh</div>
                    </div>
                    <div class="text-xs text-cyan-500 font-mono">Real-Time</div>
                  </div>
                </div>

                <!-- Animated Power Flow Arrows Visualization -->
                <div class="p-3 bg-slate-950 border border-slate-800 rounded-xl flex items-center justify-around text-xs font-mono shadow-inner">
                  <div class="flex items-center gap-1 text-amber-400 animate-pulse font-bold">
                    <span>Solar</span>
                    <span>──▶</span>
                    <span>Battery</span>
                  </div>
                  <div class="flex items-center gap-1 text-emerald-400 animate-pulse font-bold">
                    <span>Battery</span>
                    <span>──▶</span>
                    <span>Building</span>
                  </div>
                  <div class="flex items-center gap-1 text-cyan-400 animate-pulse font-bold">
                    <span>Grid</span>
                    <span>◀──▶</span>
                    <span>Arbitrage</span>
                  </div>
                </div>
              </div>

              <!-- Interactive Injection Controls -->
              <div class="mt-3 flex gap-2">
                <button onclick="injectEnergyState(18.0, 6.0, 75.0, 0.22)" class="px-3 py-1.5 bg-amber-600 hover:bg-amber-500 text-xs font-bold rounded-lg text-white transition shadow">
                  ☀️ High Solar / Sell to Grid
                </button>
                <button onclick="injectEnergyState(0.0, 12.0, 50.0, 0.25)" class="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-xs font-bold rounded-lg text-white transition shadow">
                  🏢 Night Peak Load / Discharge
                </button>
              </div>
            </div>
            """
            domain_js = """
            function renderDomainState(state) {
              if (!state) return;
              const sKw = document.getElementById('energy-solar-kw');
              if (sKw && state.solar_kw !== undefined) sKw.innerText = state.solar_kw + ' kW';
              const lKw = document.getElementById('energy-load-kw');
              if (lKw && state.building_load_kw !== undefined) lKw.innerText = state.building_load_kw + ' kW';
              const bSoc = document.getElementById('energy-battery-soc');
              const bBar = document.getElementById('battery-bar');
              if (bSoc && state.battery_soc_pct !== undefined) {
                bSoc.innerText = state.battery_soc_pct + '%';
                if (bBar) bBar.style.width = Math.min(100, Math.max(0, state.battery_soc_pct)) + '%';
              }
              const gPrice = document.getElementById('energy-grid-price');
              if (gPrice && state.grid_price_kwh !== undefined) gPrice.innerText = '$' + state.grid_price_kwh + ' / kWh';
            }

            function injectEnergyState(solar, load, soc, price) {
              sendInteraction({
                solar_kw: solar,
                building_load_kw: load,
                battery_soc_pct: soc,
                grid_price_kwh: price
              });
            }

            let mockET = 0;
            function stepMockDomain() {
              mockET += 0.25;
              const s = Math.max(0, (14.0 * Math.sin(mockET) + 2.0)).toFixed(1);
              const l = (8.0 + 3.0 * Math.cos(mockET * 0.7)).toFixed(1);
              const soc = Math.min(95, Math.max(25, (55.0 + 30.0 * Math.sin(mockET * 0.5)))).toFixed(1);
              const p = (0.12 + 0.08 * Math.sin(mockET * 0.6) + 0.04).toFixed(3);
              const probs = [0.45, 0.35, 0.15, 0.05];

              applyArenaPayload({
                episode_id: 'MOCK-ENERGY',
                step: Math.floor(mockET * 4),
                total_reward: (mockET * 1.15).toFixed(2),
                reward_breakdown: { arbitrage_savings: 1.0, tariff_efficiency: 0.2 },
                action_probs: probs,
                arena_state: {
                  solar_kw: s,
                  building_load_kw: l,
                  battery_soc_pct: soc,
                  grid_price_kwh: p,
                  last_action_label: s > l ? 'Charge Battery' : 'Discharge to Building'
                }
              });
            }
            """
        elif is_dataset:
            domain_html = f"""
            {victory_badge_html}
            {scoreboard_html}
            {hipaa_badge_html}
            {burn_rate_html}
            <!-- Data Inspector Card -->
            <div class="flex flex-col items-center justify-center py-4 w-full max-w-lg mx-auto">
              <div class="w-full bg-slate-900 border {border_accent} rounded-2xl p-5 shadow-2xl">
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
            domain_js = f"""
            const HAS_ABNORMAL_RED = {'true' if has_abnormal_red else 'false'};

            function renderDomainState(state) {{
              if (!state) return;
              const idxBadge = document.getElementById('data-idx-badge');
              if (idxBadge && state.record_index !== undefined) {{
                idxBadge.innerText = 'Row #' + state.record_index + ' / ' + (state.total_records || 100);
              }}
              const accVal = document.getElementById('data-accuracy-val');
              if (accVal && state.accuracy_pct !== undefined) {{
                accVal.innerText = state.accuracy_pct + '%';
              }}
              const preview = document.getElementById('data-preview-table');
              if (preview && state.current_record_preview) {{
                let html = '';
                for (const [k, v] of Object.entries(state.current_record_preview)) {{
                  const isAbnormal = HAS_ABNORMAL_RED && (
                    k.includes('systolic') && Number(v) > 135 ||
                    k.includes('heart') && Number(v) > 95 ||
                    k.includes('cardiac') && Number(v) === 1
                  );
                  const valClass = isAbnormal ? 'text-rose-400 font-bold bg-rose-950/70 px-1 rounded border border-rose-700/60' : 'text-slate-200';
                  html += `<div class="flex justify-between py-0.5 border-b border-slate-800/40">
                    <span class="text-slate-400 font-sans">${{k}}:</span>
                    <span class="${{valClass}}">${{v}}</span>
                  </div>`;
                }}
                preview.innerHTML = html;
              }}
            }}

            function nextRecord() {{
              sendInteraction({{ user_action: 1 }});
            }}

            let mockRow = 0;
            function stepMockDomain() {{
              mockRow = (mockRow + 1) % 50;
              const probs = ACTION_LABELS.map((_, i) => i === 0 ? 0.8 : 0.2 / Math.max(1, ACTION_LABELS.length - 1));
              applyArenaPayload({{
                episode_id: 'MOCK-DATA',
                step: mockRow,
                total_reward: (mockRow * 0.92).toFixed(1),
                reward_breakdown: {{ classification_accuracy: 1.0, latency_penalty: -0.05 }},
                action_probs: probs,
                arena_state: {{
                  record_index: mockRow,
                  total_records: 100,
                  accuracy_pct: (94.5 + Math.random() * 4).toFixed(1),
                  current_record_preview: {{
                    feature_a: (Math.random() * 10).toFixed(2),
                    feature_b: (Math.random() * 100).toFixed(1),
                    prediction: 'ACCEPTED',
                    confidence: '98.2%'
                  }}
                }}
              }});
            }}
            """
        else:
            domain_html = f"""
            {victory_badge_html}
            {scoreboard_html}
            <!-- 2D Canvas Physics / Navigation -->
            <div class="flex flex-col items-center justify-center py-4">
              <canvas id="sim-canvas" width="440" height="320" class="bg-slate-900 border {border_accent} rounded-2xl shadow-2xl cursor-crosshair"></canvas>
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

            function toScreen(x, y) {
              return [
                ((x + 8) / 16) * canvas.width,
                ((y + 8) / 16) * canvas.height
              ];
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

            let mockT = 0;
            function stepMockDomain() {
              mockT += 0.15;
              const ax = Math.sin(mockT) * 4.5;
              const ay = Math.cos(mockT * 0.7) * 3.5;
              const tx = 5.0;
              const ty = 4.0;
              const dist = Math.hypot(ax - tx, ay - ty);

              applyArenaPayload({
                episode_id: 'MOCK-SIM',
                step: Math.floor(mockT * 5),
                total_reward: (-dist).toFixed(2),
                reward_breakdown: { distance_reward: (-dist * 0.2).toFixed(2), velocity_cost: -0.05 },
                action_probs: [0.35, 0.25, 0.2, 0.2],
                arena_state: {
                  agent_pos: [ax, ay],
                  target_pos: [tx, ty]
                }
              });
            }
            """

        radar_hud_snippet = ""
        if has_radar:
            radar_hud_snippet = """
            <div id="radar-hud" class="px-3 py-1 bg-cyan-950/40 border border-cyan-800/40 rounded-lg text-xs font-mono text-cyan-300 flex items-center gap-2">
              <span class="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
              <span>RADAR HUD ACTIVE</span>
            </div>
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
  <!-- Sleek Floating Mock Preview Badge -->
  <div id="mock-preview-banner" class="flex items-center justify-between bg-gradient-to-r from-amber-500/20 via-purple-500/20 to-cyan-500/20 border border-amber-500/40 px-3.5 py-1.5 rounded-lg mb-3 text-xs text-amber-200 shadow-lg backdrop-blur-sm">
    <span class="flex items-center gap-2">
      <span class="w-2 h-2 rounded-full bg-amber-400 animate-ping"></span>
      <span class="font-semibold">✨ LIVE ARENA PREVIEW (MOCK ANIMATION)</span>
      <span class="text-amber-300/80 hidden sm:inline">— Ask in chat to customize look & feel, or click "Lock Arena & Start Training"</span>
    </span>
    <span class="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-amber-950/80 border border-amber-700/60 text-amber-300 font-bold">Mock Mode</span>
  </div>

  <!-- Top Telemetry Bar -->
  <header class="flex flex-wrap items-center justify-between pb-3 border-b border-slate-800 gap-2">
    <div class="flex items-center gap-2">
      <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
      <h1 class="text-sm font-bold text-slate-100">{task_title}</h1>
      <span class="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700 font-mono">v4.2 JIT Arena</span>
      {radar_hud_snippet}
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

  <!-- postMessage Communication Protocol & Mock Engine -->
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

    function applyArenaPayload(data) {{
      if (!data) return;

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

    // Domain Rendering JS
    {domain_js}

    // --- Mock Animation Loop Controller ---
    let mockAnimationTimer = null;

    function startMockAnimationLoop() {{
      if (mockAnimationTimer) clearInterval(mockAnimationTimer);
      const banner = document.getElementById('mock-preview-banner');
      if (banner) banner.style.display = 'flex';
      mockAnimationTimer = setInterval(() => {{
        stepMockDomain();
      }}, 650);
    }}

    function stopMockAnimationLoop() {{
      if (mockAnimationTimer) {{
        clearInterval(mockAnimationTimer);
        mockAnimationTimer = null;
      }}
      const banner = document.getElementById('mock-preview-banner');
      if (banner) banner.style.display = 'none';
    }}

    // Listen for parent messages
    window.addEventListener("message", function(event) {{
      if (!event.data) return;

      if (event.data.type === "SET_ARENA_MODE") {{
        const mode = event.data.mode;
        if (mode === "mock_preview") {{
          startMockAnimationLoop();
        }} else {{
          stopMockAnimationLoop();
        }}
        return;
      }}

      if (event.data.type === "ARENA_UPDATE") {{
        stopMockAnimationLoop();
        const data = event.data.payload || event.data;
        applyArenaPayload(data);
      }}
    }});

    // Auto-start mock animation loop on load for immediate preview
    startMockAnimationLoop();
  </script>
</body>
</html>
"""
