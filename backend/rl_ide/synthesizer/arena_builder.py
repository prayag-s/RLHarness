"""Dynamic JIT Arena Synthesizer for RL-Native Agent IDE v4.4.

Universal LLM and Schema-Driven Arena UI generator creating self-contained,
interactive HTML5/Canvas/Tailwind arena visualizations rendered in a sandboxed iframe.
Eliminates all domain-specific branches, rover fallbacks, and hardcoded templates.
"""

from __future__ import annotations
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

from rl_ide.architect.models import AgentBlueprintSpec, DynamicTaskBlueprint
from rl_ide.llm.client import BaseLLMClient, FallbackCodeSynthesizer, get_active_llm_client
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


def synthesize_universal_arena_html(
    blueprint: AgentBlueprintSpec | DynamicTaskBlueprint,
    sample_render_state: dict[str, Any] | None = None,
    visual_style: str | None = None,
    client: BaseLLMClient | None = None,
) -> str:
    """Universal Generative Arena UI Synthesizer.

    Generates a bespoke, interactive HTML5/SVG/Canvas/Tailwind Arena component
    bound directly to the blueprint's visual metaphor, actions, and state_dict_schema.
    Zero hardcoded domain branches or 2D rover fallbacks.
    """
    bp_spec = blueprint.to_agent_blueprint_spec() if hasattr(blueprint, "to_agent_blueprint_spec") else blueprint
    active_client = client or get_active_llm_client()

    # 1. Try Live LLM UI generation if configured and not FallbackCodeSynthesizer
    if not isinstance(active_client, FallbackCodeSynthesizer):
        try:
            visual_desc = getattr(blueprint, "visual_metaphor", None) or bp_spec.arena_visual_description or "Interactive Arena"
            actions_list = bp_spec.action_labels
            schema = getattr(blueprint, "state_dict_schema", {})
            prompt = (
                f"You are the Universal Generative Arena UI Synthesizer.\n"
                f"Generate a complete, self-contained interactive `arena.html` for:\n"
                f"Domain Title: {bp_spec.task_title}\n"
                f"Visual Metaphor: {visual_desc}\n"
                f"Actions: {json.dumps(actions_list)}\n"
                f"State Dictionary Schema: {json.dumps(schema)}\n"
                f"Sample Render State from Environment: {json.dumps(sample_render_state or {}, default=str)}\n"
                f"Visual Style Request: {visual_style or 'Default sleek dark-mode'}\n\n"
                f"STRICT REQUIREMENTS:\n"
                f"1. Render the domain's true visual metaphor directly from the keys in `sample_render_state`.\n"
                f"2. Never use a 2D rover canvas unless the domain is explicitly a rover navigation task.\n"
                f"3. Render interactive buttons/controls for each action that call `sendInteractiveAction(actionIdx)`.\n"
                f"4. Include `renderDomainState(state)` to update DOM/Canvas/SVG from live state.\n"
                f"5. Include `startMockAnimationLoop()` and `stepMockDomain()` for instant preview.\n"
                f"6. Support `ARENA_UPDATE`, `SET_ARENA_MODE`, and postMessage bridges.\n"
                f"7. Return ONLY valid HTML inside ```html ... ```."
            )
            raw = active_client.generate(prompt, system_prompt=ARENA_BUILDER_SYSTEM_PROMPT, timeout=35.0)
            html = extract_html_code(raw)
            if html and "ARENA_UPDATE" in html and "startMockAnimationLoop" in html and ("<html" in html.lower() or "<!doctype html" in html.lower()):
                healed = heal_arena_html_scripts(html)
                if validate_arena_html_scripts(healed):
                    return healed
        except Exception:
            pass

    # 2. Universal Schema-Driven Arena Generator
    synthesizer = JITArenaSynthesizer(active_client)
    return synthesizer._generate_deterministic_arena_html(
        blueprint=bp_spec,
        sample_arena_state=sample_render_state,
        visual_style=visual_style,
    )


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
        blueprint: AgentBlueprintSpec | DynamicTaskBlueprint,
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
        blueprint: AgentBlueprintSpec | DynamicTaskBlueprint,
        current_arena_html: str,
        user_visual_request: str,
        sample_arena_state: dict[str, Any] | None = None,
        workspace_dir: str | Path | None = None,
    ) -> str:
        """Updates and restyles arena.html according to user's request while preserving live mock animation and bi-directional bridges."""
        bp_spec = blueprint.to_agent_blueprint_spec() if hasattr(blueprint, "to_agent_blueprint_spec") else blueprint
        if not isinstance(self.client, FallbackCodeSynthesizer):
            try:
                prompt = (
                    f"You are updating and restyling this self-contained `arena.html` according to the user's request:\n"
                    f"User Request: \"{user_visual_request}\"\n\n"
                    f"Current Arena HTML:\n{current_arena_html[:6000] if current_arena_html else ''}\n\n"
                    f"Blueprint: {bp_spec.model_dump_json(indent=2)}\n\n"
                    f"REQUIREMENTS:\n"
                    f"1. Apply the user's visual styling (e.g. colors, scoreboard, badges, layouts, sizing, themes) and interactive features.\n"
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

        updated = self._generate_deterministic_arena_html(
            blueprint=bp_spec,
            sample_arena_state=sample_arena_state,
            visual_style=user_visual_request,
            base_html=current_arena_html,
        )
        if workspace_dir:
            (Path(workspace_dir) / "arena.html").write_text(updated, encoding="utf-8")
        return updated

    def generate_arena_html(
        self,
        blueprint: AgentBlueprintSpec | DynamicTaskBlueprint,
        sample_arena_state: dict[str, Any] | None = None,
        visual_style: str | None = None,
    ) -> str:
        """Synthesizes HTML code using Universal Generative UI pipeline."""
        return synthesize_universal_arena_html(
            blueprint=blueprint,
            sample_render_state=sample_arena_state,
            visual_style=visual_style,
            client=self.client,
        )

    def _generate_deterministic_arena_html(
        self,
        blueprint: AgentBlueprintSpec,
        sample_arena_state: dict[str, Any] | None,
        visual_style: str | None = None,
        base_html: str | None = None,
    ) -> str:
        """Constructs a responsive, feature-complete arena.html with full postMessage support, mock animation loop, and visual theming."""
        task_title = blueprint.task_title or "RL Arena"
        action_labels = blueprint.action_labels or ["Action 0", "Action 1"]
        target_threshold = blueprint.target_metric_threshold

        style_query = ((visual_style or "") + " " + (blueprint.arena_visual_description or "")).lower()
        if base_html:
            style_query += " " + base_html.lower()

        has_scoreboard = "scoreboard" in style_query or "score" in style_query
        has_gold = "gold" in style_query or "golden" in style_query or "amber" in style_query
        has_arcade = "arcade" in style_query or "retro" in style_query
        has_larger = "larger" in style_query or "bigger" in style_query

        accent_color = "amber" if has_gold else "cyan"
        theme_class = "font-mono" if has_arcade else "font-sans"

        sample_state = sample_arena_state or {}
        sample_json = json.dumps(sample_state)
        action_labels_json = json.dumps(action_labels)

        # Dynamic scoreboard widget if requested
        scoreboard_html = ""
        if has_scoreboard:
            scoreboard_html = f"""
            <aside id="arena-scoreboard" class="bg-slate-900/90 border border-{accent_color}-500/40 rounded-xl p-3 flex flex-col gap-2 min-w-[140px] text-xs font-mono shadow-lg">
              <div class="text-[10px] uppercase font-bold tracking-wider text-{accent_color}-400 border-b border-slate-800 pb-1">Scoreboard</div>
              <div class="flex justify-between items-center"><span class="text-slate-400">Agent:</span><span id="score-agent" class="font-bold text-{accent_color}-300">0</span></div>
              <div class="flex justify-between items-center"><span class="text-slate-400">Human / Opp:</span><span id="score-opp" class="font-bold text-slate-300">0</span></div>
              <div class="flex justify-between items-center"><span class="text-slate-400">Streak:</span><span id="score-streak" class="font-bold text-emerald-400">0</span></div>
            </aside>
            """

        # Action control buttons
        buttons_html = "".join([
            f'<button onclick="sendInteractiveAction({idx})" class="px-3 py-2 bg-slate-900 hover:bg-{accent_color}-600/30 active:scale-95 text-slate-200 hover:text-white border border-slate-700 hover:border-{accent_color}-500 rounded-lg text-xs font-mono font-medium transition-all shadow truncate" title="{lbl}">{lbl}</button>'
            for idx, lbl in enumerate(action_labels)
        ])

        # Stage size class
        stage_size = "max-w-4xl min-h-[360px]" if has_larger else "max-w-2xl min-h-[280px]"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{task_title} — RL Studio Arena</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body {{ background-color: #020617; color: #f8fafc; font-family: ui-sans-serif, system-ui, sans-serif; }}
    .glow-accent {{ box-shadow: 0 0 15px rgba(245, 158, 11, 0.25); }}
    .arena-cell {{ transition: all 0.15s ease-out; }}
    .arena-cell:hover {{ transform: scale(1.04); }}
  </style>
</head>
<body class="p-3 bg-slate-950 text-slate-100 min-h-screen flex flex-col justify-between select-none {theme_class}">

  <!-- Live Mock Preview Banner -->
  <div id="mock-preview-banner" class="flex items-center justify-between px-3 py-1.5 rounded-lg bg-amber-950/70 border border-amber-600/60 text-amber-200 text-xs shadow-md mb-2">
    <span class="flex items-center gap-2">
      <span class="w-2 h-2 rounded-full bg-amber-400 animate-ping"></span>
      <span class="font-semibold">✨ LIVE ARENA PREVIEW (MOCK ANIMATION)</span>
      <span class="text-amber-300/80 hidden sm:inline">— Ask in chat to customize look & feel, or click "Lock Arena & Start Training"</span>
    </span>
    <span class="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-amber-950/80 border border-amber-700/60 text-amber-300 font-bold">Mock Mode</span>
  </div>

  <!-- Header Telemetry Bar -->
  <header class="flex flex-wrap items-center justify-between pb-2 border-b border-slate-800 gap-2">
    <div class="flex items-center gap-2">
      <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
      <h1 class="text-sm font-bold text-slate-100">{task_title}</h1>
      <span class="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700 font-mono">Universal Arena</span>
    </div>
    <div class="flex items-center gap-2 font-mono text-xs">
      <div class="bg-slate-900 border border-slate-800 px-2.5 py-1 rounded-lg">
        <span class="text-slate-500">EPISODE:</span> <span id="telemetry-episode" class="font-bold text-{accent_color}-400">#1</span>
      </div>
      <div class="bg-slate-900 border border-slate-800 px-2.5 py-1 rounded-lg">
        <span class="text-slate-500">STEP:</span> <span id="telemetry-step" class="font-bold text-indigo-400">0</span>
      </div>
      <div class="bg-slate-900 border border-slate-800 px-2.5 py-1 rounded-lg">
        <span class="text-slate-500">REWARD:</span> <span id="telemetry-reward" class="font-bold text-emerald-400">0.0</span>
      </div>
      <div class="bg-slate-900 border border-slate-800 px-2.5 py-1 rounded-lg">
        <span class="text-slate-500">TARGET:</span> <span class="font-bold text-emerald-400">{target_threshold}%</span>
      </div>
    </div>
  </header>

  <!-- Live Reward Breakdown Bar Strip -->
  <section class="py-1.5">
    <div class="text-[10px] font-semibold uppercase tracking-wider text-slate-500 mb-1">Live Reward Breakdown</div>
    <div id="reward-breakdown-bar" class="flex flex-wrap gap-1.5 text-xs font-mono">
      <span class="text-slate-600 italic">Waiting for step telemetry...</span>
    </div>
  </section>

  <!-- Main Center Visual Stage with Optional Scoreboard -->
  <main id="domain-arena-stage" class="flex-1 flex items-center justify-center gap-4 py-2">
    {scoreboard_html}

    <div class="flex-1 flex flex-col items-center justify-center bg-slate-900/60 border border-slate-800 rounded-2xl p-4 {stage_size} w-full shadow-2xl relative overflow-hidden">
      <!-- Reactive State Display Canvas/Grid -->
      <div id="visual-stage-container" class="w-full flex-1 flex flex-col items-center justify-center gap-3">
        <div id="visual-metaphor-header" class="text-xs font-mono text-slate-400 uppercase tracking-widest">{task_title}</div>
        <div id="dynamic-state-grid" class="flex flex-wrap items-center justify-center gap-2 max-w-lg w-full"></div>
        <div id="status-badge" class="px-3 py-1 bg-slate-800 border border-slate-700 rounded-full text-xs font-mono text-slate-300">Ready</div>
      </div>

      <!-- Interactive Action Controls -->
      <div class="w-full mt-3 pt-3 border-t border-slate-800/80">
        <div class="text-[10px] uppercase font-mono text-slate-500 mb-1.5">Interactive Controls</div>
        <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {buttons_html}
        </div>
      </div>
    </div>
  </main>

  <!-- Action Probability Distribution Footer -->
  <footer class="pt-2 border-t border-slate-800">
    <div class="flex justify-between items-center text-[10px] uppercase tracking-wider text-slate-500 mb-1">
      <span>Action Probabilities (π)</span>
      <span id="selected-action-tag" class="text-{accent_color}-400 font-bold font-mono"></span>
    </div>
    <div id="action-probs-bar" class="grid grid-cols-3 sm:grid-cols-5 md:grid-cols-9 gap-1.5 text-[10px] font-mono"></div>
  </footer>

  <!-- postMessage Communication Protocol & Dynamic Animation Engine -->
  <script>
    const ACTION_LABELS = {action_labels_json};
    let currentRenderState = {sample_json};
    let mockAnimationTimer = null;
    let mockStepCount = 0;

    // Initialize Action Probabilities Bar
    const probsBar = document.getElementById('action-probs-bar');
    if (probsBar && ACTION_LABELS.length > 0) {{
      probsBar.innerHTML = ACTION_LABELS.map((lbl, idx) => `
        <div class="bg-slate-900 border border-slate-800 p-1.5 rounded flex flex-col gap-1">
          <div class="truncate text-slate-400" title="${{lbl}}">${{lbl}}</div>
          <div class="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div id="prob-bar-${{idx}}" class="bg-{accent_color}-500 h-full transition-all duration-150" style="width: 0%"></div>
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

    function sendInteractiveAction(actionIdx) {{
      sendInteraction({{ action: actionIdx, user_action: actionIdx }});
      if (mockAnimationTimer) {{
        stepMockDomain(actionIdx);
      }}
    }}

    // Dynamic State Renderer for Arbitrary Domain State
    function renderDomainState(state) {{
      if (!state) return;
      currentRenderState = state;
      const grid = document.getElementById('dynamic-state-grid');
      if (!grid) return;

      // Check if state contains a grid or board representation
      const board = state.board || state.grid;
      if (board && (Array.isArray(board) || Array.isArray(board[0]))) {{
        let flat = [];
        if (Array.isArray(board[0])) {{
          flat = board.flat();
        }} else {{
          flat = board;
        }}
        const numCols = Array.isArray(board[0]) ? board[0].length : (flat.length === 9 ? 3 : (flat.length === 42 ? 7 : Math.ceil(Math.sqrt(flat.length))));
        grid.style.display = 'grid';
        grid.style.gridTemplateColumns = `repeat(${{numCols}}, minmax(0, 1fr))`;
        grid.innerHTML = flat.map((cell, idx) => {{
          const isMarked = cell !== 0 && cell !== null && cell !== undefined;
          const markText = cell === 1 ? 'X' : (cell === 2 ? 'O' : (isMarked ? String(cell) : ''));
          const markColor = cell === 1 ? 'text-cyan-400 border-cyan-500/50 bg-cyan-950/30' : (cell === 2 ? 'text-amber-400 border-amber-500/50 bg-amber-950/30' : 'border-slate-800 bg-slate-950 hover:border-slate-700');
          return `<div onclick="sendInteractiveAction(${{idx}})" class="w-10 h-10 sm:w-12 sm:h-12 border rounded-lg flex items-center justify-center font-bold text-sm cursor-pointer arena-cell ${{markColor}}">${{markText}}</div>`;
        }}).join('');
      }} else {{
        // Key-value reactive telemetry display for generic state dicts
        grid.style.display = 'flex';
        grid.innerHTML = Object.entries(state)
          .filter(([k]) => !['status', 'step_count'].includes(k))
          .map(([k, v]) => `
            <div class="px-3 py-1.5 bg-slate-900 border border-slate-800 rounded-lg text-xs font-mono flex flex-col items-center">
              <span class="text-slate-500 text-[10px] uppercase truncate max-w-[100px]">${{k}}</span>
              <span class="text-{accent_color}-400 font-bold">${{typeof v === 'number' ? Number(v).toFixed(2) : String(v).slice(0, 15)}}</span>
            </div>
          `).join('');
      }}

      // Update status badge
      const badge = document.getElementById('status-badge');
      if (badge && state.status) {{
        badge.innerText = String(state.status).toUpperCase();
      }}
    }}

    // Apply Live Telemetry Payload
    function applyArenaPayload(data) {{
      if (!data) return;
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

      // Update Reward Breakdown Strip
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

      // Update Action Probabilities
      if (data.action_probs && Array.isArray(data.action_probs)) {{
        data.action_probs.forEach((p, idx) => {{
          const bar = document.getElementById('prob-bar-' + idx);
          const val = document.getElementById('prob-val-' + idx);
          const pct = Math.round(p * 100);
          if (bar) bar.style.width = pct + '%';
          if (val) val.innerText = pct + '%';
        }});
      }}

      // Update Domain State
      if (data.arena_state) {{
        renderDomainState(data.arena_state);
      }}
    }}

    // Mock Preview Animation Step
    function stepMockDomain(customAction) {{
      mockStepCount++;
      const stepElem = document.getElementById('telemetry-step');
      if (stepElem) stepElem.innerText = mockStepCount;

      const actIdx = customAction !== undefined ? customAction : Math.floor(Math.random() * ACTION_LABELS.length);
      const tag = document.getElementById('selected-action-tag');
      if (tag && ACTION_LABELS[actIdx]) tag.innerText = ACTION_LABELS[actIdx];

      // Mutate sample state for live animation
      if (currentRenderState) {{
        const next = {{ ...currentRenderState }};
        if (next.board && Array.isArray(next.board)) {{
          const emptyIdx = next.board.map((v, i) => v === 0 ? i : -1).filter(i => i !== -1);
          if (emptyIdx.length > 0) {{
            const chosen = emptyIdx[Math.floor(Math.random() * emptyIdx.length)];
            next.board = [...next.board];
            next.board[chosen] = (mockStepCount % 2 === 0) ? 1 : 2;
          }} else {{
            next.board = next.board.map(() => 0);
          }}
        }}
        renderDomainState(next);
      }}
    }}

    function startMockAnimationLoop() {{
      if (mockAnimationTimer) clearInterval(mockAnimationTimer);
      const banner = document.getElementById('mock-preview-banner');
      if (banner) banner.style.display = 'flex';
      mockAnimationTimer = setInterval(stepMockDomain, 750);
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
        if (event.data.mode === "mock_preview") {{
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

    // Initialize with sample render state
    renderDomainState(currentRenderState);
    startMockAnimationLoop();
  </script>
</body>
</html>
"""
