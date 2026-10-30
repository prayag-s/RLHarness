"""Dynamic JIT Arena Synthesizer for RL-Native Agent IDE v4.4.

Universal LLM and Schema-Driven Arena UI generator creating self-contained,
interactive HTML5/Canvas/Tailwind arena visualizations rendered in a sandboxed iframe.
Eliminates all domain-specific branches, rover fallbacks, and hardcoded templates.
"""

from __future__ import annotations
import html as html_lib
import json
import logging
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

from rl_ide.architect.models import AgentBlueprintSpec, DynamicTaskBlueprint
from rl_ide.llm.client import BaseLLMClient, FallbackCodeSynthesizer, get_active_llm_client
from rl_ide.llm.prompts import ARENA_SYNTHESIZER_SYSTEM_PROMPT, ARENA_BUILDER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# A full bespoke arena is a single ~30KB HTML document, so generation routinely needs several
# minutes on high-latency providers. Short timeouts silently degrade the arena to the generic
# text fallback, so this is a floor rather than a default.
ARENA_GENERATION_TIMEOUT_S = 240.0


def _arena_generation_timeout(client: BaseLLMClient) -> float:
    """Returns the arena synthesis timeout, never below the client's configured request timeout."""
    configured = getattr(getattr(client, "config", None), "timeout", None)
    try:
        configured = float(configured)
    except (TypeError, ValueError):
        configured = 0.0
    return max(ARENA_GENERATION_TIMEOUT_S, configured)


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
            except FileNotFoundError:
                # Without a JS engine we cannot prove the scripts parse: do not claim success.
                return False
            except subprocess.TimeoutExpired:
                return False
            except Exception:
                return False
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
            raw = active_client.generate(
                prompt,
                system_prompt=ARENA_BUILDER_SYSTEM_PROMPT,
                timeout=_arena_generation_timeout(active_client),
            )
            html = extract_html_code(raw)
            if html and "ARENA_UPDATE" in html and "startMockAnimationLoop" in html and ("<html" in html.lower() or "<!doctype html" in html.lower()):
                healed = heal_arena_html_scripts(html)
                if validate_arena_html_scripts(healed):
                    return healed
                logger.warning(
                    "LLM arena HTML failed JavaScript validation; falling back to the generic arena preview."
                )
            else:
                logger.warning(
                    "LLM arena HTML is missing the required preview hooks; falling back to the generic arena preview."
                )
        except Exception as exc:
            logger.warning(
                "LLM arena synthesis failed (%s: %s); falling back to the generic arena preview.",
                type(exc).__name__,
                exc,
            )

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
                    f"Current Arena HTML:\n{current_arena_html if current_arena_html else ''}\n\n"
                    f"Blueprint: {bp_spec.model_dump_json(indent=2)}\n\n"
                    f"REQUIREMENTS:\n"
                    f"1. Apply the user's visual styling (e.g. colors, scoreboard, badges, layouts, sizing, themes) and interactive features.\n"
                    f"2. PRESERVE `startMockAnimationLoop()`, `SET_ARENA_MODE`, `ARENA_UPDATE`, and `ARENA_INTERACT` functionality.\n"
                    f"3. Return ONLY valid HTML inside ```html ... ```."
                )
                raw = self.client.generate(
                    prompt,
                    system_prompt=ARENA_SYNTHESIZER_SYSTEM_PROMPT,
                    timeout=_arena_generation_timeout(self.client),
                )
                extracted = extract_html_code(raw)
                if (
                    extracted
                    and "ARENA_UPDATE" in extracted
                    and "startMockAnimationLoop" in extracted
                    and "sendInteractiveAction" in extracted
                    and "renderDomainState" in extracted
                ):
                    healed = heal_arena_html_scripts(extracted)
                    if validate_arena_html_scripts(healed):
                        if workspace_dir:
                            (Path(workspace_dir) / "arena.html").write_text(healed, encoding="utf-8")
                        return healed
                    logger.warning(
                        "Restyled arena HTML failed JavaScript validation; falling back to the generic arena preview."
                    )
                else:
                    logger.warning(
                        "Restyled arena HTML is missing the required preview hooks; falling back to the generic arena preview."
                    )
            except Exception as exc:
                logger.warning(
                    "LLM arena restyle failed (%s: %s); falling back to the generic arena preview.",
                    type(exc).__name__,
                    exc,
                )

        updated = self._generate_deterministic_arena_html(
            blueprint=bp_spec,
            sample_arena_state=sample_arena_state,
            visual_style=user_visual_request,
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
    ) -> str:
        """Builds a self-contained arena driven only by the blueprint contract and real state.

        No domain branches: visuals come from the declared action labels, the actual
        ``get_render_state()`` sample, and elements the user explicitly requested.
        """
        task_title = blueprint.task_title or "RL Arena"
        action_labels = blueprint.action_labels or ["Action 0", "Action 1"]
        target_threshold = blueprint.target_metric_threshold

        # Restyle state is explicit: only the current request and the blueprint metaphor.
        style_query = f"{visual_style or ''} {blueprint.arena_visual_description or ''}".lower()

        has_scoreboard = "scoreboard" in style_query or "score" in style_query
        has_gold = "gold" in style_query or "golden" in style_query or "amber" in style_query
        has_arcade = "arcade" in style_query or "retro" in style_query
        has_larger = "larger" in style_query or "bigger" in style_query
        accent_color = "amber" if has_gold else "cyan"
        theme_class = "font-mono" if has_arcade else "font-sans"

        # User-requested UI elements are rendered generically (text is escaped).
        requested: list[str] = []
        for match in re.finditer(
            r"(?:add|show|display|render|include|put)\s+(?:a|an)?\s*(.{3,60}?)\s+"
            r"(badge|label|meter|gauge|banner|panel|card|indicator)\b",
            visual_style or "",
            re.IGNORECASE,
        ):
            text = match.group(1).strip(" .")
            requested.append(f"{text} ({match.group(2)})")
        for quoted in re.findall(r"['\"]([^'\"]{3,60})['\"]", visual_style or ""):
            if quoted.strip() and quoted.strip() not in requested:
                requested.append(quoted.strip())
        requested = requested[:4]

        requested_html = ""
        if requested:
            chips = "".join(
                f'<span class="px-3 py-1 rounded-full bg-slate-900 border border-{accent_color}-500/50 '
                f'text-{accent_color}-300 text-xs font-mono shadow">{html_lib.escape(item)}</span>'
                for item in requested
            )
            requested_html = (
                f'<div id="requested-ui-elements" class="flex flex-wrap items-center gap-2 my-1">{chips}</div>'
            )

        scoreboard_html = ""
        if has_scoreboard:
            scoreboard_html = f"""
            <aside id="arena-scoreboard" class="bg-slate-900/90 border border-{accent_color}-500/40 rounded-xl p-3 flex flex-col gap-2 min-w-[140px] text-xs font-mono shadow-lg">
              <div class="text-[10px] uppercase font-bold tracking-wider text-{accent_color}-400 border-b border-slate-800 pb-1">Scoreboard</div>
              <div class="flex justify-between items-center"><span class="text-slate-400">Agent:</span><span id="score-agent" class="font-bold text-{accent_color}-300">0</span></div>
              <div class="flex justify-between items-center"><span class="text-slate-400">Opponent:</span><span id="score-opp" class="font-bold text-slate-300">0</span></div>
              <div class="flex justify-between items-center"><span class="text-slate-400">Streak:</span><span id="score-streak" class="font-bold text-emerald-400">0</span></div>
            </aside>
            """

        longest_label = max([len(str(lbl)) for lbl in action_labels] or [0])
        probs_cols = (
            "grid-cols-2 sm:grid-cols-2 md:grid-cols-4" if longest_label > 10
            else "grid-cols-3 sm:grid-cols-5 md:grid-cols-9"
        )

        buttons_html = "".join([
            f'<button onclick="sendInteractiveAction({idx})" class="px-3 py-2 min-h-[42px] bg-slate-900 hover:bg-{accent_color}-600/30 active:scale-95 text-slate-200 hover:text-white border border-slate-700 hover:border-{accent_color}-500 rounded-lg text-[11px] leading-tight font-mono font-medium transition-all shadow whitespace-normal break-words text-center" title="{html_lib.escape(str(lbl))}">{html_lib.escape(str(lbl))}</button>'
            for idx, lbl in enumerate(action_labels)
        ])

        stage_size = "max-w-4xl min-h-[360px]" if has_larger else "max-w-2xl min-h-[280px]"
        sample_state = sample_arena_state or {}
        sample_json = json.dumps(sample_state, default=str).replace("</", "<\\/")
        action_labels_json = json.dumps([str(x) for x in action_labels]).replace("</", "<\\/")
        safe_title = html_lib.escape(str(task_title))
        safe_metaphor = html_lib.escape(str(blueprint.arena_visual_description or task_title))
        # The generic preview cannot draw the bespoke metaphor, so keep it as a one-line caption
        # instead of letting the prose dominate the stage.
        caption_metaphor = safe_metaphor if len(safe_metaphor) <= 110 else safe_metaphor[:107].rstrip() + "…"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{safe_title} — RL Studio Arena</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body {{ background-color: #020617; color: #f8fafc; font-family: ui-sans-serif, system-ui, sans-serif; }}
    .arena-cell {{ transition: all 0.15s ease-out; }}
    .arena-cell:hover {{ transform: scale(1.04); }}
  </style>
</head>
<body class="p-3 bg-slate-950 text-slate-100 min-h-screen flex flex-col justify-between select-none {theme_class} {'arcade-theme' if has_arcade else ''}">

  <div id="mock-preview-banner" class="flex items-center justify-between px-3 py-1.5 rounded-lg bg-amber-950/70 border border-amber-600/60 text-amber-200 text-xs shadow-md mb-2">
    <span class="flex items-center gap-2">
      <span class="w-2 h-2 rounded-full bg-amber-400 animate-ping"></span>
      <span class="font-semibold">✨ LIVE ARENA PREVIEW (MOCK ANIMATION)</span>
      <span class="text-amber-300/80 hidden sm:inline">— ask in chat to customize, or lock the arena to train</span>
    </span>
    <span class="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-amber-950/80 border border-amber-700/60 text-amber-300 font-bold">Mock Mode</span>
  </div>

  <header class="flex flex-wrap items-center justify-between pb-2 border-b border-slate-800 gap-2">
    <div class="flex items-center gap-2">
      <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
      <h1 class="text-sm font-bold text-slate-100">{safe_title}</h1>
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

  {requested_html}

  <section class="py-1.5">
    <div class="text-[10px] font-semibold uppercase tracking-wider text-slate-500 mb-1">Live Reward Breakdown</div>
    <div id="reward-breakdown-bar" class="flex flex-wrap gap-1.5 text-xs font-mono">
      <span class="text-slate-600 italic">Waiting for step telemetry...</span>
    </div>
  </section>

  <main id="domain-arena-stage" class="flex-1 flex items-center justify-center gap-4 py-2">
    {scoreboard_html}

    <div class="flex-1 flex flex-col items-center justify-center bg-slate-900/60 border border-slate-800 rounded-2xl p-4 {stage_size} w-full shadow-2xl relative overflow-hidden">
      <div id="visual-stage-container" class="w-full flex-1 flex flex-col items-center justify-center gap-3">
        <div id="visual-metaphor-header" class="text-[10px] font-mono text-slate-500 uppercase tracking-widest text-center max-w-md truncate" title="{safe_metaphor}">{caption_metaphor}</div>
        <div id="dynamic-state-grid" class="flex flex-wrap items-center justify-center gap-2 max-w-lg w-full"></div>
        <div id="status-badge" class="px-3 py-1 bg-slate-800 border border-slate-700 rounded-full text-xs font-mono text-slate-300">Ready</div>
      </div>

      <div class="w-full mt-3 pt-3 border-t border-slate-800/80">
        <div class="text-[10px] uppercase font-mono text-slate-500 mb-1.5">Interactive Controls</div>
        <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {buttons_html}
        </div>
      </div>
    </div>
  </main>

  <footer class="pt-2 border-t border-slate-800">
    <div class="flex justify-between items-center text-[10px] uppercase tracking-wider text-slate-500 mb-1">
      <span>Action Probabilities (π)</span>
      <span id="selected-action-tag" class="text-{accent_color}-400 font-bold font-mono"></span>
    </div>
    <div id="action-probs-bar" class="grid {probs_cols} gap-1.5 text-[10px] font-mono"></div>
  </footer>

  <script>
    const ACTION_LABELS = {action_labels_json};
    let currentRenderState = {sample_json};
    let mockAnimationTimer = null;
    let mockStepCount = 0;

    const probsBar = document.getElementById('action-probs-bar');
    if (probsBar && ACTION_LABELS.length > 0) {{
      probsBar.innerHTML = ACTION_LABELS.map((lbl, idx) => `
        <div class="bg-slate-900 border border-slate-800 p-1.5 rounded flex flex-col gap-1">
          <div class="break-words leading-tight text-[9px] text-slate-400" title="${{lbl}}">${{lbl}}</div>
          <div class="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div id="prob-bar-${{idx}}" class="bg-{accent_color}-500 h-full transition-all duration-150" style="width: 0%"></div>
          </div>
          <div id="prob-val-${{idx}}" class="text-[9px] text-right text-slate-500">0%</div>
        </div>
      `).join('');
    }}

    function sendInteraction(payload) {{
      if (window.parent) {{
        window.parent.postMessage({{ type: "ARENA_INTERACT", payload: payload }}, "*");
      }}
    }}

    function sendInteractiveAction(actionIdx) {{
      sendInteraction({{ action: actionIdx, user_action: actionIdx }});
      if (mockAnimationTimer) {{
        stepMockDomain(actionIdx);
      }}
    }}

    function flattenStateArray(value) {{
      if (!Array.isArray(value)) return null;
      if (value.length > 0 && Array.isArray(value[0])) {{
        const cols = value[0].length;
        return {{ flat: value.flat(), cols: cols }};
      }}
      const n = value.length;
      const cols = n === 9 ? 3 : n === 42 ? 7 : Math.ceil(Math.sqrt(n));
      return {{ flat: value.slice(), cols: cols }};
    }}

    function renderDomainState(state) {{
      if (!state) return;
      currentRenderState = state;
      const grid = document.getElementById('dynamic-state-grid');
      if (!grid) return;

      let arrayKey = null;
      let arrayValue = null;
      for (const [k, v] of Object.entries(state)) {{
        if (Array.isArray(v) && v.length > 0) {{
          arrayKey = k;
          arrayValue = flattenStateArray(v);
          break;
        }}
      }}

      // Only numeric, board-shaped arrays are drawn as clickable cells; text arrays (e.g. action
      // labels) are rendered as readable entries instead of label fragments in tiny squares.
      const cellLike = Boolean(arrayValue) && arrayValue.flat.every(cell => cell === null
        || cell === undefined
        || (typeof cell === 'number' && Number.isFinite(cell))
        || (typeof cell === 'string' && /^[\\d.+-]+$/.test(cell.trim())));
      if (arrayValue && cellLike) {{
        const interactive = arrayValue.flat.length === ACTION_LABELS.length;
        grid.style.display = 'grid';
        grid.style.gridTemplateColumns = `repeat(${{arrayValue.cols}}, minmax(0, 1fr))`;
        grid.innerHTML = arrayValue.flat.map((cell, idx) => {{
          const isMarked = cell !== 0 && cell !== null && cell !== undefined && cell !== '';
          const label = isMarked ? String(cell).slice(0, 4) : '';
          const style = isMarked
            ? 'text-cyan-400 border-cyan-500/50 bg-cyan-950/30'
            : 'border-slate-800 bg-slate-950 hover:border-slate-700';
          const click = interactive ? ` onclick="sendInteractiveAction(${{idx}})"` : '';
          const cursor = interactive ? ' cursor-pointer' : '';
          return `<div${{click}} class="w-10 h-10 sm:w-12 sm:h-12 border rounded-lg flex items-center justify-center font-bold text-sm arena-cell${{cursor}} ${{style}}">${{label}}</div>`;
        }}).join('');
      }} else if (arrayValue) {{
        // Text/object arrays are labels, not board cells: show them in full as chips.
        grid.style.display = 'flex';
        grid.innerHTML = arrayValue.flat.slice(0, 12)
          .map((cell) => `<div class="px-3 py-1.5 bg-slate-900 border border-slate-800 rounded-lg text-xs font-mono text-slate-200 whitespace-nowrap">${{String(cell)}}</div>`)
          .join('');
      }} else {{
        grid.style.display = 'flex';
        grid.innerHTML = Object.entries(state)
          .filter(([k]) => !['status', 'step_count'].includes(k))
          .map(([k, v]) => `
            <div class="px-3 py-1.5 bg-slate-900 border border-slate-800 rounded-lg text-xs font-mono flex flex-col items-center">
              <span class="text-slate-500 text-[10px] uppercase truncate max-w-[110px]">${{k}}</span>
              <span class="text-{accent_color}-400 font-bold">${{typeof v === 'number' ? Number(v).toFixed(2) : String(v).slice(0, 18)}}</span>
            </div>
          `).join('');
      }}

      const badge = document.getElementById('status-badge');
      if (badge && state.status) {{
        badge.innerText = String(state.status).toUpperCase();
      }}
    }}

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
      if (data.action_probs && Array.isArray(data.action_probs)) {{
        data.action_probs.forEach((p, idx) => {{
          const bar = document.getElementById('prob-bar-' + idx);
          const val = document.getElementById('prob-val-' + idx);
          const pct = Math.round(p * 100);
          if (bar) bar.style.width = pct + '%';
          if (val) val.innerText = pct + '%';
        }});
      }}
      if (data.arena_state) {{
        renderDomainState(data.arena_state);
      }}
    }}

    // Generic preview animation: mutates the first array-valued state field, or numeric fields.
    function stepMockDomain(customAction) {{
      mockStepCount++;
      const stepElem = document.getElementById('telemetry-step');
      if (stepElem) stepElem.innerText = mockStepCount;

      const actIdx = customAction !== undefined ? customAction : Math.floor(Math.random() * Math.max(1, ACTION_LABELS.length));
      const tag = document.getElementById('selected-action-tag');
      if (tag && ACTION_LABELS[actIdx]) tag.innerText = ACTION_LABELS[actIdx];

      if (!currentRenderState) return;
      const next = {{ ...currentRenderState }};
      let mutated = false;
      for (const [k, v] of Object.entries(next)) {{
        if (Array.isArray(v) && v.length > 0) {{
          const flatInfo = flattenStateArray(v);
          if (Array.isArray(v[0])) {{
            next[k] = v.map(row => row.slice());
            const rows = next[k];
            const empty = [];
            rows.forEach((row, r) => row.forEach((cell, c) => {{ if (cell === 0) empty.push([r, c]); }}));
            if (empty.length > 0) {{
              const [r, c] = empty[Math.floor(Math.random() * empty.length)];
              rows[r][c] = (mockStepCount % 2 === 0) ? 1 : 2;
            }} else {{
              rows.forEach(row => row.fill(0));
            }}
          }} else {{
            const arr = v.slice();
            const emptyIdx = arr.map((cell, i) => cell === 0 ? i : -1).filter(i => i !== -1);
            if (emptyIdx.length > 0) {{
              arr[emptyIdx[Math.floor(Math.random() * emptyIdx.length)]] = (mockStepCount % 2 === 0) ? 1 : 2;
            }} else {{
              arr.fill(0);
            }}
            next[k] = arr;
          }}
          mutated = true;
          break;
        }}
      }}
      if (!mutated) {{
        for (const [k, v] of Object.entries(next)) {{
          if (typeof v === 'number' && k !== 'step_count') {{
            next[k] = Number((v + (Math.random() - 0.5) * 0.2).toFixed(2));
          }}
        }}
      }}
      renderDomainState(next);
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

    renderDomainState(currentRenderState);
    startMockAnimationLoop();
  </script>
</body>
</html>
"""
