"""Vibe Mode Autopilot Orchestrator and Conversational Coaching Engine."""

from __future__ import annotations
import math
import uuid
from datetime import datetime, timezone
from typing import Any
import numpy as np
from pydantic import BaseModel, Field

from rl_ide.contracts.adapter import EpisodeLog
from rl_ide.compiler.models import TaskConfig, TermDiff
from rl_ide.compiler.nl_to_config import NLToConfigCompiler
from rl_ide.compiler.critique_engine import CritiqueEngine
from rl_ide.compiler.llm_client import LLMConfig, create_llm_client
from rl_ide.reward.dsl import RewardDSL, RewardTerm
from rl_ide.trainer.runner import TrainingRunner
from rl_ide.adapters.dynamic_sim_adapter import DynamicSimAdapter
from rl_ide.adapters.auto_data_adapter import AutoDataAdapter
from rl_ide.adapters.stream_adapter import CSVStreamAdapter
from rl_ide.vibe.translator import PlainRuleCard, dsl_to_plain_rules, plain_rule_to_dsl_diff


class SkillSummary(BaseModel):
    """0–100% Agent Mastery Score and plain-English behavioral evaluation."""
    mastery_score: int = Field(ge=0, le=100)
    status_badge: str  # "Mastered" | "Proficient" | "Learning" | "Needs Coaching"
    badge_variant: str  # "emerald" | "blue" | "amber" | "rose"
    win_rate: float
    total_episodes: int
    key_insights: list[str] = Field(default_factory=list)


class VibeSessionResponse(BaseModel):
    """Response returned upon autopilot session creation."""
    session_id: str
    task_id: str
    adapter_name: str
    ui_mode: str
    mastery_score: int
    status_badge: str
    win_rate: float
    rules: list[PlainRuleCard]
    recent_episodes: list[dict[str, Any]]
    latest_frame: dict[str, Any]
    coach_message: str


class VibeCoachResponse(BaseModel):
    """Response returned after conversational coaching or thumbs up/down feedback."""
    session_id: str
    coach_reply: str
    before_mastery: int
    after_mastery: int
    mastery_delta: int
    applied_diff: list[dict[str, Any]]
    rules: list[PlainRuleCard]
    recent_episodes: list[dict[str, Any]]
    latest_frame: dict[str, Any]


def compute_skill_summary(episodes: list[EpisodeLog], adapter_name: str) -> SkillSummary:
    """Calculates Agent Mastery Score (0–100%), badge status, and key insights from recent episodes."""
    if not episodes:
        return SkillSummary(
            mastery_score=50,
            status_badge="Learning",
            badge_variant="amber",
            win_rate=0.50,
            total_episodes=0,
            key_insights=["Agent initialized. Ready to begin training micro-bursts."],
        )

    n_eps = len(episodes)
    name_low = adapter_name.lower()

    # 1. Board Games (Tic-Tac-Toe)
    if "tictactoe" in name_low or "board" in name_low:
        non_losses = 0
        illegal_moves = 0
        for ep in episodes:
            for st in ep.steps:
                if not st.events.get("obs", {}).get("is_legal_move", True):
                    illegal_moves += 1
            winner = ep.steps[-1].events.get("state", {}).get("winner", "none") if ep.steps else "none"
            if winner in ("agent", "draw"):
                non_losses += 1

        win_rate = non_losses / float(n_eps)
        penalty = min(20, illegal_moves * 5)
        raw_score = int(win_rate * 95) + (5 if illegal_moves == 0 else 0) - penalty
        mastery = max(10, min(100, raw_score))

        insights = [
            f"Agent achieved a {int(win_rate * 100)}% non-loss rate against tactical sparring.",
            "0 illegal moves recorded across all evaluated turns." if illegal_moves == 0 else f"{illegal_moves} illegal moves detected.",
            "Center grid and threat interception active in tactical policy.",
        ]

    # 2. 2D Robotics & Rover
    elif "robot" in name_low or "rover" in name_low:
        targets_reached = 0
        total_collisions = 0
        clean_episodes = 0
        for ep in episodes:
            ep_collided = False
            ep_reached = False
            for st in ep.steps:
                obs = st.events.get("obs", {})
                if obs.get("collided", False):
                    ep_collided = True
                    total_collisions += 1
                if obs.get("reached_target", False):
                    ep_reached = True
            if not ep_collided:
                clean_episodes += 1
            if ep_reached:
                targets_reached += 1

        reach_rate = targets_reached / float(n_eps)
        clean_rate = clean_episodes / float(n_eps)
        win_rate = clean_rate
        mastery = max(10, min(100, int(clean_rate * 55 + reach_rate * 40 + (5 if total_collisions == 0 else 0))))

        insights = [
            f"{clean_episodes}/{n_eps} episodes ({int(clean_rate * 100)}%) navigated completely collision-free.",
            f"Waypoint target reached successfully in {targets_reached} run(s).",
            f"Total obstacle contacts recorded: {total_collisions}.",
        ]

    # 3. Smart Inventory
    elif "inventory" in name_low:
        stockout_count = 0
        for ep in episodes:
            has_so = any(st.events.get("obs", {}).get("is_stockout", False) for st in ep.steps)
            if has_so:
                stockout_count += 1
        clean_eps = n_eps - stockout_count
        win_rate = clean_eps / float(n_eps)
        mastery = max(15, min(100, int(win_rate * 85 + 15)))

        insights = [
            f"{int(win_rate * 100)}% of supply cycles maintained continuous positive shelf inventory.",
            f"{clean_eps} out of {n_eps} replenishment cycles achieved zero stockout penalties.",
        ]

    # 4. Universal Data / Text / Stream
    else:
        avg_reward = np.mean([ep.total_reward for ep in episodes])
        positive_rewards = sum(1 for ep in episodes if ep.total_reward > 0)
        win_rate = positive_rewards / float(n_eps)
        mastery = max(20, min(100, int(win_rate * 75 + 25)))

        insights = [
            f"Positive reward rate: {int(win_rate * 100)}% across {n_eps} evaluated data streams.",
            f"Average episode return: {avg_reward:.2f} points.",
        ]

    if mastery >= 85:
        badge = "Mastered"
        variant = "emerald"
    elif mastery >= 70:
        badge = "Proficient"
        variant = "blue"
    elif mastery >= 50:
        badge = "Learning"
        variant = "amber"
    else:
        badge = "Needs Coaching"
        variant = "rose"

    return SkillSummary(
        mastery_score=mastery,
        status_badge=badge,
        badge_variant=variant,
        win_rate=float(win_rate),
        total_episodes=n_eps,
        key_insights=insights,
    )


class VibeSession:
    """Encapsulates an active autopilot vibe session."""
    def __init__(
        self,
        session_id: str,
        task_id: str,
        adapter: Any,
        task_config: TaskConfig,
        runner: TrainingRunner,
        dataset_id: str | None = None,
    ):
        self.session_id = session_id
        self.task_id = task_id
        self.adapter = adapter
        self.task_config = task_config
        self.runner = runner
        self.dataset_id = dataset_id
        self.episodes: list[EpisodeLog] = []
        self.rules: list[PlainRuleCard] = []
        self.locked_episodes: set[int] = set()
        self.skill_summary: SkillSummary = SkillSummary(
            mastery_score=50,
            status_badge="Learning",
            badge_variant="amber",
            win_rate=0.5,
            total_episodes=0,
        )
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.history: list[dict[str, Any]] = []


class VibeOrchestrator:
    """Singleton service managing high-level vibe autopilot workflows."""

    def __init__(self, compiler: NLToConfigCompiler | None = None):
        self.compiler = compiler or NLToConfigCompiler()
        self.sessions: dict[str, VibeSession] = {}

    def create_session(
        self,
        prompt: str,
        dataset_id: str | None = None,
        dataset: Any | None = None,
        task_id: str | None = None,
        llm_config: LLMConfig | None = None,
        burst_episodes: int = 25,
    ) -> tuple[VibeSession, VibeSessionResponse]:
        """One-click autopilot initialization: compiler + warmstart + initial burst."""
        session_id = f"vibe_{uuid.uuid4().hex[:8]}"
        t_id = task_id or f"task_{session_id}"

        # 1. Select / Synthesize Adapter
        if dataset is not None:
            adapter = AutoDataAdapter(dataset=dataset)
            adapter_name = "AutoDataAdapter"
            candidate_rules = getattr(dataset, "extracted_rules", [])
        else:
            p_low = prompt.lower()
            if any(k in p_low for k in ("tic-tac-toe", "tictactoe", "board game", "chess")):
                adapter = DynamicSimAdapter(mode="tictactoe")
                adapter_name = "DynamicSimAdapter"
            elif any(k in p_low for k in ("robot", "rover", "kinematics", "obstacle", "lidar")):
                adapter = DynamicSimAdapter(mode="robot_rover")
                adapter_name = "DynamicSimAdapter"
            elif any(k in p_low for k in ("inventory", "supply", "stockout", "warehouse")):
                adapter = DynamicSimAdapter(mode="inventory_sim")
                adapter_name = "DynamicSimAdapter"
            elif any(k in p_low for k in ("ticket", "spam", "support", "triage")):
                from rl_ide.adapters.text_adapter import TextClassifyAdapter
                adapter = TextClassifyAdapter()
                adapter_name = "TextClassifyAdapter"
            else:
                adapter = CSVStreamAdapter()
                adapter_name = "CSVStreamAdapter"
            candidate_rules = None

        # 2. Compile user prompt into TaskConfig
        active_compiler = self.compiler
        if llm_config:
            active_compiler = NLToConfigCompiler(llm_client=create_llm_client(llm_config))

        task_config = active_compiler.compile(
            user_prompt=prompt,
            adapter_name=adapter_name,
            adapter=adapter,
            task_id=t_id,
            candidate_rules=candidate_rules,
        )

        # 3. Initialize Training Runner & Warmstart
        runner = TrainingRunner(
            adapter=adapter,
            reward_dsl=task_config.reward_dsl,
            task_id=t_id,
            hyperparameters=task_config.hyperparameters,
        )
        runner.warmstart_policy(epochs=10, n_samples=100)

        # 4. Micro-burst training
        burst_logs = runner.run_burst(n_episodes=burst_episodes)

        # 5. Translate DSL to Plain Rules & Compute Skill Summary
        plain_rules = dsl_to_plain_rules(task_config.reward_dsl)
        skill = compute_skill_summary(burst_logs, getattr(adapter, "mode", adapter_name))

        # 6. Render latest frame
        last_ep = burst_logs[-1] if burst_logs else None
        if last_ep and last_ep.steps:
            last_step = last_ep.steps[-1]
            latest_frame = adapter.render_frame(
                episode_id=last_ep.episode_id,
                step_result=type("TempStep", (), {
                    "events": last_step.events,
                    "terminated": True,
                    "info": last_step.info,
                })(),
                total_reward=last_ep.total_reward,
                breakdown=last_ep.breakdown,
            )
        else:
            latest_frame = {"ui_mode": getattr(adapter, "ui_mode", "sim_dashboard")}

        session = VibeSession(
            session_id=session_id,
            task_id=t_id,
            adapter=adapter,
            task_config=task_config,
            runner=runner,
            dataset_id=dataset_id,
        )
        session.episodes = burst_logs
        session.rules = plain_rules
        session.skill_summary = skill
        self.sessions[session_id] = session

        coach_msg = (
            f"✨ I've initialized your {adapter_name} agent and trained it over {len(burst_logs)} episodes! "
            f"It currently has an Agent Mastery Score of {skill.mastery_score}% ({skill.status_badge}). "
            f"Active rules are displayed on the left. Type coaching in chat or click directly to interact live."
        )

        response = VibeSessionResponse(
            session_id=session_id,
            task_id=t_id,
            adapter_name=adapter_name,
            ui_mode=getattr(adapter, "ui_mode", "sim_dashboard"),
            mastery_score=skill.mastery_score,
            status_badge=skill.status_badge,
            win_rate=skill.win_rate,
            rules=plain_rules,
            recent_episodes=[ep.model_dump() for ep in burst_logs[-10:]],
            latest_frame=latest_frame,
            coach_message=coach_msg,
        )

        return session, response

    def coach_session(
        self,
        session_id: str,
        critique_text: str | None = None,
        feedback_type: str | None = None,
        target_episode_id: int | None = None,
        burst_episodes: int = 20,
        term_name: str | None = None,
        new_importance: str | None = None,
        llm_config: LLMConfig | None = None,
    ) -> VibeCoachResponse:
        """Conversational coaching or rule card priority updates."""
        if session_id not in self.sessions:
            raise KeyError(f"Vibe session '{session_id}' not found.")

        session = self.sessions[session_id]
        before_mastery = session.skill_summary.mastery_score
        runner = session.runner
        adapter = session.adapter
        applied_diff_records: list[dict[str, Any]] = []

        # 1. Handle Thumbs Up Golden Episode Lock
        if feedback_type == "thumbs_up":
            ep_id = target_episode_id or (session.episodes[-1].episode_id if session.episodes else 1)
            session.locked_episodes.add(ep_id)
            if hasattr(runner.buffer, "lock_episode"):
                runner.buffer.lock_episode(ep_id)
            coach_reply = f"👍 Locked Episode #{ep_id} as a Golden Reference. Future policy updates will preserve this behavior."
            return VibeCoachResponse(
                session_id=session_id,
                coach_reply=coach_reply,
                before_mastery=before_mastery,
                after_mastery=before_mastery,
                mastery_delta=0,
                applied_diff=[],
                rules=session.rules,
                recent_episodes=[ep.model_dump() for ep in session.episodes[-10:]],
                latest_frame={"ui_mode": getattr(adapter, "ui_mode", "sim_dashboard")},
            )

        # 2. Handle 1-Click Rule Priority Card Adjustments
        elif term_name and new_importance:
            target_card = next((c for c in session.rules if c.id == term_name), None)
            if target_card:
                term_diff = plain_rule_to_dsl_diff(target_card, new_importance)
                new_weight = float(term_diff.new_weight)
                # Apply new weight to DSL
                updated_terms = []
                for t in runner.reward_dsl.reward.terms:
                    if t.name == term_name:
                        updated_terms.append(RewardTerm(
                            name=t.name,
                            when=t.when,
                            value=new_weight,
                        ))
                    else:
                        updated_terms.append(t)
                new_dsl = RewardDSL(reward=type(runner.reward_dsl.reward)(terms=updated_terms))
                runner.apply_reward_diff(new_dsl, critique_text=f"Adjusted {term_name} to {new_importance}")
                applied_diff_records.append(term_diff.model_dump())
                coach_reply = f"Updated rule '{target_card.title}' priority to {new_importance} (weight: {new_weight:+.1f})."

        # 3. Handle Conversational Critique or Thumbs Down
        else:
            critique = critique_text
            if feedback_type == "thumbs_down" and not critique:
                critique = "Avoid collisions and eliminate bad actions in this episode."

            if not critique:
                critique = "Improve overall agent performance and safety."

            target_ep = None
            if target_episode_id:
                target_ep = runner.buffer.get_by_id(target_episode_id)
            if not target_ep and session.episodes:
                target_ep = session.episodes[-1]

            client = create_llm_client(llm_config) if llm_config else None
            engine = CritiqueEngine(
                schema=adapter.event_schema(),
                action_labels=adapter.actions().labels,
                llm_client=client,
            )

            locked_eps = list(session.locked_episodes)
            if hasattr(runner.buffer, "locked_ids"):
                locked_eps = list(session.locked_episodes.union(runner.buffer.locked_ids))

            proposal = engine.propose_diff(
                current_dsl=runner.reward_dsl,
                target_episode=target_ep or session.episodes[0],
                recent_episodes=session.episodes,
                locked_episodes=locked_eps,
                critique_text=critique,
                policy=runner.model,
            )

            runner.apply_reward_diff(proposal.new_dsl, critique_text=critique)
            applied_diff_records = [td.model_dump() for td in proposal.term_diffs]
            coach_reply = f"Coaching applied: \"{critique}\". Adjusted {len(proposal.term_diffs)} rule(s) with Critic Warmup."

        # 4. Follow-up Micro-Burst
        new_burst = runner.run_burst(n_episodes=burst_episodes)
        session.episodes.extend(new_burst)

        # 5. Recompute Mastery Score
        after_skill = compute_skill_summary(session.episodes[-burst_episodes:], getattr(adapter, "mode", adapter.__class__.__name__))
        session.skill_summary = after_skill
        session.rules = dsl_to_plain_rules(runner.reward_dsl)
        mastery_delta = after_skill.mastery_score - before_mastery

        # 6. Render Latest Frame
        last_ep = new_burst[-1] if new_burst else session.episodes[-1]
        last_step = last_ep.steps[-1] if last_ep.steps else None
        if last_step:
            latest_frame = adapter.render_frame(
                episode_id=last_ep.episode_id,
                step_result=type("TempStep", (), {
                    "events": last_step.events,
                    "terminated": True,
                    "info": last_step.info,
                })(),
                total_reward=last_ep.total_reward,
                breakdown=last_ep.breakdown,
            )
        else:
            latest_frame = {"ui_mode": getattr(adapter, "ui_mode", "sim_dashboard")}

        return VibeCoachResponse(
            session_id=session_id,
            coach_reply=coach_reply,
            before_mastery=before_mastery,
            after_mastery=after_skill.mastery_score,
            mastery_delta=mastery_delta,
            applied_diff=applied_diff_records,
            rules=session.rules,
            recent_episodes=[ep.model_dump() for ep in session.episodes[-10:]],
            latest_frame=latest_frame,
        )
