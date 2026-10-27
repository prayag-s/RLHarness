"""Plain-English Behavioral Rule Translator for Vibe Studio."""

from __future__ import annotations
import re
from typing import Any
from pydantic import BaseModel, Field

from rl_ide.compiler.models import TermDiff
from rl_ide.reward.dsl import RewardDSL, RewardTerm


class PlainRuleCard(BaseModel):
    """User-facing plain-English rule card for the Vibe Studio Rules panel."""
    id: str  # maps to term.name
    title: str
    description: str
    importance: str  # "Must Do" | "High Priority" | "Bonus" | "Avoid" | "Strict Ban"
    badge_variant: str  # "emerald" | "blue" | "indigo" | "amber" | "rose"
    weight: float
    expression: str
    direction: str  # "reward" | "penalty"
    category: str = "behavior"  # "safety" | "goal" | "efficiency" | "compliance" | "behavior"
    icon: str = "sparkles"


IMPORTANCE_LEVELS = {
    "Must Do": {"weight": 10.0, "badge": "emerald", "direction": "reward", "icon": "shield-check"},
    "High Priority": {"weight": 5.0, "badge": "blue", "direction": "reward", "icon": "zap"},
    "Bonus": {"weight": 1.5, "badge": "indigo", "direction": "reward", "icon": "star"},
    "Avoid": {"weight": -2.0, "badge": "amber", "direction": "penalty", "icon": "alert-triangle"},
    "Strict Ban": {"weight": -10.0, "badge": "rose", "direction": "penalty", "icon": "slash"},
}


def _classify_importance(weight: float) -> tuple[str, str, str, str]:
    """Returns (importance, badge_variant, direction, icon) from weight."""
    if weight >= 8.0:
        return "Must Do", "emerald", "reward", "shield-check"
    if weight >= 2.0:
        return "High Priority", "blue", "reward", "zap"
    if weight > 0.0:
        return "Bonus", "indigo", "reward", "star"
    if weight >= -4.0:
        return "Avoid", "amber", "penalty", "alert-triangle"
    return "Strict Ban", "rose", "penalty", "slash"


def _generate_human_title_and_desc(term: RewardTerm) -> tuple[str, str, str, str]:
    """Inspects term name & AST expression to synthesize friendly title, description, category, and icon."""
    name_low = term.name.lower()
    expr_low = term.expression.lower()

    # 1. Board Games / Tic-Tac-Toe
    if "center" in name_low or "center" in expr_low:
        return (
            "Take Center Grid",
            "Seize the center square immediately when open to dominate board geometry.",
            "goal",
            "target",
        )
    if "block" in name_low or "opponent_win" in expr_low:
        return (
            "Block Opponent Threats",
            "Immediately intercept opponent alignments before they complete a 3-in-a-row.",
            "safety",
            "shield-alert",
        )
    if "win" in name_low or "took_win" in expr_low or "state.winner == 'agent'" in expr_low:
        return (
            "Execute Winning Move",
            "Instantly claim open winning lines to secure victory.",
            "goal",
            "trophy",
        )
    if "draw" in name_low:
        return (
            "Hold Tactical Stalemate",
            "Secure a draw when direct victory is blocked by opponent defense.",
            "goal",
            "shield",
        )
    if "illegal" in name_low:
        return (
            "Zero Illegal Moves",
            "Never select an already occupied cell or execute invalid actions.",
            "safety",
            "ban",
        )
    if "corner" in name_low:
        return (
            "Prioritize Outer Corners",
            "Establish strategic outer corners for dual-fork tactical advantages.",
            "efficiency",
            "maximize-2",
        )

    # 2. 2D Robotics & Rover Navigation
    if "reach" in name_low or "reached_target" in expr_low:
        return (
            "Reach Docking Beacon",
            "Guide the rover cleanly into the active docking target beacon.",
            "goal",
            "map-pin",
        )
    if "dist" in name_low or "distance_reduced" in expr_low:
        return (
            "Approach Target Beacon",
            "Continuously reduce spatial distance toward active target waypoint.",
            "efficiency",
            "navigation",
        )
    if "obstacle" in name_low or "collid" in name_low or "min_obstacle_dist" in expr_low:
        return (
            "Obstacle Safety Clearance",
            "Maintain a generous safety buffer from obstacles and arena borders.",
            "safety",
            "shield-alert",
        )
    if "jerk" in name_low or "smooth" in name_low or "jerk_magnitude" in expr_low:
        return (
            "Smooth Motion Control",
            "Minimize abrupt thruster direction changes and harsh kinematic jerk.",
            "efficiency",
            "activity",
        )
    if "battery" in name_low or "power" in name_low:
        return (
            "Conserve Battery",
            "Avoid excessive thruster burns to protect remaining battery reserves.",
            "efficiency",
            "battery-charging",
        )

    # 3. Smart Inventory
    if "stockout" in name_low or "is_stockout" in expr_low:
        return (
            "Zero Stockout Mandate",
            "Keep safety stock replenished to never miss customer demand spikes.",
            "safety",
            "alert-circle",
        )
    if "profit" in name_low or "margin" in name_low or "units_sold" in expr_low:
        return (
            "Maximize Sales Profit",
            "Fulfill incoming customer orders at strong positive profit margins.",
            "goal",
            "trending-up",
        )
    if "holding" in name_low or "holding_cost" in expr_low:
        return (
            "Control Holding Costs",
            "Avoid excessive warehouse overstock that accumulates carrying fees.",
            "efficiency",
            "package",
        )

    # 4. Universal Document Auditing
    if "dollar" in name_low or "currency" in name_low or "max_dollar" in expr_low:
        return (
            "High-Dollar Audit Escalation",
            "Flag and escalate high-value transactions exceeding authorization limits.",
            "compliance",
            "dollar-sign",
        )
    if "po" in name_low or "missing_po" in expr_low:
        return (
            "Require Purchase Order",
            "Ensure every invoice is matched with a valid authorized purchase order.",
            "compliance",
            "file-check",
        )
    if "risk" in name_low or "liability" in name_low or "risk_keyword" in expr_low:
        return (
            "Flag Compliance Risk",
            "Escalate documents containing critical liability or breach keywords.",
            "compliance",
            "alert-triangle",
        )

    # 5. Financial Stream
    if "volatility" in name_low or "spike" in name_low or "anomaly" in name_low:
        return (
            "Flag Market Anomaly",
            "Trigger rapid alerts upon extreme volatility and volume bursts.",
            "goal",
            "flame",
        )
    if "chop" in name_low or "false" in name_low:
        return (
            "Filter Noise & Chop",
            "Suppress false alarms during routine baseline market consolidation.",
            "safety",
            "volume-x",
        )

    # Generic Fallback
    friendly_title = term.name.replace("_", " ").title()
    friendly_desc = f"Maintain policy constraint: {term.expression}."
    return friendly_title, friendly_desc, "behavior", "sliders"


def dsl_to_plain_rules(dsl: RewardDSL) -> list[PlainRuleCard]:
    """Translates AST RewardDSL into a collection of human-friendly PlainRuleCards."""
    cards: list[PlainRuleCard] = []
    for term in dsl.reward.terms:
        importance, badge, direction, fallback_icon = _classify_importance(term.weight)
        title, desc, category, icon = _generate_human_title_and_desc(term)

        cards.append(PlainRuleCard(
            id=term.name,
            title=title,
            description=desc,
            importance=importance,
            badge_variant=badge,
            weight=float(term.weight),
            expression=term.expression,
            direction=direction,
            category=category,
            icon=icon or fallback_icon,
        ))
    return cards


def plain_rule_to_dsl_diff(card: PlainRuleCard, new_importance: str) -> TermDiff:
    """Creates a TermDiff when a user adjusts a rule's importance level."""
    target_spec = IMPORTANCE_LEVELS.get(new_importance, IMPORTANCE_LEVELS["High Priority"])
    new_weight = float(target_spec["weight"])
    return TermDiff(
        action="modified",
        term_name=card.id,
        old_term={"name": card.id, "when": card.expression, "value": card.weight},
        new_term={"name": card.id, "when": card.expression, "value": new_weight},
        rationale=f"User updated rule '{card.title}' importance to {new_importance}",
    )
