"""Per-role generation budgets: how much a call may think, and how much it may emit.

Reasoning tokens are billed as output tokens. DeepSeek enables thinking by default at high effort,
so a classification call that returns 150 tokens of JSON can still be billed for thousands of
reasoning tokens. Most calls in this product are classification or a bounded rewrite, where
chain-of-thought buys nothing and multiplies the bill.

A role is a string the caller passes to ``generate(..., role=...)``. It is also recorded on every
usage row, so cost can be attributed to the call site that caused it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationProfile:
    """The budget a role is allowed for one call."""

    thinking: bool
    effort: str = "low"
    max_tokens: int = 4096
    # Floor for the request timeout. A large ceiling is useless if the request is cut off before the
    # model can reach it: the tokens are paid for and the answer is lost.
    timeout_s: float = 0.0


# Roles whose answer is a classification, a short explanation or a style spec get no reasoning and
# a ceiling small enough that a runaway generation cannot become a large charge. Roles that write
# code may think, but at low effort: the provider default is "high", which costs more than the code.
_PROFILES: dict[str, GenerationProfile] = {
    "intake_blueprint": GenerationProfile(False, "low", 1800),
    "chat_planner": GenerationProfile(False, "low", 700),
    "coach_judge": GenerationProfile(False, "low", 1200),
    "diagnostic_chat": GenerationProfile(False, "low", 700),
    "arena_ui": GenerationProfile(False, "low", 1400),
    # Escape hatch for ``RL_IDE_ARENA_MODE=bespoke``, which still emits a whole HTML document.
    "arena_ui_bespoke": GenerationProfile(False, "low", 12000),
    # For a thinking role the ceiling must cover reasoning *and* the answer: a cap that reasoning
    # alone exhausts pays for tokens and returns nothing. These two are bounded function rewrites
    # with an explicit diagnosis, where reasoning measured 600-3,000 variable tokens for the same
    # answer, so it is switched off rather than budgeted for.
    "env_synthesizer": GenerationProfile(True, "low", 8000, timeout_s=300.0),
    "verifier_repair": GenerationProfile(True, "low", 6000, timeout_s=240.0),
    "auto_tuner": GenerationProfile(False, "low", 1200),
    "reward_rewrite": GenerationProfile(False, "low", 900),
}

_DEFAULT = GenerationProfile(False, "low", 4096)


def _env_off(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"0", "off", "false", "no", "disabled"}


def profile_for(role: str | None) -> GenerationProfile:
    """Returns the budget for ``role``, honouring the global overrides.

    ``RL_IDE_LLM_THINKING=0`` disables reasoning everywhere, which is the safest switch for a
    cost-sensitive deployment. ``RL_IDE_LLM_MAX_TOKENS`` is a hard ceiling applied on top of every
    role budget.
    """
    profile = _PROFILES.get((role or "").strip(), _DEFAULT)

    if not profile.thinking or _env_off("RL_IDE_LLM_THINKING"):
        profile = GenerationProfile(False, "low", profile.max_tokens, profile.timeout_s)

    ceiling = os.environ.get("RL_IDE_LLM_MAX_TOKENS", "").strip()
    if ceiling.isdigit() and int(ceiling) > 0:
        profile = GenerationProfile(
            profile.thinking, profile.effort, min(profile.max_tokens, int(ceiling)), profile.timeout_s
        )
    return profile


def timeout_for(role: str | None, explicit: float | None, configured: float) -> float:
    """The request timeout: an explicit caller value wins, otherwise the role floor or config."""
    if explicit is not None:
        return float(explicit)
    return max(float(configured or 0.0), profile_for(role).timeout_s)


def max_tokens_for(role: str | None, configured: int | None) -> int:
    """The role budget, never above what the user configured for the provider."""
    budget = profile_for(role).max_tokens
    if configured and configured > 0:
        return min(budget, int(configured))
    return budget


__all__ = ["GenerationProfile", "profile_for", "max_tokens_for", "timeout_for"]
