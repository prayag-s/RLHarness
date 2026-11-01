"""Per-call LLM usage accounting: what a run actually cost.

Nothing here reaches the network. Each provider call reports the token counts it was billed for;
this module prices them with a local table and persists one row per call, so a conversation can be
invoiced, a runaway customer can be capped, and an enterprise buyer can be told what a run costs.

Prices are USD per 1M tokens and are deliberately coarse — override them for a real contract with
``RL_IDE_LLM_PRICES='{"deepseek-flash": [0.15, 0.60, 0.003]}'``. A third element is the cached-input
price, which is what makes prompt caching visible in a report instead of silently discounted.
"""

from __future__ import annotations

import json
import os
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# Request-scoped attribution. Set by the API middleware; a missing conversation is still accounted
# for at owner level, which is what a budget cap needs.
current_conversation_id: ContextVar[str | None] = ContextVar("llm_usage_conversation", default=None)
current_owner_id: ContextVar[str] = ContextVar("llm_usage_owner", default="local")

# provider/model -> (input USD per 1M, output USD per 1M, cached input USD per 1M)
_DEFAULT_PRICES: dict[str, tuple[float, float, float]] = {
    "deepseek/deepseek-chat": (0.27, 1.10, 0.027),
    "deepseek/deepseek-reasoner": (0.55, 2.19, 0.055),
    # DeepSeek V4 flash: peak hours cost twice the off-peak rate, and a cache hit is ~50x cheaper
    # than a miss on input. Both are applied automatically by ``estimate_cost``.
    "deepseek/deepseek-flash": (0.30, 1.20, 0.006),
    "deepseek/deepseek-flash@offpeak": (0.15, 0.60, 0.003),
    "openai/gpt-4o": (2.50, 10.00, 1.25),
    "openai/gpt-4o-mini": (0.15, 0.60, 0.075),
    "anthropic/claude-3-5-sonnet": (3.00, 15.00, 0.30),
    # Local weights cost nothing to run; recording them keeps the totals complete.
    "ollama/": (0.0, 0.0, 0.0),
    "fallback/": (0.0, 0.0, 0.0),
}


class BudgetExceeded(RuntimeError):
    """Raised before a call that would push a tenant past its configured budget."""


def _prices() -> dict[str, tuple[float, float, float]]:
    table = dict(_DEFAULT_PRICES)
    raw = os.environ.get("RL_IDE_LLM_PRICES")
    if raw:
        try:
            for key, value in json.loads(raw).items():
                table[key.lower()] = (
                    float(value[0]),
                    float(value[1]),
                    float(value[2]) if len(value) > 2 else 0.0,
                )
        except (ValueError, TypeError, IndexError):
            pass  # a malformed override must never break a run
    return table


# Peak hours are 01:00-04:00 and 06:00-10:00 UTC on weekdays; everything else is half price.
_DEEPSEEK_PEAK_HOURS = ((1, 4), (6, 10))


def is_deepseek_offpeak(at: datetime | None = None) -> bool:
    """Whether DeepSeek currently bills at its off-peak rate."""
    moment = at.astimezone(timezone.utc) if at else datetime.now(timezone.utc)
    if moment.weekday() >= 5:
        return True
    return not any(start <= moment.hour < end for start, end in _DEEPSEEK_PEAK_HOURS)


def price_for(
    provider: str, model: str, at: datetime | None = None
) -> tuple[float, float, float]:
    """Returns (input, output, cached-input) USD per 1M tokens for a provider/model pair."""
    table = _prices()
    key = f"{provider}/{model}".lower()
    if key in table:
        price = table[key]
    else:
        price = next(
            (value for candidate, value in table.items() if candidate.endswith("/") and key.startswith(candidate)),
            (0.0, 0.0, 0.0),
        )
    if key.startswith("deepseek/") and "flash" in key and is_deepseek_offpeak(at):
        return table.get("deepseek/deepseek-flash@offpeak", price)
    return price


def estimate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    at: datetime | None = None,
) -> float:
    price_in, price_out, price_cached = price_for(provider, model, at)
    cached = max(0, min(int(cached_input_tokens or 0), int(input_tokens or 0)))
    missed = max(0, int(input_tokens or 0) - cached)
    return round(
        (missed / 1_000_000) * price_in
        + (cached / 1_000_000) * price_cached
        + (output_tokens / 1_000_000) * price_out,
        8,
    )


def budget_usd() -> float | None:
    """Per-tenant spend cap; unset means uncapped."""
    raw = os.environ.get("RL_IDE_OWNER_BUDGET_USD", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _store():
    from rl_ide.state.conversation_store import get_conversation_store

    return get_conversation_store()


def owner_spend(owner_id: str | None = None) -> float:
    owner = owner_id or current_owner_id.get()
    return float(_store().usage_summary(owner_id=owner, limit=1)["totals"]["cost_usd"])


def enforce_budget(owner_id: str | None = None) -> None:
    """Fails a job cleanly before it spends more than its tenant is allowed."""
    cap = budget_usd()
    if cap is None:
        return
    spent = owner_spend(owner_id)
    if spent >= cap:
        raise BudgetExceeded(
            f"LLM budget of ${cap:.2f} exhausted for this tenant (${spent:.2f} spent). "
            "Raise RL_IDE_OWNER_BUDGET_USD or start a new billing period."
        )


def record_usage(
    *,
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: float = 0.0,
    conversation_id: str | None = None,
    owner_id: str | None = None,
    cost_usd: float | None = None,
    role: str | None = None,
    reasoning_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> dict[str, Any]:
    """Persists one call and returns the row that was written.

    ``reasoning_tokens`` is a *subset* of ``output_tokens``: providers bill it as output, so it is
    recorded for attribution rather than added to the cost a second time.
    """
    cost = (
        float(cost_usd)
        if cost_usd is not None
        else estimate_cost(
            provider,
            model,
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
        )
    )
    row = {
        "provider": provider,
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cost_usd": cost,
        "latency_ms": round(latency_ms, 3),
        "conversation_id": conversation_id or current_conversation_id.get(),
        "owner_id": owner_id or current_owner_id.get(),
        "role": role,
        "reasoning_tokens": int(reasoning_tokens),
        "cached_input_tokens": int(cached_input_tokens),
    }
    try:
        _store().record_usage(**row)
    except Exception:
        # Accounting must never take down a run that already paid for its tokens.
        pass
    return row


def capture_response(
    provider: str,
    model: str,
    response: dict[str, Any],
    started_at: float,
    role: str | None = None,
) -> dict[str, Any] | None:
    """Reads a provider response's usage block and records it (None when the provider sent none)."""
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0

    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else 0
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_input_tokens = (
        usage.get("prompt_cache_hit_tokens")
        or (prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else 0)
        or 0
    )

    return record_usage(
        provider=provider,
        model=model,
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        latency_ms=(time.perf_counter() - started_at) * 1000.0,
        role=role,
        reasoning_tokens=int(reasoning_tokens or 0),
        cached_input_tokens=int(cached_input_tokens or 0),
    )


def usage_report(
    conversation_id: str | None = None,
    owner_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Totals, per-model split and recent calls for one conversation or one tenant."""
    report = _store().usage_summary(conversation_id=conversation_id, owner_id=owner_id, limit=limit)
    cap = budget_usd()
    report["budget"] = {
        "cap_usd": cap,
        "spent_usd": report["totals"]["cost_usd"],
        "remaining_usd": None if cap is None else round(max(0.0, cap - report["totals"]["cost_usd"]), 6),
    }
    return report
