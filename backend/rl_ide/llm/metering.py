"""Per-call LLM usage accounting: what a run actually cost.

Nothing here reaches the network. Each provider call reports the token counts it was billed for;
this module prices them with a local table and persists one row per call, so a conversation can be
invoiced, a runaway customer can be capped, and an enterprise buyer can be told what a run costs.

Prices are USD per 1M tokens and are deliberately coarse — override them for a real contract with
``RL_IDE_LLM_PRICES='{"deepseek/deepseek-chat": [0.27, 1.10]}'``.
"""

from __future__ import annotations

import json
import os
import time
from contextvars import ContextVar
from typing import Any

# Request-scoped attribution. Set by the API middleware; a missing conversation is still accounted
# for at owner level, which is what a budget cap needs.
current_conversation_id: ContextVar[str | None] = ContextVar("llm_usage_conversation", default=None)
current_owner_id: ContextVar[str] = ContextVar("llm_usage_owner", default="local")

# provider/model -> (input USD per 1M, output USD per 1M)
_DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-chat": (0.27, 1.10),
    "deepseek/deepseek-reasoner": (0.55, 2.19),
    "deepseek/deepseek-flash": (0.27, 1.10),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "anthropic/claude-3-5-sonnet": (3.00, 15.00),
    # Local weights cost nothing to run; recording them keeps the totals complete.
    "ollama/": (0.0, 0.0),
    "fallback/": (0.0, 0.0),
}


class BudgetExceeded(RuntimeError):
    """Raised before a call that would push a tenant past its configured budget."""


def _prices() -> dict[str, tuple[float, float]]:
    table = dict(_DEFAULT_PRICES)
    raw = os.environ.get("RL_IDE_LLM_PRICES")
    if raw:
        try:
            for key, value in json.loads(raw).items():
                table[key.lower()] = (float(value[0]), float(value[1]))
        except (ValueError, TypeError, IndexError):
            pass  # a malformed override must never break a run
    return table


def price_for(provider: str, model: str) -> tuple[float, float]:
    """Returns (input, output) USD per 1M tokens for a provider/model pair."""
    table = _prices()
    key = f"{provider}/{model}".lower()
    if key in table:
        return table[key]
    for candidate, price in table.items():
        if candidate.endswith("/") and key.startswith(candidate):
            return price
    return (0.0, 0.0)


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = price_for(provider, model)
    return round((input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out, 8)


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
) -> dict[str, Any]:
    """Persists one call and returns the row that was written."""
    cost = (
        float(cost_usd)
        if cost_usd is not None
        else estimate_cost(provider, model, input_tokens, output_tokens)
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
) -> dict[str, Any] | None:
    """Reads a provider response's usage block and records it (None when the provider sent none)."""
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    return record_usage(
        provider=provider,
        model=model,
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        latency_ms=(time.perf_counter() - started_at) * 1000.0,
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
