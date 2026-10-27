"""Sandboxed AST Reward Evaluator and Static Schema Validator."""

from __future__ import annotations
import ast
from typing import Any
from rl_ide.contracts.adapter import EpisodeLog
from rl_ide.reward.dsl import RewardDSL, RewardTerm


class RewardValidationError(Exception):
    """Raised when Reward DSL fails structural, semantic, or schema validation."""
    pass


class RewardSecurityError(RewardValidationError):
    """Raised when Reward DSL contains disallowed AST nodes, dunders, or malicious code."""
    pass


ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Attribute,
    ast.Constant,
    ast.Load,
    ast.List,
    ast.Tuple,
    # Bool operators
    ast.And,
    ast.Or,
    # Unary operators
    ast.Not,
    ast.UAdd,
    ast.USub,
    # Comparison operators
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    # Binary operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)

STANDARD_NAME_CONSTANTS = {"True", "False", "None"}


def _get_dotted_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent_path = _get_dotted_path(node.value)
        if parent_path:
            return f"{parent_path}.{node.attr}"
    return None


class RewardValidator:
    """Validates expressions against an AST whitelist and adapter event_schema."""

    @classmethod
    def validate_expression(
        cls,
        expr: str | bool | int | float,
        schema: dict[str, str] | None = None
    ) -> ast.Expression | None:
        if isinstance(expr, (bool, int, float)):
            return None

        if not isinstance(expr, str):
            raise RewardValidationError(f"Expression must be a string or literal, got {type(expr)}")

        expr = expr.strip()
        if not expr:
            return None

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise RewardValidationError(f"Invalid Python syntax in expression '{expr}': {e}") from e

        # Build parent map for AST nodes to check whether an identifier is root or child
        parent_map: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent

        # 1. Whitelist check and security inspection
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                raise RewardSecurityError(
                    f"Forbidden AST node '{type(node).__name__}' in expression: '{expr}'"
                )

            # Reject private / dunder attribute access
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise RewardSecurityError(
                    f"Private or dunder attribute '{node.attr}' forbidden in expression: '{expr}'"
                )

            # Reject private / dunder names
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                raise RewardSecurityError(
                    f"Private or dunder name '{node.id}' forbidden in expression: '{expr}'"
                )

        # 2. Static variable verification against schema
        if schema is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    # Check if this node is the outermost attribute in a dotted path
                    parent = parent_map.get(node)
                    if not isinstance(parent, ast.Attribute):
                        path = _get_dotted_path(node)
                        if path and path not in schema:
                            raise RewardValidationError(
                                f"Undefined variable '{path}' in expression: '{expr}'. "
                                f"Allowed fields: {sorted(schema.keys())}"
                            )

                elif isinstance(node, ast.Name):
                    # If this name is the root of an Attribute, it is validated by the attribute check
                    parent = parent_map.get(node)
                    if isinstance(parent, ast.Attribute) and parent.value is node:
                        continue

                    # Bare name not in constants
                    if node.id not in STANDARD_NAME_CONSTANTS:
                        raise RewardValidationError(
                            f"Undefined variable or identifier '{node.id}' in expression: '{expr}'. "
                            f"Namespace qualification required (e.g. 'obs.{node.id}')."
                        )

        return tree

    @classmethod
    def validate(cls, dsl: RewardDSL, schema: dict[str, str]) -> None:
        """Validates all terms and pinned episodes in a RewardDSL against schema."""
        for term in dsl.reward.terms:
            # Validate 'when'
            cls.validate_expression(term.when, schema=schema)
            # Validate 'value' if string
            if isinstance(term.value, str):
                cls.validate_expression(term.value, schema=schema)


class _NamespaceWrapper:
    """Ultra-fast attribute accessor for events dictionary."""
    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any] | None):
        self._data = data or {}

    def __getattr__(self, name: str) -> Any:
        try:
            val = self._data[name]
            if isinstance(val, dict):
                return _NamespaceWrapper(val)
            return val
        except KeyError:
            return None

    def __getitem__(self, name: str) -> Any:
        return self._data[name]


class _CompiledTerm:
    __slots__ = ("name", "when_code", "when_literal", "val_code", "val_literal")

    def __init__(
        self,
        name: str,
        when_code: Any,
        when_literal: bool | None,
        val_code: Any,
        val_literal: float | None
    ):
        self.name = name
        self.when_code = when_code
        self.when_literal = when_literal
        self.val_code = val_code
        self.val_literal = val_literal


class RewardEvaluator:
    """Evaluates Reward DSL terms and pinned episodes in a sandboxed execution context."""

    def __init__(self, dsl: RewardDSL, schema: dict[str, str] | None = None):
        self.dsl = dsl
        self.schema = schema
        if schema is not None:
            RewardValidator.validate(dsl, schema)

        self._compiled_terms: list[_CompiledTerm] = []
        self._compile_terms()

    def _compile_terms(self) -> None:
        self._compiled_terms.clear()
        for term in self.dsl.reward.terms:
            # 'when' compilation
            when_code = None
            when_literal = None
            if isinstance(term.when, bool):
                when_literal = term.when
            elif isinstance(term.when, str):
                tree = RewardValidator.validate_expression(term.when, schema=self.schema)
                if tree is not None:
                    when_code = compile(tree, filename=f"<term_{term.name}_when>", mode="eval")
                else:
                    when_literal = True
            else:
                when_literal = bool(term.when)

            # 'value' compilation
            val_code = None
            val_literal = None
            if isinstance(term.value, (int, float)):
                val_literal = float(term.value)
            elif isinstance(term.value, str):
                try:
                    val_literal = float(term.value)
                except ValueError:
                    tree = RewardValidator.validate_expression(term.value, schema=self.schema)
                    if tree is not None:
                        val_code = compile(tree, filename=f"<term_{term.name}_val>", mode="eval")
                    else:
                        val_literal = 0.0
            else:
                val_literal = float(term.value)

            self._compiled_terms.append(
                _CompiledTerm(
                    name=term.name,
                    when_code=when_code,
                    when_literal=when_literal,
                    val_code=val_code,
                    val_literal=val_literal,
                )
            )

    def evaluate(self, episode_id: int, events: dict[str, Any]) -> tuple[float, dict[str, float]]:
        """Evaluates reward for a single step/event frame.
        
        Returns:
            (total_clamped_reward, per_term_breakdown)
        """
        obs_ns = _NamespaceWrapper(events.get("obs"))
        action_ns = _NamespaceWrapper(events.get("action"))
        state_ns = _NamespaceWrapper(events.get("state"))
        truth_ns = _NamespaceWrapper(events.get("truth"))

        context = {
            "obs": obs_ns,
            "action": action_ns,
            "state": state_ns,
            "truth": truth_ns,
            "True": True,
            "False": False,
            "None": None,
        }

        breakdown: dict[str, float] = {}

        for ct in self._compiled_terms:
            condition_met = False
            if ct.when_literal is not None:
                condition_met = ct.when_literal
            elif ct.when_code is not None:
                try:
                    res = eval(ct.when_code, {"__builtins__": {}}, context)
                    condition_met = bool(res)
                except Exception:
                    condition_met = False

            if condition_met:
                if ct.val_literal is not None:
                    breakdown[ct.name] = ct.val_literal
                elif ct.val_code is not None:
                    try:
                        res = eval(ct.val_code, {"__builtins__": {}}, context)
                        breakdown[ct.name] = float(res)
                    except Exception:
                        breakdown[ct.name] = 0.0
                else:
                    breakdown[ct.name] = 0.0
            else:
                breakdown[ct.name] = 0.0

        # Check pinned episodes constraints
        for pinned in self.dsl.reward.pinned_episodes:
            if pinned.episode_id == episode_id:
                act = events.get("action", {})
                act_label = act.get("label") if isinstance(act, dict) else None
                if act_label == pinned.forbid_action:
                    term_key = f"pinned_ep_{pinned.episode_id}_penalty"
                    breakdown[term_key] = pinned.penalty

        # Aggregate
        agg = self.dsl.reward.aggregate
        if agg == "mean":
            total = sum(breakdown.values()) / max(1, len(breakdown))
        else:
            total = sum(breakdown.values())

        # Clamp
        clamp_bounds = self.dsl.reward.clamp
        if clamp_bounds and len(clamp_bounds) == 2:
            low, high = clamp_bounds
            total = max(low, min(high, total))

        return total, breakdown

    def rescore_episodes(self, episodes: list[EpisodeLog]) -> list[EpisodeLog]:
        """Retroactively recomputes rewards across historical episodes in memory."""
        for ep in episodes:
            total = 0.0
            breakdown_acc: dict[str, float] = {}
            step_rewards: list[float] = []
            step_breakdowns: list[dict[str, float]] = []

            for step in ep.steps:
                r, b = self.evaluate(ep.episode_id, step.events)
                total += r
                for k, v in b.items():
                    breakdown_acc[k] = breakdown_acc.get(k, 0.0) + v
                step_rewards.append(r)
                step_breakdowns.append(b)

            ep.total_reward = total
            ep.breakdown = breakdown_acc
            ep.step_rewards = step_rewards
            ep.step_breakdowns = step_breakdowns

        return episodes
