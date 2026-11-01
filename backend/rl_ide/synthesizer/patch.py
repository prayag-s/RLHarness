"""Function-level code patches.

Regenerating a whole 8 KB module to change one reward term is the single most expensive thing this
product does: the model re-emits every unrelated method, and every re-emitted line is a new chance
to break something that already worked. A patch replaces exactly one function and leaves the rest of
the module byte-for-byte untouched.
"""

from __future__ import annotations

import ast
import re
import textwrap


class PatchError(ValueError):
    """The model's patch could not be applied safely."""


def _find_unique_function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if not matches:
        raise PatchError(f"No function named '{name}' was found.")
    if len(matches) > 1:
        raise PatchError(f"'{name}' is defined more than once; refusing to guess which one to replace.")
    return matches[0]


def _top_level_shape(source: str) -> list[str]:
    tree = ast.parse(source)
    return [getattr(node, "name", type(node).__name__) for node in tree.body]


def replace_function(module_source: str, replacement_source: str, name: str) -> str:
    """Returns ``module_source`` with the function ``name`` replaced by ``replacement_source``.

    The replacement must be a single function definition with that name. Its body is re-indented to
    the original position, so a method stays a method.
    """
    dedented = textwrap.dedent(replacement_source).strip("\n")
    new_tree = ast.parse(dedented + "\n")
    definitions = [
        node
        for node in new_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(new_tree.body) != 1 or len(definitions) != 1:
        raise PatchError("The patch must contain exactly one function definition and nothing else.")
    if definitions[0].name != name:
        raise PatchError(f"The patch defined '{definitions[0].name}' instead of '{name}'.")

    module_tree = ast.parse(module_source)
    target = _find_unique_function(module_tree, name)

    start_line = min(
        [target.lineno] + [decorator.lineno for decorator in getattr(target, "decorator_list", [])]
    )
    end_line = target.end_lineno or target.lineno
    if not end_line:
        raise PatchError(f"Could not determine the extent of '{name}'.")

    indent = " " * target.col_offset
    replacement = textwrap.indent(dedented, indent) + "\n"

    lines = module_source.splitlines(keepends=True)
    updated = "".join(lines[: start_line - 1]) + replacement + "".join(lines[end_line:])

    try:
        ast.parse(updated)
    except SyntaxError as exc:  # pragma: no cover - guarded by the caller as well
        raise PatchError(f"The patched module does not parse: {exc}") from exc

    if _top_level_shape(module_source) != _top_level_shape(updated):
        raise PatchError("The patch changed the module's top-level structure.")
    return updated


def extract_function_block(raw: str, name: str) -> str | None:
    """Pulls the single function definition out of a model response, fenced or bare."""
    text = raw or ""
    for pattern in (r"```(?:python)?\s*([\s\S]*?)```", r"(?m)^(def\s+" + re.escape(name) + r"\s*\([\s\S]*)"):
        for match in re.finditer(pattern, text):
            candidate = match.group(1).strip()
            if f"def {name}(" not in candidate:
                continue
            try:
                tree = ast.parse(candidate + "\n")
            except SyntaxError:
                continue
            if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
                return candidate
    return None


def extract_functions(raw: str) -> dict[str, str]:
    """Maps every top-level function in a model answer to its source.

    A targeted repair may legitimately return two functions (a method and the helper it uses), so
    this accepts multi-function answers while still refusing anything that is not a definition.
    """
    source = (raw or "").strip()
    fences = re.findall(r"```(?:python)?\s*([\s\S]*?)```", source)
    for candidate in [*fences, source]:
        try:
            tree = ast.parse(textwrap.dedent(candidate).strip("\n") + "\n")
        except SyntaxError:
            continue
        found: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                segment = ast.get_source_segment(candidate, node)
                if segment:
                    found[node.name] = textwrap.dedent(segment)
        if found:
            return found
    return {}


__all__ = ["PatchError", "replace_function", "extract_function_block", "extract_functions"]
