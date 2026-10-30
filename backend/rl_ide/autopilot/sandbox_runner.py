"""Runs verification and training inside the sandbox, falling back only when none exists.

The generated environment module is the untrusted component. Everything that needs it in-process
(training with PPO, runtime smoke checks) is executed in a worker with its own directory, no network
and no view of the other sessions' files; only the results are copied back.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from rl_ide.security.sandbox import (
    SandboxError,
    SandboxUnavailable,
    isolation_available,
    run_worker,
    stage_case,
)


class SandboxedTrainingError(RuntimeError):
    """Training inside the sandbox failed; the message carries the worker's own error."""


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def sandbox_workspace_root() -> Path:
    return Path("runs") / "sandbox"


def run_smoke_in_sandbox(
    env_module: Path | str,
    *,
    records: Iterable[dict[str, Any]] | None = None,
    expected_state_keys: list[str] | None = None,
    expected_obs_dim: int | None = None,
    expected_n_actions: int | None = None,
    timeout: float = 180.0,
    case_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Imports the generated module and runs the smoke test inside the sandbox."""
    if not isolation_available():
        raise SandboxUnavailable("No isolation mechanism available for the smoke test.")

    case = (Path(case_dir) if case_dir else sandbox_workspace_root() / f"smoke_{uuid.uuid4().hex[:8]}").resolve()
    stage_case(case, env_module=env_module, records=records)
    staged = case / "env_module.py"
    job = {
        "env_module": str(staged),
        "records_path": str(case / "records.json") if records is not None else None,
        "expected_state_keys": list(expected_state_keys or []),
        "expected_obs_dim": expected_obs_dim,
        "expected_n_actions": expected_n_actions,
    }
    try:
        outcome = run_worker("smoke", job, case, timeout=timeout)
        return outcome.result
    finally:
        shutil.rmtree(case, ignore_errors=True)


def sample_observations_in_sandbox(
    env_module: Path | str,
    *,
    records: Iterable[dict[str, Any]] | None = None,
    episodes: int = 5,
    steps: int = 10,
    timeout: float = 180.0,
) -> list[list[float]]:
    """Replays the environment in the sandbox and returns real observation vectors."""
    if not isolation_available():
        raise SandboxUnavailable("No isolation mechanism available for observation sampling.")

    case = (sandbox_workspace_root() / f"obs_{uuid.uuid4().hex[:8]}").resolve()
    stage_case(case, env_module=env_module, records=records)
    job = {
        "env_module": str(case / "env_module.py"),
        "records_path": str(case / "records.json") if records is not None else None,
        "episodes": episodes,
        "steps": steps,
    }
    try:
        outcome = run_worker("observations", job, case, timeout=timeout)
        return list(outcome.result.get("observations") or [])
    finally:
        shutil.rmtree(case, ignore_errors=True)


def behavioural_parity_in_sandbox(
    env_module: Path | str,
    *,
    model_path: Path | str,
    onnx_path: Path | str,
    records: Iterable[dict[str, Any]] | None = None,
    episodes: int = 10,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Runs the ONNX-vs-SB3 episode comparison in the sandbox."""
    if not isolation_available():
        raise SandboxUnavailable("No isolation mechanism available for behavioural parity.")

    case = (sandbox_workspace_root() / f"parity_{uuid.uuid4().hex[:8]}").resolve()
    stage_case(case, env_module=env_module, records=records, model_path=model_path)
    shutil.copy2(str(onnx_path), str(case / "model.onnx"))
    job = {
        "env_module": str(case / "env_module.py"),
        "records_path": str(case / "records.json") if records is not None else None,
        "model_path": str(case / "model.zip"),
        "onnx_path": str(case / "model.onnx"),
        "episodes": episodes,
    }
    try:
        outcome = run_worker("parity", job, case, timeout=timeout)
        return dict(outcome.result)
    finally:
        shutil.rmtree(case, ignore_errors=True)


def run_training_in_sandbox(
    *,
    session_id: str,
    blueprint: dict[str, Any],
    env_module: Path | str,
    records: Iterable[dict[str, Any]] | None,
    base_dir: Path | str,
    max_rounds: int | None,
    episodes_per_burst: int,
    additional_rounds: int | None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    timeout: float = 3600.0,
) -> dict[str, Any]:
    """Trains in a worker; returns the session summary after copying artefacts back.

    ``base_dir`` is the real workspace root (``workspaces/``). The worker gets a private case
    directory that contains a staged copy of everything it needs, so it never sees another session.
    """
    if not isolation_available():
        raise SandboxUnavailable("No isolation mechanism available for training.")

    case = (sandbox_workspace_root() / f"train_{session_id}_{uuid.uuid4().hex[:6]}").resolve()
    stage_case(case, env_module=env_module, records=records)

    job = {
        "session_id": session_id,
        "blueprint": blueprint,
        "env_module": str(case / "env_module.py"),
        "records_path": str(case / "records.json") if records is not None else None,
        "base_dir": str(case / "workspaces"),
        "max_rounds": max_rounds,
        "episodes_per_burst": episodes_per_burst,
        "additional_rounds": additional_rounds,
    }
    try:
        try:
            outcome = run_worker("train", job, case, timeout=timeout, on_event=on_event)
        except SandboxError as exc:
            raise SandboxedTrainingError(str(exc)) from exc

        # Bring back only this session's artefacts.
        _copy_tree(case / "workspaces" / session_id, Path(base_dir) / session_id)
        _copy_tree(case / "runs" / session_id, Path("runs") / session_id)
        return outcome.result
    finally:
        shutil.rmtree(case, ignore_errors=True)
