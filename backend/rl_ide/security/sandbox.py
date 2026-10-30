"""OS-level isolation for model-generated code.

Generated environment modules are ordinary Python that the LLM wrote. Running them inside the API
process gives them the server's UID, filesystem and network — a denylist is not a boundary. This
module runs such code in a separate process under ``bwrap`` (Bubblewrap) with only the interpreter,
the library tree, the read-only application code and the current case directory visible, and with
the network namespace unshared.

When no isolation mechanism is available the caller gets ``SandboxUnavailable`` and decides whether
to fall back (single-user local studio) or refuse (hosted mode).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = BACKEND_ROOT / "rl_ide"
EVENT_PREFIX = "EVENT "
RESULT_PREFIX = "RESULT "


class SandboxUnavailable(RuntimeError):
    """No usable isolation mechanism on this host."""


class SandboxError(RuntimeError):
    """The sandboxed process failed, timed out, or returned no result."""


def _bindable(path: Path) -> bool:
    return path.exists()


def isolation_tool() -> str | None:
    """Returns the isolation binary, or None.

    Only ``bwrap`` counts. ``unshare -rn`` gives network and PID isolation but leaves the host
    filesystem fully visible, so treating it as equivalent would be security theatre: a module could
    still read another tenant's workspace. Hosts without bwrap get the explicit single-user
    fallback instead.
    """
    return shutil.which("bwrap")


def isolation_available() -> bool:
    return isolation_tool() is not None


def _bwrap_argv(case_dir: Path, python: str) -> list[str]:
    # Use the interpreter path inside the *bound* prefix. Resolving the symlink all the way to
    # /usr/bin/python3 would drop the venv (sys.prefix=/usr) and with it every installed package.
    venv_python = Path(sys.prefix).resolve() / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else str(Path(python).resolve())
    argv = ["bwrap"]
    for path in ("/usr", "/lib", "/lib64"):
        if _bindable(Path(path)):
            argv += ["--ro-bind", path, path]
    # The interpreter's own prefix (the venv) carries site-packages.
    prefix = Path(sys.prefix).resolve()
    if str(prefix) not in ("/usr", "/usr/local") and _bindable(prefix):
        argv += ["--ro-bind", str(prefix), str(prefix)]

    # Application code is mounted read-only at a fixed path so the working directory can stay the
    # case dir. `rl_ide` is bound alone: the backend root also holds workspaces/ and runs/, which
    # belong to other sessions.
    argv += ["--dir", "/opt", "--ro-bind", str(PACKAGE_DIR), "/opt/rl_ide"]

    # `--tmpfs /tmp` must come *before* the case bind: when the case directory lives under /tmp
    # (tests, temp workspaces) a later tmpfs would shadow it and bwrap could not chdir into it.
    argv += ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    argv += ["--bind", str(case_dir), str(case_dir)]
    argv += [
        "--unshare-net",
        "--unshare-pid",
        "--die-with-parent",
        "--chdir", str(case_dir),
        python,
    ]
    return argv


def build_command(mode: str, case_dir: Path, job_path: Path) -> list[str]:
    """Builds the isolated argv for a worker invocation."""
    tool = isolation_tool()
    if tool is None:
        raise SandboxUnavailable(
            "bwrap is not installed; refusing to execute generated code outside the API process."
        )
    argv = _bwrap_argv(case_dir, sys.executable)
    argv += ["-m", "rl_ide.security.worker", "--mode", mode, "--job", str(job_path)]
    return argv


def sandbox_env(case_dir: Path) -> dict[str, str]:
    """Minimal environment: no provider keys are forwarded into the untrusted process."""
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(case_dir),
        "PYTHONPATH": "/opt",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "RL_IDE_SANDBOX": "1",
    }


@dataclass
class SandboxResult:
    result: dict[str, Any]
    events: list[dict[str, Any]]


def run_worker(
    mode: str,
    job: dict[str, Any],
    case_dir: Path | str,
    *,
    timeout: float = 1800.0,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> SandboxResult:
    """Runs one worker to completion and returns its result payload."""
    case_path = Path(case_dir).resolve()
    case_path.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:8]
    job_path = case_path / f"job_{job_id}.json"
    out_path = case_path / f"result_{job_id}.json"
    payload = dict(job)
    payload["out_path"] = str(out_path)
    job_path.write_text(json.dumps(payload), encoding="utf-8")

    argv = build_command(mode, case_path, job_path)
    events: list[dict[str, Any]] = []

    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(case_path),
            env=sandbox_env(case_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SandboxUnavailable(str(exc)) from exc

    timer = threading.Timer(timeout, proc.kill)
    timer.start()
    stderr_tail: list[str] = []
    try:
        # stderr is merged into stdout: reading one pipe while the other fills is a deadlock.
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith(EVENT_PREFIX):
                try:
                    event = json.loads(line[len(EVENT_PREFIX):])
                except json.JSONDecodeError:
                    continue
                events.append(event)
                if on_event:
                    try:
                        on_event(event)
                    except Exception:
                        pass
            elif line.startswith(RESULT_PREFIX):
                continue  # authoritative payload is read from the out file
            elif line.strip():
                stderr_tail.append(line.rstrip("\n"))
                if len(stderr_tail) > 40:
                    stderr_tail.pop(0)
        proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()

    if not out_path.exists():
        detail = "\n".join(stderr_tail[-15:]) or f"exit code {proc.returncode}"
        raise SandboxError(f"{mode} worker produced no result: {detail}")

    result = json.loads(out_path.read_text(encoding="utf-8"))
    try:
        job_path.unlink()
    except OSError:
        pass
    if not result.get("ok", False):
        raise SandboxError(str(result.get("error") or f"{mode} worker failed"))
    return SandboxResult(result=result.get("result") or {}, events=events)


def stage_case(
    case_dir: Path | str,
    *,
    env_module: Path | str | None = None,
    records: Iterable[dict[str, Any]] | None = None,
    model_path: Path | str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Copies exactly what one worker needs into its private case directory."""
    case = Path(case_dir)
    case.mkdir(parents=True, exist_ok=True)
    if env_module is not None:
        shutil.copy2(str(env_module), str(case / "env_module.py"))
    if records is not None:
        (case / "records.json").write_text(json.dumps(list(records), default=str), encoding="utf-8")
    if model_path is not None and Path(model_path).exists():
        shutil.copy2(str(model_path), str(case / "model.zip"))
    if extra:
        (case / "job_extra.json").write_text(json.dumps(extra, default=str), encoding="utf-8")
    return case
