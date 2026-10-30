"""Client for a long-lived sandboxed environment session.

Interactive play needs the environment to keep its state between requests, so unlike the one-shot
worker modes this keeps a `bwrap` process alive and speaks JSON lines to it. The process never sees
the API's memory, filesystem or network.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from rl_ide.security.sandbox import (
    RESULT_PREFIX,
    SandboxError,
    build_command,
    sandbox_env,
    stage_case,
)


class SandboxedEnvSession:
    """A persistent, isolated environment process for one conversation."""

    def __init__(self, timeout: float = 120.0) -> None:
        self.timeout = timeout
        self.last_used = time.monotonic()
        self.case_dir: Path | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    def start(
        self,
        *,
        env_module: Path | str,
        records: Iterable[dict[str, Any]] | None = None,
        model_path: Path | str | None = None,
    ) -> None:
        case = (Path("runs") / "sandbox" / f"serve_{uuid.uuid4().hex[:8]}").resolve()
        stage_case(case, env_module=env_module, records=records, model_path=model_path)
        self.case_dir = case
        self._spawn()

    def _spawn(self) -> None:
        assert self.case_dir is not None
        job_path = self.case_dir / "serve_job.json"
        job_path.write_text(
            json.dumps(
                {
                    "env_module": str(self.case_dir / "env_module.py"),
                    "records_path": (
                        str(self.case_dir / "records.json")
                        if (self.case_dir / "records.json").exists()
                        else None
                    ),
                    "model_path": (
                        str(self.case_dir / "model.zip")
                        if (self.case_dir / "model.zip").exists()
                        else None
                    ),
                }
            ),
            encoding="utf-8",
        )
        argv = build_command("serve", self.case_dir, job_path)
        self._proc = subprocess.Popen(
            argv,
            cwd=str(self.case_dir),
            env=sandbox_env(self.case_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                    proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        if self.case_dir is not None:
            shutil.rmtree(self.case_dir, ignore_errors=True)
            self.case_dir = None

    # -------------------------------------------------------------------- request
    def request(self, payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Sends one command and returns its result, restarting a dead worker once."""
        deadline = timeout or self.timeout
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                if self.case_dir is None:
                    raise SandboxError("sandboxed environment session was never started")
                self._spawn()

            assert self._proc is not None and self._proc.stdin and self._proc.stdout
            self.last_used = time.monotonic()
            killer = threading.Timer(deadline, self._kill)
            killer.start()
            try:
                self._proc.stdin.write(json.dumps(payload, default=str) + "\n")
                self._proc.stdin.flush()
                while True:
                    line = self._proc.stdout.readline()
                    if not line:
                        raise SandboxError("sandboxed environment session exited unexpectedly")
                    if not line.startswith(RESULT_PREFIX):
                        continue  # worker logs / warnings
                    data = json.loads(line[len(RESULT_PREFIX):])
                    if not data.get("ok"):
                        raise SandboxError(str(data.get("error") or "interactive step failed"))
                    return dict(data.get("result") or {})
            finally:
                killer.cancel()

    def _kill(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    def interact(self, payload: dict[str, Any], mode: str = "agent_vs_env") -> dict[str, Any]:
        return self.request({"cmd": "interact", "payload": payload, "mode": mode})
