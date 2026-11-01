"""Test-suite defaults.

Interactive sessions are reaped aggressively in production (15 min idle, 8 concurrent). Tests must
not inherit that: a reaped worker is respawned on the next request, and spawning imports torch, which
dominated the suite runtime. The test process closes every worker on exit anyway (stdin EOF).

The suite also runs from a scratch directory. Every store (`runs/`, `workspaces/`,
`exported_bundles/`) resolves its root from those relative paths, and the API tests drive the real
app singletons, so a test that synthesizes an agent would otherwise leave it in the developer's
studio. The chdir has to happen before collection imports `rl_ide.api.server`, whose module-level
stores resolve their roots then, which is why it runs in `pytest_sessionstart` rather than a fixture.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("RL_IDE_INTERACT_IDLE_SECONDS", "3600")
os.environ.setdefault("RL_IDE_MAX_INTERACT_SESSIONS", "16")

_ORIGINAL_CWD = Path.cwd()
_scratch_root: Path | None = None


def pytest_sessionstart(session: pytest.Session) -> None:
    global _scratch_root
    _scratch_root = Path(tempfile.mkdtemp(prefix="rlharness-tests-"))
    os.chdir(_scratch_root)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    os.chdir(_ORIGINAL_CWD)
    if _scratch_root is not None:
        shutil.rmtree(_scratch_root, ignore_errors=True)
