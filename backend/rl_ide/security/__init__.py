"""Isolation for model-generated code (see :mod:`rl_ide.security.sandbox`)."""

from rl_ide.security.sandbox import (  # noqa: F401
    SandboxError,
    SandboxResult,
    SandboxUnavailable,
    isolation_available,
    isolation_tool,
    run_worker,
    stage_case,
)


def enforce_isolation_requirement() -> None:
    """Fails fast when a deployment promises isolation it cannot deliver.

    Set ``RL_IDE_REQUIRE_ISOLATION=1`` (the container image does) so a host without ``bwrap`` —
    hardened runtimes that forbid unprivileged user namespaces, for example — refuses to start
    instead of silently serving a hosted tenant with in-process execution.
    """
    import os

    if os.environ.get("RL_IDE_REQUIRE_ISOLATION") != "1":
        return
    tool = isolation_tool()
    if tool is None:
        raise RuntimeError(
            "RL_IDE_REQUIRE_ISOLATION=1 but no isolation mechanism is available: install bubblewrap "
            "and allow unprivileged user namespaces, or unset the variable for single-user use."
        )
