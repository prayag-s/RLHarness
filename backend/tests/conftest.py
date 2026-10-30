"""Test-suite defaults.

Interactive sessions are reaped aggressively in production (15 min idle, 8 concurrent). Tests must
not inherit that: a reaped worker is respawned on the next request, and spawning imports torch, which
dominated the suite runtime. The test process closes every worker on exit anyway (stdin EOF).
"""

import os

os.environ.setdefault("RL_IDE_INTERACT_IDLE_SECONDS", "3600")
os.environ.setdefault("RL_IDE_MAX_INTERACT_SESSIONS", "16")
