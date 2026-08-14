"""Exceptions importable before (and without) config — the CLI bootstrap
must be able to catch a broken data contract without itself triggering a
config import — plus cross-module control-flow signals that need the same
import-light home."""


class ConfigError(RuntimeError):
    """The show data directory doesn't satisfy the data contract."""


class ThreadResumeError(RuntimeError):
    """--thread points at a checkpoint that doesn't exist or doesn't match
    the invoked run (wrong scene/episode, or a typo'd id)."""


class BudgetPause(Exception):
    """Control flow, not a failure: the spend cap stopped a --planner run
    at a node boundary. Raised by the planner's node gates before any
    billed call and caught in planner._run_graph — it never escapes to the
    CLI (the pause tests pin that the runners return normally). Raising
    (instead of routing to END) is what keeps the thread resumable: END
    marks a checkpoint complete for good."""
