"""Exceptions importable before (and without) config — the CLI bootstrap
must be able to catch a broken data contract without itself triggering a
config import."""


class ConfigError(RuntimeError):
    """The show data directory doesn't satisfy the data contract."""


class ThreadResumeError(RuntimeError):
    """--thread points at a checkpoint that doesn't exist or doesn't match
    the invoked run (wrong scene/episode, or a typo'd id)."""
