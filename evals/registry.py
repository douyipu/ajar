"""Inspect-AI task registry entrypoint (replaces ajar.tasks._registry)."""

from evals.crescendo import crescendo_harmbench  # noqa: F401
from evals.actorattack import actorattack_harmbench  # noqa: F401
from evals.xteaming import xteaming_harmbench  # noqa: F401
