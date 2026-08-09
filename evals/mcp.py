"""MCP strategy server wiring for Inspect / Petri evals."""

from __future__ import annotations

import os
import sys
from typing import Literal

from inspect_ai.tool import ToolSource, mcp_server_stdio

AttackAlgorithm = Literal["crescendo", "actor-attack", "x-teaming"]

_MODULE_BY_ALGORITHM: dict[str, str] = {
    "crescendo": "servers.crescendo",
    "actor-attack": "servers.actorattack",
    "x-teaming": "servers.x_teaming",
}

_NAME_BY_ALGORITHM: dict[str, str] = {
    "crescendo": "Crescendo",
    "actor-attack": "ActorAttack",
    "x-teaming": "X-Teaming",
}

_MODEL_ENV_BY_ALGORITHM: dict[str, str] = {
    "crescendo": "CRESCENDO_MODEL",
    "actor-attack": "ACTORATTACK_MODEL",
    "x-teaming": "X_TEAMING_MODEL",
}


def get_instructions(algorithm: AttackAlgorithm) -> str:
    """Load INSTRUCTIONS from the corresponding MCP server package."""
    module_name = _MODULE_BY_ALGORITHM[algorithm]
    module = __import__(module_name, fromlist=["INSTRUCTIONS"])
    return getattr(module, "INSTRUCTIONS")


def get_attack_mcp(
    algorithm: AttackAlgorithm,
    *,
    model: str | None = None,
) -> ToolSource:
    """Create a stdio MCP ToolSource for the given attack algorithm."""
    env = dict(os.environ)
    if model:
        env[_MODEL_ENV_BY_ALGORITHM[algorithm]] = model

    return mcp_server_stdio(
        name=_NAME_BY_ALGORITHM[algorithm],
        command=sys.executable,
        args=["-m", _MODULE_BY_ALGORITHM[algorithm]],
        env=env,
    )


def crescendo_mcp(*, model: str | None = None) -> ToolSource:
    return get_attack_mcp("crescendo", model=model)


def actorattack_mcp(*, model: str | None = None) -> ToolSource:
    return get_attack_mcp("actor-attack", model=model)


def xteaming_mcp(*, model: str | None = None) -> ToolSource:
    return get_attack_mcp("x-teaming", model=model)
