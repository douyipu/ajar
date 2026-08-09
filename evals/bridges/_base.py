"""Shared helpers for Inspect store_as ↔ attack-handler bridges."""

from __future__ import annotations

import json
from typing import Any

from inspect_ai.util import StoreModel, store_as
from pydantic import Field


class AttackBridgeStore(StoreModel):
    """Per-sample opaque attack state dict (mirrors MCP envelope.state)."""

    state: dict[str, Any] = Field(default_factory=dict)


def parse_envelope(raw: str) -> tuple[str, dict[str, Any]]:
    data = json.loads(raw)
    if not isinstance(data, dict) or "result" not in data:
        raise ValueError(f"Unexpected MCP envelope: {raw[:200]}")
    state = data.get("state") or {}
    if not isinstance(state, dict):
        raise ValueError("MCP envelope state must be an object")
    return str(data["result"]), state


def dump_store_state(store_type: type[AttackBridgeStore] = AttackBridgeStore) -> str:
    return json.dumps(store_as(store_type).state, ensure_ascii=False)


def save_store_state(
    state: dict[str, Any],
    store_type: type[AttackBridgeStore] = AttackBridgeStore,
) -> None:
    store_as(store_type).state = state


async def call_handler(
    handler,
    *,
    store_type: type[AttackBridgeStore] = AttackBridgeStore,
    pass_state: bool = True,
    **kwargs: Any,
) -> str:
    """Invoke an MCP tool handler, syncing state through store_as."""
    if pass_state:
        kwargs = {**kwargs, "state": dump_store_state(store_type)}
    result, state = parse_envelope(await handler(**kwargs))
    save_store_state(state, store_type)
    return result
