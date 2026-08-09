"""Shared helpers for Inspect store_as ↔ attack-handler bridges."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from inspect_ai.model import ChatMessage, ChatMessageAssistant
from inspect_ai.tool import ToolError
from inspect_ai.util import StoreModel, store_as
from inspect_petri.target import controller
from pydantic import Field

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


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


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks; keep the visible reply text."""
    return _THINK_RE.sub("", text).strip()


def target_response_from_messages(
    messages: Sequence[ChatMessage],
    real_id: str,
    *,
    display_id: str | None = None,
) -> str:
    """Look up an assistant reply by real message id and return cleaned text."""
    label = display_id or real_id
    for message in messages:
        if getattr(message, "id", None) != real_id:
            continue
        if not isinstance(message, ChatMessageAssistant):
            raise ToolError(
                f"Message {label!r} is role={getattr(message, 'role', None)!r}, "
                "expected a target assistant response."
            )
        return strip_think_tags(message.text)
    raise ToolError(
        f"Unknown message_id {label!r}. Use a short id from resume output "
        "(e.g. M8), or a real target message id still present in the conversation."
    )


def resolve_target_response(message_id: str) -> str:
    """Resolve a Petri short/real message id to the Target's reply text."""
    if not str(message_id).strip():
        raise ToolError("message_id is required (e.g. M8 from resume output).")
    ctrl = controller()
    real_id = ctrl.resolve_id(message_id)
    return target_response_from_messages(
        ctrl.state.messages,
        real_id,
        display_id=message_id,
    )
