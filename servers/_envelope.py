"""Shared JSON envelope helpers for client-carried MCP state."""

from __future__ import annotations

import json
import re
from typing import Any


def ensure_json_keyword(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """DeepSeek (and some OpenAI-compatible APIs) require the word 'json' in the
    prompt when using response_format=json_object."""
    blob = " ".join(str(m.get("content", "")) for m in messages).lower()
    if "json" in blob:
        return messages
    return [
        *messages,
        {"role": "user", "content": "Return a valid json object."},
    ]


def loads_llm_json(content: str | None) -> dict[str, Any]:
    """Parse model JSON output; tolerate fences / leading prose / trailing junk."""
    if content is None or not str(content).strip():
        raise ValueError("API returned empty content")
    text = str(content).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if not text:
        raise ValueError("API returned empty content after stripping fences")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise ValueError(f"API returned non-JSON content: {text[:120]!r}") from None
        data, _idx = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    return data


def dump_envelope(result: str, state: dict[str, Any] | None) -> str:
    """Serialize tool output as {"result": ..., "state": ...}."""
    return json.dumps({"result": result, "state": state or {}}, ensure_ascii=False)


def _unwrap_state_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Accept either a bare state object or a full {result, state} envelope."""
    inner = data.get("state")
    if "result" in data and isinstance(inner, dict):
        return inner
    return data


def parse_state(state: str | dict[str, Any] | None) -> dict[str, Any]:
    """Parse client-echoed state JSON; empty/missing → {}.

    Tolerates common auditor mistakes: wrapping the full envelope, or appending
    trailing characters after a valid JSON object (e.g. an extra '}').
    """
    if state is None:
        return {}
    if isinstance(state, dict):
        return _unwrap_state_dict(state)
    text = str(state).strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data, _idx = json.JSONDecoder().raw_decode(text)
    if not isinstance(data, dict):
        raise ValueError("state must be a JSON object")
    return _unwrap_state_dict(data)
