"""Crescendo MCP Server state management - matches src/ajar/tools/crescendo.py CrescendoState"""

import json
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    """Single message in conversation context"""

    role: str = ""
    content: str = ""


@dataclass
class CrescendoState:
    """Crescendo state - matches src/ajar/tools/crescendo.py CrescendoState"""

    # Conversation context: system msg, user msg for question generation, assistant responses
    conversation_context: list[Message] = field(default_factory=list)

    # Fixed after init
    objective: Optional[str] = None
    max_turns: Optional[int] = None

    # Incremented when response is not refusal; unchanged on refusal
    executed_turns: int = 0

    # Last generated question (conversation_context does not include it)
    last_question: Optional[str] = None

    # Last question: refused?, success?, reason (set by evaluate only)
    is_refusal: Optional[bool] = None
    success_score: Optional[float] = None
    success_rationale: Optional[str] = None

    # Session ID for MCP
    session_id: str = ""

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict"""
        return {
            "conversation_context": [
                {"role": m.role, "content": m.content} for m in self.conversation_context
            ],
            "objective": self.objective,
            "max_turns": self.max_turns,
            "executed_turns": self.executed_turns,
            "last_question": self.last_question,
            "is_refusal": self.is_refusal,
            "success_score": self.success_score,
            "success_rationale": self.success_rationale,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CrescendoState":
        """Create from dict"""
        conversation_context = []
        for m in data.get("conversation_context", []):
            conversation_context.append(
                Message(role=m.get("role", ""), content=m.get("content", ""))
            )

        return cls(
            conversation_context=conversation_context,
            objective=data.get("objective"),
            max_turns=data.get("max_turns"),
            executed_turns=data.get("executed_turns", 0),
            last_question=data.get("last_question"),
            is_refusal=data.get("is_refusal"),
            success_score=data.get("success_score"),
            success_rationale=data.get("success_rationale"),
            session_id=data.get("session_id", ""),
        )


def _get_session_file_path(session_id: str) -> Path:
    """Get session state file path"""
    return Path(tempfile.gettempdir()) / f"crescendo_session_{session_id}.json"


# Cache current session state
_current_state: CrescendoState | None = None
_current_session_id: str | None = None


def get_state() -> CrescendoState:
    """Get current session state"""
    global _current_state, _current_session_id

    session_id = os.environ.get("CRESCENDO_SESSION_ID", "default")

    if _current_session_id != session_id or _current_state is None:
        state_file = _get_session_file_path(session_id)
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                _current_state = CrescendoState.from_dict(data)
            except Exception:
                _current_state = CrescendoState(session_id=session_id)
        else:
            _current_state = CrescendoState(session_id=session_id)
        _current_state.session_id = session_id
        _current_session_id = session_id

    return _current_state


def save_state() -> None:
    """Save current session state"""
    global _current_state

    if _current_state:
        state_file = _get_session_file_path(_current_state.session_id)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=state_file.parent, delete=False
        ) as tf:
            json.dump(_current_state.to_dict(), tf)
        os.replace(tf.name, state_file)
