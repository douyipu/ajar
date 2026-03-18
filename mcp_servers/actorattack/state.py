"""ActorAttack MCP Server state management - matches src/ajar/tools/actorattack.py ActorAttackState"""

import json
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """Single message in conversation history"""

    role: str = ""
    content: str = ""


@dataclass
class ActorItem:
    """Actor data"""

    actor_name: str = ""
    relationship: str = ""


@dataclass
class ActorAttackState:
    """ActorAttack state - matches src/ajar/tools/actorattack.py ActorAttackState"""

    # Global config
    original_query: str | None = None
    harm_target: str | None = None
    query_details: dict = field(default_factory=dict)
    max_actors: int | None = None

    # Actor Network state
    network_history: list[Message] = field(default_factory=list)
    used_actor_names: list[str] = field(default_factory=list)
    tried_actors_count: int = 0

    # Current Actor attack state
    current_actor: ActorItem | None = None
    current_queries: list[str] = field(default_factory=list)
    current_query_index: int = 0

    # Interaction state
    last_target_response: str | None = None
    last_successful_response: str | None = None
    last_response_type: str = "unknown"
    is_refusal_pending: bool = False
    current_question_modified: bool = False

    # Summary phase state
    in_summary_phase: bool = False
    summary_attempt: int = 0
    summary_needs_retry: bool = False

    # Actor switch flag
    need_new_actor: bool = False

    # Success state
    success: bool = False
    success_score: int | None = None
    success_reason: str | None = None

    # Session ID for MCP
    session_id: str = ""

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict"""
        return {
            "original_query": self.original_query,
            "harm_target": self.harm_target,
            "query_details": self.query_details,
            "max_actors": self.max_actors,
            "network_history": [
                {"role": m.role, "content": m.content} for m in self.network_history
            ],
            "used_actor_names": self.used_actor_names,
            "tried_actors_count": self.tried_actors_count,
            "current_actor": (
                {
                    "actor_name": self.current_actor.actor_name,
                    "relationship": self.current_actor.relationship,
                }
                if self.current_actor
                else None
            ),
            "current_queries": self.current_queries,
            "current_query_index": self.current_query_index,
            "last_target_response": self.last_target_response,
            "last_successful_response": self.last_successful_response,
            "last_response_type": self.last_response_type,
            "is_refusal_pending": self.is_refusal_pending,
            "current_question_modified": self.current_question_modified,
            "in_summary_phase": self.in_summary_phase,
            "summary_attempt": self.summary_attempt,
            "summary_needs_retry": self.summary_needs_retry,
            "need_new_actor": self.need_new_actor,
            "success": self.success,
            "success_score": self.success_score,
            "success_reason": self.success_reason,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActorAttackState":
        """Create from dict"""
        network_history = []
        for m in data.get("network_history", []):
            network_history.append(Message(role=m.get("role", ""), content=m.get("content", "")))

        current_actor = None
        if data.get("current_actor"):
            ca = data["current_actor"]
            current_actor = ActorItem(
                actor_name=ca.get("actor_name", ""),
                relationship=ca.get("relationship", ""),
            )

        return cls(
            original_query=data.get("original_query"),
            harm_target=data.get("harm_target"),
            query_details=data.get("query_details", {}),
            max_actors=data.get("max_actors"),
            network_history=network_history,
            used_actor_names=data.get("used_actor_names", []),
            tried_actors_count=data.get("tried_actors_count", 0),
            current_actor=current_actor,
            current_queries=data.get("current_queries", []),
            current_query_index=data.get("current_query_index", 0),
            last_target_response=data.get("last_target_response"),
            last_successful_response=data.get("last_successful_response"),
            last_response_type=data.get("last_response_type", "unknown"),
            is_refusal_pending=data.get("is_refusal_pending", False),
            current_question_modified=data.get("current_question_modified", False),
            in_summary_phase=data.get("in_summary_phase", False),
            summary_attempt=data.get("summary_attempt", 0),
            summary_needs_retry=data.get("summary_needs_retry", False),
            need_new_actor=data.get("need_new_actor", False),
            success=data.get("success", False),
            success_score=data.get("success_score"),
            success_reason=data.get("success_reason"),
            session_id=data.get("session_id", ""),
        )


def _get_session_file_path(session_id: str) -> Path:
    """Get session state file path"""
    return Path(tempfile.gettempdir()) / f"actorattack_session_{session_id}.json"


# Cache current session state
_current_state: ActorAttackState | None = None
_current_session_id: str | None = None


def get_state() -> ActorAttackState:
    """Get current session state"""
    global _current_state, _current_session_id

    session_id = os.environ.get("ACTORATTACK_SESSION_ID", "default")

    if _current_session_id != session_id or _current_state is None:
        state_file = _get_session_file_path(session_id)
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                _current_state = ActorAttackState.from_dict(data)
            except Exception:
                _current_state = ActorAttackState(session_id=session_id)
        else:
            _current_state = ActorAttackState(session_id=session_id)
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
