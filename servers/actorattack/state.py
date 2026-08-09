"""ActorAttack state models — client-carried, no server persistence."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = ""
    content: str = ""


class ActorItem(BaseModel):
    actor_name: str = ""
    relationship: str = ""


class ActorAttackState(BaseModel):
    """ActorAttack state — matches mcp_servers fields except session_id."""

    # Global config
    original_query: Optional[str] = None
    harm_target: Optional[str] = None
    query_details: dict[str, Any] = Field(default_factory=dict)
    max_actors: Optional[int] = None

    # Actor Network state
    network_history: list[Message] = Field(default_factory=list)
    used_actor_names: list[str] = Field(default_factory=list)
    tried_actors_count: int = 0

    # Current Actor attack state
    current_actor: Optional[ActorItem] = None
    current_queries: list[str] = Field(default_factory=list)
    current_query_index: int = 0

    # Interaction state
    last_target_response: Optional[str] = None
    last_successful_response: Optional[str] = None
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
    success_score: Optional[int] = None
    success_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActorAttackState:
        return cls.model_validate(data)
