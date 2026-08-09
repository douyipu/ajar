"""Crescendo state models — client-carried, no server persistence."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class CrescendoState(BaseModel):
    conversation_context: list[Message] = Field(default_factory=list)
    objective: Optional[str] = None
    max_turns: Optional[int] = None
    executed_turns: int = 0
    last_question: Optional[str] = None
    is_refusal: Optional[bool] = None
    success_score: Optional[float] = None
    success_rationale: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrescendoState:
        return cls.model_validate(data)
