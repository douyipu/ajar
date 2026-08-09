"""X-Teaming state models — client-carried, no server persistence."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class CandidateAttempt(BaseModel):
    attack_message: str = ""
    target_response: str = ""
    score: int = 0
    reason: str = ""


class XTeamingState(BaseModel):
    # Fixed parameters after init
    objective: Optional[str] = None
    max_turns: Optional[int] = None
    max_strategies: Optional[int] = None
    max_turns_per_phase: Optional[int] = None
    textgrad_enabled: bool = True

    # Current strategy
    strategy: Optional[dict[str, Any]] = None
    strategy_history: list[dict[str, Any]] = Field(default_factory=list)

    # Phase tracking
    plan_phase: int = 1
    num_phases: int = 0
    attempts_this_phase: int = 0
    plan_score_history: list[int] = Field(default_factory=list)
    plan_reason_history: list[str] = Field(default_factory=list)

    # Committed conversation history: {"role": str, "content": str}
    committed_history: list[dict[str, Any]] = Field(default_factory=list)

    # Phase candidates (CandidateAttempt as dict)
    phase_candidates: list[dict[str, Any]] = Field(default_factory=list)

    # Current turn state
    last_attack_message: Optional[str] = None
    last_score: Optional[int] = None
    last_reason: Optional[str] = None
    last_target_response: Optional[str] = None
    last_message_index: Optional[int] = None

    # Success flag
    jailbreak_achieved: bool = False

    # TextGrad optimization signal
    needs_optimization: bool = False
    pending_optimization: Optional[dict[str, Any]] = None

    # Best candidate selection
    return_best: bool = False

    # TextGrad optimization status
    last_optimization_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> XTeamingState:
        return cls.model_validate(data)


def candidate_to_dict(c: CandidateAttempt) -> dict[str, Any]:
    return c.model_dump()


def dict_to_candidate(d: dict[str, Any]) -> CandidateAttempt:
    return CandidateAttempt.model_validate(d)
