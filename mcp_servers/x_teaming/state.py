"""X-Teaming MCP Server state management - matches src/ajar/tools/xteaming.py XTeamingState"""

import json
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """Single message in conversation history"""
    role: str | None = None
    content: str | None = None


@dataclass
class CandidateAttempt:
    """Candidate attempt within a phase"""
    attack_message: str = ""
    target_response: str = ""
    score: int = 0
    reason: str = ""


@dataclass
class XTeamingState:
    """X-Teaming state - matches src/ajar/tools/xteaming.py XTeamingState"""

    # Fixed parameters after init
    objective: str | None = None
    max_turns: int | None = None
    max_strategies: int | None = None
    max_turns_per_phase: int | None = None
    textgrad_enabled: bool = True

    # Current strategy
    strategy: dict | None = None
    strategy_history: list[dict] = field(default_factory=list)

    # Phase tracking
    plan_phase: int = 1
    num_phases: int = 0
    attempts_this_phase: int = 0
    plan_score_history: list[int] = field(default_factory=list)
    plan_reason_history: list[str] = field(default_factory=list)

    # Committed conversation history
    committed_history: list[dict] = field(default_factory=list)  # {"role": str, "content": str}

    # Phase candidates (for best candidate selection)
    phase_candidates: list[dict] = field(default_factory=list)  # CandidateAttempt as dict

    # Current turn state
    last_attack_message: str | None = None
    last_score: int | None = None
    last_reason: str | None = None
    last_target_response: str | None = None
    last_message_index: int | None = None

    # Success flag
    jailbreak_achieved: bool = False

    # TextGrad optimization signal
    needs_optimization: bool = False
    pending_optimization: dict | None = None

    # Best candidate selection
    return_best: bool = False

    # TextGrad optimization status
    last_optimization_failed: bool = False

    # Session ID for MCP
    session_id: str = ""

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict"""
        return {
            "objective": self.objective,
            "max_turns": self.max_turns,
            "max_strategies": self.max_strategies,
            "max_turns_per_phase": self.max_turns_per_phase,
            "textgrad_enabled": self.textgrad_enabled,
            "strategy": self.strategy,
            "strategy_history": self.strategy_history,
            "plan_phase": self.plan_phase,
            "num_phases": self.num_phases,
            "attempts_this_phase": self.attempts_this_phase,
            "plan_score_history": self.plan_score_history,
            "plan_reason_history": self.plan_reason_history,
            "committed_history": self.committed_history,
            "phase_candidates": self.phase_candidates,
            "last_attack_message": self.last_attack_message,
            "last_score": self.last_score,
            "last_reason": self.last_reason,
            "last_target_response": self.last_target_response,
            "last_message_index": self.last_message_index,
            "jailbreak_achieved": self.jailbreak_achieved,
            "needs_optimization": self.needs_optimization,
            "pending_optimization": self.pending_optimization,
            "return_best": self.return_best,
            "last_optimization_failed": self.last_optimization_failed,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "XTeamingState":
        """Create from dict"""
        return cls(
            objective=data.get("objective"),
            max_turns=data.get("max_turns"),
            max_strategies=data.get("max_strategies"),
            max_turns_per_phase=data.get("max_turns_per_phase"),
            textgrad_enabled=data.get("textgrad_enabled", True),
            strategy=data.get("strategy"),
            strategy_history=data.get("strategy_history", []),
            plan_phase=data.get("plan_phase", 1),
            num_phases=data.get("num_phases", 0),
            attempts_this_phase=data.get("attempts_this_phase", 0),
            plan_score_history=data.get("plan_score_history", []),
            plan_reason_history=data.get("plan_reason_history", []),
            committed_history=data.get("committed_history", []),
            phase_candidates=data.get("phase_candidates", []),
            last_attack_message=data.get("last_attack_message"),
            last_score=data.get("last_score"),
            last_reason=data.get("last_reason"),
            last_target_response=data.get("last_target_response"),
            last_message_index=data.get("last_message_index"),
            jailbreak_achieved=data.get("jailbreak_achieved", False),
            needs_optimization=data.get("needs_optimization", False),
            pending_optimization=data.get("pending_optimization"),
            return_best=data.get("return_best", False),
            last_optimization_failed=data.get("last_optimization_failed", False),
            session_id=data.get("session_id", ""),
        )


def _get_session_file_path(session_id: str) -> Path:
    """Get session state file path"""
    return Path(tempfile.gettempdir()) / f"x_teaming_session_{session_id}.json"


# Cache current session state
_current_state: XTeamingState | None = None
_current_session_id: str | None = None


def get_state() -> XTeamingState:
    """Get current session state"""
    global _current_state, _current_session_id

    session_id = os.environ.get("X_TEAMING_SESSION_ID", "default")

    if _current_session_id != session_id or _current_state is None:
        state_file = _get_session_file_path(session_id)
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                _current_state = XTeamingState.from_dict(data)
            except Exception:
                _current_state = XTeamingState(session_id=session_id)
        else:
            _current_state = XTeamingState(session_id=session_id)
        _current_state.session_id = session_id
        _current_session_id = session_id

    return _current_state


def save_state() -> None:
    """Save current session state"""
    global _current_state

    if _current_state:
        state_file = _get_session_file_path(_current_state.session_id)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=state_file.parent, delete=False) as tf:
            json.dump(_current_state.to_dict(), tf)
        os.replace(tf.name, state_file)


def candidate_to_dict(c: CandidateAttempt) -> dict:
    """Convert CandidateAttempt to dict"""
    return {
        "attack_message": c.attack_message,
        "target_response": c.target_response,
        "score": c.score,
        "reason": c.reason,
    }


def dict_to_candidate(d: dict) -> CandidateAttempt:
    """Convert dict to CandidateAttempt"""
    return CandidateAttempt(
        attack_message=d.get("attack_message", ""),
        target_response=d.get("target_response", ""),
        score=int(d.get("score", 0)),
        reason=d.get("reason", ""),
    )
