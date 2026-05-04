"""
Pydantic models for the FastAPI request/response schema.
"""

from typing import List, Dict, Any, Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Payload sent by the client to the /chat endpoint."""

    session_id: str = Field(
        ...,
        description=(
            "Unique identifier for the conversation session. "
            "The same session_id must be used across turns to maintain context."
        ),
        examples=["user-session-abc123"],
    )
    message: str = Field(
        ...,
        description="The patient's message text.",
        examples=["I'd like to check my upcoming appointments."],
    )


class ReviewVerdict(BaseModel):
    """
    Quality-control verdict produced by the reviewer agent on each turn.

    The reviewer evaluates the primary assistant's response against five criteria
    and reports its findings here.

    ``verdict`` values
    ------------------
    ``"pass"``  — No issues detected; the response is safe to deliver as-is.
    ``"flag"``  — Non-critical issues noted (e.g. user stalling, gibberish input,
                  minor scope drift).  The response is still delivered but the
                  flag is recorded for audit purposes.
    ``"block"`` — Critical issues found (e.g. sensitive data exposure, severe
                  hallucination, or clear out-of-scope content).  The original
                  response is suppressed and a safe replacement is delivered
                  to the user instead.

    ``action`` values
    -----------------
    ``"none"``    — Nothing beyond logging was required.
    ``"warn"``    — The issue was flagged in the audit trail.
    ``"replace"`` — The user received a safe replacement message.
    """

    verdict: Literal["pass", "flag", "block"] = Field(
        ...,
        description="Overall quality-control verdict for this turn.",
    )
    flags: List[str] = Field(
        default_factory=list,
        description="List of specific issues raised by the review tools.",
    )
    action: Literal["none", "warn", "replace"] = Field(
        ...,
        description="Action taken by the reviewer based on the verdict.",
    )
    summary: str = Field(
        default="",
        description="Brief human-readable explanation of the review outcome.",
    )


class ChatResponse(BaseModel):
    """Response returned by the /chat endpoint."""

    session_id: str = Field(..., description="Echo of the session identifier.")
    reply: str = Field(..., description="The assistant's reply to the patient.")
    verified: bool = Field(
        ...,
        description="Whether the patient has been successfully verified in this session.",
    )
    state_id: Optional[int] = Field(
        None,
        description="The ID of the state created by this interaction.",
    )
    review: Optional[ReviewVerdict] = Field(
        None,
        description=(
            "Quality-control verdict from the reviewer agent. "
            "Present on every turn once the review agent has run. "
            "When verdict is 'block' the reply field already contains the "
            "safe replacement message chosen by the reviewer."
        ),
    )


class StateMetadata(BaseModel):
    """Metadata for a conversation state."""

    session_id: str = Field(..., description="The session identifier.")
    state_id: int = Field(..., description="The state identifier within the session.")
    created_at: str = Field(..., description="ISO timestamp when the state was created.")


class SessionStatesResponse(BaseModel):
    """Response for the session states endpoint."""

    session_id: str = Field(..., description="The session identifier.")
    states: List[StateMetadata] = Field(..., description="List of states in this session.")


class StateTransitionData(BaseModel):
    """Data about a transition between states."""

    id: int = Field(..., description="Transition identifier.")
    session_id: str = Field(..., description="The session identifier.")
    from_state_id: int = Field(..., description="Source state ID.")
    to_state_id: int = Field(..., description="Destination state ID.")
    transition_type: Optional[str] = Field(None, description="Type of transition.")
    created_at: str = Field(..., description="When the transition occurred.")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional transition data.")


class SessionTransitionsResponse(BaseModel):
    """Response for the session transitions endpoint."""

    session_id: str = Field(..., description="The session identifier.")
    transitions: List[StateTransitionData] = Field(
        ..., description="List of transitions in this session."
    )