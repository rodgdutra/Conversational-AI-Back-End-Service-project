"""
Pydantic models for the FastAPI request/response schema.
"""

from typing import List, Dict, Any, Optional
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