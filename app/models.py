"""
Pydantic models for the FastAPI request/response schema.
"""

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
