"""
FastAPI application — Conversational AI Appointment Assistant.

Run with:
    uvicorn app.main:app --reload
"""

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from app.agent.graph import compiled_graph
from app.logger import get_logger
from app.models import ChatRequest, ChatResponse
from app.config import config

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()

if not config.OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY environment variable is not set. "
        "Copy .env.example to .env and fill in your OpenRouter API key."
    )

logger.info("Service starting | OPENROUTER_API_KEY configured ✓")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Conversational AI Appointment Assistant",
    description=(
        "A healthcare back-end service that uses a LangGraph multi-agent "
        "workflow to help patients manage their appointments through a "
        "conversational interface. Identity verification is required before "
        "any appointment action is performed."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# In-memory session store
# Keyed by session_id → the latest AgentState snapshot returned by the graph.
# In production this should be replaced with a persistent store (e.g. Redis).
# ---------------------------------------------------------------------------

session_store: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_last_ai_reply(state: dict) -> str:
    """Extract the last AI message text from the graph state."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            # AIMessage.content can be a str or a list of content blocks
            if isinstance(msg.content, str):
                return msg.content
            if isinstance(msg.content, list):
                texts = [
                    block["text"]
                    for block in msg.content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return " ".join(texts)
    return "I'm sorry, something went wrong. Please try again."


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def health_check():
    """Simple health-check endpoint."""
    return {"status": "ok", "service": "Conversational AI Appointment Assistant"}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a single conversational turn.

    The client must supply a stable **session_id** across all turns of the
    same conversation.  The service maintains state (including whether the
    patient has been verified) in memory for the lifetime of the session.

    **Flow**
    1. First turn: the assistant greets the patient and requests identity info.
    2. Once the patient provides their full name, phone, and date of birth the
       assistant verifies them automatically.
    3. After successful verification the patient may list, confirm, or cancel
       appointments freely.
    """
    session_id = request.session_id
    user_message = request.message.strip()

    if not user_message:
        logger.warning("POST /chat | Empty message received | session_id='%s'", session_id)
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # -----------------------------------------------------------------------
    # Retrieve or initialise session state
    # -----------------------------------------------------------------------
    current_state = session_store.get(session_id)
    is_new_session = current_state is None

    if is_new_session:
        logger.info("POST /chat | New session started | session_id='%s'", session_id)
        current_state = {
            "messages": [],
            "verified": False,
            "patient_id": None,
            "patient_name": None,
            "pending_action": None,
        }
    else:
        turn = len(current_state["messages"]) + 1
        logger.info(
            "POST /chat | Continuing session | session_id='%s' turn=%d verified=%s",
            session_id, turn, current_state.get("verified", False),
        )

    logger.info("POST /chat | User message | session_id='%s' msg='%.100s'", session_id, user_message)

    # -----------------------------------------------------------------------
    # Append the user's new message and invoke the graph
    # -----------------------------------------------------------------------
    input_state = {
        **current_state,
        "messages": current_state["messages"] + [HumanMessage(content=user_message)],
    }

    try:
        new_state: dict = compiled_graph.invoke(input_state)
    except Exception as exc:
        logger.error(
            "POST /chat | Agent error | session_id='%s' error='%s'",
            session_id, str(exc), exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(exc)}",
        ) from exc

    # -----------------------------------------------------------------------
    # Persist updated state
    # -----------------------------------------------------------------------
    session_store[session_id] = new_state

    # -----------------------------------------------------------------------
    # Build and return the response
    # -----------------------------------------------------------------------
    reply = _get_last_ai_reply(new_state)
    verified = bool(new_state.get("verified", False))

    logger.info(
        "POST /chat | Response sent | session_id='%s' verified=%s reply_len=%d",
        session_id, verified, len(reply),
    )
    logger.debug("POST /chat | Reply preview | session_id='%s' reply='%.120s'", session_id, reply)

    return ChatResponse(session_id=session_id, reply=reply, verified=verified)


@app.delete("/chat/{session_id}", tags=["Chat"])
async def clear_session(session_id: str):
    """
    Clear the conversation history for a given session.
    Useful for starting a fresh conversation without changing the session ID.
    """
    if session_id in session_store:
        del session_store[session_id]
        logger.info("DELETE /chat | Session cleared | session_id='%s'", session_id)
        return {"detail": f"Session '{session_id}' has been cleared."}
    logger.warning("DELETE /chat | Session not found | session_id='%s'", session_id)
    raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
