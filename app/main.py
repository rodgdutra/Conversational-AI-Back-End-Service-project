"""
FastAPI application — Conversational AI Appointment Assistant.

Run with:
    uvicorn app.main:app --reload
"""

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage

from app.agent.graph import compiled_graph
from app.agent.persistence import AsyncStatePersistenceService
from app.db import init_db
from app.data import initialize_mock_data
from app.logger import get_logger
from app.models import (
    ChatRequest, ChatResponse, SessionStatesResponse, 
    SessionTransitionsResponse, StateMetadata, StateTransitionData
)
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

# Initialize the database and mock data
try:
    # Create database schema
    init_db()
    logger.info("Database schema initialized successfully")
    
    # Seed database with sample data if needed
    initialize_mock_data()
    logger.info("Mock data initialization complete")
except Exception as e:
    logger.error(f"Database initialization failed: {str(e)}")
    raise

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

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session state persistence
# ---------------------------------------------------------------------------

# Dependency for accessing the state persistence service
def get_state_service():
    return AsyncStatePersistenceService()


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


@app.get("/sessions/{session_id}/states", response_model=SessionStatesResponse, tags=["Session History"])
async def get_session_states(
    session_id: str,
    state_service: AsyncStatePersistenceService = Depends(get_state_service)
):
    """
    Get all states for a session.
    
    This endpoint allows you to see the history of states for a particular
    conversation session, which is useful for debugging or analysis.
    """
    states = await state_service.list_states(session_id)
    
    if not states:
        logger.warning("GET /sessions/states | No states found | session_id='%s'", session_id)
        raise HTTPException(status_code=404, detail=f"No states found for session '{session_id}'")
    
    logger.info("GET /sessions/states | Retrieved %d states | session_id='%s'", len(states), session_id)
    return SessionStatesResponse(session_id=session_id, states=states)


@app.get("/sessions/{session_id}/states/{state_id}", tags=["Session History"])
async def get_session_state(
    session_id: str,
    state_id: int,
    state_service: AsyncStatePersistenceService = Depends(get_state_service)
):
    """
    Get a specific state for a session.
    
    This endpoint allows you to retrieve a particular state in the conversation history.
    """
    state = await state_service.load_state(session_id, state_id)
    
    if not state:
        logger.warning("GET /sessions/state | State not found | session_id='%s' state_id=%d", session_id, state_id)
        raise HTTPException(
            status_code=404, 
            detail=f"State {state_id} not found for session '{session_id}'"
        )
    
    logger.info("GET /sessions/state | Retrieved state | session_id='%s' state_id=%d", session_id, state_id)
    return state


@app.get("/sessions/{session_id}/transitions", response_model=SessionTransitionsResponse, tags=["Session History"])
async def get_session_transitions(
    session_id: str,
    state_service: AsyncStatePersistenceService = Depends(get_state_service)
):
    """
    Get all transitions for a session.
    
    This endpoint allows you to see how the conversation progressed from one state to another,
    including metadata about each transition.
    """
    transitions = await state_service.get_state_transitions(session_id)
    
    if not transitions:
        logger.warning(
            "GET /sessions/transitions | No transitions found | session_id='%s'", 
            session_id
        )
        raise HTTPException(
            status_code=404, 
            detail=f"No transitions found for session '{session_id}'"
        )
    
    logger.info(
        "GET /sessions/transitions | Retrieved %d transitions | session_id='%s'", 
        len(transitions), 
        session_id
    )
    return SessionTransitionsResponse(session_id=session_id, transitions=transitions)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    state_service: AsyncStatePersistenceService = Depends(get_state_service)
) -> ChatResponse:
    """
    Process a single conversational turn.

    The client must supply a stable **session_id** across all turns of the
    same conversation. The service persists state (including whether the
    patient has been verified) in PostgreSQL for the lifetime of the session.

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
    current_state = await state_service.load_state(session_id)
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
        # Get metadata if available
        state_id = current_state.get("_metadata", {}).get("state_id", 0)
        logger.info(
            "POST /chat | Continuing session | session_id='%s' state_id=%d verified=%s",
            session_id, state_id, current_state.get("verified", False),
        )

    logger.info("POST /chat | User message | session_id='%s' msg='%.100s'", session_id, user_message)

    # -----------------------------------------------------------------------
    # Append the user's new message and invoke the graph
    # -----------------------------------------------------------------------
    input_state = {
        **current_state,
        "messages": current_state.get("messages", []) + [HumanMessage(content=user_message)],
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
    # Persist updated state with transition information
    # -----------------------------------------------------------------------
    transition_data = {
        "user_message": user_message,
        "reply_length": len(_get_last_ai_reply(new_state)),
        "is_verified": bool(new_state.get("verified", False)),
    }
    
    save_result = await state_service.save_state(
        session_id=session_id, 
        state=new_state,
        transition_type="user_message",
        transition_data=transition_data
    )
    
    if not save_result:
        logger.warning(
            "POST /chat | Failed to persist state | session_id='%s'",
            session_id
        )

    # -----------------------------------------------------------------------
    # Build and return the response
    # -----------------------------------------------------------------------
    reply = _get_last_ai_reply(new_state)
    verified = bool(new_state.get("verified", False))

    # Get state metadata if available
    state_metadata = new_state.get("_metadata", {})
    state_id = state_metadata.get("state_id", 0)

    logger.info(
        "POST /chat | Response sent | session_id='%s' state_id=%d verified=%s reply_len=%d",
        session_id, state_id, verified, len(reply),
    )
    logger.debug("POST /chat | Reply preview | session_id='%s' reply='%.120s'", session_id, reply)

    return ChatResponse(
        session_id=session_id, 
        reply=reply, 
        verified=verified,
        state_id=state_id
    )


@app.delete("/chat/{session_id}", tags=["Chat"])
async def clear_session(
    session_id: str,
    state_service: AsyncStatePersistenceService = Depends(get_state_service)
):
    """
    Clear the conversation history for a given session.
    Useful for starting a fresh conversation without changing the session ID.
    """
    # First check if the session exists
    current_state = await state_service.load_state(session_id)
    if not current_state:
        logger.warning("DELETE /chat | Session not found | session_id='%s'", session_id)
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    # Delete the session
    delete_result = await state_service.delete_session(session_id)
    if delete_result:
        logger.info("DELETE /chat | Session cleared | session_id='%s'", session_id)
        return {"detail": f"Session '{session_id}' has been cleared."}
    else:
        logger.error("DELETE /chat | Failed to delete session | session_id='%s'", session_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session '{session_id}'. Database error."
        )
