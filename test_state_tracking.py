#!/usr/bin/env python
"""
Test script for the langgraph state persistence with state tracking capabilities.

This script tests:
1. Creating and persisting multiple states for a session
2. Retrieving state history
3. Verifying state transitions are tracked
4. State metadata is properly recorded 

Usage:
    python test_state_tracking.py
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.db import init_db
from app.agent.persistence import AsyncStatePersistenceService
from app.logger import get_logger

logger = get_logger(__name__)


def create_test_state(message_content: str, verified: bool = False) -> Dict[str, Any]:
    """Create a test LangGraph state for testing."""
    return {
        "messages": [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=message_content),
            AIMessage(content=f"I understand that you said: '{message_content}'")
        ],
        "verified": verified,
        "patient_id": "P001" if verified else None,
        "patient_name": "Alice Johnson" if verified else None,
        "pending_action": None,
        "test_timestamp": datetime.now().isoformat(),
        "test_metadata": {"source": "test_script", "version": "1.0"}
    }


async def create_test_session_with_multiple_states(
    service: AsyncStatePersistenceService, 
    num_states: int = 3
) -> Tuple[str, List[Dict[str, Any]]]:
    """Create a test session with multiple states."""
    session_id = f"test-{uuid.uuid4()}"
    logger.info(f"Creating test session: {session_id} with {num_states} states")
    
    states = []
    
    # Create initial state
    initial_state = create_test_state("Hello, I need help with my appointments.")
    save_result = await service.save_state(session_id, initial_state)
    
    if not save_result:
        logger.error(f"Failed to save initial state for session {session_id}")
        return session_id, states
    
    loaded_state = await service.load_state(session_id)
    states.append(loaded_state)
    
    # Create additional states with transitions
    messages = [
        "My name is Alice Johnson",
        "My phone number is 555-1234",
        "My date of birth is 1985-03-15",
        "I want to check my appointments",
        "Please confirm appointment A001"
    ]
    
    for i in range(min(num_states-1, len(messages))):
        message = messages[i]
        verified = i >= 2  # After the third message, we'll assume verification
        
        next_state = create_test_state(message, verified)
        transition_type = "verification" if i == 2 else "user_message"
        transition_data = {
            "message": message,
            "changed_verification": i == 2,
            "timestamp": datetime.now().isoformat()
        }
        
        save_result = await service.save_state(
            session_id, 
            next_state,
            transition_type=transition_type,
            transition_data=transition_data
        )
        
        if not save_result:
            logger.error(f"Failed to save state {i+1} for session {session_id}")
            continue
        
        loaded_state = await service.load_state(session_id)
        states.append(loaded_state)
    
    return session_id, states


async def test_state_transitions(service: AsyncStatePersistenceService, session_id: str) -> bool:
    """Test that state transitions were recorded."""
    transitions = await service.get_state_transitions(session_id)
    
    if not transitions:
        logger.error(f"No transitions found for session {session_id}")
        return False
    
    logger.info(f"Found {len(transitions)} transitions for session {session_id}")
    for i, transition in enumerate(transitions):
        logger.info(f"Transition {i+1}: {transition['from_state_id']} → {transition['to_state_id']} ({transition['transition_type']})")
        if transition.get("data"):
            logger.info(f"  Data: {json.dumps(transition['data'])}")
    
    return True


async def test_state_retrieval(service: AsyncStatePersistenceService, session_id: str) -> bool:
    """Test retrieving specific states by ID."""
    states_meta = await service.list_states(session_id)
    
    if not states_meta:
        logger.error(f"No states found for session {session_id}")
        return False
    
    logger.info(f"Found {len(states_meta)} states for session {session_id}")
    logger.info(f"States meta {states_meta}")
    
    
    # Try to load each state specifically
    for state_meta in states_meta:
        state_id = state_meta["state_id"]
        state = await service.load_state(session_id, state_id)
        
        if not state:
            logger.error(f"Failed to load state {state_id} for session {session_id}")
            return False
        
        metadata = state.get("_metadata", {})
        logger.info(f"Loaded state {state_id}: created at {metadata.get('created_at')}")
        
        # Check if state contains message history
        messages = state.get("messages", [])
        if messages:
            logger.info(f"  State has {len(messages)} messages")
            
        # Check verification status
        verified = state.get("verified", False)
        logger.info(f"  Verified: {verified}")
        
        if verified:
            logger.info(f"  Patient: {state.get('patient_name')} (ID: {state.get('patient_id')})")
    
    return True


async def cleanup_session(service: AsyncStatePersistenceService, session_id: str) -> bool:
    """Clean up the test session."""
    logger.info(f"Cleaning up test session: {session_id}")
    delete_result = await service.delete_session(session_id)
    
    if not delete_result:
        logger.error(f"Failed to delete session {session_id}")
        return False
    
    # Verify session is gone
    state = await service.load_state(session_id)
    if state:
        logger.error(f"Session {session_id} still exists after deletion!")
        return False
    
    logger.info(f"Session {session_id} successfully deleted")
    return True


async def run_tests():
    """Run the full test suite."""
    logger.info("Initializing the database...")
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        return
    
    service = AsyncStatePersistenceService()
    
    # Test 1: Create a session with multiple states
    session_id, states = await create_test_session_with_multiple_states(service, 5)
    
    if not states:
        logger.error("Failed to create test states")
        return
    
    logger.info(f"Created {len(states)} states for session {session_id}")
    
    # Test 2: Check state transitions
    transitions_ok = await test_state_transitions(service, session_id)
    
    if not transitions_ok:
        logger.error("State transitions test failed")
    
    # Test 3: Retrieve specific states
    retrieval_ok = await test_state_retrieval(service, session_id)
    
    if not retrieval_ok:
        logger.error("State retrieval test failed")
    
    # Test 4: Load by specific state ID
    second_state_id = 2  # Try to load the second state
    second_state = await service.load_state(session_id, second_state_id)
    
    if not second_state:
        logger.error(f"Failed to load specific state {second_state_id}")
    else:
        logger.info(f"Successfully loaded state {second_state_id}")
        # If the state contains a user message, print its content
        if second_state.get("messages") and len(second_state["messages"]) > 1:
            message = second_state["messages"][1]
            if hasattr(message, "content"):
                logger.info(f"Message: {message.content}")
    
    # Clean up
    cleanup_ok = await cleanup_session(service, session_id)
    
    if not cleanup_ok:
        logger.error("Session cleanup failed")
    
    # Overall test result
    all_passed = states and transitions_ok and retrieval_ok and cleanup_ok
    
    if all_passed:
        logger.info("All tests PASSED! 🎉")
    else:
        logger.error("Some tests FAILED! 😞")


def main():
    """Entry point for the test script."""
    try:
        asyncio.run(run_tests())
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()