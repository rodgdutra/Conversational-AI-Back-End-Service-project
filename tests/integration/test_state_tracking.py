"""
Integration tests for langgraph state persistence with state tracking capabilities.

These tests verify:
1. Creating and persisting multiple states for a session
2. Retrieving state history
3. Verifying state transitions are tracked
4. State metadata is properly recorded 
"""

import json
import pytest
import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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


class TestStateTracking:
    """Tests for state tracking functionality."""

    @pytest.fixture
    def service(self) -> AsyncStatePersistenceService:
        """Create an AsyncStatePersistenceService for testing."""
        return AsyncStatePersistenceService()

    @pytest.fixture
    def session_id(self) -> str:
        """Create a unique session ID for testing."""
        return f"test-{uuid.uuid4()}"
    
    @pytest.fixture
    def test_messages(self) -> List[str]:
        """Sample messages for testing state transitions."""
        return [
            "My name is Alice Johnson",
            "My phone number is 555-1234",
            "My date of birth is 1985-03-15",
            "I want to check my appointments",
            "Please confirm appointment A001"
        ]

    @pytest.mark.asyncio
    async def test_create_multiple_states(self, service: AsyncStatePersistenceService, session_id: str, test_messages: List[str]):
        """Test creating a session with multiple states."""
        # Create initial state
        initial_state = create_test_state("Hello, I need help with my appointments.")
        save_result = await service.save_state(session_id, initial_state)
        assert save_result is not None, "Failed to save initial state"
        
        # Create additional states
        for i in range(min(4, len(test_messages))):
            message = test_messages[i]
            verified = i >= 1  # After the third message, we'll assume verification
            
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
            assert save_result is not None, f"Failed to save state {i+1} for session {session_id}"
        
        # Check how many states we have
        states_meta = await service.list_states(session_id)
        assert len(states_meta) == 5, f"Expected 5 states, got {len(states_meta)}"
    
    @pytest.mark.asyncio
    async def test_state_transitions(self, service: AsyncStatePersistenceService, session_id: str, test_messages: List[str]):
        """Test that state transitions are properly recorded."""
        # Create initial state plus transitions
        await self.test_create_multiple_states(service, session_id, test_messages)
        
        # Get the transitions
        transitions = await service.get_state_transitions(session_id)
        
        # Validate transitions
        assert transitions, "No transitions found"
        assert len(transitions) == 4, f"Expected 4 transitions, got {len(transitions)}"
        
        # Check verification transition specifically
        verification_transitions = [t for t in transitions if t["transition_type"] == "verification"]
        assert verification_transitions, "No verification transition found"
        assert verification_transitions[0]["data"]["changed_verification"] is True, "Verification transition data incorrect"
        
        # Check correct order (from_state_id always less than to_state_id)
        for transition in transitions:
            assert transition["from_state_id"] < transition["to_state_id"], "Transition state order incorrect"
    
    @pytest.mark.asyncio  
    async def test_state_retrieval_by_id(self, service: AsyncStatePersistenceService, session_id: str, test_messages: List[str]):
        """Test retrieving specific states by ID."""
        # Create the states
        await self.test_create_multiple_states(service, session_id, test_messages)
        
        # List the states
        states_meta = await service.list_states(session_id)
        assert states_meta, "No states found"
        
        # Load each state by ID and verify contents
        for state_meta in states_meta:
            state_id = state_meta["state_id"]
            state = await service.load_state(session_id, state_id)
            
            # Check the specific state was loaded
            assert state is not None, f"Failed to load state {state_id}"
            assert state.get("_metadata", {}).get("state_id") == state_id, "State ID mismatch"
            
            # Check verification status is correct (states 3, 4, 5 should be verified)
            expected_verified = state_id >= 3
            logger.info(f"Loading state {state_id} - Verified: {state.get('verified')} - Expected: {expected_verified}")

            assert state.get("verified") == expected_verified, f"Verification status incorrect for state {state_id}"
            
            # Check message content
            messages = state.get("messages", [])
            assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}"
    
    @pytest.mark.asyncio
    async def test_state_metadata(self, service: AsyncStatePersistenceService, session_id: str):
        """Test state metadata is properly recorded."""
        # Create a state with metadata
        test_state = create_test_state("Test message with metadata")
        await service.save_state(session_id, test_state)
        
        # Load the state
        loaded_state = await service.load_state(session_id)
        assert loaded_state is not None, "Failed to load state"
        
        # Check metadata
        metadata = loaded_state.get("_metadata", {})
        assert metadata, "No metadata found"
        assert "state_id" in metadata, "state_id missing from metadata"
        assert "created_at" in metadata, "created_at missing from metadata"
        assert "session_id" in metadata, "session_id missing from metadata"
        assert metadata["session_id"] == session_id, "Session ID mismatch in metadata"
    
    @pytest.mark.asyncio
    async def test_session_cleanup(self, service: AsyncStatePersistenceService, session_id: str):
        """Test cleaning up a session."""
        # Create a state
        test_state = create_test_state("Test message for cleanup")
        await service.save_state(session_id, test_state)
        
        # Delete the session
        delete_result = await service.delete_session(session_id)
        assert delete_result is True, "Failed to delete session"
        
        # Verify session is gone
        loaded_state = await service.load_state(session_id)
        assert loaded_state is None, "Session still exists after deletion"
        
        # Verify no transitions exist
        transitions = await service.get_state_transitions(session_id)
        assert not transitions, "Transitions still exist after session deletion"