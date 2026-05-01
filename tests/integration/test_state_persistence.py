"""
Integration tests for the PostgreSQL state persistence layer.

These tests verify the storage and retrieval of LangGraph states from
the PostgreSQL database.
"""

import uuid
import pytest
from datetime import datetime
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.agent.persistence import AsyncStatePersistenceService, StatePersistenceService
from app.logger import get_logger

logger = get_logger(__name__)


def create_test_state(session_id: str) -> Dict[str, Any]:
    """Create a test LangGraph state for testing."""
    return {
        "messages": [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello, how can you help me?"),
            AIMessage(content="I can help you manage your appointments, but first I need to verify your identity.")
        ],
        "verified": False,
        "patient_id": None,
        "patient_name": None,
        "pending_action": None,
        "test_timestamp": datetime.now().isoformat(),
        "test_metadata": {"source": "test_script", "version": "1.0"}
    }


@pytest.fixture
def session_id() -> str:
    """Generate a unique session ID for testing."""
    return f"test-{uuid.uuid4()}"


@pytest.fixture
def sync_session_id() -> str:
    """Generate a unique session ID for sync testing."""
    return f"test-sync-{uuid.uuid4()}"


class TestAsyncPersistenceService:
    """Tests for AsyncStatePersistenceService."""

    @pytest.fixture
    def service(self) -> AsyncStatePersistenceService:
        """Create an AsyncStatePersistenceService instance."""
        return AsyncStatePersistenceService()

    @pytest.mark.asyncio
    async def test_save_load_state(self, service: AsyncStatePersistenceService, session_id: str):
        """Test saving and loading a state."""
        # Create a test state
        test_state = create_test_state(session_id)
        
        # Save state
        logger.info(f"Saving state for session {session_id}...")
        save_result = await service.save_state(session_id, test_state)
        assert save_result is True, "Failed to save state"
        
        # Load state
        loaded_state = await service.load_state(session_id)
        assert loaded_state is not None, "Failed to load state"
        
        # Verify state content
        assert loaded_state.get("test_timestamp") == test_state.get("test_timestamp"), "State content does not match"
        
        # Verify messages are preserved
        assert len(loaded_state.get("messages", [])) == 3, "Messages were not properly preserved"
        assert loaded_state.get("verified") is False, "Verified flag was not properly preserved"

    @pytest.mark.asyncio
    async def test_delete_state(self, service: AsyncStatePersistenceService, session_id: str):
        """Test deleting a state."""
        # Create a test state
        test_state = create_test_state(session_id)
        
        # Save state
        await service.save_state(session_id, test_state)
        
        # Delete state
        delete_result = await service.delete_session(session_id)
        assert delete_result is True, "Failed to delete state"
        
        # Verify deletion
        should_be_none = await service.load_state(session_id)
        assert should_be_none is None, "State was not properly deleted"

    @pytest.mark.asyncio
    async def test_state_transitions(self, service: AsyncStatePersistenceService, session_id: str):
        """Test state transitions tracking."""
        # Create initial state
        initial_state = create_test_state(session_id)
        await service.save_state(session_id, initial_state)
        
        # Create updated state with transition info
        updated_state = create_test_state(session_id)
        updated_state["verified"] = True
        updated_state["patient_id"] = "P123"
        
        # Save with transition data
        transition_data = {"action": "verify_identity", "changed": ["verified", "patient_id"]}
        await service.save_state(
            session_id, 
            updated_state, 
            transition_type="verification",
            transition_data=transition_data
        )
        
        # Get transitions
        transitions = await service.get_state_transitions(session_id)
        assert len(transitions) == 1, "Expected 1 transition"
        assert transitions[0]["transition_type"] == "verification", "Transition type not preserved"
        assert "data" in transitions[0], "Transition data missing"
        assert transitions[0]["data"]["action"] == "verify_identity", "Transition data content incorrect"


class TestSyncPersistenceService:
    """Tests for StatePersistenceService."""

    @pytest.fixture
    def service(self) -> StatePersistenceService:
        """Create a StatePersistenceService instance."""
        return StatePersistenceService()

    def test_save_load_state(self, service: StatePersistenceService, sync_session_id: str):
        """Test saving and loading a state."""
        # Create a test state
        test_state = create_test_state(sync_session_id)
        
        # Save state
        logger.info(f"Saving state for session {sync_session_id}...")
        save_result = service.save_state(sync_session_id, test_state)
        assert save_result is True, "Failed to save state"
        
        # Load state
        loaded_state = service.load_state(sync_session_id)
        assert loaded_state is not None, "Failed to load state"
        
        # Verify state content
        assert loaded_state.get("test_timestamp") == test_state.get("test_timestamp"), "State content does not match"
        
        # Verify messages are preserved
        assert len(loaded_state.get("messages", [])) == 3, "Messages were not properly preserved"
        assert loaded_state.get("verified") is False, "Verified flag was not properly preserved"

    def test_delete_state(self, service: StatePersistenceService, sync_session_id: str):
        """Test deleting a state."""
        # Create a test state
        test_state = create_test_state(sync_session_id)
        
        # Save state
        service.save_state(sync_session_id, test_state)
        
        # Delete state
        delete_result = service.delete_session(sync_session_id)
        assert delete_result is True, "Failed to delete state"
        
        # Verify deletion
        should_be_none = service.load_state(sync_session_id)
        assert should_be_none is None, "State was not properly deleted"

    def test_state_history(self, service: StatePersistenceService, sync_session_id: str):
        """Test listing state history."""
        # Create initial state
        initial_state = create_test_state(sync_session_id)
        service.save_state(sync_session_id, initial_state)
        
        # Create a second state
        second_state = create_test_state(sync_session_id)
        second_state["verified"] = True
        service.save_state(sync_session_id, second_state)
        
        # List states
        states = service.list_states(sync_session_id)
        assert len(states) == 2, "Expected 2 states in history"
        assert states[0]["state_id"] == 1, "First state should have ID 1"
        assert states[1]["state_id"] == 2, "Second state should have ID 2"