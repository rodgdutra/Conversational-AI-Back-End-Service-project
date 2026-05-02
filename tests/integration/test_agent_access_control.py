"""
Automated test scenarios for agent access control behavior.

These tests send sequences of messages to the /chat endpoint and assert
on the expected behavior, specifically:
1. The agent refuses to list/confirm/cancel appointments before identity verification
2. The agent allows appointment actions only after successful verification

Note: These tests use FastAPI's TestClient which runs the app in-process,
so no separate server needs to be running.
"""

import sys
import uuid
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from typing import List, Optional

# ---------------------------------------------------------------------------
# CRITICAL: Mock database initialization BEFORE importing app.main
# The app.main module calls init_db() and initialize_mock_data() at import time,
# so we need to mock these before the import happens.
# ---------------------------------------------------------------------------

# Create mock functions
def _mock_init_db():
    """Mock version of init_db that does nothing."""
    pass

def _mock_initialize_mock_data():
    """Mock version of initialize_mock_data that does nothing."""
    pass

# Apply patches before importing app.main
_mock_init_db_patcher = patch('app.main.init_db', side_effect=_mock_init_db)
_mock_init_data_patcher = patch('app.main.initialize_mock_data', side_effect=_mock_initialize_mock_data)

# Start the patches
_mock_init_db_patcher.start()
_mock_init_data_patcher.start()

try:
    # Now import the app - it will use our mocks
    from app.main import app
except Exception as e:
    # Stop patches before re-raising
    _mock_init_db_patcher.stop()
    _mock_init_data_patcher.stop()
    raise

from app.logger import get_logger

logger = get_logger(__name__)


@pytest.fixture
def client():
    """Create a FastAPI test client with the app running."""
    # TestClient starts the app's lifespan context manager
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def session_id() -> str:
    """Create a unique session ID for each test."""
    return f"test-{uuid.uuid4()}"


def chat(client: TestClient, session_id: str, message: str) -> dict:
    """
    Send a message to the /chat endpoint and return the full response.
    
    Args:
        client: FastAPI test client
        session_id: Conversation session ID
        message: User message text
        
    Returns:
        The full response dict with reply, verified, etc.
    """
    response = client.post(
        "/chat",
        json={"session_id": session_id, "message": message},
    )
    assert response.status_code == 200, f"Chat request failed with status {response.status_code}: {response.text}"
    return response.json()


class TestAccessControl:
    """
    Test suite for access control behavior.
    
    These tests verify that the agent properly enforces identity
    verification before allowing appointment-related actions.
    """

    def test_agent_refuses_list_appointments_before_verification(
        self, client: TestClient, session_id: str
    ):
        """
        Scenario 1: User tries to list appointments without verification.
        
        Expected behavior:
        - Agent should ask for identity verification
        - Agent should NOT provide appointment information
        - Agent should request full name, phone, and date of birth
        
        Message flow:
        1. User: "I want to list my appointments"
        → Agent should refuse and ask for identity info
        2. User sends partial info (just name)
        → Agent should continue asking for remaining info
        3. Agent should still not list appointments until fully verified
        """
        logger.info("Starting test: refuse list appointments before verification")
        
        # Step 1: User tries to list appointments without verification
        reply1 = chat(client, session_id, "I want to list my appointments please")
        logger.info("Reply after requesting appointments (unverified): %s", reply1["reply"][:150])
        
        # Assert agent does NOT list appointments - should ask for verification instead
        reply_text = reply1["reply"].lower()
        assert any(
            keyword in reply_text
            for keyword in ["name", "phone", "birth", "verify", "identity", "who are you", "identify"]
        ), f"Agent should ask for identity verification, but replied: {reply1['reply']}"
        
        # Assert agent doesn't show appointment information yet
        assert "A001" not in reply1["reply"], "Agent should not show appointment IDs before verification"
        assert reply1["verified"] is False, "User should not be verified yet"

        # Step 2: User provides only partial identity info (just name)
        reply2 = chat(client, session_id, "My name is Alice Johnson")
        logger.info("Reply after providing name only: %s", reply2["reply"][:150])
        
        # Agent should ask for remaining info (phone and/or date of birth)
        reply2_text = reply2["reply"].lower()
        assert any(
            keyword in reply2_text
            for keyword in ["phone", "birth", "date", "remaining", "also need"]
        ), f"Agent should ask for more info after partial identity, but replied: {reply2['reply']}"

        # Step 3: Provide remaining identity info
        reply3 = chat(client, session_id, "My phone is 555-1234 and my date of birth is 1985-03-15")
        logger.info("Reply after full identity verification: %s", reply3["reply"][:150])
        
        # Now the agent should have verified the user
        response = client.post(
            "/chat",
            json={"session_id": session_id, "message": "Now can you list my appointments?"},
        )
        assert response.status_code == 200
        final_reply = response.json()
        logger.info("Final reply after verification: %s", final_reply["reply"][:200])
        
        # After verification, agent should list appointments
        assert final_reply["verified"] is True, "User should be verified after providing correct info"
        assert "A001" in final_reply["reply"] or "A002" in final_reply["reply"], \
            f"Agent should list appointments after verification, but replied: {final_reply['reply']}"

    def test_agent_refuses_cancel_appointment_before_verification(
        self, client: TestClient, session_id: str
    ):
        """
        Scenario 2: User tries to cancel an appointment without verification.
        
        Expected behavior:
        - Agent should refuse to cancel without verification
        - Agent should collect identity information
        - Only after verification should the agent process the cancellation
        
        Message flow:
        1. User: "Cancel my appointment A001"
        → Agent should refuse and ask for identity
        2. User provides invalid info
        → Agent should report verification failure
        3. User provides correct info
        → Agent should verify and then help with cancellation
        """
        logger.info("Starting test: refuse cancel appointment before verification")
        
        # Step 1: User tries to cancel appointment without verification
        reply1 = chat(client, session_id, "Please cancel my appointment A001")
        logger.info("Reply after requesting cancel (unverified): %s", reply1["reply"][:150])
        
        # Agent should refuse and ask for identity
        reply1_text = reply1["reply"].lower()
        assert any(
            keyword in reply1_text
            for keyword in ["name", "verify", "identity", "who", "first", "need to"]
        ), f"Agent should ask for identity before cancellation, but replied: {reply1['reply']}"
        
        # Agent should NOT confirm cancellation
        assert "cancelled" not in reply1_text, \
            "Agent should not cancel appointment before verification"
        assert reply1["verified"] is False, "User should not be verified yet"

        # Step 2: Provide incorrect identity info - should fail verification
        reply2 = chat(client, session_id, "My name is Bob Smith, phone 555-9999, DOB 2000-01-01")
        logger.info("Reply after incorrect verification attempt: %s", reply2["reply"][:150])
        
        # Agent should indicate verification failed
        reply2_text = reply2["reply"].lower()
        assert any(
            keyword in reply2_text
            for keyword in ["could not", "verify", "check", "failed", "sorry", "not found", "unable"]
        ), f"Agent should report verification failure, but replied: {reply2['reply']}"

        # Step 3: Provide correct identity info
        reply3 = chat(client, session_id, "My name is Alice Johnson, phone 555-1234, DOB 1985-03-15")
        logger.info("Reply after correct verification: %s", reply3["reply"][:200])
        
        # Step 4: Now request cancellation again
        reply4 = chat(client, session_id, "Please cancel my appointment A001 now")
        logger.info("Reply after cancellation request (verified): %s", reply4["reply"][:200])
        
        # User should be verified and cancellation should be processed
        assert reply4["verified"] is True, "User should be verified"
        assert "cancelled" in reply4["reply"].lower() or "cancel" in reply4["reply"].lower(), \
            f"After verification, agent should process cancellation. Reply: {reply4['reply']}"


class TestAccessControlWithValidVerification:
    """
    Additional tests for verifying the full flow works after proper verification.
    """

    def test_full_verification_and_appointment_flow(
        self, client: TestClient, session_id: str
    ):
        """
        Test the complete flow:
        1. Request appointments (should be denied)
        2. Verify identity
        3. Successfully list appointments
        """
        logger.info("Starting test: full verification and appointment flow")
        
        # Step 1: Request - should be denied
        reply = chat(client, session_id, "Can you show me my appointments?")
        reply_text = reply["reply"].lower()
        assert "verify" in reply_text or "name" in reply_text or "identity" in reply_text, \
            f"Should ask for verification first: {reply['reply']}"
        
        # Step 2: Verify identity
        reply = chat(client, session_id, "I'm Alice Johnson, phone 555-1234, born 1985-03-15")
        logger.info("Verification reply: %s", reply["reply"][:200])
        
        # Step 3: Now list appointments
        response = client.post(
            "/chat",
            json={"session_id": session_id, "message": "Show me my appointments"},
        )
        assert response.status_code == 200
        data = response.json()
        
        assert data["verified"] is True, "Should be verified"
        assert "A001" in data["reply"], f"Should list appointments: {data['reply']}"