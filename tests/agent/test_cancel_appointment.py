"""
Test scenario: Cancel Appointment

Verifies that cancelling an appointment is gated behind identity verification:
  1. The agent refuses to cancel an appointment before the user is verified.
  2. The agent successfully cancels an appointment once the user is verified.

Note: These tests use FastAPI's TestClient which runs the app in-process.
The database and state-persistence layers are replaced with in-memory mocks
to avoid event-loop conflicts (asyncpg pool vs. TestClient's anyio loop).
"""

import uuid
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# In-memory mock for AsyncStatePersistenceService
# ---------------------------------------------------------------------------


class MockStatePersistenceService:
    """
    Drop-in async replacement for AsyncStatePersistenceService.

    State is stored in a class-level dictionary so that every instance
    created within the same process (FastAPI creates one per request via
    the Depends factory) shares the same store.
    """

    _states: dict = {}

    @classmethod
    def _reset(cls) -> None:
        cls._states.clear()

    async def load_state(self, session_id: str, state_id: int = None):
        return self._states.get(session_id)

    async def save_state(
        self,
        session_id: str,
        state: dict,
        transition_type: str = None,
        transition_data: dict = None,
    ) -> bool:
        self._states[session_id] = state
        return True

    async def list_states(self, session_id: str):
        if session_id in self._states:
            return [{"session_id": session_id, "state_id": 1, "created_at": ""}]
        return []

    async def get_state_transitions(self, session_id: str):
        return []

    async def delete_session(self, session_id: str) -> bool:
        self._states.pop(session_id, None)
        return True


# ---------------------------------------------------------------------------
# Apply patches BEFORE importing app.main
# ---------------------------------------------------------------------------

_patcher_init_db = patch("app.main.init_db", side_effect=lambda: None)
_patcher_mock_data = patch("app.main.initialize_mock_data", side_effect=lambda: None)
_patcher_persistence = patch(
    "app.main.AsyncStatePersistenceService", MockStatePersistenceService
)

_patcher_init_db.start()
_patcher_mock_data.start()
_patcher_persistence.start()

try:
    from app.main import app
except Exception:
    _patcher_init_db.stop()
    _patcher_mock_data.stop()
    _patcher_persistence.stop()
    raise

from app.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_mock_persistence():
    """Wipe in-memory state before and after every test for isolation."""
    MockStatePersistenceService._reset()
    yield
    MockStatePersistenceService._reset()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def session_id() -> str:
    return f"test-{uuid.uuid4()}"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def chat(client: TestClient, session_id: str, message: str) -> dict:
    """Send a message and return the parsed JSON response."""
    response = client.post(
        "/chat",
        json={"session_id": session_id, "message": message},
    )
    assert response.status_code == 200, (
        f"Chat request failed ({response.status_code}): {response.text}"
    )
    return response.json()


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestCancelAppointment:
    """
    Tests for the 'cancel appointment' action.

    Rule: the agent must refuse to cancel any appointment before identity
    verification succeeds; after verification it must be able to cancel
    the requested appointment.
    """

    def test_refuses_to_cancel_appointment_before_verification(
        self, client: TestClient, session_id: str
    ):
        """
        Scenario: User asks to cancel appointment A001 without verifying first.

        Expected:
        - Agent asks for identity (does NOT cancel the appointment).
        - The word "cancelled" does not appear in the reply.
        - The session remains unverified.
        """
        logger.info("Starting test: refuse cancel appointment before verification")

        reply = chat(client, session_id, "Please cancel my appointment A001")
        logger.info("Agent reply: %s", reply["reply"][:200])

        reply_text = reply["reply"].lower()

        # Agent must request identity, not perform the action
        assert any(
            kw in reply_text
            for kw in ["name", "phone", "birth", "verify", "identity", "who", "first"]
        ), f"Expected identity request, got: {reply['reply']}"

        # Cancellation must not have happened
        assert "cancelled" not in reply_text, (
            "Agent must not cancel an appointment before verification"
        )
        assert reply["verified"] is False, "Session must remain unverified"

    def test_refuses_to_cancel_with_failed_verification(
        self, client: TestClient, session_id: str
    ):
        """
        Scenario: User requests cancellation, then provides *wrong* identity
        details.

        Expected:
        - Agent reports that verification failed.
        - Appointment is NOT cancelled.
        - Session remains unverified.
        """
        logger.info("Starting test: refuse cancel with failed verification")

        # Step 1 – request cancellation (triggers identity request)
        reply1 = chat(client, session_id, "I want to cancel my appointment A002")
        assert reply1["verified"] is False

        # Step 2 – provide wrong identity details
        reply2 = chat(
            client,
            session_id,
            "My name is Fake User, phone 111-1111, DOB 2010-06-15",
        )
        logger.info("Reply after wrong identity: %s", reply2["reply"][:200])

        reply2_text = reply2["reply"].lower()
        assert any(
            kw in reply2_text
            for kw in [
                "could not",
                "couldn't",
                "unable",
                "failed",
                "sorry",
                "not found",
                "verify",
                "check",
            ]
        ), f"Expected verification failure message, got: {reply2['reply']}"
        assert "cancelled" not in reply2_text, (
            "Appointment must not be cancelled on failed verification"
        )
        assert reply2["verified"] is False

    def test_cancels_appointment_after_successful_verification(
        self, client: TestClient, session_id: str
    ):
        """
        Scenario: User verifies identity correctly and then cancels A002.

        Expected:
        - After verification, the agent processes the cancellation.
        - The reply mentions "cancelled" or "cancel".
        - The session is marked verified.
        """
        logger.info("Starting test: cancel appointment after verification")

        # Step 1 – verify identity
        verify_reply = chat(
            client,
            session_id,
            "I'm Alice Johnson, phone 555-1234, date of birth 1985-03-15",
        )
        logger.info("Verification reply: %s", verify_reply["reply"][:200])

        # Step 2 – cancel a specific appointment
        cancel_reply = chat(client, session_id, "Please cancel my appointment A002")
        logger.info("Cancel reply: %s", cancel_reply["reply"][:300])

        assert cancel_reply["verified"] is True, "Session must be verified"
        assert (
            "cancelled" in cancel_reply["reply"].lower()
            or "cancel" in cancel_reply["reply"].lower()
        ), f"Expected cancellation message, got: {cancel_reply['reply']}"

    def test_cancels_appointment_via_full_conversation_flow(
        self, client: TestClient, session_id: str
    ):
        """
        Scenario: User opens the conversation with a cancellation intent,
        provides wrong credentials (first attempt), then provides the correct
        credentials and the cancellation is finally processed.

        Expected:
        - First verification attempt fails gracefully.
        - Second verification attempt (correct credentials) succeeds.
        - Cancellation is carried out after the verified state is established.
        """
        logger.info("Starting test: cancel via full conversation flow (wrong → correct)")

        # Step 1 – intent without credentials
        reply1 = chat(client, session_id, "I need to cancel appointment A001")
        logger.info("Step 1 reply: %s", reply1["reply"][:150])
        assert reply1["verified"] is False

        # Step 2 – wrong credentials
        reply2 = chat(
            client,
            session_id,
            "My name is Bob Smith, phone 555-9999, DOB 2000-01-01",
        )
        logger.info("Step 2 reply (wrong creds): %s", reply2["reply"][:150])
        reply2_text = reply2["reply"].lower()
        assert any(
            kw in reply2_text
            for kw in ["sorry", "could not", "couldn't", "failed", "verify", "unable"]
        ), f"Expected failure message, got: {reply2['reply']}"
        assert reply2["verified"] is False

        # Step 3 – correct credentials
        reply3 = chat(
            client,
            session_id,
            "My name is Alice Johnson, phone 555-1234, DOB 1985-03-15",
        )
        logger.info("Step 3 reply (correct creds): %s", reply3["reply"][:200])

        # Step 4 – request cancellation now that the session is verified
        reply4 = chat(client, session_id, "Now please cancel my appointment A001")
        logger.info("Step 4 reply (cancel): %s", reply4["reply"][:300])

        assert reply4["verified"] is True, "Session must be verified after correct credentials"
        assert (
            "cancelled" in reply4["reply"].lower()
            or "cancel" in reply4["reply"].lower()
        ), f"Expected cancellation after verification, got: {reply4['reply']}"
