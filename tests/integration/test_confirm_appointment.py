"""
Test scenario: Confirm Appointment

Verifies that confirming an appointment is gated behind identity verification:
  1. The agent refuses to confirm an appointment before the user is verified.
  2. The agent successfully confirms an appointment once the user is verified.

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


class TestConfirmAppointment:
    """
    Tests for the 'confirm appointment' action.

    Rule: the agent must refuse to confirm any appointment before identity
    verification succeeds; after verification it must be able to confirm
    the requested appointment.
    """

    def test_refuses_to_confirm_appointment_before_verification(
        self, client: TestClient, session_id: str
    ):
        """
        Scenario: User asks to confirm appointment A001 without verifying first.

        Expected:
        - Agent asks for identity (does NOT confirm the appointment).
        - The word "confirmed" does not appear in the reply.
        - The session remains unverified.
        """
        logger.info("Starting test: refuse confirm appointment before verification")

        reply = chat(client, session_id, "Please confirm my appointment A001")
        logger.info("Agent reply: %s", reply["reply"][:200])

        reply_text = reply["reply"].lower()

        # Agent must request identity, not perform the action
        assert any(
            kw in reply_text
            for kw in ["name", "phone", "birth", "verify", "identity", "who", "first"]
        ), f"Expected identity request, got: {reply['reply']}"

        # Confirmation must not have happened
        assert "confirmed" not in reply_text, (
            "Agent must not confirm an appointment before verification"
        )
        assert reply["verified"] is False, "Session must remain unverified"

    def test_refuses_to_confirm_with_failed_verification(
        self, client: TestClient, session_id: str
    ):
        """
        Scenario: User requests confirmation, then provides *wrong* identity
        details.

        Expected:
        - Agent reports that verification failed.
        - Appointment is NOT confirmed.
        - Session remains unverified.
        """
        logger.info("Starting test: refuse confirm with failed verification")

        # Step 1 – request confirmation (triggers identity request)
        reply1 = chat(client, session_id, "I want to confirm appointment A001")
        assert reply1["verified"] is False

        # Step 2 – provide wrong identity details
        reply2 = chat(
            client,
            session_id,
            "My name is Nobody Special, phone 000-0000, DOB 1900-01-01",
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
        assert "confirmed" not in reply2_text, "Appointment must not be confirmed on failed verification"
        assert reply2["verified"] is False

    def test_confirms_appointment_after_retry_with_correct_credentials(
        self, client: TestClient, session_id: str
    ):
        """
        Verification edge-case: User provides *wrong* credentials first, is
        told verification failed, then provides the *correct* credentials and
        successfully confirms an appointment.

        Expected:
        - First attempt: agent reports verification failure, appointment NOT confirmed.
        - Second attempt (correct credentials): session reaches verified=True.
        - Appointment confirmation is processed once verified.
        """
        logger.info("Starting test: confirm after retry with correct credentials")

        # Step 1 – request confirmation (triggers identity request)
        reply1 = chat(client, session_id, "Please confirm appointment A001 for me")
        assert reply1["verified"] is False

        # Step 2 – wrong credentials
        reply2 = chat(
            client,
            session_id,
            "My name is Wrong Person, phone 000-0000, DOB 2099-12-31",
        )
        logger.info("Reply after wrong credentials: %s", reply2["reply"][:200])
        reply2_text = reply2["reply"].lower()
        assert any(
            kw in reply2_text
            for kw in ["could not", "couldn't", "unable", "failed", "sorry", "verify"]
        ), f"Expected failure message, got: {reply2['reply']}"
        assert "confirmed" not in reply2_text, "Appointment must not be confirmed on failed verification"
        assert reply2["verified"] is False

        # Step 3 – correct credentials
        reply3 = chat(
            client,
            session_id,
            "My name is Alice Johnson, phone 555-1234, DOB 1985-03-15",
        )
        logger.info("Reply after correct credentials: %s", reply3["reply"][:200])

        # Step 4 – confirm the appointment (session should now be verified)
        confirm_reply = chat(client, session_id, "Please confirm my appointment A001 now")
        logger.info("Confirm reply: %s", confirm_reply["reply"][:300])

        assert confirm_reply["verified"] is True, "Session must be verified after correct credentials"
        assert (
            "confirmed" in confirm_reply["reply"].lower()
            or "confirm" in confirm_reply["reply"].lower()
        ), f"Expected confirmation after retry verification, got: {confirm_reply['reply']}"

    def test_confirms_appointment_after_successful_verification(
        self, client: TestClient, session_id: str
    ):
        """
        Scenario: User verifies identity correctly and then confirms A001.

        Expected:
        - After verification, the agent processes the confirmation.
        - The reply mentions "confirmed" or "confirm".
        - The session is marked verified.
        """
        logger.info("Starting test: confirm appointment after verification")

        # Step 1 – verify identity
        verify_reply = chat(
            client,
            session_id,
            "I'm Alice Johnson, phone 555-1234, date of birth 1985-03-15",
        )
        logger.info("Verification reply: %s", verify_reply["reply"][:200])

        # Step 2 – confirm a specific appointment
        confirm_reply = chat(client, session_id, "Please confirm my appointment A001")
        logger.info("Confirm reply: %s", confirm_reply["reply"][:300])

        assert confirm_reply["verified"] is True, "Session must be verified"
        assert (
            "confirmed" in confirm_reply["reply"].lower()
            or "confirm" in confirm_reply["reply"].lower()
        ), f"Expected confirmation message, got: {confirm_reply['reply']}"

    def test_confirms_appointment_listed_then_confirmed(
        self, client: TestClient, session_id: str
    ):
        """
        Scenario: User verifies, lists appointments to find the IDs, and then
        confirms a specific one — the natural real-world flow.

        Expected:
        - Appointment list is returned after verification.
        - The agent successfully confirms A001 when asked.
        """
        logger.info("Starting test: list then confirm flow")

        # Step 1 – verify
        chat(
            client,
            session_id,
            "My name is Alice Johnson, phone 555-1234, DOB 1985-03-15",
        )

        # Step 2 – list to retrieve IDs
        list_reply = chat(client, session_id, "Show me my appointments")
        logger.info("List reply: %s", list_reply["reply"][:300])
        assert list_reply["verified"] is True
        assert "A001" in list_reply["reply"] or "A002" in list_reply["reply"], (
            f"Expected appointment list: {list_reply['reply']}"
        )

        # Step 3 – confirm the first appointment
        confirm_reply = chat(
            client, session_id, "Please confirm appointment A001"
        )
        logger.info("Confirm reply: %s", confirm_reply["reply"][:300])

        assert confirm_reply["verified"] is True
        assert (
            "confirmed" in confirm_reply["reply"].lower()
            or "confirm" in confirm_reply["reply"].lower()
        ), f"Expected confirmation message: {confirm_reply['reply']}"
