"""
Test scenario: List Appointments

Verifies that listing appointments is gated behind identity verification:
  1. The agent refuses to list appointments before the user is verified.
  2. The agent successfully lists appointments once the user is verified.

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


class TestListAppointments:
    """
    Tests for the 'list appointments' action.

    Rule: the agent must NEVER reveal appointment data before identity
    verification succeeds; after verification it must list all appointments
    for the authenticated patient.
    """

    def test_refuses_to_list_appointments_before_verification(
        self, client: TestClient, session_id: str
    ):
        """
        Scenario: User asks to list appointments without verifying first.

        Expected:
        - Agent asks for identity information (name / phone / date of birth).
        - No appointment IDs appear in the reply.
        - The session remains unverified.
        """
        logger.info("Starting test: refuse list appointments before verification")

        reply = chat(client, session_id, "Can you show me my upcoming appointments?")
        logger.info("Agent reply: %s", reply["reply"][:200])

        reply_text = reply["reply"].lower()

        # Agent must ask for identity, not serve data
        assert any(
            kw in reply_text
            for kw in ["name", "phone", "birth", "verify", "identity", "identify"]
        ), f"Expected identity request, got: {reply['reply']}"

        # No appointment IDs should be visible
        assert "A001" not in reply["reply"], "Appointment A001 must not appear before verification"
        assert "A002" not in reply["reply"], "Appointment A002 must not appear before verification"
        assert reply["verified"] is False, "Session must remain unverified"

    def test_refuses_to_list_appointments_with_partial_identity(
        self, client: TestClient, session_id: str
    ):
        """
        Scenario: User provides only their name (partial identity) and then
        asks to list appointments.

        Expected:
        - After the name alone the agent asks for the remaining fields.
        - Appointments are still not revealed until all three fields are
          provided and verification succeeds.
        """
        logger.info("Starting test: refuse list with partial identity")

        # Turn 1 – ask to list
        reply1 = chat(client, session_id, "I'd like to see my appointments")
        assert reply1["verified"] is False

        # Turn 2 – provide only name
        reply2 = chat(client, session_id, "My name is Alice Johnson")
        logger.info("After name only: %s", reply2["reply"][:200])

        reply2_text = reply2["reply"].lower()
        # Agent should ask for the remaining fields
        assert any(
            kw in reply2_text
            for kw in ["phone", "birth", "date", "also need", "missing", "remaining"]
        ), f"Expected request for more info, got: {reply2['reply']}"
        assert "A001" not in reply2["reply"], "No appointment data should be shown yet"
        assert reply2["verified"] is False

    def test_lists_appointments_after_successful_verification(
        self, client: TestClient, session_id: str
    ):
        """
        Scenario: User verifies correctly and then asks for their appointments.

        Expected:
        - After successful verification the agent lists Alice Johnson's
          appointments, including A001 and A002.
        - The session is marked as verified.
        """
        logger.info("Starting test: list appointments after verification")

        # Step 1 – verify identity
        verify_reply = chat(
            client,
            session_id,
            "I'm Alice Johnson, phone 555-1234, date of birth 1985-03-15",
        )
        logger.info("Verification reply: %s", verify_reply["reply"][:200])

        # Step 2 – request appointment list
        list_reply = chat(client, session_id, "Please list my appointments")
        logger.info("List reply: %s", list_reply["reply"][:300])

        assert list_reply["verified"] is True, "Session must be verified"
        assert "A001" in list_reply["reply"] or "A002" in list_reply["reply"], (
            f"Expected appointment listings, got: {list_reply['reply']}"
        )

    def test_lists_appointments_after_retry_with_correct_credentials(
        self, client: TestClient, session_id: str
    ):
        """
        Verification edge-case: User provides *wrong* credentials first, is
        told verification failed, then provides the *correct* credentials and
        successfully lists appointments.

        Expected:
        - First attempt: agent reports verification failure, no data shown.
        - Second attempt (correct credentials): session reaches verified=True.
        - Appointment list is returned once verified.
        """
        logger.info("Starting test: list after retry with correct credentials")

        # Step 1 – trigger the flow (agent asks for identity)
        reply1 = chat(client, session_id, "I want to see my appointments")
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
        assert "A001" not in reply2["reply"], "No appointments should appear after failed verification"
        assert reply2["verified"] is False

        # Step 3 – correct credentials
        reply3 = chat(
            client,
            session_id,
            "My name is Alice Johnson, phone 555-1234, DOB 1985-03-15",
        )
        logger.info("Reply after correct credentials: %s", reply3["reply"][:200])

        # Step 4 – list appointments (session should now be verified)
        list_reply = chat(client, session_id, "Now please show me my appointments")
        logger.info("List reply: %s", list_reply["reply"][:300])

        assert list_reply["verified"] is True, "Session must be verified after correct credentials"
        assert "A001" in list_reply["reply"] or "A002" in list_reply["reply"], (
            f"Expected appointment list after retry verification, got: {list_reply['reply']}"
        )

    def test_lists_appointments_immediately_after_inline_verification(
        self, client: TestClient, session_id: str
    ):
        """
        Scenario: User requests appointments while providing all identity
        fields in a single message.

        Expected:
        - The agent verifies the user and then returns the appointment list
          without requiring an additional turn.
        """
        logger.info("Starting test: inline verification + list")

        # Turn 1 – single message with intent + identity
        reply = chat(
            client,
            session_id,
            "Show my appointments – I'm Alice Johnson, 555-1234, born 1985-03-15",
        )
        logger.info("Reply: %s", reply["reply"][:300])

        # The agent may need one more turn to list after verifying; handle both cases
        if reply["verified"] and ("A001" in reply["reply"] or "A002" in reply["reply"]):
            # One-shot success
            return

        # Otherwise, ask explicitly after verification is confirmed on any turn
        if reply["verified"]:
            list_reply = chat(client, session_id, "Show me my appointments")
            logger.info("Follow-up list reply: %s", list_reply["reply"][:300])
            assert "A001" in list_reply["reply"] or "A002" in list_reply["reply"], (
                f"Expected appointment list after verification: {list_reply['reply']}"
            )
        else:
            # Not yet verified – provide full details explicitly
            verify_reply = chat(
                client,
                session_id,
                "My name is Alice Johnson, phone 555-1234, DOB 1985-03-15",
            )
            list_reply = chat(client, session_id, "Now list my appointments")
            logger.info("List reply: %s", list_reply["reply"][:300])
            assert list_reply["verified"] is True
            assert "A001" in list_reply["reply"] or "A002" in list_reply["reply"]
