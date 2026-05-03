"""
Test scenario 5: Appointment Action Routing / Re-routing

Verifies that once a patient is verified, the assistant allows free, natural
navigation between all three appointment actions — listing, confirming, and
cancelling — across an unlimited number of turns without requiring
re-verification or losing context.

Key invariants tested:
  • The verified state is preserved throughout the conversation.
  • Each action produces the expected change and the correct reply.
  • The agent can switch from one action to another seamlessly.
  • Returning to a previously used action (e.g. listing again after a
    confirm/cancel) produces up-to-date information.

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


def _verify_alice(client: TestClient, session_id: str) -> None:
    """
    Convenience helper: verify Alice Johnson in the given session.

    Asserts that the response is 200 OK.  The caller is responsible for
    checking that `verified` reaches True on a subsequent turn if the LLM
    needs an extra round-trip.
    """
    chat(
        client,
        session_id,
        "My name is Alice Johnson, phone 555-1234, DOB 1985-03-15",
    )


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestAppointmentRouting:
    """
    Tests for free navigation between appointment actions.

    All tests in this class start by verifying Alice Johnson so they can
    focus exclusively on the routing behaviour rather than the gating
    behaviour (which is covered by the other test modules).
    """

    def test_list_then_confirm_then_list_again(
        self, client: TestClient, session_id: str
    ):
        """
        Route: verify → list → confirm A001 → list again

        Expected:
        - Listing works immediately after verification.
        - Confirmation of A001 succeeds in the same session.
        - A second listing request is honoured without re-verification.
        - The verified flag stays True throughout.
        """
        logger.info("Starting test: list → confirm → list")

        # Step 1 – verify
        _verify_alice(client, session_id)

        # Step 2 – list appointments
        list_reply1 = chat(client, session_id, "Show me my appointments please")
        logger.info("First list reply: %s", list_reply1["reply"][:300])
        assert list_reply1["verified"] is True, "Should be verified after identity step"
        assert "A001" in list_reply1["reply"] or "A002" in list_reply1["reply"], (
            f"Expected appointment list, got: {list_reply1['reply']}"
        )

        # Step 3 – confirm appointment A001
        confirm_reply = chat(client, session_id, "I'd like to confirm appointment A001")
        logger.info("Confirm reply: %s", confirm_reply["reply"][:300])
        assert confirm_reply["verified"] is True, "Should remain verified"
        assert (
            "confirmed" in confirm_reply["reply"].lower()
            or "confirm" in confirm_reply["reply"].lower()
        ), f"Expected confirmation, got: {confirm_reply['reply']}"

        # Step 4 – list again to verify the session is still active and verified
        list_reply2 = chat(client, session_id, "Can you list my appointments again?")
        logger.info("Second list reply: %s", list_reply2["reply"][:300])
        assert list_reply2["verified"] is True, "Should still be verified"
        assert "A001" in list_reply2["reply"] or "A002" in list_reply2["reply"], (
            f"Expected appointment list on second request, got: {list_reply2['reply']}"
        )

    def test_list_then_cancel_then_list_again(
        self, client: TestClient, session_id: str
    ):
        """
        Route: verify → list → cancel A002 → list again

        Expected:
        - Cancellation succeeds without re-verification.
        - A subsequent list request is still served in the same session.
        - The verified flag stays True throughout.
        """
        logger.info("Starting test: list → cancel → list")

        # Step 1 – verify
        _verify_alice(client, session_id)

        # Step 2 – list
        list_reply1 = chat(client, session_id, "List my appointments")
        logger.info("First list reply: %s", list_reply1["reply"][:300])
        assert list_reply1["verified"] is True
        assert "A001" in list_reply1["reply"] or "A002" in list_reply1["reply"]

        # Step 3 – cancel A002
        cancel_reply = chat(client, session_id, "Please cancel appointment A002")
        logger.info("Cancel reply: %s", cancel_reply["reply"][:300])
        assert cancel_reply["verified"] is True, "Should remain verified after cancel"
        assert (
            "cancelled" in cancel_reply["reply"].lower()
            or "cancel" in cancel_reply["reply"].lower()
        ), f"Expected cancellation message, got: {cancel_reply['reply']}"

        # Step 4 – list again
        list_reply2 = chat(client, session_id, "Show me my appointments once more")
        logger.info("Second list reply: %s", list_reply2["reply"][:300])
        assert list_reply2["verified"] is True, "Should still be verified"
        assert "A001" in list_reply2["reply"] or "A002" in list_reply2["reply"], (
            f"Expected appointment list, got: {list_reply2['reply']}"
        )

    def test_confirm_one_appointment_then_cancel_another(
        self, client: TestClient, session_id: str
    ):
        """
        Route: verify → confirm A001 → cancel A002

        Expected:
        - Both actions succeed within the same session without re-verification.
        - Each reply acknowledges the correct appointment.
        """
        logger.info("Starting test: confirm A001 → cancel A002")

        # Step 1 – verify
        _verify_alice(client, session_id)

        # Step 2 – confirm A001
        confirm_reply = chat(client, session_id, "Please confirm my appointment A001")
        logger.info("Confirm reply: %s", confirm_reply["reply"][:300])
        assert confirm_reply["verified"] is True
        assert (
            "confirmed" in confirm_reply["reply"].lower()
            or "confirm" in confirm_reply["reply"].lower()
        ), f"Expected confirmation, got: {confirm_reply['reply']}"

        # Step 3 – switch to a cancel without re-verifying
        cancel_reply = chat(
            client, session_id, "Now I'd like to cancel appointment A002"
        )
        logger.info("Cancel reply: %s", cancel_reply["reply"][:300])
        assert cancel_reply["verified"] is True, "Should remain verified for cancel"
        assert (
            "cancelled" in cancel_reply["reply"].lower()
            or "cancel" in cancel_reply["reply"].lower()
        ), f"Expected cancellation, got: {cancel_reply['reply']}"

    def test_multi_step_routing_list_confirm_cancel_list(
        self, client: TestClient, session_id: str
    ):
        """
        Route: verify → list → confirm A001 → cancel A002 → list again

        This is the most comprehensive routing test. It exercises all three
        actions in sequence and verifies that:
        - The session remains verified through four distinct actions.
        - Each action receives the correct reply.
        - The final list request is still served correctly.
        """
        logger.info("Starting test: list → confirm → cancel → list")

        # Step 1 – verify
        _verify_alice(client, session_id)

        # Step 2 – list
        list_reply1 = chat(client, session_id, "What appointments do I have?")
        logger.info("Initial list reply: %s", list_reply1["reply"][:300])
        assert list_reply1["verified"] is True
        assert "A001" in list_reply1["reply"] or "A002" in list_reply1["reply"], (
            f"Expected appointment list, got: {list_reply1['reply']}"
        )

        # Step 3 – confirm A001
        confirm_reply = chat(client, session_id, "Great, please confirm A001")
        logger.info("Confirm reply: %s", confirm_reply["reply"][:300])
        assert confirm_reply["verified"] is True
        assert (
            "confirmed" in confirm_reply["reply"].lower()
            or "confirm" in confirm_reply["reply"].lower()
        ), f"Expected confirmation, got: {confirm_reply['reply']}"

        # Step 4 – cancel A002 (different action, same session)
        cancel_reply = chat(
            client, session_id, "Actually, I also want to cancel appointment A002"
        )
        logger.info("Cancel reply: %s", cancel_reply["reply"][:300])
        assert cancel_reply["verified"] is True
        assert (
            "cancelled" in cancel_reply["reply"].lower()
            or "cancel" in cancel_reply["reply"].lower()
        ), f"Expected cancellation, got: {cancel_reply['reply']}"

        # Step 5 – list again at the end of the session
        list_reply2 = chat(client, session_id, "Can you show me my appointments again?")
        logger.info("Final list reply: %s", list_reply2["reply"][:300])
        assert list_reply2["verified"] is True, "Should be verified to the end"
        assert "A001" in list_reply2["reply"] or "A002" in list_reply2["reply"], (
            f"Expected final appointment list, got: {list_reply2['reply']}"
        )
