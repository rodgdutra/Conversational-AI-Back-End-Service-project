"""
Unit tests for app.models — Pydantic request/response schemas.

All tests are pure Python — no database, no network, no I/O.
"""
import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from app.models import (
    ChatRequest,
    ChatResponse,
    StateMetadata,
    SessionStatesResponse,
    StateTransitionData,
    SessionTransitionsResponse,
)


# ---------------------------------------------------------------------------
# ChatRequest
# ---------------------------------------------------------------------------


class TestChatRequest:
    """Validate the /chat request payload model."""

    def test_valid_request_accepted(self):
        req = ChatRequest(session_id="sess-001", message="Hello")
        assert req.session_id == "sess-001"
        assert req.message == "Hello"

    def test_missing_session_id_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="Hello")

    def test_missing_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(session_id="sess-001")

    def test_empty_strings_are_accepted_by_pydantic(self):
        # FastAPI validates emptiness at the route level; Pydantic only enforces type
        req = ChatRequest(session_id="", message="")
        assert req.session_id == ""
        assert req.message == ""

    def test_long_session_id_accepted(self):
        long_id = "x" * 256
        req = ChatRequest(session_id=long_id, message="Hi")
        assert req.session_id == long_id

    def test_multiline_message_accepted(self):
        msg = "line1\nline2\nline3"
        req = ChatRequest(session_id="s1", message=msg)
        assert req.message == msg


# ---------------------------------------------------------------------------
# ChatResponse
# ---------------------------------------------------------------------------


class TestChatResponse:
    """Validate the /chat response payload model."""

    def test_valid_response_without_state_id(self):
        resp = ChatResponse(
            session_id="sess-001",
            reply="Hello, how can I help?",
            verified=False,
        )
        assert resp.session_id == "sess-001"
        assert resp.verified is False
        assert resp.state_id is None

    def test_valid_response_with_state_id(self):
        resp = ChatResponse(
            session_id="sess-001",
            reply="Your appointments are …",
            verified=True,
            state_id=42,
        )
        assert resp.verified is True
        assert resp.state_id == 42

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            ChatResponse(session_id="s1")  # missing reply and verified

    def test_verified_must_be_bool(self):
        # Pydantic v2 coerces int → bool; v1 also accepts it
        resp = ChatResponse(session_id="s", reply="hi", verified=1)
        assert resp.verified is True

    def test_state_id_defaults_to_none(self):
        resp = ChatResponse(session_id="s", reply="hi", verified=False)
        assert resp.state_id is None


# ---------------------------------------------------------------------------
# StateMetadata
# ---------------------------------------------------------------------------


class TestStateMetadata:
    """Validate the state metadata model."""

    def test_valid_metadata(self):
        meta = StateMetadata(
            session_id="sess-1",
            state_id=3,
            created_at="2026-01-01T10:00:00",
        )
        assert meta.state_id == 3
        assert meta.created_at == "2026-01-01T10:00:00"

    def test_missing_fields_raises(self):
        with pytest.raises(ValidationError):
            StateMetadata(session_id="s")


# ---------------------------------------------------------------------------
# SessionStatesResponse
# ---------------------------------------------------------------------------


class TestSessionStatesResponse:
    """Validate the session-states list response."""

    def test_empty_states_list_accepted(self):
        resp = SessionStatesResponse(session_id="sess-1", states=[])
        assert resp.states == []

    def test_multiple_states(self):
        states = [
            StateMetadata(session_id="s", state_id=i, created_at="2026-01-01T00:00:00")
            for i in range(3)
        ]
        resp = SessionStatesResponse(session_id="s", states=states)
        assert len(resp.states) == 3
        assert resp.states[2].state_id == 2


# ---------------------------------------------------------------------------
# StateTransitionData
# ---------------------------------------------------------------------------


class TestStateTransitionData:
    """Validate the state-transition data model."""

    def test_valid_transition_without_extra_data(self):
        t = StateTransitionData(
            id=1,
            session_id="s",
            from_state_id=0,
            to_state_id=1,
            transition_type="user_message",
            created_at="2026-01-01T10:00:00",
        )
        assert t.from_state_id == 0
        assert t.to_state_id == 1
        assert t.data is None

    def test_valid_transition_with_extra_data(self):
        t = StateTransitionData(
            id=2,
            session_id="s",
            from_state_id=1,
            to_state_id=2,
            transition_type="tool_call",
            created_at="2026-01-01T10:01:00",
            data={"key": "value"},
        )
        assert t.data == {"key": "value"}

    def test_optional_transition_type(self):
        t = StateTransitionData(
            id=3,
            session_id="s",
            from_state_id=0,
            to_state_id=1,
            created_at="2026-01-01T10:02:00",
        )
        assert t.transition_type is None


# ---------------------------------------------------------------------------
# SessionTransitionsResponse
# ---------------------------------------------------------------------------


class TestSessionTransitionsResponse:
    """Validate the session-transitions list response."""

    def test_empty_transitions_accepted(self):
        resp = SessionTransitionsResponse(session_id="s", transitions=[])
        assert resp.transitions == []

    def test_multiple_transitions(self):
        transitions = [
            StateTransitionData(
                id=i,
                session_id="s",
                from_state_id=i - 1,
                to_state_id=i,
                created_at="2026-01-01T00:00:00",
            )
            for i in range(1, 4)
        ]
        resp = SessionTransitionsResponse(session_id="s", transitions=transitions)
        assert len(resp.transitions) == 3
