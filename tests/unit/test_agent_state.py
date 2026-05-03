"""
Unit tests for app.agent.state.AgentState.

AgentState extends LangGraph's MessagesState with domain-specific fields.
These tests confirm the field defaults and that values can be assigned.
"""
import pytest


class TestAgentStateDefaults:
    """AgentState must have sensible defaults for all fields."""

    def test_verified_defaults_to_false(self):
        from app.agent.state import AgentState
        state = AgentState()
        
        assert state.get("verified", False) is False

    def test_patient_id_defaults_to_none(self):
        from app.agent.state import AgentState
        state = AgentState()
        assert state.get("patient_id", None) is None

    def test_patient_name_defaults_to_none(self):
        from app.agent.state import AgentState
        state = AgentState()
        assert state.get("patient_name", None) is None

    def test_pending_action_defaults_to_none(self):
        from app.agent.state import AgentState
        state = AgentState()
        assert state.get("pending_action", None) is None

    def test_messages_defaults_to_empty_list(self):
        from app.agent.state import AgentState
        state = AgentState()
        assert state.get("messages", []) == []


class TestAgentStateAssignment:
    """AgentState values can be set and retrieved."""

    def test_set_verified_true(self):
        from app.agent.state import AgentState
        state = AgentState(verified=True)
        assert state["verified"] is True

    def test_set_patient_id(self):
        from app.agent.state import AgentState
        state = AgentState(patient_id="P001")
        assert state["patient_id"] == "P001"

    def test_set_patient_name(self):
        from app.agent.state import AgentState
        state = AgentState(patient_name="Alice Johnson")
        assert state["patient_name"] == "Alice Johnson"

    def test_set_pending_action(self):
        from app.agent.state import AgentState
        state = AgentState(pending_action="list_appointments")
        assert state["pending_action"] == "list_appointments"

    def test_full_verified_state(self):
        from app.agent.state import AgentState
        state = AgentState(
            verified=True,
            patient_id="P001",
            patient_name="Alice Johnson",
            pending_action=None,
        )
        assert state["verified"] is True
        assert state["patient_id"] == "P001"
        assert state["patient_name"] == "Alice Johnson"
        assert state["pending_action"] is None
