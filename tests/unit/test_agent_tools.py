"""
Unit tests for app.agent.tools — LangChain tool wrappers.

The underlying data-layer functions (find_patient, get_appointments, …) are
replaced with unittest.mock stubs so no database or network is needed.

Each tool is invoked via its `.invoke({…})` interface, which is the same
API used by LangGraph nodes.
"""
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# verify_patient_tool
# ---------------------------------------------------------------------------


class TestVerifyPatientTool:
    """Tests for verify_patient_tool."""

    def _invoke(self, full_name: str, phone: str, date_of_birth: str) -> dict:
        from app.agent.tools import verify_patient_tool
        return verify_patient_tool.invoke(
            {"full_name": full_name, "phone": phone, "date_of_birth": date_of_birth}
        )

    def test_success_when_patient_found(self):
        fake_patient = {
            "id": "P001",
            "full_name": "Alice Johnson",
            "phone": "555-1234",
            "date_of_birth": "1985-03-15",
        }
        with patch("app.agent.tools.find_patient", return_value=fake_patient):
            result = self._invoke("Alice Johnson", "555-1234", "1985-03-15")

        assert result["verified"] is True
        assert result["patient_id"] == "P001"
        assert result["patient_name"] == "Alice Johnson"
        assert "verified" in result["message"].lower() or "identity" in result["message"].lower()

    def test_failure_when_patient_not_found(self):
        with patch("app.agent.tools.find_patient", return_value=None):
            result = self._invoke("Ghost User", "000-0000", "2000-01-01")

        assert result["verified"] is False
        assert result["patient_id"] is None
        assert result["patient_name"] is None
        assert len(result["message"]) > 0

    def test_result_contains_required_keys(self):
        with patch("app.agent.tools.find_patient", return_value=None):
            result = self._invoke("A", "B", "C")

        for key in ("verified", "patient_id", "patient_name", "message"):
            assert key in result

    def test_calls_find_patient_with_correct_arguments(self):
        with patch("app.agent.tools.find_patient", return_value=None) as mock_find:
            self._invoke("Alice Johnson", "555-1234", "1985-03-15")

        mock_find.assert_called_once_with("Alice Johnson", "555-1234", "1985-03-15")


# ---------------------------------------------------------------------------
# list_appointments_tool
# ---------------------------------------------------------------------------


class TestListAppointmentsTool:
    """Tests for list_appointments_tool."""

    _FAKE_APPOINTMENTS = [
        {
            "id": "A001",
            "patient_id": "P001",
            "date": "2026-05-10",
            "time": "09:00",
            "doctor": "Dr. Carter",
            "specialty": "General Practice",
            "status": "scheduled",
        },
        {
            "id": "A002",
            "patient_id": "P001",
            "date": "2026-05-20",
            "time": "14:30",
            "doctor": "Dr. Lee",
            "specialty": "Dermatology",
            "status": "scheduled",
        },
    ]

    def _invoke(self, patient_id: str) -> dict:
        from app.agent.tools import list_appointments_tool
        return list_appointments_tool.invoke({"patient_id": patient_id})

    def test_returns_appointments_when_patient_has_some(self):
        with patch("app.agent.tools.get_appointments", return_value=self._FAKE_APPOINTMENTS):
            result = self._invoke("P001")

        assert result["appointments"] == self._FAKE_APPOINTMENTS
        assert "A001" in result["message"]
        assert "A002" in result["message"]

    def test_returns_empty_list_message_when_no_appointments(self):
        with patch("app.agent.tools.get_appointments", return_value=[]):
            result = self._invoke("P999")

        assert result["appointments"] == []
        assert "no appointments" in result["message"].lower()

    def test_result_contains_required_keys(self):
        with patch("app.agent.tools.get_appointments", return_value=[]):
            result = self._invoke("P001")

        assert "appointments" in result
        assert "message" in result

    def test_message_includes_appointment_details(self):
        with patch("app.agent.tools.get_appointments", return_value=self._FAKE_APPOINTMENTS):
            result = self._invoke("P001")

        # Message should include the appointment IDs and dates
        assert "A001" in result["message"]
        assert "2026-05-10" in result["message"]

    def test_calls_get_appointments_with_correct_patient_id(self):
        with patch("app.agent.tools.get_appointments", return_value=[]) as mock_get:
            self._invoke("P001")

        mock_get.assert_called_once_with("P001")


# ---------------------------------------------------------------------------
# confirm_appointment_tool
# ---------------------------------------------------------------------------


class TestConfirmAppointmentTool:
    """Tests for confirm_appointment_tool."""

    _CONFIRMED_APPT = {
        "id": "A001",
        "patient_id": "P001",
        "date": "2026-05-10",
        "time": "09:00",
        "doctor": "Dr. Carter",
        "specialty": "General Practice",
        "status": "confirmed",
    }

    def _invoke(self, patient_id: str, appointment_id: str) -> dict:
        from app.agent.tools import confirm_appointment_tool
        return confirm_appointment_tool.invoke(
            {"patient_id": patient_id, "appointment_id": appointment_id}
        )

    def test_success_when_appointment_found(self):
        with patch("app.agent.tools.confirm_appointment", return_value=self._CONFIRMED_APPT):
            result = self._invoke("P001", "A001")

        assert result["success"] is True
        assert result["appointment"] == self._CONFIRMED_APPT
        assert "confirmed" in result["message"].lower()

    def test_failure_when_appointment_not_found(self):
        with patch("app.agent.tools.confirm_appointment", return_value=None):
            result = self._invoke("P001", "A999")

        assert result["success"] is False
        assert result["appointment"] is None
        assert "could not" in result["message"].lower() or "make sure" in result["message"].lower()

    def test_result_contains_required_keys(self):
        with patch("app.agent.tools.confirm_appointment", return_value=None):
            result = self._invoke("P001", "A001")

        for key in ("success", "appointment", "message"):
            assert key in result

    def test_message_includes_appointment_id_on_success(self):
        with patch("app.agent.tools.confirm_appointment", return_value=self._CONFIRMED_APPT):
            result = self._invoke("P001", "A001")

        assert "A001" in result["message"]

    def test_calls_confirm_appointment_with_correct_arguments(self):
        with patch("app.agent.tools.confirm_appointment", return_value=None) as mock_confirm:
            self._invoke("P001", "A001")

        # Tool passes arguments as (appointment_id, patient_id) to the data layer
        mock_confirm.assert_called_once_with("A001", "P001")


# ---------------------------------------------------------------------------
# cancel_appointment_tool
# ---------------------------------------------------------------------------


class TestCancelAppointmentTool:
    """Tests for cancel_appointment_tool."""

    _CANCELLED_APPT = {
        "id": "A002",
        "patient_id": "P001",
        "date": "2026-05-20",
        "time": "14:30",
        "doctor": "Dr. Lee",
        "specialty": "Dermatology",
        "status": "cancelled",
    }

    def _invoke(self, patient_id: str, appointment_id: str) -> dict:
        from app.agent.tools import cancel_appointment_tool
        return cancel_appointment_tool.invoke(
            {"patient_id": patient_id, "appointment_id": appointment_id}
        )

    def test_success_when_appointment_found(self):
        with patch("app.agent.tools.cancel_appointment", return_value=self._CANCELLED_APPT):
            result = self._invoke("P001", "A002")

        assert result["success"] is True
        assert result["appointment"] == self._CANCELLED_APPT
        assert "cancelled" in result["message"].lower()

    def test_failure_when_appointment_not_found(self):
        with patch("app.agent.tools.cancel_appointment", return_value=None):
            result = self._invoke("P001", "A999")

        assert result["success"] is False
        assert result["appointment"] is None
        assert "could not" in result["message"].lower() or "make sure" in result["message"].lower()

    def test_result_contains_required_keys(self):
        with patch("app.agent.tools.cancel_appointment", return_value=None):
            result = self._invoke("P001", "A002")

        for key in ("success", "appointment", "message"):
            assert key in result

    def test_message_includes_appointment_id_on_success(self):
        with patch("app.agent.tools.cancel_appointment", return_value=self._CANCELLED_APPT):
            result = self._invoke("P001", "A002")

        assert "A002" in result["message"]

    def test_calls_cancel_appointment_with_correct_arguments(self):
        with patch("app.agent.tools.cancel_appointment", return_value=None) as mock_cancel:
            self._invoke("P001", "A002")

        # Tool passes arguments as (appointment_id, patient_id) to the data layer
        mock_cancel.assert_called_once_with("A002", "P001")


# ---------------------------------------------------------------------------
# ALL_TOOLS export
# ---------------------------------------------------------------------------


class TestAllToolsExport:
    """The ALL_TOOLS list must export exactly the four expected tools."""

    def test_all_tools_has_four_entries(self):
        from app.agent.tools import ALL_TOOLS
        assert len(ALL_TOOLS) == 4

    def test_all_tools_contains_verify_patient_tool(self):
        from app.agent.tools import ALL_TOOLS, verify_patient_tool
        assert verify_patient_tool in ALL_TOOLS

    def test_all_tools_contains_list_appointments_tool(self):
        from app.agent.tools import ALL_TOOLS, list_appointments_tool
        assert list_appointments_tool in ALL_TOOLS

    def test_all_tools_contains_confirm_appointment_tool(self):
        from app.agent.tools import ALL_TOOLS, confirm_appointment_tool
        assert confirm_appointment_tool in ALL_TOOLS

    def test_all_tools_contains_cancel_appointment_tool(self):
        from app.agent.tools import ALL_TOOLS, cancel_appointment_tool
        assert cancel_appointment_tool in ALL_TOOLS

    def test_each_tool_has_a_name(self):
        from app.agent.tools import ALL_TOOLS
        for tool in ALL_TOOLS:
            assert hasattr(tool, "name")
            assert isinstance(tool.name, str)
            assert len(tool.name) > 0

    def test_each_tool_has_a_description(self):
        from app.agent.tools import ALL_TOOLS
        for tool in ALL_TOOLS:
            assert hasattr(tool, "description")
            assert isinstance(tool.description, str)
            assert len(tool.description) > 0
