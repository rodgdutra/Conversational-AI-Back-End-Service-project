"""
Integration tests for patient and appointment data persistence.

These tests verify:
1. CRUD operations for patients
2. CRUD operations for appointments
3. Both sync and async versions of the services
"""

import uuid
import pytest
from datetime import datetime
from typing import Dict, Any, List, Tuple

from app.data_persistence import (
    PatientService, AppointmentService, 
    AsyncPatientService, AsyncAppointmentService
)
from app.logger import get_logger

logger = get_logger(__name__)


def generate_test_id() -> str:
    """Generate a unique ID for test data."""
    return f"TEST-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_patient_data() -> Dict[str, Any]:
    """Create test patient data."""
    test_id = generate_test_id()
    return {
        "full_name": f"Test Patient {test_id}",
        "phone": f"555-{test_id[-4:]}",
        "date_of_birth": "2000-01-01",
    }


@pytest.fixture
def test_appointment_data(patient_id: str) -> Dict[str, Any]:
    """Create test appointment data."""
    return {
        "patient_id": patient_id,
        "date": "2026-12-31",
        "time": "12:00",
        "doctor": "Dr. Test Doctor",
        "specialty": "Test Specialty",
        "status": "scheduled",
    }


@pytest.fixture
def patient_id(test_patient_data: Dict[str, Any]) -> str:
    """Create a test patient and return its ID."""
    created_patient = PatientService.create_patient(test_patient_data)
    assert created_patient is not None, "Failed to create test patient"
    return created_patient["id"]


@pytest.fixture
def appointment_id(patient_id: str) -> str:
    """Create a test appointment and return its ID."""
    appointment_data = {
        "patient_id": patient_id,
        "date": "2026-12-31",
        "time": "12:00",
        "doctor": "Dr. Test Doctor",
        "specialty": "Test Specialty",
        "status": "scheduled",
    }
    created_appointment = AppointmentService.create_appointment(appointment_data)
    assert created_appointment is not None, "Failed to create test appointment"
    return created_appointment["id"]


class TestPatientService:
    """Tests for synchronous PatientService."""

    def test_create_patient(self, test_patient_data: Dict[str, Any]):
        """Test creating a patient."""
        created_patient = PatientService.create_patient(test_patient_data)
        assert created_patient is not None, "Failed to create patient"
        assert "id" in created_patient, "Created patient missing ID"
        assert created_patient["id"].startswith("P"), "Invalid patient ID format"
        assert created_patient["full_name"] == test_patient_data["full_name"], "Patient name mismatch"

    def test_get_patient(self, patient_id: str, test_patient_data: Dict[str, Any]):
        """Test retrieving a patient by ID."""
        retrieved_patient = PatientService.get_patient(patient_id)
        assert retrieved_patient is not None, f"Failed to retrieve patient with ID: {patient_id}"
        assert retrieved_patient["full_name"] == test_patient_data["full_name"], "Patient name mismatch"

    def test_list_patients(self, patient_id: str):
        """Test listing all patients."""
        patients = PatientService.list_patients()
        assert patients, "No patients found"
        assert any(p["id"] == patient_id for p in patients), f"Test patient {patient_id} not found in list"

    def test_find_patient(self, patient_id: str, test_patient_data: Dict[str, Any]):
        """Test finding a patient by identity details."""
        found_patient = PatientService.find_patient(
            test_patient_data["full_name"],
            test_patient_data["phone"],
            test_patient_data["date_of_birth"],
        )
        assert found_patient is not None, "Failed to find patient by identity details"
        assert found_patient["id"] == patient_id, f"Found wrong patient: {found_patient['id']} vs {patient_id}"


class TestAppointmentService:
    """Tests for synchronous AppointmentService."""

    def test_create_appointment(self, patient_id: str):
        """Test creating an appointment."""
        appointment_data = {
            "patient_id": patient_id,
            "date": "2026-12-31",
            "time": "12:00",
            "doctor": "Dr. Test Doctor",
            "specialty": "Test Specialty",
            "status": "scheduled",
        }
        created_appointment = AppointmentService.create_appointment(appointment_data)
        assert created_appointment is not None, "Failed to create appointment"
        assert "id" in created_appointment, "Created appointment missing ID"
        assert created_appointment["id"].startswith("A"), "Invalid appointment ID format"
        assert created_appointment["patient_id"] == patient_id, "Appointment patient ID mismatch"

    def test_get_appointments(self, patient_id: str, appointment_id: str):
        """Test getting all appointments for a patient."""
        appointments = AppointmentService.get_appointments(patient_id)
        assert appointments, f"No appointments found for patient {patient_id}"
        assert any(a["id"] == appointment_id for a in appointments), f"Test appointment {appointment_id} not found"

    def test_get_appointment_by_id(self, patient_id: str, appointment_id: str):
        """Test getting a specific appointment by ID."""
        appointment = AppointmentService.get_appointment_by_id(appointment_id, patient_id)
        assert appointment is not None, f"Failed to retrieve appointment {appointment_id}"
        assert appointment["id"] == appointment_id, "Appointment ID mismatch"

    def test_confirm_appointment(self, patient_id: str, appointment_id: str):
        """Test confirming an appointment."""
        confirmed = AppointmentService.confirm_appointment(appointment_id, patient_id)
        assert confirmed is not None, f"Failed to confirm appointment {appointment_id}"
        assert confirmed["status"] == "confirmed", f"Appointment status not updated: {confirmed['status']}"
        
        # Verify the change persisted
        appointment = AppointmentService.get_appointment_by_id(appointment_id, patient_id)
        assert appointment["status"] == "confirmed", "Status change not persisted"

    def test_cancel_appointment(self, patient_id: str, appointment_id: str):
        """Test cancelling an appointment."""
        # First confirm it to ensure it's not already cancelled
        AppointmentService.confirm_appointment(appointment_id, patient_id)
        
        # Now cancel it
        cancelled = AppointmentService.cancel_appointment(appointment_id, patient_id)
        assert cancelled is not None, f"Failed to cancel appointment {appointment_id}"
        assert cancelled["status"] == "cancelled", f"Appointment status not updated: {cancelled['status']}"
        
        # Verify the change persisted
        appointment = AppointmentService.get_appointment_by_id(appointment_id, patient_id)
        assert appointment["status"] == "cancelled", "Status change not persisted"


class TestAsyncPatientService:
    """Tests for asynchronous PatientService."""

    @pytest.mark.asyncio
    async def test_find_patient(self, patient_id: str, test_patient_data: Dict[str, Any]):
        """Test finding a patient by identity details."""
        found_patient = await AsyncPatientService.find_patient(
            test_patient_data["full_name"],
            test_patient_data["phone"],
            test_patient_data["date_of_birth"],
        )
        assert found_patient is not None, "Failed to find patient by identity details (async)"
        assert found_patient["id"] == patient_id, f"Found wrong patient: {found_patient['id']} vs {patient_id}"


class TestAsyncAppointmentService:
    """Tests for asynchronous AppointmentService."""

    @pytest.mark.asyncio
    async def test_get_appointments(self, patient_id: str, appointment_id: str):
        """Test getting all appointments for a patient."""
        appointments = await AsyncAppointmentService.get_appointments(patient_id)
        assert appointments, f"No appointments found for patient {patient_id}"
        assert any(a["id"] == appointment_id for a in appointments), f"Test appointment {appointment_id} not found"

    @pytest.mark.asyncio
    async def test_get_appointment_by_id(self, patient_id: str, appointment_id: str):
        """Test getting a specific appointment by ID."""
        appointment = await AsyncAppointmentService.get_appointment_by_id(appointment_id, patient_id)
        assert appointment is not None, f"Failed to retrieve appointment {appointment_id}"
        assert appointment["id"] == appointment_id, "Appointment ID mismatch"

    @pytest.mark.asyncio
    async def test_confirm_appointment(self, patient_id: str, appointment_id: str):
        """Test confirming an appointment."""
        confirmed = await AsyncAppointmentService.confirm_appointment(appointment_id, patient_id)
        assert confirmed is not None, f"Failed to confirm appointment {appointment_id}"
        assert confirmed["status"] == "confirmed", f"Appointment status not updated: {confirmed['status']}"

    @pytest.mark.asyncio
    async def test_cancel_appointment(self, patient_id: str, appointment_id: str):
        """Test cancelling an appointment."""
        # First confirm it to ensure it's not already cancelled
        await AsyncAppointmentService.confirm_appointment(appointment_id, patient_id)
        
        # Now cancel it
        cancelled = await AsyncAppointmentService.cancel_appointment(appointment_id, patient_id)
        assert cancelled is not None, f"Failed to cancel appointment {appointment_id}"
        assert cancelled["status"] == "cancelled", f"Appointment status not updated: {cancelled['status']}"