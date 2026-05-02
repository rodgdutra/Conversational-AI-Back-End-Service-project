"""
Data access layer for patients and appointments.
This module provides a thin wrapper around the database services.

When USE_MOCK_DATA=true  → data is served from the in-memory lists below.
When USE_MOCK_DATA=false → data is served from PostgreSQL (the same lists
                           are used by initialize_mock_data() to seed the DB).
"""

from typing import Optional, List, Dict, Any
import copy

from app.logger import get_logger
from app.config import config
from app.data_persistence import (
    PatientService, AppointmentService,
    AsyncPatientService, AsyncAppointmentService,
)

logger = get_logger(__name__)

# Determine if we should use mock data or real database
USE_MOCK_DATA = config.USE_MOCK_DATA

# ---------------------------------------------------------------------------
# Canonical mock data set
# Always defined so that:
#   • initialize_mock_data() can seed PostgreSQL regardless of USE_MOCK_DATA
#   • In-memory dispatch (when USE_MOCK_DATA=true) has the data available
# ---------------------------------------------------------------------------

PATIENTS: List[Dict[str, Any]] = [
    {
        "id": "P001",
        "full_name": "Alice Johnson",
        "phone": "555-1234",
        "date_of_birth": "1985-03-15",
    },
    {
        "id": "P002",
        "full_name": "Bob Smith",
        "phone": "555-5678",
        "date_of_birth": "1990-07-22",
    },
    {
        "id": "P003",
        "full_name": "Carol White",
        "phone": "555-9012",
        "date_of_birth": "1978-11-30",
    },
]

APPOINTMENTS: List[Dict[str, Any]] = [
    {
        "id": "A001",
        "patient_id": "P001",
        "date": "2026-05-10",
        "time": "09:00",
        "doctor": "Dr. Emily Carter",
        "specialty": "General Practice",
        "status": "scheduled",
    },
    {
        "id": "A002",
        "patient_id": "P001",
        "date": "2026-05-20",
        "time": "14:30",
        "doctor": "Dr. Michael Lee",
        "specialty": "Dermatology",
        "status": "scheduled",
    },
    {
        "id": "A003",
        "patient_id": "P002",
        "date": "2026-05-12",
        "time": "10:00",
        "doctor": "Dr. Sarah Brown",
        "specialty": "Cardiology",
        "status": "scheduled",
    },
    {
        "id": "A004",
        "patient_id": "P002",
        "date": "2026-06-01",
        "time": "11:00",
        "doctor": "Dr. Emily Carter",
        "specialty": "General Practice",
        "status": "scheduled",
    },
    {
        "id": "A005",
        "patient_id": "P003",
        "date": "2026-05-15",
        "time": "08:30",
        "doctor": "Dr. James Wilson",
        "specialty": "Orthopedics",
        "status": "scheduled",
    },
]


# ---------------------------------------------------------------------------
# Initialize database with mock data
# ---------------------------------------------------------------------------

def initialize_mock_data() -> None:
    """
    Seed PostgreSQL with the canonical mock data set if the DB is empty.

    When USE_MOCK_DATA=true this is a no-op (data lives in-memory above).
    When USE_MOCK_DATA=false this ensures P001-P003 and A001-A005 always
    exist in the database so that the agent and tests can rely on them.
    """
    if USE_MOCK_DATA:
        logger.info("USE_MOCK_DATA=true — skipping PostgreSQL seeding (using in-memory data)")
        return

    # Only seed when the patients table is empty
    existing_patients = PatientService.list_patients()
    if existing_patients:
        logger.info(
            "Database already has %d patient(s) — skipping mock-data seed",
            len(existing_patients),
        )
        return

    logger.info("Seeding database with mock data …")

    for patient_data in PATIENTS:
        PatientService.create_patient(patient_data)

    for appointment_data in APPOINTMENTS:
        AppointmentService.create_appointment(appointment_data)

    logger.info("Mock data seeded: %d patients, %d appointments", len(PATIENTS), len(APPOINTMENTS))


# ---------------------------------------------------------------------------
# Data access helpers — dispatch to in-memory OR database
# ---------------------------------------------------------------------------

def find_patient(full_name: str, phone: str, date_of_birth: str) -> Optional[Dict[str, Any]]:
    """Return a patient record if identity details match, else None."""
    logger.debug(
        "Identity lookup | name='%s' phone='%s' dob='%s'",
        full_name, phone, date_of_birth,
    )

    if USE_MOCK_DATA:
        for patient in PATIENTS:
            name_match = patient["full_name"].strip().lower() == full_name.strip().lower()
            phone_match = (
                patient["phone"].replace("-", "").replace(" ", "")
                == phone.replace("-", "").replace(" ", "")
            )
            dob_match = patient["date_of_birth"] == date_of_birth.strip()
            if name_match and phone_match and dob_match:
                logger.info(
                    "Identity verified | patient_id='%s' name='%s'",
                    patient["id"], patient["full_name"],
                )
                return patient
        logger.warning(
            "Identity verification failed | name='%s' phone='%s' dob='%s'",
            full_name, phone, date_of_birth,
        )
        return None

    return PatientService.find_patient(full_name, phone, date_of_birth)


def get_all_patients() -> List[Dict[str, Any]]:
    """Return all patients."""
    if USE_MOCK_DATA:
        return list(PATIENTS)
    return PatientService.list_patients()


def get_appointments(patient_id: str) -> List[Dict[str, Any]]:
    """Return all appointments for a given patient."""
    if USE_MOCK_DATA:
        results = [a for a in APPOINTMENTS if a["patient_id"] == patient_id]
        logger.debug(
            "Fetched appointments | patient_id='%s' count=%d", patient_id, len(results)
        )
        return results
    return AppointmentService.get_appointments(patient_id)


def get_appointment_by_id(
    appointment_id: str, patient_id: str
) -> Optional[Dict[str, Any]]:
    """Return a specific appointment if it belongs to the patient, else None."""
    if USE_MOCK_DATA:
        for appt in APPOINTMENTS:
            if appt["id"] == appointment_id and appt["patient_id"] == patient_id:
                logger.debug(
                    "Appointment found | appointment_id='%s' patient_id='%s'",
                    appointment_id, patient_id,
                )
                return appt
        logger.debug(
            "Appointment not found | appointment_id='%s' patient_id='%s'",
            appointment_id, patient_id,
        )
        return None
    return AppointmentService.get_appointment_by_id(appointment_id, patient_id)


def confirm_appointment(
    appointment_id: str, patient_id: str
) -> Optional[Dict[str, Any]]:
    """Mark an appointment as confirmed. Returns the updated record or None."""
    if USE_MOCK_DATA:
        for appt in APPOINTMENTS:
            if appt["id"] == appointment_id and appt["patient_id"] == patient_id:
                appt["status"] = "confirmed"
                logger.info(
                    "Appointment confirmed | appointment_id='%s' patient_id='%s' date='%s' doctor='%s'",
                    appointment_id, patient_id, appt["date"], appt["doctor"],
                )
                return appt
        logger.warning(
            "Confirm failed — not found | appointment_id='%s' patient_id='%s'",
            appointment_id, patient_id,
        )
        return None
    return AppointmentService.confirm_appointment(appointment_id, patient_id)


def cancel_appointment(
    appointment_id: str, patient_id: str
) -> Optional[Dict[str, Any]]:
    """Mark an appointment as cancelled. Returns the updated record or None."""
    if USE_MOCK_DATA:
        for appt in APPOINTMENTS:
            if appt["id"] == appointment_id and appt["patient_id"] == patient_id:
                appt["status"] = "cancelled"
                logger.info(
                    "Appointment cancelled | appointment_id='%s' patient_id='%s' date='%s' doctor='%s'",
                    appointment_id, patient_id, appt["date"], appt["doctor"],
                )
                return appt
        logger.warning(
            "Cancel failed — not found | appointment_id='%s' patient_id='%s'",
            appointment_id, patient_id,
        )
        return None
    return AppointmentService.cancel_appointment(appointment_id, patient_id)


# ---------------------------------------------------------------------------
# Async versions — for FastAPI route handlers
# ---------------------------------------------------------------------------

async def async_find_patient(
    full_name: str, phone: str, date_of_birth: str
) -> Optional[Dict[str, Any]]:
    """Async version of find_patient."""
    if USE_MOCK_DATA:
        return find_patient(full_name, phone, date_of_birth)
    return await AsyncPatientService.find_patient(full_name, phone, date_of_birth)


async def async_get_appointments(patient_id: str) -> List[Dict[str, Any]]:
    """Async version of get_appointments."""
    if USE_MOCK_DATA:
        return get_appointments(patient_id)
    return await AsyncAppointmentService.get_appointments(patient_id)


async def async_get_appointment_by_id(
    appointment_id: str, patient_id: str
) -> Optional[Dict[str, Any]]:
    """Async version of get_appointment_by_id."""
    if USE_MOCK_DATA:
        return get_appointment_by_id(appointment_id, patient_id)
    return await AsyncAppointmentService.get_appointment_by_id(appointment_id, patient_id)


async def async_confirm_appointment(
    appointment_id: str, patient_id: str
) -> Optional[Dict[str, Any]]:
    """Async version of confirm_appointment."""
    if USE_MOCK_DATA:
        return confirm_appointment(appointment_id, patient_id)
    return await AsyncAppointmentService.confirm_appointment(appointment_id, patient_id)


async def async_cancel_appointment(
    appointment_id: str, patient_id: str
) -> Optional[Dict[str, Any]]:
    """Async version of cancel_appointment."""
    if USE_MOCK_DATA:
        return cancel_appointment(appointment_id, patient_id)
    return await AsyncAppointmentService.cancel_appointment(appointment_id, patient_id)
