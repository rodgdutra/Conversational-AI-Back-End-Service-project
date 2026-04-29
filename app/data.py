"""
Mock database for patients and appointments.
In a real system this would be replaced with actual DB queries.
"""

from typing import Optional

from app.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Mock patient records
# ---------------------------------------------------------------------------
PATIENTS = [
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

# ---------------------------------------------------------------------------
# Mock appointment records (mutable so confirm/cancel persist in-memory)
# ---------------------------------------------------------------------------
APPOINTMENTS = [
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
# Data access helpers
# ---------------------------------------------------------------------------

def find_patient(full_name: str, phone: str, date_of_birth: str) -> Optional[dict]:
    """Return a patient record if identity details match, else None."""
    logger.debug(
        "Identity lookup attempt | name='%s' phone='%s' dob='%s'",
        full_name, phone, date_of_birth,
    )
    for patient in PATIENTS:
        name_match = patient["full_name"].strip().lower() == full_name.strip().lower()
        phone_match = patient["phone"].replace("-", "").replace(" ", "") == phone.replace("-", "").replace(" ", "")
        dob_match = patient["date_of_birth"] == date_of_birth.strip()
        if name_match and phone_match and dob_match:
            logger.info("Identity verified | patient_id='%s' name='%s'", patient["id"], patient["full_name"])
            return patient
    logger.warning("Identity verification failed | name='%s' phone='%s' dob='%s'", full_name, phone, date_of_birth)
    return None


def get_appointments(patient_id: str) -> list[dict]:
    """Return all appointments for a given patient."""
    results = [a for a in APPOINTMENTS if a["patient_id"] == patient_id]
    logger.debug("Fetched appointments | patient_id='%s' count=%d", patient_id, len(results))
    return results


def get_appointment_by_id(appointment_id: str, patient_id: str) -> Optional[dict]:
    """Return a specific appointment if it belongs to the patient, else None."""
    for appt in APPOINTMENTS:
        if appt["id"] == appointment_id and appt["patient_id"] == patient_id:
            logger.debug("Appointment found | appointment_id='%s' patient_id='%s'", appointment_id, patient_id)
            return appt
    logger.debug("Appointment not found | appointment_id='%s' patient_id='%s'", appointment_id, patient_id)
    return None


def confirm_appointment(appointment_id: str, patient_id: str) -> Optional[dict]:
    """Mark an appointment as confirmed. Returns the updated record or None."""
    for appt in APPOINTMENTS:
        if appt["id"] == appointment_id and appt["patient_id"] == patient_id:
            appt["status"] = "confirmed"
            logger.info(
                "Appointment confirmed | appointment_id='%s' patient_id='%s' date='%s' doctor='%s'",
                appointment_id, patient_id, appt["date"], appt["doctor"],
            )
            return appt
    logger.warning("Confirm failed — appointment not found | appointment_id='%s' patient_id='%s'", appointment_id, patient_id)
    return None


def cancel_appointment(appointment_id: str, patient_id: str) -> Optional[dict]:
    """Mark an appointment as cancelled. Returns the updated record or None."""
    for appt in APPOINTMENTS:
        if appt["id"] == appointment_id and appt["patient_id"] == patient_id:
            appt["status"] = "cancelled"
            logger.info(
                "Appointment cancelled | appointment_id='%s' patient_id='%s' date='%s' doctor='%s'",
                appointment_id, patient_id, appt["date"], appt["doctor"],
            )
            return appt
    logger.warning("Cancel failed — appointment not found | appointment_id='%s' patient_id='%s'", appointment_id, patient_id)
    return None
