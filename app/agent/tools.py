"""
LangChain tools that wrap the mock data layer.

These tools are called by the LLM inside the agent graph nodes to perform
real actions (lookup, confirm, cancel) against the in-memory data store.
"""

from langchain_core.tools import tool
from app.data import (
    find_patient,
    get_appointments,
    get_appointment_by_id,
    confirm_appointment,
    cancel_appointment,
)
from app.logger import get_logger

logger = get_logger(__name__)


@tool
def verify_patient_tool(full_name: str, phone: str, date_of_birth: str) -> dict:
    """
    Verify a patient's identity using their full name, phone number, and
    date of birth (YYYY-MM-DD format).

    Returns a dict with:
      - verified (bool): True if the identity matched a record.
      - patient_id (str | None): Internal patient ID when verified.
      - patient_name (str | None): Patient's full name when verified.
      - message (str): Human-readable outcome.
    """
    logger.info("Tool called: verify_patient_tool | name='%s'", full_name)
    patient = find_patient(full_name, phone, date_of_birth)
    if patient:
        logger.info("Tool result: verify_patient_tool | SUCCESS patient_id='%s'", patient["id"])
        return {
            "verified": True,
            "patient_id": patient["id"],
            "patient_name": patient["full_name"],
            "message": f"Identity verified for {patient['full_name']}.",
        }
    logger.warning("Tool result: verify_patient_tool | FAILED name='%s'", full_name)
    return {
        "verified": False,
        "patient_id": None,
        "patient_name": None,
        "message": (
            "Could not verify identity. Please check your full name, "
            "phone number, and date of birth and try again."
        ),
    }


@tool
def list_appointments_tool(patient_id: str) -> dict:
    """
    List all appointments for the verified patient.

    Args:
        patient_id: The internal patient ID returned after verification.

    Returns a dict with:
      - appointments (list): List of appointment records.
      - message (str): Human-readable summary.
    """
    logger.info("Tool called: list_appointments_tool | patient_id='%s'", patient_id)
    appointments = get_appointments(patient_id)
    logger.info("Tool result: list_appointments_tool | patient_id='%s' count=%d", patient_id, len(appointments))
    if not appointments:
        return {
            "appointments": [],
            "message": "You have no appointments on record.",
        }
    lines = []
    for appt in appointments:
        lines.append(
            f"[{appt['id']}] {appt['date']} at {appt['time']} – "
            f"{appt['specialty']} with {appt['doctor']} (Status: {appt['status']})"
        )
    return {
        "appointments": appointments,
        "message": "Here are your appointments:\n" + "\n".join(lines),
    }


@tool
def confirm_appointment_tool(patient_id: str, appointment_id: str) -> dict:
    """
    Confirm a specific appointment for the verified patient.

    Args:
        patient_id: The internal patient ID returned after verification.
        appointment_id: The appointment ID (e.g. 'A001').

    Returns a dict with:
      - success (bool): Whether the operation succeeded.
      - appointment (dict | None): The updated appointment record.
      - message (str): Human-readable outcome.
    """
    logger.info(
        "Tool called: confirm_appointment_tool | appointment_id='%s' patient_id='%s'",
        appointment_id, patient_id,
    )
    appt = confirm_appointment(appointment_id, patient_id)
    if appt:
        logger.info(
            "Tool result: confirm_appointment_tool | SUCCESS appointment_id='%s' date='%s'",
            appointment_id, appt["date"],
        )
        return {
            "success": True,
            "appointment": appt,
            "message": (
                f"Appointment {appointment_id} on {appt['date']} at {appt['time']} "
                f"with {appt['doctor']} has been confirmed."
            ),
        }
    logger.warning(
        "Tool result: confirm_appointment_tool | NOT FOUND appointment_id='%s' patient_id='%s'",
        appointment_id, patient_id,
    )
    return {
        "success": False,
        "appointment": None,
        "message": (
            f"Could not confirm appointment {appointment_id}. "
            "Please make sure the appointment ID is correct."
        ),
    }


@tool
def cancel_appointment_tool(patient_id: str, appointment_id: str) -> dict:
    """
    Cancel a specific appointment for the verified patient.

    Args:
        patient_id: The internal patient ID returned after verification.
        appointment_id: The appointment ID (e.g. 'A001').

    Returns a dict with:
      - success (bool): Whether the operation succeeded.
      - appointment (dict | None): The updated appointment record.
      - message (str): Human-readable outcome.
    """
    logger.info(
        "Tool called: cancel_appointment_tool | appointment_id='%s' patient_id='%s'",
        appointment_id, patient_id,
    )
    appt = cancel_appointment(appointment_id, patient_id)
    if appt:
        logger.info(
            "Tool result: cancel_appointment_tool | SUCCESS appointment_id='%s' date='%s'",
            appointment_id, appt["date"],
        )
        return {
            "success": True,
            "appointment": appt,
            "message": (
                f"Appointment {appointment_id} on {appt['date']} at {appt['time']} "
                f"with {appt['doctor']} has been cancelled."
            ),
        }
    logger.warning(
        "Tool result: cancel_appointment_tool | NOT FOUND appointment_id='%s' patient_id='%s'",
        appointment_id, patient_id,
    )
    return {
        "success": False,
        "appointment": None,
        "message": (
            f"Could not cancel appointment {appointment_id}. "
            "Please make sure the appointment ID is correct."
        ),
    }


# Convenience export so nodes can import a single list
ALL_TOOLS = [
    verify_patient_tool,
    list_appointments_tool,
    confirm_appointment_tool,
    cancel_appointment_tool,
]
