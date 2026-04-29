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
    patient = find_patient(full_name, phone, date_of_birth)
    if patient:
        return {
            "verified": True,
            "patient_id": patient["id"],
            "patient_name": patient["full_name"],
            "message": f"Identity verified for {patient['full_name']}.",
        }
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
    appointments = get_appointments(patient_id)
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
    appt = confirm_appointment(appointment_id, patient_id)
    if appt:
        return {
            "success": True,
            "appointment": appt,
            "message": (
                f"Appointment {appointment_id} on {appt['date']} at {appt['time']} "
                f"with {appt['doctor']} has been confirmed."
            ),
        }
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
    appt = cancel_appointment(appointment_id, patient_id)
    if appt:
        return {
            "success": True,
            "appointment": appt,
            "message": (
                f"Appointment {appointment_id} on {appt['date']} at {appt['time']} "
                f"with {appt['doctor']} has been cancelled."
            ),
        }
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
