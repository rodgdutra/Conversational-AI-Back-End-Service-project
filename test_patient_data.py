#!/usr/bin/env python
"""
Test script for patient and appointment data persistence.

This script tests:
1. CRUD operations for patients
2. CRUD operations for appointments
3. Both sync and async versions of the services

Usage:
    python test_patient_data.py
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple

from app.db import init_db
from app.data_persistence import (
    PatientService, AppointmentService, 
    AsyncPatientService, AsyncAppointmentService
)
from app.logger import get_logger

logger = get_logger(__name__)


def generate_test_id() -> str:
    """Generate a unique ID for test data."""
    return f"TEST-{uuid.uuid4().hex[:8]}"


def create_test_patient() -> Dict[str, Any]:
    """Create test patient data."""
    test_id = generate_test_id()
    return {
        "full_name": f"Test Patient {test_id}",
        "phone": f"555-{test_id[-4:]}",
        "date_of_birth": "2000-01-01",
    }


def create_test_appointment(patient_id: str) -> Dict[str, Any]:
    """Create test appointment data."""
    return {
        "patient_id": patient_id,
        "date": "2026-12-31",
        "time": "12:00",
        "doctor": "Dr. Test Doctor",
        "specialty": "Test Specialty",
        "status": "scheduled",
    }


def test_sync_patient_service():
    """Test synchronous patient service."""
    logger.info("Testing PatientService...")
    
    # Create a patient
    patient_data = create_test_patient()
    created_patient = PatientService.create_patient(patient_data)
    
    if not created_patient:
        logger.error("Failed to create test patient")
        return False
    
    patient_id = created_patient["id"]
    logger.info(f"Created test patient with ID: {patient_id}")
    
    # Get the patient
    retrieved_patient = PatientService.get_patient(patient_id)
    
    if not retrieved_patient:
        logger.error(f"Failed to retrieve patient with ID: {patient_id}")
        return False
    
    if retrieved_patient["full_name"] != patient_data["full_name"]:
        logger.error(f"Patient name mismatch: {retrieved_patient['full_name']} vs {patient_data['full_name']}")
        return False
    
    # Find by identity
    found_patient = PatientService.find_patient(
        patient_data["full_name"],
        patient_data["phone"],
        patient_data["date_of_birth"],
    )
    
    if not found_patient:
        logger.error("Failed to find patient by identity details")
        return False
    
    if found_patient["id"] != patient_id:
        logger.error(f"Found wrong patient: {found_patient['id']} vs {patient_id}")
        return False
    
    logger.info("Patient service tests passed")
    return patient_id


def test_sync_appointment_service(patient_id: str):
    """Test synchronous appointment service."""
    logger.info("Testing AppointmentService...")
    
    # Create an appointment
    appointment_data = create_test_appointment(patient_id)
    created_appointment = AppointmentService.create_appointment(appointment_data)
    
    if not created_appointment:
        logger.error("Failed to create test appointment")
        return False
    
    appointment_id = created_appointment["id"]
    logger.info(f"Created test appointment with ID: {appointment_id}")
    
    # Get appointments for patient
    appointments = AppointmentService.get_appointments(patient_id)
    
    if not appointments:
        logger.error(f"Failed to retrieve appointments for patient {patient_id}")
        return False
    
    if len(appointments) == 0:
        logger.error(f"No appointments found for patient {patient_id}")
        return False
    
    # Get specific appointment
    appointment = AppointmentService.get_appointment_by_id(appointment_id, patient_id)
    
    if not appointment:
        logger.error(f"Failed to retrieve appointment {appointment_id}")
        return False
    
    # Confirm appointment
    confirmed = AppointmentService.confirm_appointment(appointment_id, patient_id)
    
    if not confirmed:
        logger.error(f"Failed to confirm appointment {appointment_id}")
        return False
    
    if confirmed["status"] != "confirmed":
        logger.error(f"Appointment status not updated: {confirmed['status']}")
        return False
    
    # Cancel appointment
    cancelled = AppointmentService.cancel_appointment(appointment_id, patient_id)
    
    if not cancelled:
        logger.error(f"Failed to cancel appointment {appointment_id}")
        return False
    
    if cancelled["status"] != "cancelled":
        logger.error(f"Appointment status not updated: {cancelled['status']}")
        return False
    
    logger.info("Appointment service tests passed")
    return True


async def test_async_patient_service():
    """Test asynchronous patient service."""
    logger.info("Testing AsyncPatientService...")
    
    # Generate test data
    patient_data = create_test_patient()
    
    # We need to create the patient using the sync service first
    # since the async service doesn't have a create method
    created_patient = PatientService.create_patient(patient_data)
    
    if not created_patient:
        logger.error("Failed to create test patient")
        return False
    
    patient_id = created_patient["id"]
    logger.info(f"Created test patient with ID: {patient_id}")
    
    # Find by identity
    found_patient = await AsyncPatientService.find_patient(
        patient_data["full_name"],
        patient_data["phone"],
        patient_data["date_of_birth"],
    )
    
    if not found_patient:
        logger.error("Failed to find patient by identity details (async)")
        return False
    
    if found_patient["id"] != patient_id:
        logger.error(f"Found wrong patient: {found_patient['id']} vs {patient_id}")
        return False
    
    logger.info("Async patient service tests passed")
    return patient_id


async def test_async_appointment_service(patient_id: str):
    """Test asynchronous appointment service."""
    logger.info("Testing AsyncAppointmentService...")
    
    # Generate test data
    appointment_data = create_test_appointment(patient_id)
    
    # Create the appointment using sync service first
    created_appointment = AppointmentService.create_appointment(appointment_data)
    
    if not created_appointment:
        logger.error("Failed to create test appointment")
        return False
    
    appointment_id = created_appointment["id"]
    logger.info(f"Created test appointment with ID: {appointment_id}")
    
    # Get appointments for patient
    appointments = await AsyncAppointmentService.get_appointments(patient_id)
    
    if not appointments:
        logger.error(f"Failed to retrieve appointments for patient {patient_id}")
        return False
    
    if len(appointments) == 0:
        logger.error(f"No appointments found for patient {patient_id}")
        return False
    
    # Get specific appointment
    appointment = await AsyncAppointmentService.get_appointment_by_id(appointment_id, patient_id)
    
    if not appointment:
        logger.error(f"Failed to retrieve appointment {appointment_id}")
        return False
    
    # Confirm appointment
    confirmed = await AsyncAppointmentService.confirm_appointment(appointment_id, patient_id)
    
    if not confirmed:
        logger.error(f"Failed to confirm appointment {appointment_id}")
        return False
    
    if confirmed["status"] != "confirmed":
        logger.error(f"Appointment status not updated: {confirmed['status']}")
        return False
    
    # Cancel appointment
    cancelled = await AsyncAppointmentService.cancel_appointment(appointment_id, patient_id)
    
    if not cancelled:
        logger.error(f"Failed to cancel appointment {appointment_id}")
        return False
    
    if cancelled["status"] != "cancelled":
        logger.error(f"Appointment status not updated: {cancelled['status']}")
        return False
    
    logger.info("Async appointment service tests passed")
    return True


async def run_all_tests():
    """Run all tests."""
    logger.info("Starting patient and appointment data tests...")
    
    try:
        # Initialize database
        init_db()
        logger.info("Database schema initialized")
        
        # Run synchronous tests
        patient_id = test_sync_patient_service()
        if not patient_id:
            logger.error("Patient service tests failed!")
            return False
        
        appointment_test = test_sync_appointment_service(patient_id)
        if not appointment_test:
            logger.error("Appointment service tests failed!")
            return False
        
        # Run asynchronous tests
        async_patient_id = await test_async_patient_service()
        if not async_patient_id:
            logger.error("Async patient service tests failed!")
            return False
        
        async_appointment_test = await test_async_appointment_service(async_patient_id)
        if not async_appointment_test:
            logger.error("Async appointment service tests failed!")
            return False
        
        logger.info("All patient and appointment data tests passed! 🎉")
        return True
    
    except Exception as e:
        logger.error(f"Unexpected error in tests: {str(e)}", exc_info=True)
        return False


def main():
    """Entry point."""
    success = asyncio.run(run_all_tests())
    if not success:
        logger.error("Some tests failed! 😞")


if __name__ == "__main__":
    main()