"""
PostgreSQL persistence layer for patient and appointment data.

This module provides services to store and retrieve patient and appointment data from PostgreSQL.
It replaces the in-memory implementation from app.data.py.
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from sqlalchemy import select, update, delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.db import Patient, Appointment, AppointmentStatus, get_db, get_async_db
from app.logger import get_logger

logger = get_logger(__name__)


class PatientService:
    """
    Service for managing patient data in PostgreSQL.
    """
    
    @staticmethod
    def find_patient(full_name: str, phone: str, date_of_birth: str) -> Optional[Dict[str, Any]]:
        """
        Find a patient by identity details.
        
        Args:
            full_name: Patient's full name
            phone: Patient's phone number
            date_of_birth: Patient's date of birth (YYYY-MM-DD format)
        
        Returns:
            Dict or None: Patient data if found, None otherwise
        """
        try:
            db = get_db()
            
            # Normalize inputs for comparison
            normalized_name = full_name.strip().lower()
            normalized_phone = phone.replace("-", "").replace(" ", "")
            normalized_dob = date_of_birth.strip()
            
            # Use the ORM to query patients
            stmt = select(Patient).where(
                Patient.full_name.ilike(f"%{normalized_name}%")  # Case insensitive search
            )
            result = db.execute(stmt)
            patients = result.scalars().all()
            
            # Manual verification to handle phone normalization properly
            for patient in patients:
                patient_phone = patient.phone.replace("-", "").replace(" ", "")
                name_match = patient.full_name.strip().lower() == normalized_name
                phone_match = patient_phone == normalized_phone
                dob_match = patient.date_of_birth == normalized_dob
                
                if name_match and phone_match and dob_match:
                    logger.info(
                        "Identity verified | patient_id='%s' name='%s'",
                        patient.id, patient.full_name
                    )
                    return patient.to_dict()
            
            logger.warning(
                "Identity verification failed | name='%s' phone='%s' dob='%s'",
                full_name, phone, date_of_birth
            )
            return None
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during patient search: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during patient search: {str(e)}")
            return None
    
    @staticmethod
    def get_patient(patient_id: str) -> Optional[Dict[str, Any]]:
        """Get a patient by ID."""
        try:
            db = get_db()
            patient = db.query(Patient).filter(Patient.id == patient_id).first()
            
            if patient:
                return patient.to_dict()
            return None
            
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving patient {patient_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving patient {patient_id}: {str(e)}")
            return None
    
    @staticmethod
    def list_patients() -> List[Dict[str, Any]]:
        """Get all patients."""
        try:
            db = get_db()
            result = db.query(Patient).all()
            
            return [patient.to_dict() for patient in result]
            
        except SQLAlchemyError as e:
            logger.error(f"Database error listing patients: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing patients: {str(e)}")
            return []
    
    @staticmethod
    def create_patient(patient_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new patient."""
        try:
            db = get_db()
            
            # Check if ID already exists
            if "id" in patient_data:
                existing = db.query(Patient).filter(Patient.id == patient_data["id"]).first()
                if existing:
                    logger.warning(f"Patient ID '{patient_data['id']}' already exists")
                    return None
            
            # Generate ID if not provided
            if "id" not in patient_data:
                # Get the highest existing ID and increment
                highest = db.query(Patient).order_by(Patient.id.desc()).first()
                if highest:
                    # Assuming IDs are in format "PXXX" where XXX is a number
                    highest_num = int(highest.id[1:])
                    patient_data["id"] = f"P{(highest_num + 1):03d}"
                else:
                    # If no patients exist yet, start with P001
                    patient_data["id"] = "P001"
            
            patient = Patient.from_dict(patient_data)
            db.add(patient)
            db.commit()
            
            logger.info(f"Created patient: {patient.id} - {patient.full_name}")
            return patient.to_dict()
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database error creating patient: {str(e)}")
            return None
        except Exception as e:
            db.rollback()
            logger.error(f"Unexpected error creating patient: {str(e)}")
            return None


class AppointmentService:
    """
    Service for managing appointment data in PostgreSQL.
    """
    
    @staticmethod
    def get_appointments(patient_id: str) -> List[Dict[str, Any]]:
        """
        Get all appointments for a patient.
        
        Args:
            patient_id: The patient's ID
            
        Returns:
            List of appointment dictionaries
        """
        try:
            db = get_db()
            
            result = db.query(Appointment).filter(
                Appointment.patient_id == patient_id
            ).all()
            
            appointments = [appointment.to_dict() for appointment in result]
            logger.debug(f"Fetched appointments | patient_id='{patient_id}' count={len(appointments)}")
            return appointments
            
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving appointments for {patient_id}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error retrieving appointments for {patient_id}: {str(e)}")
            return []
    
    @staticmethod
    def get_appointment_by_id(appointment_id: str, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific appointment by ID.
        
        Args:
            appointment_id: The appointment's ID
            patient_id: The patient's ID (for verification)
            
        Returns:
            Dict or None: Appointment data if found, None otherwise
        """
        try:
            db = get_db()
            
            appointment = db.query(Appointment).filter(
                Appointment.id == appointment_id,
                Appointment.patient_id == patient_id
            ).first()
            
            if appointment:
                logger.debug(f"Appointment found | appointment_id='{appointment_id}' patient_id='{patient_id}'")
                return appointment.to_dict()
            
            logger.debug(f"Appointment not found | appointment_id='{appointment_id}' patient_id='{patient_id}'")
            return None
            
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving appointment {appointment_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving appointment {appointment_id}: {str(e)}")
            return None
    
    @staticmethod
    def confirm_appointment(appointment_id: str, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Mark an appointment as confirmed.
        
        Args:
            appointment_id: The appointment's ID
            patient_id: The patient's ID (for verification)
            
        Returns:
            Dict or None: Updated appointment data if successful, None otherwise
        """
        try:
            db = get_db()
            
            appointment = db.query(Appointment).filter(
                Appointment.id == appointment_id,
                Appointment.patient_id == patient_id
            ).first()
            
            if not appointment:
                logger.warning(
                    f"Confirm failed — appointment not found | " 
                    f"appointment_id='{appointment_id}' patient_id='{patient_id}'"
                )
                return None
            
            # Update status
            appointment.status = AppointmentStatus.CONFIRMED
            appointment.updated_at = datetime.utcnow()
            db.commit()
            
            logger.info(
                f"Appointment confirmed | appointment_id='{appointment_id}' "
                f"patient_id='{patient_id}' date='{appointment.date}' doctor='{appointment.doctor}'"
            )
            return appointment.to_dict()
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database error confirming appointment {appointment_id}: {str(e)}")
            return None
        except Exception as e:
            db.rollback()
            logger.error(f"Unexpected error confirming appointment {appointment_id}: {str(e)}")
            return None
    
    @staticmethod
    def cancel_appointment(appointment_id: str, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Mark an appointment as cancelled.
        
        Args:
            appointment_id: The appointment's ID
            patient_id: The patient's ID (for verification)
            
        Returns:
            Dict or None: Updated appointment data if successful, None otherwise
        """
        try:
            db = get_db()
            
            appointment = db.query(Appointment).filter(
                Appointment.id == appointment_id,
                Appointment.patient_id == patient_id
            ).first()
            
            if not appointment:
                logger.warning(
                    f"Cancel failed — appointment not found | " 
                    f"appointment_id='{appointment_id}' patient_id='{patient_id}'"
                )
                return None
            
            # Update status
            appointment.status = AppointmentStatus.CANCELLED
            appointment.updated_at = datetime.utcnow()
            db.commit()
            
            logger.info(
                f"Appointment cancelled | appointment_id='{appointment_id}' "
                f"patient_id='{patient_id}' date='{appointment.date}' doctor='{appointment.doctor}'"
            )
            return appointment.to_dict()
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database error cancelling appointment {appointment_id}: {str(e)}")
            return None
        except Exception as e:
            db.rollback()
            logger.error(f"Unexpected error cancelling appointment {appointment_id}: {str(e)}")
            return None
    
    @staticmethod
    def create_appointment(appointment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new appointment."""
        try:
            db = get_db()
            
            # Check if ID already exists
            if "id" in appointment_data:
                existing = db.query(Appointment).filter(Appointment.id == appointment_data["id"]).first()
                if existing:
                    logger.warning(f"Appointment ID {appointment_data['id']} already exists")
                    return None
            
            # Generate ID if not provided
            if "id" not in appointment_data:
                # Get the highest existing ID and increment
                highest = db.query(Appointment).order_by(Appointment.id.desc()).first()
                if highest:
                    # Assuming IDs are in format "AXXX" where XXX is a number
                    highest_num = int(highest.id[1:])
                    appointment_data["id"] = f"A{(highest_num + 1):03d}"
                else:
                    # If no appointments exist yet, start with A001
                    appointment_data["id"] = "A001"
            
            # Ensure patient exists
            if "patient_id" in appointment_data:
                patient = db.query(Patient).filter(Patient.id == appointment_data["patient_id"]).first()
                if not patient:
                    logger.warning(f"Patient {appointment_data['patient_id']} does not exist")
                    return None
            
            appointment = Appointment.from_dict(appointment_data)
            db.add(appointment)
            db.commit()
            
            logger.info(
                f"Created appointment: {appointment.id} - Patient: {appointment.patient_id} "
                f"Date: {appointment.date} Doctor: {appointment.doctor}"
            )
            return appointment.to_dict()
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database error creating appointment: {str(e)}")
            return None
        except Exception as e:
            db.rollback()
            logger.error(f"Unexpected error creating appointment: {str(e)}")
            return None


class AsyncPatientService:
    """
    Async service for managing patient data in PostgreSQL.
    """
    
    @staticmethod
    async def find_patient(full_name: str, phone: str, date_of_birth: str) -> Optional[Dict[str, Any]]:
        """
        Find a patient by identity details.
        
        Args:
            full_name: Patient's full name
            phone: Patient's phone number
            date_of_birth: Patient's date of birth (YYYY-MM-DD format)
        
        Returns:
            Dict or None: Patient data if found, None otherwise
        """
        try:
            # Normalize inputs for comparison
            normalized_name = full_name.strip().lower()
            normalized_phone = phone.replace("-", "").replace(" ", "")
            normalized_dob = date_of_birth.strip()
            
            async with get_async_db() as db:
                # Use SQLAlchemy to query patients
                stmt = select(Patient).where(
                    Patient.full_name.ilike(f"%{normalized_name}%")  # Case insensitive search
                )
                result = await db.execute(stmt)
                patients = result.scalars().all()
                
                # Manual verification to handle phone normalization properly
                for patient in patients:
                    patient_phone = patient.phone.replace("-", "").replace(" ", "")
                    name_match = patient.full_name.strip().lower() == normalized_name
                    phone_match = patient_phone == normalized_phone
                    dob_match = patient.date_of_birth == normalized_dob
                    
                    if name_match and phone_match and dob_match:
                        logger.info(
                            "Identity verified | patient_id='%s' name='%s'",
                            patient.id, patient.full_name
                        )
                        return patient.to_dict()
                
                logger.warning(
                    "Identity verification failed | name='%s' phone='%s' dob='%s'",
                    full_name, phone, date_of_birth
                )
                return None
                
        except SQLAlchemyError as e:
            logger.error(f"Database error during patient search: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during patient search: {str(e)}")
            return None


class AsyncAppointmentService:
    """
    Async service for managing appointment data in PostgreSQL.
    """
    
    @staticmethod
    async def get_appointments(patient_id: str) -> List[Dict[str, Any]]:
        """
        Get all appointments for a patient.
        
        Args:
            patient_id: The patient's ID
            
        Returns:
            List of appointment dictionaries
        """
        try:
            async with get_async_db() as db:
                result = await db.execute(
                    select(Appointment).where(Appointment.patient_id == patient_id)
                )
                appointments = result.scalars().all()
                
                appointment_dicts = [appointment.to_dict() for appointment in appointments]
                logger.debug(f"Fetched appointments | patient_id='{patient_id}' count={len(appointment_dicts)}")
                return appointment_dicts
                
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving appointments for {patient_id}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error retrieving appointments for {patient_id}: {str(e)}")
            return []
    
    @staticmethod
    async def get_appointment_by_id(appointment_id: str, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific appointment by ID.
        
        Args:
            appointment_id: The appointment's ID
            patient_id: The patient's ID (for verification)
            
        Returns:
            Dict or None: Appointment data if found, None otherwise
        """
        try:
            async with get_async_db() as db:
                result = await db.execute(
                    select(Appointment).where(
                        Appointment.id == appointment_id,
                        Appointment.patient_id == patient_id
                    )
                )
                appointment = result.scalars().first()
                
                if appointment:
                    logger.debug(f"Appointment found | appointment_id='{appointment_id}' patient_id='{patient_id}'")
                    return appointment.to_dict()
                
                logger.debug(f"Appointment not found | appointment_id='{appointment_id}' patient_id='{patient_id}'")
                return None
                
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving appointment {appointment_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving appointment {appointment_id}: {str(e)}")
            return None
    
    @staticmethod
    async def confirm_appointment(appointment_id: str, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Mark an appointment as confirmed.
        
        Args:
            appointment_id: The appointment's ID
            patient_id: The patient's ID (for verification)
            
        Returns:
            Dict or None: Updated appointment data if successful, None otherwise
        """
        try:
            async with get_async_db() as db:
                result = await db.execute(
                    select(Appointment).where(
                        Appointment.id == appointment_id,
                        Appointment.patient_id == patient_id
                    )
                )
                appointment = result.scalars().first()
                
                if not appointment:
                    logger.warning(
                        f"Confirm failed — appointment not found | " 
                        f"appointment_id='{appointment_id}' patient_id='{patient_id}'"
                    )
                    return None
                
                # Update status
                appointment.status = AppointmentStatus.CONFIRMED
                appointment.updated_at = datetime.utcnow()
                await db.commit()
                
                logger.info(
                    f"Appointment confirmed | appointment_id='{appointment_id}' "
                    f"patient_id='{patient_id}' date='{appointment.date}' doctor='{appointment.doctor}'"
                )
                return appointment.to_dict()
                
        except SQLAlchemyError as e:
            if "db" in locals():
                await db.rollback()
            logger.error(f"Database error confirming appointment {appointment_id}: {str(e)}")
            return None
        except Exception as e:
            if "db" in locals():
                await db.rollback()
            logger.error(f"Unexpected error confirming appointment {appointment_id}: {str(e)}")
            return None
    
    @staticmethod
    async def cancel_appointment(appointment_id: str, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Mark an appointment as cancelled.
        
        Args:
            appointment_id: The appointment's ID
            patient_id: The patient's ID (for verification)
            
        Returns:
            Dict or None: Updated appointment data if successful, None otherwise
        """
        try:
            async with get_async_db() as db:
                result = await db.execute(
                    select(Appointment).where(
                        Appointment.id == appointment_id,
                        Appointment.patient_id == patient_id
                    )
                )
                appointment = result.scalars().first()
                
                if not appointment:
                    logger.warning(
                        f"Cancel failed — appointment not found | " 
                        f"appointment_id='{appointment_id}' patient_id='{patient_id}'"
                    )
                    return None
                
                # Update status
                appointment.status = AppointmentStatus.CANCELLED
                appointment.updated_at = datetime.utcnow()
                await db.commit()
                
                logger.info(
                    f"Appointment cancelled | appointment_id='{appointment_id}' "
                    f"patient_id='{patient_id}' date='{appointment.date}' doctor='{appointment.doctor}'"
                )
                return appointment.to_dict()
                
        except SQLAlchemyError as e:
            if "db" in locals():
                await db.rollback()
            logger.error(f"Database error cancelling appointment {appointment_id}: {str(e)}")
            return None
        except Exception as e:
            if "db" in locals():
                await db.rollback()
            logger.error(f"Unexpected error cancelling appointment {appointment_id}: {str(e)}")
            return None