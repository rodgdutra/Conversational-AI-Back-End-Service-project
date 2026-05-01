"""
Database connection and session management.

This module defines:
1. Database models for LangGraph state persistence
2. Database models for patient and appointment data
3. Connection management and initialization functions
"""
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any, List
from datetime import datetime, date
from enum import Enum

from sqlalchemy import Column, String, Text, Integer, DateTime, Date, Time, ForeignKey, create_engine, MetaData, desc, func, Enum as SQLEnum
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

from app.config import config
from app.logger import get_logger

logger = get_logger(__name__)

# Initialize SQLAlchemy base
metadata = MetaData(schema=config.POSTGRES_SCHEMA)
Base = declarative_base(metadata=metadata)


class Session(Base):
    """
    Represents a conversation session.
    
    Each conversation has a unique session_id and can contain multiple state snapshots.
    """
    __tablename__ = "sessions"
    
    session_id = Column(String(64), primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationship to GraphState - one session has many states
    states = relationship("GraphState", back_populates="session", cascade="all, delete-orphan")


class GraphState(Base):
    """
    Persistent storage for LangGraph state snapshots.
    
    Each row represents a state snapshot associated with a session_id and state_id.
    The state is stored as serialized JSON in the state_data column.
    """
    __tablename__ = "graph_states"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.session_id"), nullable=False, index=True)
    state_id = Column(Integer, nullable=False, index=True)
    state_data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship to Session - many states belong to one session
    session = relationship("Session", back_populates="states")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the stored state data back to a dictionary."""
        state_dict = json.loads(self.state_data)
        # Add metadata about the state
        state_dict["_metadata"] = {
            "state_id": self.state_id,
            "created_at": self.created_at.isoformat(),
            "session_id": self.session_id
        }
        return state_dict
    
    @classmethod
    def from_dict(cls, session_id: str, state_id: int, state_data: Dict[str, Any]) -> "GraphState":
        """Create a new GraphState instance from a dictionary."""
        # Remove metadata if present to avoid duplication
        if "_metadata" in state_data:
            state_data = {k: v for k, v in state_data.items() if k != "_metadata"}
        
        return cls(
            session_id=session_id,
            state_id=state_id,
            state_data=json.dumps(state_data),
        )


class StateTransition(Base):
    """
    Tracks transitions between states in a conversation.
    
    This allows for analyzing how the conversation flows from one state to another.
    """
    __tablename__ = "state_transitions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.session_id"), nullable=False, index=True)
    from_state_id = Column(Integer, nullable=False)
    to_state_id = Column(Integer, nullable=False)
    transition_type = Column(String(64), nullable=True)  # e.g., "user_message", "tool_execution", etc.
    transition_data = Column(Text, nullable=True)  # Additional JSON data about the transition
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the transition data to a dictionary."""
        result = {
            "id": self.id,
            "session_id": self.session_id,
            "from_state_id": self.from_state_id,
            "to_state_id": self.to_state_id,
            "transition_type": self.transition_type,
            "created_at": self.created_at.isoformat(),
        }
        
        if self.transition_data:
            result["data"] = json.loads(self.transition_data)
            
        return result
    
    @classmethod
    def from_dict(cls, session_id: str, from_state_id: int, to_state_id: int, 
                 transition_type: str = None, transition_data: Dict[str, Any] = None) -> "StateTransition":
        """Create a new StateTransition instance from data."""
        return cls(
            session_id=session_id,
            from_state_id=from_state_id,
            to_state_id=to_state_id,
            transition_type=transition_type,
            transition_data=json.dumps(transition_data) if transition_data else None,
        )


# ---------------------------------------------------------------------------
# Patient & Appointment Models
# ---------------------------------------------------------------------------

class Patient(Base):
    """
    Patient record for the appointment system.
    
    Matches the structure of the original mock data with proper SQLAlchemy typing.
    """
    __tablename__ = "patients"
    
    id = Column(String(10), primary_key=True)  # We'll keep the "P001" style IDs
    full_name = Column(String(100), nullable=False, index=True)
    phone = Column(String(20), nullable=False)
    date_of_birth = Column(String(10), nullable=False)  # Stored as YYYY-MM-DD string
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationship to appointments
    appointments = relationship("Appointment", back_populates="patient", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert patient to dictionary for API responses."""
        return {
            "id": self.id,
            "full_name": self.full_name,
            "phone": self.phone,
            "date_of_birth": self.date_of_birth,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Patient":
        """Create a Patient instance from a dictionary."""
        return cls(
            id=data.get("id"),
            full_name=data.get("full_name"),
            phone=data.get("phone"),
            date_of_birth=data.get("date_of_birth"),
        )


class AppointmentStatus(str, Enum):
    """Valid appointment statuses."""
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    MISSED = "missed"


class Appointment(Base):
    """
    Appointment record for the scheduling system.
    
    Matches the structure of the original mock data with proper SQLAlchemy typing.
    """
    __tablename__ = "appointments"
    
    id = Column(String(10), primary_key=True)  # We'll keep the "A001" style IDs
    patient_id = Column(String(10), ForeignKey("patients.id"), nullable=False, index=True)
    date = Column(String(10), nullable=False)  # Stored as YYYY-MM-DD string
    time = Column(String(5), nullable=False)   # Stored as HH:MM string
    doctor = Column(String(100), nullable=False)
    specialty = Column(String(100), nullable=False)
    status = Column(SQLEnum(AppointmentStatus), nullable=False, default=AppointmentStatus.SCHEDULED)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationship to patient
    patient = relationship("Patient", back_populates="appointments")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert appointment to dictionary for API responses."""
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "date": self.date,
            "time": self.time,
            "doctor": self.doctor,
            "specialty": self.specialty,
            "status": self.status.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Appointment":
        """Create an Appointment instance from a dictionary."""
        # Handle status - it might come as a string or enum
        status = data.get("status", AppointmentStatus.SCHEDULED)
        if isinstance(status, str):
            status = AppointmentStatus(status)
            
        return cls(
            id=data.get("id"),
            patient_id=data.get("patient_id"),
            date=data.get("date"),
            time=data.get("time"),
            doctor=data.get("doctor"),
            specialty=data.get("specialty"),
            status=status,
        )


# Synchronous engine and session factory
engine = create_engine(
    config.DATABASE_URL,
    echo=config.LOG_LEVEL == "DEBUG",
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Asynchronous engine and session factory
async_url = config.ASYNC_DATABASE_URL
# Ensure the URL has the asyncpg driver
if "postgresql+asyncpg://" not in async_url:
    logger.warning(f"ASYNC_DATABASE_URL doesn't have asyncpg driver, attempting to fix")
    async_url = async_url.replace("postgresql://", "postgresql+asyncpg://")
    async_url = async_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")

logger.info(f"Creating async engine with URL driver: {async_url.split('://', 1)[0]}")
try:
    async_engine = create_async_engine(
        async_url,
        echo=config.LOG_LEVEL == "DEBUG",
        pool_pre_ping=True,
    )
    logger.info("Async database engine created successfully")
except Exception as e:
    logger.error(f"Failed to create async database engine: {str(e)}")
    # Fallback to a placeholder async engine that will raise clear errors if used
    class FailingAsyncEngine:
        def connect(self):
            raise RuntimeError(f"Async database engine creation failed: {str(e)}")
    async_engine = FailingAsyncEngine()
AsyncSessionLocal = async_sessionmaker(
    autocommit=False, 
    autoflush=False, 
    expire_on_commit=False,
    bind=async_engine,
)


def create_tables() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    logger.info(f"Tables created in schema '{config.POSTGRES_SCHEMA}'")


def get_db() -> Session:
    """Get a database session for synchronous operations."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


@asynccontextmanager
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session for asynchronous operations."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Database initialization function
def init_db() -> None:
    """Initialize the database schema if it doesn't exist."""
    create_tables()
    logger.info("Database initialized")