"""
Database connection and session management.
"""
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any

from sqlalchemy import Column, String, Text, create_engine, MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from app.config import config
from app.logger import get_logger

logger = get_logger(__name__)

# Initialize SQLAlchemy base
metadata = MetaData(schema=config.POSTGRES_SCHEMA)
Base = declarative_base(metadata=metadata)


class GraphState(Base):
    """
    Persistent storage for LangGraph state snapshots.
    
    Each row represents a state snapshot associated with a session_id.
    The state is stored as serialized JSON in the state_data column.
    """
    __tablename__ = "graph_states"
    
    session_id = Column(String(64), primary_key=True, index=True)
    state_data = Column(Text, nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the stored state data back to a dictionary."""
        return json.loads(self.state_data)
    
    @classmethod
    def from_dict(cls, session_id: str, state_data: Dict[str, Any]) -> "GraphState":
        """Create a new GraphState instance from a dictionary."""
        return cls(
            session_id=session_id,
            state_data=json.dumps(state_data),
        )


# Synchronous engine and session factory
engine = create_engine(
    config.DATABASE_URL,
    echo=config.LOG_LEVEL == "DEBUG",
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Asynchronous engine and session factory
async_engine = create_async_engine(
    config.ASYNC_DATABASE_URL,
    echo=config.LOG_LEVEL == "DEBUG",
    pool_pre_ping=True,
)
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