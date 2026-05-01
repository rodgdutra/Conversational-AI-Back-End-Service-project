import os
import sys
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from sqlalchemy import text

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.db import init_db, engine, Base, get_db, get_async_db
from app.logger import get_logger

logger = get_logger(__name__)


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def initialize_database() -> None:
    """Initialize the database schema before running any tests."""
    from app.config import config
    
    logger.info("Setting up test database...")
    logger.info(f"Using database: {config.DATABASE_URL}")
    logger.info(f"Using schema: {config.POSTGRES_SCHEMA}")
    
    try:
        # Clear existing tables and recreate schema
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        
        logger.info("Test database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise


@pytest.fixture(scope="function")
def db_session() -> Generator:
    """Get a database session for tests."""
    db = get_db()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


@pytest.fixture(scope="function")
async def async_db_session() -> AsyncGenerator:
    """Get an async database session for tests."""
    async with get_async_db() as session:
        try:
            yield session
        finally:
            await session.rollback()


@pytest.fixture(scope="function", autouse=True)
def cleanup_database(db_session) -> None:
    """Clean up the database after each test."""
    # This will be executed after each test function
    yield
    
    try:
        # Clear the tables to start with a clean state for next test
        logger.info("Cleaning up database after test...")
        
        # Disable triggers and foreign key constraints temporarily
        db_session.execute(text("SET session_replication_role = 'replica'"))
        
        # Get list of tables to clean
        tables_to_clean = list(reversed(Base.metadata.sorted_tables))
        logger.info(f"Tables to clean: {[table.name for table in tables_to_clean]}")
        
        # Clean each table
        for table in tables_to_clean:
            try:
                # Try TRUNCATE first
                logger.info(f"Cleaning table {table.name}...")
                db_session.execute(text(f'TRUNCATE TABLE "{table.schema}.{table.name}" CASCADE'))
            except Exception as e:
                # If truncate fails, try DELETE
                logger.warning(f"Failed to TRUNCATE table {table.name}, trying DELETE: {str(e)}")
                try:
                    db_session.execute(text(f'DELETE FROM "{table.schema}.{table.name}"'))
                except Exception as e2:
                    logger.error(f"Also failed to DELETE from {table.name}: {str(e2)}")
                    # Continue to next table
                    continue
        
        # Re-enable triggers and constraints
        db_session.execute(text("SET session_replication_role = 'origin'"))
        
        # Commit changes
        db_session.commit()
        logger.info("Database cleanup complete")
        
    except Exception as e:
        logger.error(f"Error during database cleanup: {str(e)}")
        db_session.rollback()
