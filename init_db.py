#!/usr/bin/env python
"""
Database initialization script for the Conversational AI Appointment Assistant.

This script initializes the PostgreSQL database for the application by:
1. Creating the database if it doesn't exist
2. Creating the necessary tables using SQLAlchemy models

Usage:
    python init_db.py
"""

import sys
import time
from sqlalchemy_utils import database_exists, create_database
import psycopg2
from psycopg2 import OperationalError

from app.config import config
from app.db import init_db
from app.logger import get_logger

logger = get_logger(__name__)

def wait_for_postgres(retries=5, delay=10):
    """Wait for PostgreSQL to be available."""
    conn_string = (
        f"host={config.POSTGRES_HOST} "
        f"port={config.POSTGRES_PORT} "
        f"user={config.POSTGRES_USER} "
        f"password={config.POSTGRES_PASSWORD}"
    )
    logger.debug("PostgreSQL connection string: %s", conn_string)
    
    for i in range(retries):
        try:
            conn = psycopg2.connect(conn_string)
            conn.close()
            logger.info("PostgreSQL is available!")
            return True
        except OperationalError:
            logger.warning(
                f"PostgreSQL not available yet. Retrying in {delay} seconds... "
                f"({i+1}/{retries})"
            )
            time.sleep(delay)
    
    logger.error(f"Could not connect to PostgreSQL after {retries} attempts")
    return False

def create_db_if_not_exists():
    """Create the database if it doesn't exist."""
    # Always use the synchronous database URL for creation (sqlalchemy_utils doesn't support asyncpg)
    db_url = config.DATABASE_URL
    
    # Make sure we're using psycopg2 for database creation
    if "postgresql+psycopg2://" not in db_url and "postgresql://" not in db_url:
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        logger.warning(f"Changed database URL driver to psycopg2 for database creation")
    
    logger.info(f"Checking if database '{config.POSTGRES_DB}' exists using URL: {db_url}")
    
    if not database_exists(db_url):
        logger.info(f"Creating database '{config.POSTGRES_DB}'...")
        create_database(db_url)
        logger.info(f"Database '{config.POSTGRES_DB}' created successfully")
    else:
        logger.info(f"Database '{config.POSTGRES_DB}' already exists")

def main():
    """Initialize the database."""
    logger.info("Starting database initialization")
    
    # Wait for PostgreSQL to be available
    if not wait_for_postgres():
        logger.error("Could not connect to PostgreSQL. Exiting...")
        sys.exit(1)
    
    # Create database if it doesn't exist
    try:
        create_db_if_not_exists()
    except Exception as e:
        logger.error(f"Failed to create database: {str(e)}")
        sys.exit(1)
    
    # Initialize tables
    try:
        init_db()
        logger.info("Database initialization completed successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize tables: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()