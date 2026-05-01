#!/usr/bin/env python
"""
Test script for the PostgreSQL state persistence layer.

This script tests the storage and retrieval of LangGraph states from
the PostgreSQL database.

Usage:
    python test_persistence.py
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.db import init_db
from app.agent.persistence import AsyncStatePersistenceService, StatePersistenceService
from app.logger import get_logger

logger = get_logger(__name__)


def create_test_state(session_id: str) -> Dict[str, Any]:
    """Create a test LangGraph state for testing."""
    return {
        "messages": [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello, how can you help me?"),
            AIMessage(content="I can help you manage your appointments, but first I need to verify your identity.")
        ],
        "verified": False,
        "patient_id": None,
        "patient_name": None,
        "pending_action": None,
        "test_timestamp": datetime.now().isoformat(),
        "test_metadata": {"source": "test_script", "version": "1.0"}
    }


async def test_async_persistence():
    """Test the async persistence service."""
    logger.info("Testing AsyncStatePersistenceService...")
    
    # Generate a unique session ID for testing
    session_id = f"test-{uuid.uuid4()}"
    logger.info(f"Using test session_id: {session_id}")
    
    # Create service
    service = AsyncStatePersistenceService()
    
    # Create a test state
    test_state = create_test_state(session_id)
    
    # Save state
    logger.info("Saving state...")
    save_result = await service.save_state(session_id, test_state)
    if not save_result:
        logger.error("Failed to save state!")
        return False
    logger.info("State saved successfully.")
    
    # Load state
    logger.info("Loading state...")
    loaded_state = await service.load_state(session_id)
    if not loaded_state:
        logger.error("Failed to load state!")
        return False
    logger.info("State loaded successfully.")
    
    # Verify state content
    logger.info("Verifying state content...")
    if loaded_state.get("test_timestamp") == test_state.get("test_timestamp"):
        logger.info("State content verification successful.")
    else:
        logger.error("State content does not match!")
        return False
    
    # Delete state
    logger.info("Deleting state...")
    delete_result = await service.delete_state(session_id)
    if not delete_result:
        logger.error("Failed to delete state!")
        return False
    logger.info("State deleted successfully.")
    
    # Verify deletion
    logger.info("Verifying deletion...")
    should_be_none = await service.load_state(session_id)
    if should_be_none is None:
        logger.info("Deletion verification successful.")
    else:
        logger.error("State was not properly deleted!")
        return False
    
    logger.info("All async tests passed!")
    return True


def test_sync_persistence():
    """Test the sync persistence service."""
    logger.info("Testing StatePersistenceService...")
    
    # Generate a unique session ID for testing
    session_id = f"test-sync-{uuid.uuid4()}"
    logger.info(f"Using test session_id: {session_id}")
    
    # Create service
    service = StatePersistenceService()
    
    # Create a test state
    test_state = create_test_state(session_id)
    
    # Save state
    logger.info("Saving state...")
    save_result = service.save_state(session_id, test_state)
    if not save_result:
        logger.error("Failed to save state!")
        return False
    logger.info("State saved successfully.")
    
    # Load state
    logger.info("Loading state...")
    loaded_state = service.load_state(session_id)
    if not loaded_state:
        logger.error("Failed to load state!")
        return False
    logger.info("State loaded successfully.")
    
    # Verify state content
    logger.info("Verifying state content...")
    if loaded_state.get("test_timestamp") == test_state.get("test_timestamp"):
        logger.info("State content verification successful.")
    else:
        logger.error("State content does not match!")
        return False
    
    # Delete state
    logger.info("Deleting state...")
    delete_result = service.delete_state(session_id)
    if not delete_result:
        logger.error("Failed to delete state!")
        return False
    logger.info("State deleted successfully.")
    
    # Verify deletion
    logger.info("Verifying deletion...")
    should_be_none = service.load_state(session_id)
    if should_be_none is None:
        logger.info("Deletion verification successful.")
    else:
        logger.error("State was not properly deleted!")
        return False
    
    logger.info("All sync tests passed!")
    return True


async def main():
    """Run all tests."""
    logger.info("Initializing database...")
    try:
        init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return
    
    # Run tests
    async_result = await test_async_persistence()
    sync_result = test_sync_persistence()
    
    if async_result and sync_result:
        logger.info("All tests passed! The PostgreSQL state persistence is working correctly.")
    else:
        logger.error("Some tests failed. Please check the logs for details.")


if __name__ == "__main__":
    asyncio.run(main())