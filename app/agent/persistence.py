"""
LangGraph state persistence service for PostgreSQL.

This module provides methods to save and load agent state from a Postgres database,
allowing conversational state to survive application restarts.
"""

import json
import logging
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.db import GraphState, get_db, get_async_db
from app.logger import get_logger

logger = get_logger(__name__)


class StatePersistenceService:
    """
    Service for persisting LangGraph state to a PostgreSQL database.
    
    Provides methods for saving and loading graph states associated with session IDs.
    """
    
    @staticmethod
    def save_state(session_id: str, state: Dict[str, Any]) -> bool:
        """
        Save a graph state to the database using session_id as the key.
        
        Args:
            session_id: Unique identifier for the conversation session
            state: The LangGraph state dictionary to persist
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            db = get_db()
            # Convert any non-serializable objects to strings
            serializable_state = _prepare_state_for_serialization(state)
            
            # Check if state already exists
            graph_state_db = db.query(GraphState).filter(
                GraphState.session_id == session_id
            ).first()
            
            if graph_state_db:
                # Update existing state
                graph_state_db.state_data = json.dumps(serializable_state)
            else:
                # Create new state
                graph_state_db = GraphState.from_dict(session_id, serializable_state)
                db.add(graph_state_db)
                
            db.commit()
            logger.debug(f"State saved | session_id='{session_id}'")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to save state | session_id='{session_id}' error='{str(e)}'")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving state | session_id='{session_id}' error='{str(e)}'")
            return False
    
    @staticmethod
    def load_state(session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a graph state from the database by session_id.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Returns:
            Dict or None: The state dictionary if found, None otherwise
        """
        try:
            db = get_db()
            graph_state_db = db.query(GraphState).filter(
                GraphState.session_id == session_id
            ).first()
            
            if not graph_state_db:
                logger.debug(f"No state found | session_id='{session_id}'")
                return None
                
            state = graph_state_db.to_dict()
            logger.debug(f"State loaded | session_id='{session_id}'")
            return state
        except SQLAlchemyError as e:
            logger.error(f"Failed to load state | session_id='{session_id}' error='{str(e)}'")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading state | session_id='{session_id}' error='{str(e)}'")
            return None
    
    @staticmethod
    def delete_state(session_id: str) -> bool:
        """
        Delete a graph state from the database by session_id.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Returns:
            bool: True if successfully deleted or if state didn't exist, False on error
        """
        try:
            db = get_db()
            graph_state_db = db.query(GraphState).filter(
                GraphState.session_id == session_id
            ).first()
            
            if graph_state_db:
                db.delete(graph_state_db)
                db.commit()
                logger.info(f"State deleted | session_id='{session_id}'")
            else:
                logger.warning(f"Delete requested for non-existent state | session_id='{session_id}'")
                
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete state | session_id='{session_id}' error='{str(e)}'")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting state | session_id='{session_id}' error='{str(e)}'")
            return False


class AsyncStatePersistenceService:
    """
    Async version of the StatePersistenceService for use with FastAPI.
    
    Provides async methods for saving and loading graph states associated with session IDs.
    """
    
    @staticmethod
    async def save_state(session_id: str, state: Dict[str, Any]) -> bool:
        """
        Asynchronously save a graph state to the database.
        
        Args:
            session_id: Unique identifier for the conversation session
            state: The LangGraph state dictionary to persist
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Convert any non-serializable objects to strings
            serializable_state = _prepare_state_for_serialization(state)
            
            async with get_async_db() as db:
                # Check if state already exists
                result = await db.execute(
                    select(GraphState).where(GraphState.session_id == session_id)
                )
                graph_state_db = result.scalars().first()
                
                if graph_state_db:
                    # Update existing state
                    graph_state_db.state_data = json.dumps(serializable_state)
                else:
                    # Create new state
                    graph_state_db = GraphState.from_dict(session_id, serializable_state)
                    db.add(graph_state_db)
                    
                await db.commit()
                logger.debug(f"State saved async | session_id='{session_id}'")
                return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to save state async | session_id='{session_id}' error='{str(e)}'")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving state async | session_id='{session_id}' error='{str(e)}'")
            return False
    
    @staticmethod
    async def load_state(session_id: str) -> Optional[Dict[str, Any]]:
        """
        Asynchronously load a graph state from the database by session_id.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Returns:
            Dict or None: The state dictionary if found, None otherwise
        """
        try:
            async with get_async_db() as db:
                result = await db.execute(
                    select(GraphState).where(GraphState.session_id == session_id)
                )
                graph_state_db = result.scalars().first()
                
                if not graph_state_db:
                    logger.debug(f"No state found async | session_id='{session_id}'")
                    return None
                    
                state = graph_state_db.to_dict()
                logger.debug(f"State loaded async | session_id='{session_id}'")
                return state
        except SQLAlchemyError as e:
            logger.error(f"Failed to load state async | session_id='{session_id}' error='{str(e)}'")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading state async | session_id='{session_id}' error='{str(e)}'")
            return None
    
    @staticmethod
    async def delete_state(session_id: str) -> bool:
        """
        Asynchronously delete a graph state from the database by session_id.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Returns:
            bool: True if successfully deleted or if state didn't exist, False on error
        """
        try:
            async with get_async_db() as db:
                result = await db.execute(
                    select(GraphState).where(GraphState.session_id == session_id)
                )
                graph_state_db = result.scalars().first()
                
                if graph_state_db:
                    await db.delete(graph_state_db)
                    await db.commit()
                    logger.info(f"State deleted async | session_id='{session_id}'")
                else:
                    logger.warning(f"Delete requested for non-existent state async | session_id='{session_id}'")
                    
                return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete state async | session_id='{session_id}' error='{str(e)}'")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting state async | session_id='{session_id}' error='{str(e)}'")
            return False


def _prepare_state_for_serialization(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a LangGraph state for serialization, handling any non-serializable objects.
    
    Args:
        state: The original state dictionary
        
    Returns:
        Dict: A serializable version of the state
    """
    # Create a shallow copy of the state to avoid modifying the original
    serializable_state = dict(state)
    
    # Handle conversion of messages
    if "messages" in serializable_state:
        serializable_state["messages"] = [
            message.dict() if hasattr(message, "dict") else message
            for message in serializable_state["messages"]
        ]
        
    return serializable_state