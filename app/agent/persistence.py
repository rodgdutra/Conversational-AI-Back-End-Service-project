"""
LangGraph state persistence service for PostgreSQL.

This module provides methods to save and load agent state from a Postgres database,
allowing conversational state to survive application restarts.
"""

import json
import logging
from typing import Any, Dict, Optional, List, Tuple, Union
from datetime import datetime

from sqlalchemy import select, desc, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.db import GraphState, get_db, get_async_db, Session as DBSession, StateTransition
from app.logger import get_logger

logger = get_logger(__name__)


class StatePersistenceService:
    """
    Service for persisting LangGraph state to a PostgreSQL database.
    
    Provides methods for saving and loading graph states associated with session IDs.
    Supports multiple states per session and state transitions.
    """
    
    @staticmethod
    def save_state(session_id: str, state: Dict[str, Any], 
                  transition_type: str = None, 
                  transition_data: Dict[str, Any] = None) -> bool:
        """
        Save a graph state to the database using session_id as the key.
        
        Args:
            session_id: Unique identifier for the conversation session
            state: The LangGraph state dictionary to persist
            transition_type: Optional type of transition (e.g., "user_message", "tool_execution")
            transition_data: Optional data about the transition
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            db = get_db()
            # Convert any non-serializable objects to strings
            serializable_state = _prepare_state_for_serialization(state)
            
            # Get or create session
            db_session = db.query(DBSession).filter(
                DBSession.session_id == session_id
            ).first()
            
            if not db_session:
                db_session = DBSession(session_id=session_id)
                db.add(db_session)
                db.flush()
                
            # Get the latest state_id for this session
            latest_state = db.query(GraphState).filter(
                GraphState.session_id == session_id
            ).order_by(desc(GraphState.state_id)).first()
            
            # Determine new state_id
            current_state_id = 0
            if latest_state:
                current_state_id = latest_state.state_id
            
            # New state will be the next one
            new_state_id = current_state_id + 1
            
            # Create the new state
            graph_state_db = GraphState.from_dict(
                session_id=session_id, 
                state_id=new_state_id, 
                state_data=serializable_state,
            )
            db.add(graph_state_db)
            
            # Record state transition if applicable
            if current_state_id > 0:
                transition = StateTransition.from_dict(
                    session_id=session_id,
                    from_state_id=current_state_id,
                    to_state_id=new_state_id,
                    transition_type=transition_type,
                    transition_data=transition_data,
                )
                db.add(transition)
            
            # Update the session's updated_at timestamp
            db_session.updated_at = datetime.utcnow()
            
            # Commit changes
            db.commit()
            logger.debug(f"State saved | session_id='{session_id}' state_id={new_state_id}")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to save state | session_id='{session_id}' error='{str(e)}'")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving state | session_id='{session_id}' error='{str(e)}'")
            return False
    
    @staticmethod
    def load_state(session_id: str, state_id: int = None) -> Optional[Dict[str, Any]]:
        """
        Load a graph state from the database by session_id.
        
        Args:
            session_id: Unique identifier for the conversation session
            state_id: Optional specific state ID to load. If None, loads the latest state.
            
        Returns:
            Dict or None: The state dictionary if found, None otherwise
        """
        try:
            db = get_db()
            
            if state_id is not None:
                # Load specific state
                graph_state_db = db.query(GraphState).filter(
                    GraphState.session_id == session_id,
                    GraphState.state_id == state_id
                ).first()
            else:
                # Load latest state
                graph_state_db = db.query(GraphState).filter(
                    GraphState.session_id == session_id
                ).order_by(desc(GraphState.state_id)).first()
            
            if not graph_state_db:
                logger.debug(f"No state found | session_id='{session_id}' state_id={state_id or 'latest'}")
                return None
                
            state = graph_state_db.to_dict()
            logger.debug(f"State loaded | session_id='{session_id}' state_id={graph_state_db.state_id}")
            return state
        except SQLAlchemyError as e:
            logger.error(f"Failed to load state | session_id='{session_id}' error='{str(e)}'")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading state | session_id='{session_id}' error='{str(e)}'")
            return None
    
    @staticmethod
    def list_states(session_id: str) -> List[Dict[str, Any]]:
        """
        List all states for a session.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Returns:
            List of state metadata dictionaries
        """
        try:
            db = get_db()
            
            states = db.query(GraphState).filter(
                GraphState.session_id == session_id
            ).order_by(GraphState.state_id).all()
            
            result = []
            for state in states:
                result.append({
                    "session_id": state.session_id,
                    "state_id": state.state_id,
                    "created_at": state.created_at.isoformat()
                })
                
            logger.debug(f"Listed {len(result)} states | session_id='{session_id}'")
            return result
        except SQLAlchemyError as e:
            logger.error(f"Failed to list states | session_id='{session_id}' error='{str(e)}'")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing states | session_id='{session_id}' error='{str(e)}'")
            return []
    
    @staticmethod
    def get_state_transitions(session_id: str) -> List[Dict[str, Any]]:
        """
        Get all state transitions for a session.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Returns:
            List of transition dictionaries
        """
        try:
            db = get_db()
            
            transitions = db.query(StateTransition).filter(
                StateTransition.session_id == session_id
            ).order_by(StateTransition.id).all()
            
            result = [transition.to_dict() for transition in transitions]
            
            logger.debug(f"Retrieved {len(result)} transitions | session_id='{session_id}'")
            return result
        except SQLAlchemyError as e:
            logger.error(f"Failed to get transitions | session_id='{session_id}' error='{str(e)}'")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting transitions | session_id='{session_id}' error='{str(e)}'")
            return []
    
    @staticmethod
    def delete_session(session_id: str) -> bool:
        """
        Delete all states and transitions for a session.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Returns:
            bool: True if successfully deleted or if session didn't exist, False on error
        """
        try:
            db = get_db()
            
            # Find the session
            db_session = db.query(DBSession).filter(
                DBSession.session_id == session_id
            ).first()
            
            if db_session:
                # Session deletion will cascade to states and transitions due to relationship config
                db.delete(db_session)
                db.commit()
                logger.info(f"Session deleted | session_id='{session_id}'")
            else:
                logger.warning(f"Delete requested for non-existent session | session_id='{session_id}'")
                
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete session | session_id='{session_id}' error='{str(e)}'")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting session | session_id='{session_id}' error='{str(e)}'")
            return False


class AsyncStatePersistenceService:
    """
    Async version of the StatePersistenceService for use with FastAPI.
    
    Provides async methods for saving and loading graph states associated with session IDs.
    Supports multiple states per session and state transitions.
    """
    
    @staticmethod
    async def save_state(session_id: str, state: Dict[str, Any],
                        transition_type: str = None,
                        transition_data: Dict[str, Any] = None) -> bool:
        """
        Asynchronously save a graph state to the database.
        
        Args:
            session_id: Unique identifier for the conversation session
            state: The LangGraph state dictionary to persist
            transition_type: Optional type of transition (e.g., "user_message", "tool_execution")
            transition_data: Optional data about the transition
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Convert any non-serializable objects to strings
            serializable_state = _prepare_state_for_serialization(state)
            
            async with get_async_db() as db:
                # Get or create session
                result = await db.execute(
                    select(DBSession).where(DBSession.session_id == session_id)
                )
                db_session = result.scalars().first()
                
                if not db_session:
                    db_session = DBSession(session_id=session_id)
                    db.add(db_session)
                    await db.flush()
                
                # Get the latest state_id for this session
                result = await db.execute(
                    select(GraphState)
                    .where(GraphState.session_id == session_id)
                    .order_by(desc(GraphState.state_id))
                )
                latest_state = result.scalars().first()
                
                # Determine new state_id
                current_state_id = 0
                if latest_state:
                    current_state_id = latest_state.state_id
                
                # New state will be the next one
                new_state_id = current_state_id + 1
                
                # Create the new state
                graph_state_db = GraphState.from_dict(
                    session_id=session_id, 
                    state_id=new_state_id, 
                    state_data=serializable_state,
                )
                db.add(graph_state_db)
                
                # Record state transition if applicable
                if current_state_id > 0:
                    transition = StateTransition.from_dict(
                        session_id=session_id,
                        from_state_id=current_state_id,
                        to_state_id=new_state_id,
                        transition_type=transition_type,
                        transition_data=transition_data,
                    )
                    db.add(transition)
                
                # Update the session's updated_at timestamp
                db_session.updated_at = datetime.utcnow()
                
                # Commit changes
                await db.commit()
                logger.debug(f"State saved async | session_id='{session_id}' state_id={new_state_id}")
                return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to save state async | session_id='{session_id}' error='{str(e)}'")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving state async | session_id='{session_id}' error='{str(e)}'")
            return False
    
    @staticmethod
    async def load_state(session_id: str, state_id: int = None) -> Optional[Dict[str, Any]]:
        """
        Asynchronously load a graph state from the database by session_id.
        
        Args:
            session_id: Unique identifier for the conversation session
            state_id: Optional specific state ID to load. If None, loads the latest state.
            
        Returns:
            Dict or None: The state dictionary if found, None otherwise
        """
        try:
            async with get_async_db() as db:
                if state_id is not None:
                    # Load specific state
                    result = await db.execute(
                        select(GraphState)
                        .where(GraphState.session_id == session_id, GraphState.state_id == state_id)
                    )
                else:
                    # Load latest state
                    result = await db.execute(
                        select(GraphState)
                        .where(GraphState.session_id == session_id)
                        .order_by(desc(GraphState.state_id))
                    )
                
                graph_state_db = result.scalars().first()
                
                if not graph_state_db:
                    logger.debug(f"No state found async | session_id='{session_id}' state_id={state_id or 'latest'}")
                    return None
                    
                state = graph_state_db.to_dict()
                logger.debug(f"State loaded async | session_id='{session_id}' state_id={graph_state_db.state_id}")
                return state
        except SQLAlchemyError as e:
            logger.error(f"Failed to load state async | session_id='{session_id}' error='{str(e)}'")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading state async | session_id='{session_id}' error='{str(e)}'")
            return None
    
    @staticmethod
    async def list_states(session_id: str) -> List[Dict[str, Any]]:
        """
        Asynchronously list all states for a session.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Returns:
            List of state metadata dictionaries
        """
        try:
            async with get_async_db() as db:
                result = await db.execute(
                    select(GraphState)
                    .where(GraphState.session_id == session_id)
                    .order_by(GraphState.state_id)
                )
                states = result.scalars().all()
                
                result_list = []
                for state in states:
                    result_list.append({
                        "session_id": state.session_id,
                        "state_id": state.state_id,
                        "created_at": state.created_at.isoformat()
                    })
                    
                logger.debug(f"Listed {len(result_list)} states async | session_id='{session_id}'")
                return result_list
        except SQLAlchemyError as e:
            logger.error(f"Failed to list states async | session_id='{session_id}' error='{str(e)}'")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing states async | session_id='{session_id}' error='{str(e)}'")
            return []
    
    @staticmethod
    async def get_state_transitions(session_id: str) -> List[Dict[str, Any]]:
        """
        Asynchronously get all state transitions for a session.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Returns:
            List of transition dictionaries
        """
        try:
            async with get_async_db() as db:
                result = await db.execute(
                    select(StateTransition)
                    .where(StateTransition.session_id == session_id)
                    .order_by(StateTransition.id)
                )
                transitions = result.scalars().all()
                
                result_list = [transition.to_dict() for transition in transitions]
                
                logger.debug(f"Retrieved {len(result_list)} transitions async | session_id='{session_id}'")
                return result_list
        except SQLAlchemyError as e:
            logger.error(f"Failed to get transitions async | session_id='{session_id}' error='{str(e)}'")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting transitions async | session_id='{session_id}' error='{str(e)}'")
            return []
    
    @staticmethod
    async def delete_session(session_id: str) -> bool:
        """
        Asynchronously delete all states and transitions for a session.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Returns:
            bool: True if successfully deleted or if session didn't exist, False on error
        """
        try:
            async with get_async_db() as db:
                result = await db.execute(
                    select(DBSession).where(DBSession.session_id == session_id)
                )
                db_session = result.scalars().first()
                
                if db_session:
                    # Session deletion will cascade to states and transitions due to relationship config
                    await db.delete(db_session)
                    await db.commit()
                    logger.info(f"Session deleted async | session_id='{session_id}'")
                else:
                    logger.warning(f"Delete requested for non-existent session async | session_id='{session_id}'")
                    
                return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete session async | session_id='{session_id}' error='{str(e)}'")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting session async | session_id='{session_id}' error='{str(e)}'")
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