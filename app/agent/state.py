"""
LangGraph state definition for the conversational appointment assistant.
"""

from typing import Annotated, Optional
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """
    Shared state that flows through every node in the graph.

    Fields
    ------
    verified : bool
        Whether the patient has been successfully identified.
    patient_id : str | None
        The internal patient ID once verified.
    patient_name : str | None
        The patient's full name once verified (used for personalised replies).
    pending_action : str | None
        When the user tries to do something before being verified, this stores
        what they wanted so the graph can redirect them after verification.
    """

    verified: bool = False
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    pending_action: Optional[str] = None
