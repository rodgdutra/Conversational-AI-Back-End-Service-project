"""
LangGraph state definition for the conversational appointment assistant.
"""

from typing import Any, Dict, List, Optional
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
    review_result : dict | None
        The verdict produced by the ReviewerAgent on the most recent turn.
        Schema::

            {
              "verdict":              "pass" | "flag" | "block",
              "flags":                list[str],        # Empty when verdict is "pass"
              "action":               "none" | "warn" | "replace",
              "replacement_message":  str | None,       # Set when action == "replace"
              "tool_results": {                         # Raw tool outputs
                  "scope":            dict,
                  "sensitive_data":   dict,
                  "hallucination":    dict,
                  "stalling":         dict,
                  "gibberish":        dict,
              },
              "summary":              str,
            }
    """

    verified: bool = False
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    pending_action: Optional[str] = None
    review_result: Optional[Dict[str, Any]] = None
