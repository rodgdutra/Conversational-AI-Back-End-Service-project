"""
LangGraph agent graph for the conversational appointment assistant.

Graph topology
--------------
                         ┌──────────────┐
           START ──────► │   assistant  │ ◄─────────────────┐
                         └──────┬───────┘                    │
                                │                            │
                    has tool calls?                          │
                    ┌─────┴──────┐                          │
                   YES            NO                        │
                    │              │                        │
                    ▼              ▼                        │
              ┌──────────┐      END                        │
              │  tools   │                                  │
              └────┬─────┘                                  │
                   │                                        │
                   ▼                                        │
          ┌─────────────────┐                              │
          │  update_state   │──────────────────────────────┘
          └─────────────────┘

The `assistant` node calls the LLM (with all tools bound).  The system prompt
instructs the model to:
  1. Always collect full_name, phone, and date_of_birth before anything else.
  2. Call verify_patient_tool once all three are collected.
  3. Only call appointment tools after verification.
  4. Allow free re-routing between actions.

The `tools` node executes tool calls.

The `update_state` node inspects tool results and promotes verification data
(patient_id, patient_name, verified) into the graph state so subsequent turns
have access to it.
"""

import json
import os
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.agent.state import AgentState
from app.agent.tools import ALL_TOOLS

# ---------------------------------------------------------------------------
# OpenRouter configuration
# Reads OPENROUTER_API_KEY and OPENROUTER_MODEL from the environment.
# Falls back to sensible defaults so the service fails fast with a clear error
# rather than a cryptic downstream exception.
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful and empathetic healthcare scheduling assistant \
for a medical clinic. Your primary responsibilities are:

1. **Identity Verification (mandatory first step)**
   - Before any appointment-related action, you MUST verify the patient's identity.
   - Politely greet the patient and ask for their:
       • Full name
       • Phone number
       • Date of birth (in YYYY-MM-DD format, e.g. 1990-07-22)
   - You may collect these details across multiple turns if needed.
   - Once you have all three, call `verify_patient_tool` immediately.
   - If verification fails, apologise and ask the patient to check their details.
   - Do NOT proceed to appointment actions until verification succeeds.

2. **Appointment Actions (available only after successful verification)**
   - **List appointments**: call `list_appointments_tool` with the patient_id \
from the verification result.
   - **Confirm an appointment**: ask the patient which appointment they want to \
confirm (use the appointment ID shown in the list, e.g. A001), then call \
`confirm_appointment_tool`.
   - **Cancel an appointment**: ask the patient which appointment they want to \
cancel, then call `cancel_appointment_tool`.

3. **Free navigation**
   - After completing one action, always ask what else you can help with.
   - Allow the patient to switch freely between listing, confirming, and \
cancelling — guide them naturally.

4. **Tone & style**
   - Be warm, professional, and concise.
   - Always confirm back the result of every action clearly.
   - When listing appointments, format them in an easy-to-read way.

IMPORTANT: You have access to the following tools:
- verify_patient_tool
- list_appointments_tool
- confirm_appointment_tool
- cancel_appointment_tool

Never reveal internal patient IDs in your responses; use the patient's name instead.
"""

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def _build_llm() -> ChatOpenAI:
    """
    Build the LLM instance pointed at OpenRouter, then bind all tools.

    Configuration is driven entirely by environment variables:
      - OPENROUTER_API_KEY   : Your OpenRouter API key (required)
      - OPENROUTER_BASE_URL  : API base URL (default: https://openrouter.ai/api/v1)
      - OPENROUTER_MODEL     : Model name recognised by OpenRouter
                               (default: openai/gpt-4o-mini)
    """
    llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        temperature=0,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        default_headers={
            # Recommended by OpenRouter for analytics / prioritisation
            "HTTP-Referer": "https://github.com/rodgdutra/Conversational-AI-Back-End-Service-project",
            "X-Title": "Conversational AI Appointment Assistant",
        },
    )
    return llm.bind_tools(ALL_TOOLS)


# ---------------------------------------------------------------------------
# Node: assistant
# ---------------------------------------------------------------------------

def assistant_node(state: AgentState) -> dict:
    """
    Call the LLM.  Prepend the system prompt if it is not already the first
    message in the conversation history.
    """
    llm = _build_llm()

    messages = list(state["messages"])

    # Ensure the system prompt is always the first message
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response: AIMessage = llm.invoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Node: update_state
# ---------------------------------------------------------------------------

def update_state_node(state: AgentState) -> dict:
    """
    After tool execution, scan ToolMessage results for verification data and
    promote them to top-level state fields so the assistant can use them in
    subsequent turns without re-parsing tool messages.
    """
    updates: dict = {}

    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            break  # only look at the most-recent batch of tool messages

        # Tool messages store their content as a JSON string (from ToolNode)
        try:
            result = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue

        # Pick up verification data from verify_patient_tool
        if "verified" in result and result["verified"] and not state.get("verified"):
            updates["verified"] = True
            updates["patient_id"] = result.get("patient_id")
            updates["patient_name"] = result.get("patient_name")

    return updates if updates else {}


# ---------------------------------------------------------------------------
# Router: decide whether to call tools or finish
# ---------------------------------------------------------------------------

def should_use_tools(state: AgentState) -> Literal["tools", "__end__"]:
    """Route to the tool node if the last AI message contains tool calls."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph."""
    tool_node = ToolNode(ALL_TOOLS)

    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", tool_node)
    builder.add_node("update_state", update_state_node)

    # Entry point
    builder.set_entry_point("assistant")

    # After assistant: either call tools or end the turn
    builder.add_conditional_edges(
        "assistant",
        should_use_tools,
        {"tools": "tools", "__end__": END},
    )

    # After tools: update state, then return to assistant for follow-up
    builder.add_edge("tools", "update_state")
    builder.add_edge("update_state", "assistant")

    return builder.compile()


# Singleton compiled graph — imported by the FastAPI layer
compiled_graph = build_graph()
