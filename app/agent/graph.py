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
              │  tools   │  (access-controlled)            │
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

The `tools` node is guarded: if the patient has not been verified yet and the
LLM attempts to call an appointment tool, the call is intercepted and a clear
error ToolMessage is returned instead.  This makes the access-control rule
deterministic and independent of LLM behaviour.

The `update_state` node inspects tool results and promotes verification data
(patient_id, patient_name, verified) into the graph state so subsequent turns
have access to it.
"""

import json
import os
from typing import Literal
from app.config import config

# ---------------------------------------------------------------------------
# OpenRouter configuration
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = config.OPENROUTER_API_KEY
OPENROUTER_BASE_URL = config.OPENROUTER_BASE_URL
OPENROUTER_MODEL = config.OPENROUTER_MODEL

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.tools import ALL_TOOLS
from app.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Access-control constants
# ---------------------------------------------------------------------------

# Tools that require a verified patient — any call to these while
# verified=False will be intercepted at the graph level.
_APPOINTMENT_TOOLS = {
    "list_appointments_tool",
    "confirm_appointment_tool",
    "cancel_appointment_tool",
}

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
   - If verification fails, apologise and ask the patient to re-check their details.
   - Do NOT call list_appointments_tool, confirm_appointment_tool, or \
cancel_appointment_tool until verification has succeeded.
   - IMPORTANT: Even if the patient mentions a specific appointment (e.g. "cancel A001") \
before their identity is verified, you must first collect their identity information and \
call verify_patient_tool. Only after successful verification may you proceed with the \
requested appointment action.

2. **Appointment Actions (available ONLY after verify_patient_tool returns verified=True)**
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

def _build_llm() -> ChatOpenRouter:
    """
    Build the LLM instance pointed at OpenRouter, then bind all tools.

    Configuration is driven entirely by environment variables:
      - OPENROUTER_API_KEY   : Your OpenRouter API key (required)
      - OPENROUTER_BASE_URL  : API base URL (default: https://openrouter.ai/api/v1)
      - OPENROUTER_MODEL     : Model name recognised by OpenRouter
                               (default: openai/gpt-4o-mini)
    """
    if config.USE_OLLAMA:
        llm = ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
        )
        logger.info("LLM initialized using Ollama.")
        return llm.bind_tools(ALL_TOOLS)

    llm = ChatOpenRouter(
        model=OPENROUTER_MODEL,
        temperature=0,
        max_tokens=1024,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    logger.info("LLM initialized using OpenRouter.")

    return llm.bind_tools(ALL_TOOLS)


# ---------------------------------------------------------------------------
# Node: assistant
# ---------------------------------------------------------------------------

def assistant_node(state: AgentState) -> dict:
    """
    Call the LLM.  Prepend the system prompt if it is not already the first
    message in the conversation history.

    A short verification-status message is injected immediately after the
    system prompt on every turn so the LLM has an unambiguous, up-to-date
    view of whether the patient has been verified.
    """
    llm = _build_llm()

    messages = list(state["messages"])
    verified = state.get("verified", False)
    patient_id = state.get("patient_id")
    patient_name = state.get("patient_name") or "unverified"

    logger.debug(
        "Node: assistant | verified=%s patient='%s' history_len=%d model='%s'",
        verified, patient_name, len(messages), config.model_name,
    )

    # Ensure the system prompt is always the first message
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    # Inject a dynamic verification-status reminder right after the system
    # prompt.  This lets the LLM know *with certainty* whether it should be
    # collecting identity info or executing appointment actions.
    if verified:
        status_content = (
            f"[VERIFICATION STATUS] ✅ Patient VERIFIED. "
            f"patient_id='{patient_id}' | name='{patient_name}'. "
            "You may now use list_appointments_tool, confirm_appointment_tool, "
            "and cancel_appointment_tool."
        )
    else:
        status_content = (
            "[VERIFICATION STATUS] ❌ Patient NOT verified. "
            "You MUST collect the patient's full name, phone number, and date of birth, "
            "then call verify_patient_tool. "
            "Do NOT call list_appointments_tool, confirm_appointment_tool, or "
            "cancel_appointment_tool until verify_patient_tool returns verified=True."
        )

    # Replace a previously-injected status message (always the second message
    # when present) to avoid growing the context with stale status blocks.
    if len(messages) >= 2 and isinstance(messages[1], SystemMessage):
        messages = [messages[0], SystemMessage(content=status_content)] + messages[2:]
    else:
        messages = [messages[0], SystemMessage(content=status_content)] + messages[1:]

    response: AIMessage = llm.invoke(messages)

    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    if tool_calls:
        tool_names = [tc["name"] for tc in tool_calls]
        logger.info("Node: assistant | LLM requesting tools: %s", tool_names)
    else:
        preview = (response.content or "")[:120].replace("\n", " ")
        logger.info("Node: assistant | LLM reply (preview): '%s...'", preview)

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Node: guarded_tools
# ---------------------------------------------------------------------------

# Build a fast name→callable lookup for our tools
_TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def guarded_tools_node(state: AgentState) -> dict:
    """
    Execute tool calls requested by the LLM, enforcing access control.

    Rules
    -----
    • `verify_patient_tool` is always allowed.
    • `list_appointments_tool`, `confirm_appointment_tool`, and
      `cancel_appointment_tool` are only allowed when `state['verified']` is
      True.  If the LLM tries to call one of these while the patient is still
      unverified, the call is intercepted and a descriptive error ToolMessage
      is returned so the LLM can correct itself on the next turn.
    """
    verified = state.get("verified", False)
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    new_messages: list = []

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc.get("args", {})
        tool_call_id = tc["id"]

        if not verified and tool_name in _APPOINTMENT_TOOLS:
            # ----------------------------------------------------------------
            # ACCESS DENIED — return a synthetic error ToolMessage
            # ----------------------------------------------------------------
            logger.warning(
                "Node: guarded_tools | BLOCKED '%s' — patient not verified", tool_name
            )
            result = {
                "error": "access_denied",
                "message": (
                    f"Cannot execute '{tool_name}': the patient has not been verified yet. "
                    "You must first collect the patient's full name, phone number, and date "
                    "of birth, then call verify_patient_tool. Proceed with verification before "
                    "attempting any appointment action."
                ),
            }
        else:
            # ----------------------------------------------------------------
            # ALLOWED — invoke the real tool
            # ----------------------------------------------------------------
            try:
                tool = _TOOL_MAP[tool_name]
                raw = tool.invoke(tool_args)
                # LangChain tools can return str or dict; normalise to dict
                if isinstance(raw, str):
                    result = {"message": raw}
                else:
                    result = raw
                logger.debug(
                    "Node: guarded_tools | executed '%s' successfully", tool_name
                )
            except Exception as exc:
                logger.error(
                    "Node: guarded_tools | error executing '%s': %s", tool_name, exc
                )
                result = {"error": str(exc)}

        new_messages.append(
            ToolMessage(
                content=json.dumps(result),
                tool_call_id=tool_call_id,
            )
        )

    return {"messages": new_messages}


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

        try:
            result = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue

        # Pick up verification data from verify_patient_tool
        if "verified" in result and result["verified"] and not state.get("verified"):
            updates["verified"] = True
            updates["patient_id"] = result.get("patient_id")
            updates["patient_name"] = result.get("patient_name")
            logger.info(
                "Node: update_state | Patient verified and promoted to state | "
                "patient_id='%s' name='%s'",
                updates["patient_id"], updates["patient_name"],
            )

    if not updates:
        logger.debug("Node: update_state | No state updates required")
    return updates if updates else {}


# ---------------------------------------------------------------------------
# Router: decide whether to call tools or finish
# ---------------------------------------------------------------------------

def should_use_tools(state: AgentState) -> Literal["tools", "__end__"]:
    """Route to the tool node if the last AI message contains tool calls."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.debug("Router: should_use_tools → 'tools'")
        return "tools"
    logger.debug("Router: should_use_tools → '__end__' (replying to user)")
    return "__end__"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph."""

    if config.USE_OLLAMA:
        logger.info(
            "Building LangGraph | model='%s' base_url='%s'",
            config.model_name, config.OLLAMA_BASE_URL,
        )
    else:
        logger.info(
            "Building LangGraph | model='%s' base_url='%s'",
            OPENROUTER_MODEL, OPENROUTER_BASE_URL,
        )

    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", guarded_tools_node)   # access-controlled
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

    graph = builder.compile()
    logger.info("LangGraph compiled successfully | nodes=%s", list(graph.nodes.keys()))
    return graph


# Singleton compiled graph — imported by the FastAPI layer
compiled_graph = build_graph()
