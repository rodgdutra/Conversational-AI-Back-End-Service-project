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
import re
import uuid
import os
from typing import Literal, Optional, Tuple
from app.config import config

# ---------------------------------------------------------------------------
# OpenRouter configuration
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = config.OPENROUTER_API_KEY
OPENROUTER_BASE_URL = config.OPENROUTER_BASE_URL
OPENROUTER_MODEL = config.OPENROUTER_MODEL

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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
   - Once you have all three pieces of information, call `verify_patient_tool` immediately.
   - If verification fails, apologise and ask the patient to re-check their details.
   - When the patient provides new identity information after a failed attempt, you MUST \
call verify_patient_tool again using ONLY the information from the patient's LATEST message. \
Never re-use identity details from a previous failed verification attempt.
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

IMPORTANT RULES FOR TOOL USAGE:
- You have access to: verify_patient_tool, list_appointments_tool, \
confirm_appointment_tool, cancel_appointment_tool.
- NEVER write or describe tool results in your response text. \
Tool results come ONLY from actual tool calls, not from your narrative.
- NEVER write phrases like "[VERIFICATION RESULT]", "verification succeeded", \
"you are now verified", or similar in your text. Verification is only valid when \
verify_patient_tool has been called and its ToolMessage result confirms it.
- NEVER pre-empt, predict, or simulate tool outputs in your text response.
- If you have collected the patient's name, phone, and date of birth, CALL \
verify_patient_tool immediately — do not describe calling it, actually call it.

Never reveal internal patient IDs in your responses; use the patient's name instead.
"""

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def _build_llm() -> ChatOpenRouter:
    """
    Build the LLM instance pointed at OpenRouter, then bind all tools.
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
# Hallucination detection helpers
# ---------------------------------------------------------------------------

# Phrases that indicate the LLM is *claiming* verification succeeded or is
# actively calling the tool without actually emitting a tool_call.
_HALLUCINATION_POSITIVE = (
    "you are now verified",
    "your identity has been verified",
    "identity has been verified",
    "identity verified",
    "verified successfully",
    "verification succeeded",
    "verification successful",
    "verification result",
    "result is: verified",
    "you've been verified",
    "you have been verified",
    "congratulations",
    "i'll go ahead and verify",
    "let me verify your identity",
    "going to verify your identity",
    "i will verify your identity",
)

# Phrases that indicate the LLM is correctly reporting a *failed* attempt —
# these must NOT trigger the interceptor, otherwise a legitimate "Sorry,
# verification failed — please try again" response would cause an infinite loop.
_HALLUCINATION_NEGATIVE = (
    "could not verify",
    "couldn't verify",
    "unable to verify",
    "failed to verify",
    "verification failed",
    "verify failed",
    "not been verified",
    "not verified",
    "verification was unsuccessful",
    "was unable to verify",
    "sorry",
    "apologize",
    "apologies",
    "please check",
    "please try again",
    "double-check",
    "try again",
)


def _extract_identity_from_text(text: str) -> Optional[Tuple[str, str, str]]:
    """
    Try to extract (full_name, phone, date_of_birth) from a single text string.
    Returns None if any field is missing.
    """
    # Date of birth — strict YYYY-MM-DD
    dob_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if not dob_match:
        return None
    date_of_birth = dob_match.group(1)

    # Phone number — labelled then bare XXX-XXXX
    phone_match = re.search(
        r"(?:phone|phone\s+number|number|tel(?:ephone)?)\s*[:\s]+\s*(\d{3}[-\s]?\d{4})",
        text,
        re.IGNORECASE,
    )
    if not phone_match:
        phone_match = re.search(r"\b(\d{3}[-\s]\d{4})\b", text)
    if not phone_match:
        return None
    phone_raw = phone_match.group(1).strip()
    phone = re.sub(r"\s", "-", phone_raw)

    # Full name — common introductory phrases
    name_match = re.search(
        r"(?:my name is|name is|i['\u2019]?m|i am|name\s*[:=])\s+"
        r"([A-Za-z][A-Za-z ]{2,40}?)(?:\s*,|\s*\.|phone|$)",
        text,
        re.IGNORECASE,
    )
    if not name_match:
        return None
    full_name = name_match.group(1).strip()

    return full_name, phone, date_of_birth


def _extract_identity_from_messages(
    messages: list,
) -> Optional[Tuple[str, str, str]]:
    """
    Attempt to extract (full_name, phone, date_of_birth) from recent user
    messages in the conversation history.

    Strategy (most-recent-first to avoid stale data from failed attempts):
    1. Try the single most recent HumanMessage.
    2. If any field is missing, widen to the last 2 messages combined.
    3. If still incomplete, widen to the last 3 messages combined.

    Returns None if all three fields cannot be resolved.
    """
    user_texts = [m.content for m in messages if isinstance(m, HumanMessage)]
    if not user_texts:
        return None

    # Try progressively wider windows, starting with only the latest message
    for n in range(1, min(4, len(user_texts) + 1)):
        combined = " ".join(user_texts[-n:])
        result = _extract_identity_from_text(combined)
        if result is not None:
            if n > 1:
                logger.debug(
                    "_extract_identity_from_messages | found identity using last %d messages",
                    n,
                )
            return result

    return None


def _intercept_hallucination(
    state: AgentState, response: AIMessage
) -> AIMessage:
    """
    Detect whether the LLM hallucinated a verification outcome and, if so,
    replace the response with a real verify_patient_tool call constructed
    from information already present in the conversation.

    Detection logic:
      1. Skip if patient is already verified or LLM actually made a tool call.
      2. Skip if the response contains *negative* indicators — the LLM is
         correctly reporting a failed attempt ("sorry, couldn't verify…").
         Intercepting here would cause an infinite retry loop.
      3. Intercept only when the response contains *positive* indicators —
         the LLM is claiming success or claiming to be calling the tool right
         now, without an actual tool_call being emitted.
    """
    if state.get("verified", False):
        return response  # already verified — nothing to do

    if response.tool_calls:
        return response  # tool was properly requested — nothing to do

    content_lower = (response.content or "").lower()

    # -----------------------------------------------------------------------
    # Guard: if the LLM is legitimately reporting a failure, do NOT intercept.
    # Without this check, "I'm sorry, I couldn't verify…" would trigger an
    # infinite loop because it re-runs the same (wrong) credentials forever.
    # -----------------------------------------------------------------------
    if any(neg in content_lower for neg in _HALLUCINATION_NEGATIVE):
        return response

    # -----------------------------------------------------------------------
    # Only intercept on unmistakable positive hallucination signals.
    # -----------------------------------------------------------------------
    is_hallucination = any(pos in content_lower for pos in _HALLUCINATION_POSITIVE)
    if not is_hallucination:
        return response

    identity = _extract_identity_from_messages(state["messages"])
    if identity is None:
        logger.warning(
            "Node: assistant | Hallucination detected but could not extract identity info"
        )
        return response

    full_name, phone, date_of_birth = identity
    tool_call_id = f"call_{uuid.uuid4().hex[:20]}"

    synthetic = AIMessage(
        content="Let me verify your identity now.",
        tool_calls=[
            {
                "name": "verify_patient_tool",
                "args": {
                    "full_name": full_name,
                    "phone": phone,
                    "date_of_birth": date_of_birth,
                },
                "id": tool_call_id,
                "type": "tool_call",
            }
        ],
    )

    logger.warning(
        "Node: assistant | Hallucination intercepted — synthetic verify_patient_tool "
        "call injected | name='%s' phone='%s' dob='%s'",
        full_name,
        phone,
        date_of_birth,
    )
    return synthetic


# ---------------------------------------------------------------------------
# Node: assistant
# ---------------------------------------------------------------------------

def assistant_node(state: AgentState) -> dict:
    """
    Call the LLM.  Prepend the system prompt and inject a dynamic verification-
    status message on every turn so the LLM has an unambiguous view of whether
    the patient has been verified.

    After the LLM responds, a hallucination interceptor checks whether the
    model described a verification result in text without actually calling
    verify_patient_tool.  When detected, the hallucinated response is replaced
    with a real tool call constructed from identity info in the conversation.
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

    # ------------------------------------------------------------------
    # Dynamic verification-status message
    # ------------------------------------------------------------------
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
            "cancel_appointment_tool until verify_patient_tool returns verified=True. "
            "IMPORTANT: If a previous verification attempt failed and the patient has now "
            "provided new identity information, call verify_patient_tool using ONLY the "
            "details from the patient's MOST RECENT message — never re-use information "
            "from an earlier failed attempt. "
            "CRITICAL: Do NOT write verification results in your text. "
            "Do NOT write '[VERIFICATION RESULT]' or similar phrases. "
            "Call verify_patient_tool and let the system return the actual result."
        )

    # Replace a previously-injected status message (second position) to
    # avoid stale status blocks accumulating in the context.
    if len(messages) >= 2 and isinstance(messages[1], SystemMessage):
        messages = [messages[0], SystemMessage(content=status_content)] + messages[2:]
    else:
        messages = [messages[0], SystemMessage(content=status_content)] + messages[1:]

    response: AIMessage = llm.invoke(messages)

    # ------------------------------------------------------------------
    # Hallucination guard
    # ------------------------------------------------------------------
    response = _intercept_hallucination(state, response)

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

_TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def guarded_tools_node(state: AgentState) -> dict:
    """
    Execute tool calls requested by the LLM, enforcing access control.

    Rules
    -----
    • verify_patient_tool is always allowed.
    • Appointment tools are only allowed when state['verified'] is True.
      Blocked calls receive a descriptive error ToolMessage so the LLM can
      correct itself on the next turn.
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
            logger.warning(
                "Node: guarded_tools | BLOCKED '%s' — patient not verified", tool_name
            )
            result = {
                "error": "access_denied",
                "message": (
                    f"Cannot execute '{tool_name}': the patient has not been verified yet. "
                    "Collect full name, phone, and date of birth, then call "
                    "verify_patient_tool first."
                ),
            }
        else:
            try:
                tool = _TOOL_MAP[tool_name]
                raw = tool.invoke(tool_args)
                result = {"message": raw} if isinstance(raw, str) else raw
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
    promote them to top-level state fields.
    """
    updates: dict = {}

    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            break

        try:
            result = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue

        if "verified" in result and result["verified"] and not state.get("verified"):
            updates["verified"] = True
            updates["patient_id"] = result.get("patient_id")
            updates["patient_name"] = result.get("patient_name")
            logger.info(
                "Node: update_state | Patient verified | patient_id='%s' name='%s'",
                updates["patient_id"], updates["patient_name"],
            )

    if not updates:
        logger.debug("Node: update_state | No state updates required")
    return updates if updates else {}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def should_use_tools(state: AgentState) -> Literal["tools", "__end__"]:
    """Route to the tool node if the last AI message contains tool calls."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.debug("Router: should_use_tools → 'tools'")
        return "tools"
    logger.debug("Router: should_use_tools → '__end__'")
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

    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", guarded_tools_node)
    builder.add_node("update_state", update_state_node)

    builder.set_entry_point("assistant")

    builder.add_conditional_edges(
        "assistant",
        should_use_tools,
        {"tools": "tools", "__end__": END},
    )

    builder.add_edge("tools", "update_state")
    builder.add_edge("update_state", "assistant")

    graph = builder.compile()
    logger.info("LangGraph compiled successfully | nodes=%s", list(graph.nodes.keys()))
    return graph


# Singleton compiled graph
compiled_graph = build_graph()
