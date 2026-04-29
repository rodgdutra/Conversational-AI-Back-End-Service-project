# Conversational AI Appointment Assistant — Back-End Service

A FastAPI back-end service that exposes a conversational endpoint powered by a
**LangGraph** multi-agent workflow. Patients interact with the assistant through
a chat interface to manage their clinic appointments.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Setup](#setup)
6. [Running the Service](#running-the-service)
7. [API Reference](#api-reference)
8. [Conversation Flow](#conversation-flow)
9. [Mock Data](#mock-data)
10. [Configuration](#configuration)

---

## Overview

The service implements the following interaction flow:

| Step | Feature | Gate |
|------|---------|------|
| 1 | **User Verification** — patient identity confirmed via full name, phone, and date of birth | Always required first |
| 2 | **List Appointments** | After successful verification |
| 3 | **Confirm Appointment** | After successful verification |
| 4 | **Cancel Appointment** | After successful verification |
| 5 | **Free Navigation** — patient may move between actions naturally | After successful verification |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Layer                        │
│  POST /chat  ──►  session_store  ──►  LangGraph Graph       │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │         LangGraph Graph           │
          │                                   │
          │  START ──► assistant_node         │
          │                 │                 │
          │         has tool calls?           │
          │          ┌──────┴──────┐          │
          │         YES            NO         │
          │          │              │         │
          │          ▼              ▼         │
          │      tools_node        END        │
          │          │                        │
          │          ▼                        │
          │    update_state_node              │
          │          │                        │
          │          └──────► assistant_node  │
          └───────────────────────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │          Tool Layer              │
          │  verify_patient_tool             │
          │  list_appointments_tool          │
          │  confirm_appointment_tool        │
          │  cancel_appointment_tool         │
          └──────────────────────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │        Mock Data Layer           │
          │  app/data.py  (in-memory)        │
          └──────────────────────────────────┘
```

### Graph Nodes

| Node | Role |
|------|------|
| `assistant_node` | Calls the LLM (via OpenRouter) with the full conversation history + system prompt. Decides whether to call a tool or reply. |
| `tools_node` | Executes tool calls requested by the LLM (`ToolNode` from `langgraph.prebuilt`). |
| `update_state_node` | Inspects tool results and promotes verification data (`verified`, `patient_id`, `patient_name`) into the graph state. |

### Tools

| Tool | Description |
|------|-------------|
| `verify_patient_tool` | Looks up the patient by full name, phone, and date of birth. |
| `list_appointments_tool` | Returns all appointments for the verified patient. |
| `confirm_appointment_tool` | Marks an appointment as **confirmed**. |
| `cancel_appointment_tool` | Marks an appointment as **cancelled**. |

---

## Project Structure

```
.
├── .env.example              # Environment variable template
├── .gitignore
├── requirements.txt
├── README.md
└── app/
    ├── __init__.py
    ├── main.py               # FastAPI application & session management
    ├── models.py             # Pydantic request / response models
    ├── data.py               # Mock patient & appointment data + helpers
    └── agent/
        ├── __init__.py
        ├── state.py          # LangGraph AgentState definition
        ├── tools.py          # LangChain tools wrapping the data layer
        └── graph.py          # LangGraph graph (nodes, edges, compiler)
```

---

## Prerequisites

- [Miniforge / Conda](https://github.com/conda-forge/miniforge) (or any conda distribution)
- An [OpenRouter](https://openrouter.ai) account and API key

---

## Setup

### 1. Activate the conda environment

```bash
conda activate conversational-ai-service
```

> If the environment does not exist yet, create it first:
> ```bash
> conda create -n conversational-ai-service python=3.12 -y
> conda activate conversational-ai-service
> pip install -r requirements.txt
> ```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your OpenRouter API key:

```dotenv
OPENROUTER_API_KEY=sk-or-...your-key-here...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1   # optional, default shown
OPENROUTER_MODEL=openai/gpt-4o-mini                # optional, default shown
```

---

## Running the Service

```bash
conda activate conversational-ai-service
uvicorn app.main:app --reload
```

The service starts on **http://localhost:8000**.

- Interactive docs (Swagger UI): http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## API Reference

### `GET /`

Health check.

**Response**
```json
{ "status": "ok", "service": "Conversational AI Appointment Assistant" }
```

---

### `POST /chat`

Send a conversational message. The same `session_id` must be used for all
turns of a single conversation.

**Request body**
```json
{
  "session_id": "patient-session-001",
  "message": "Hi, I need help with my appointments."
}
```

**Response**
```json
{
  "session_id": "patient-session-001",
  "reply": "Hello! I'd be happy to help you manage your appointments...",
  "verified": false
}
```

The `verified` field becomes `true` once the patient's identity has been
successfully confirmed.

---

### `DELETE /chat/{session_id}`

Clear the conversation history for a session (start fresh).

---

## Conversation Flow

Below is a sample interaction showing the full flow:

```
User  : Hi, I need help with my appointments.
Bot   : Hello! I'd be happy to help you manage your appointments. Before I
        can assist you, I need to verify your identity. Could you please
        provide your full name?

User  : Alice Johnson
Bot   : Thank you, Alice! Could you please share your phone number?

User  : 555-1234
Bot   : And finally, your date of birth in YYYY-MM-DD format please.

User  : 1985-03-15
Bot   : ✅ I've verified your identity, Alice! How can I help you today?
        You can:
          • List your appointments
          • Confirm an appointment
          • Cancel an appointment

User  : Show me my appointments.
Bot   : Here are your upcoming appointments:
          [A001] 2026-05-10 at 09:00 – General Practice with Dr. Emily Carter (scheduled)
          [A002] 2026-05-20 at 14:30 – Dermatology with Dr. Michael Lee (scheduled)
        What would you like to do?

User  : Please confirm appointment A001.
Bot   : ✅ Appointment on 2026-05-10 at 09:00 with Dr. Emily Carter has been
        confirmed. Is there anything else I can help you with?

User  : Actually, cancel appointment A002.
Bot   : ✅ Appointment on 2026-05-20 at 14:30 with Dr. Michael Lee has been
        cancelled. Would you like to see your updated appointments or need
        anything else?
```

---

## Mock Data

Three test patients are available:

| Name | Phone | Date of Birth | Patient ID |
|------|-------|---------------|------------|
| Alice Johnson | 555-1234 | 1985-03-15 | P001 |
| Bob Smith | 555-5678 | 1990-07-22 | P002 |
| Carol White | 555-9012 | 1978-11-30 | P003 |

Phone numbers are normalised (dashes and spaces are ignored during matching).

---

## Configuration

All configuration lives in the `.env` file:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | **Yes** | — | Your OpenRouter API key |
| `OPENROUTER_BASE_URL` | No | `https://openrouter.ai/api/v1` | OpenRouter base URL |
| `OPENROUTER_MODEL` | No | `openai/gpt-4o-mini` | Model identifier on OpenRouter |

Any [model listed on OpenRouter](https://openrouter.ai/models) that supports
**function/tool calling** can be used (e.g. `openai/gpt-4o`,
`anthropic/claude-3.5-sonnet`, `google/gemini-flash-1.5`).
