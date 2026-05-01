# Conversational AI Back-End Service with LangGraph

A healthcare back-end service that uses a LangGraph multi-agent workflow to help patients manage their appointments through a conversational interface.

## Features

- Conversational AI interface for healthcare appointments
- Patient identity verification
- Appointment listing, confirmation, and cancellation
- Persistent state tracking with PostgreSQL
- Multi-agent workflow using LangGraph

## Setup

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- PostgreSQL database

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Conversational-AI-Back-End-Service-project.git
   cd Conversational-AI-Back-End-Service-project
   ```

2. Set up environment variables:
   ```
   cp .env.example .env
   ```
   Edit `.env` with your API keys and configuration.

3. Start with Docker:
   ```
   docker-compose up -d
   ```

4. Or install locally:
   ```
   pip install -r requirements.txt
   python init_db.py
   uvicorn app.main:app --reload
   ```

## Usage

The API has the following endpoints:

- `GET /`: Health check
- `POST /chat`: Send a message to the assistant
- `DELETE /chat/{session_id}`: Clear a conversation session

Example request:
```json
{
  "session_id": "user123",
  "message": "I'd like to check my appointments please"
}
```

## Testing

The project includes a comprehensive test suite using pytest. Tests are organized into:

- Unit tests: Test individual components
- Integration tests: Test component interactions

### Running Tests with Docker

Use the provided script to run tests in a Docker environment:

```bash
./run-tests.sh
```

This script:
- Creates a dedicated test database
- Runs all tests in an isolated environment
- Cleans up containers when done

Additional options:
```bash
# Run specific tests
./run-tests.sh "-xvs tests/integration/test_state_tracking.py"

# Skip rebuilding the container
./run-tests.sh --no-rebuild "-xvs tests/unit/"
```

### Running Tests Locally

You can also run tests directly with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/integration/test_state_tracking.py
```

For more test options, see the documentation in `tests/README.md`.

## Architecture

The service follows a layered architecture:

- `app/main.py`: FastAPI application entry point
- `app/agent/graph.py`: LangGraph workflow definition
- `app/agent/tools.py`: LangChain tools for appointment management
- `app/agent/state.py`: State definitions and transitions
- `app/agent/persistence.py`: PostgreSQL state persistence
- `app/data.py`: Data access layer
- `app/models.py`: Pydantic models for API

## Database Schema

The service uses PostgreSQL with the following schema:

- `sessions`: Conversation sessions
- `graph_states`: Persistent LangGraph states
- `state_transitions`: State transition history
- `patients`: Patient records
- `appointments`: Appointment records

## License

[MIT License](LICENSE)