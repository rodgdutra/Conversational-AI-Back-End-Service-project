# Tests for Conversational AI Back-End Service

This directory contains tests for the Conversational AI Back-End Service project. The tests are organized into different categories and use pytest as the testing framework.

## Directory Structure

```
tests/
├── conftest.py         # Common fixtures and test setup
├── unit/               # Unit tests (no DB required, only used if `USE_MOCK_DATA=false` set in the .env)
│   ├── test_config.py          # Config properties and URL construction
│   ├── test_models.py          # Pydantic request/response schema validation
│   ├── test_data.py            # In-memory data layer (find, get, confirm, cancel)
│   ├── test_logger.py          # Logger factory
│   ├── test_agent_state.py     # AgentState defaults and field assignment
│   └── test_agent_tools.py     # LangChain tool wrappers (mocked data layer)
├── agent/              # Agent behaviour tests (no DB required, only used if `USE_MOCK_DATA=false` set in the .env)
│   ├── test_agent_access_control.py      # Identity verification gate
│   ├── test_appointment_routing.py       # Multi-action routing within a session
│   ├── test_cancel_appointment.py        # Cancel appointment flow
│   ├── test_confirm_appointment.py       # Confirm appointment flow
│   ├── test_conversational_robustness.py # Off-topic, gibberish, prompt injection
│   ├── test_list_appointments.py         # List appointments flow
│   └── test_safety_data_leakage.py       # Data isolation & prompt confidentiality
└── integration/        # Integration tests (require a live PostgreSQL database)
    ├── test_state_persistence.py  # Tests for state persistence
    ├── test_state_tracking.py     # Tests for state tracking
    └── test_patient_data.py       # Tests for patient data persistence
```

## Prerequisites

Before running the tests, make sure you have installed all required dependencies:

```
pip install -r requirements.txt
```

## Running the Tests

### Running All Tests

To run all tests:

```bash
pytest tests/
```

### Running Specific Test Categories

To run only agent tests:

```bash
pytest tests/agent/
```

To run only integration tests:

```bash
pytest tests/integration/
```

To run only unit tests:

```bash
pytest tests/unit/
```

### Running Specific Test Files

To run tests from a specific file:

```bash
pytest tests/agent/test_list_appointments.py
pytest tests/integration/test_patient_data.py
```

### Running Specific Test Classes or Methods

To run a specific test class:

```bash
pytest tests/agent/test_list_appointments.py::TestListAppointments
pytest tests/integration/test_patient_data.py::TestPatientService
```

To run a specific test method:

```bash
pytest tests/agent/test_list_appointments.py::TestListAppointments::test_lists_appointments_after_successful_verification
pytest tests/integration/test_patient_data.py::TestPatientService::test_create_patient
```

## Test Coverage

To generate a test coverage report:

```bash
pytest --cov=app tests/
```

For a more detailed HTML coverage report:

```bash
pytest --cov=app --cov-report=html tests/
```

This will create a directory named `htmlcov` with the coverage report, which you can view by opening `htmlcov/index.html` in a browser.

## Asynchronous Tests

The project includes asynchronous tests that use the `pytest-asyncio` plugin. These tests are marked with the `@pytest.mark.asyncio` decorator.

## Test Configuration

The test configuration is managed through fixtures defined in `conftest.py`. These fixtures provide:

1. Database initialization and cleanup
2. Test session data
3. Common test utilities

## Database Setup for Tests

The integration tests automatically set up and tear down the database tables as needed. The database connection is configured through environment variables, which can be set in `.env`.

Agent tests do **not** require a database — all persistence and database calls are replaced with in-memory mocks.

## Running Tests in Docker

We provide dedicated Docker Compose configurations for running tests in isolation.

---

All Docker Compose files live in the `docker/` directory at the repo root.

| Compose file | Runs | Needs DB? |
|---|---|---|
| `docker/docker-compose.agent-test.yml` | `tests/agent/` | No ( will depend on `USE_MOCK_DATA` env variable if will use or not) |
| `docker/docker-compose.unit-test.yml` | `tests/unit/` | No ( will depend on `USE_MOCK_DATA` env variable if will use or not) |
| `docker/docker-compose.integration-test.yml` | `tests/integration/` | Yes |
| `docker/docker-compose.test.yml` | All tests | Yes |

---

### Running Agent Tests (`docker/docker-compose.agent-test.yml`)

Agent tests are fully mocked and require **no database**. This makes them fast and dependency-free. Nevertheless, it is possible to test with database using `USE_MOCK_DATA=false`. 

1. To run all agent tests:

```bash
docker compose -f docker/docker-compose.agent-test.yml up
```

2. To run a specific agent test file:

```bash
PYTEST_ARGS="-xvs tests/agent/test_list_appointments.py" docker compose -f docker/docker-compose.agent-test.yml up
```

3. To run with coverage:

```bash
PYTEST_ARGS="--cov=app --cov-report=html" docker compose -f docker/docker-compose.agent-test.yml up
```

4. To clean up the agent test container:

```bash
docker compose -f docker/docker-compose.agent-test.yml down
```

---

### Running Unit Tests (`docker/docker-compose.unit-test.yml`)

Unit tests are also fully mocked and require **no database**. However one can set `USE_MOCK_DATA=false` in order to depend on the dataset to check the tests.

```bash
docker compose -f docker/docker-compose.unit-test.yml up
docker compose -f docker/docker-compose.unit-test.yml down
```

---

### Running Integration Tests (`docker/docker-compose.integration-test.yml`)

Integration tests require a live PostgreSQL instance (started automatically).

```bash
docker compose -f docker/docker-compose.integration-test.yml up
docker compose -f docker/docker-compose.integration-test.yml down -v
```

---

### Running All Tests (`docker/docker-compose.test.yml`)

The general test configuration spins up a dedicated PostgreSQL instance for the integration tests.

1. To run all tests in the Docker container:

```bash
docker compose -f docker/docker-compose.test.yml up
```

2. To run specific tests or use additional pytest arguments:

```bash
PYTEST_ARGS="-xvs tests/integration/test_patient_data.py" docker compose -f docker/docker-compose.test.yml up
```

3. To run with coverage:

```bash
PYTEST_ARGS="--cov=app --cov-report=html" docker compose -f docker/docker-compose.test.yml up
```

### Benefits of the Test Docker Setup

- Agent and unit test containers need no database — starts instantly
- Dedicated test database on port 5433 for integration tests (avoids conflicts with development)
- Clean environment for reproducible test results
- Independent of your local Python environment
- Tests run the same way in CI/CD as they do locally

### Cleanup

To clean up the integration test containers and volumes:

```bash
docker compose -f docker/docker-compose.test.yml down -v
```
