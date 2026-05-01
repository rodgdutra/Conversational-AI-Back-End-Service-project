# Tests for Conversational AI Back-End Service

This directory contains tests for the Conversational AI Back-End Service project. The tests are organized into different categories and use pytest as the testing framework.

## Directory Structure

```
tests/
├── conftest.py         # Common fixtures and test setup
├── unit/               # Unit tests
└── integration/        # Integration tests
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
pytest tests/integration/test_patient_data.py
```

### Running Specific Test Classes or Methods

To run a specific test class:

```bash
pytest tests/integration/test_patient_data.py::TestPatientService
```

To run a specific test method:

```bash
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

The tests automatically set up and tear down the database tables as needed. The database connection is configured through environment variables, which can be set in `.env`.

## Running Tests in Docker

We provide a dedicated Docker Compose configuration for running tests. This keeps your test environment isolated from your development environment.

### Using the Testing Docker Compose

1. To run all tests in the Docker container:

```bash
docker-compose -f docker-compose.test.yml up
```

2. To run specific tests or use additional pytest arguments:

```bash
PYTEST_ARGS="-xvs tests/integration/test_patient_data.py" docker-compose -f docker-compose.test.yml up
```

3. To run with coverage:

```bash
PYTEST_ARGS="--cov=app --cov-report=html" docker-compose -f docker-compose.test.yml up
```

### Benefits of the Test Docker Setup

- Dedicated test database on port 5433 (avoiding conflicts with development)
- Clean environment for reproducible test results
- Independent of your local Python environment
- Tests run the same way in CI/CD as they do locally

### Cleanup

To clean up the test containers and volumes:

```bash
docker-compose -f docker-compose.test.yml down -v
```
