"""
Shared pytest fixtures for the test suite.

Behaviour is driven by the USE_MOCK_DATA environment variable
(read via app.config.Config.USE_MOCK_DATA):

  USE_MOCK_DATA=true  → No PostgreSQL involvement.  All DB fixtures
                         yield None / are no-ops.  Tests that require
                         a live database are automatically skipped.

  USE_MOCK_DATA=false → A real PostgreSQL database is used.  The schema
                         is created once per session, then seeded with
                         the canonical mock data (P001-P003, A001-A005).
                         After every test the tables are truncated and
                         re-seeded so each test starts with clean data.
"""

import os
import sys
import pytest
import asyncio
from typing import Generator, AsyncGenerator, Optional
from unittest.mock import patch
from sqlalchemy import text

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ---------------------------------------------------------------------------
# CRITICAL: Prevent app.main from calling init_db / initialize_mock_data
# at import time.  These patches must be started before any test module
# imports app.main.
# ---------------------------------------------------------------------------


def _noop():
    pass


_mock_init_db_patcher = patch("app.main.init_db", side_effect=_noop)
_mock_init_data_patcher = patch("app.main.initialize_mock_data", side_effect=_noop)

_mock_init_db_patcher.start()
_mock_init_data_patcher.start()

# ---------------------------------------------------------------------------
# Safe to import app modules now
# ---------------------------------------------------------------------------

from app.config import config          # noqa: E402
from app.db import Base                # noqa: E402
from app.logger import get_logger      # noqa: E402

logger = get_logger(__name__)
logger.info(
    "conftest loaded | USE_MOCK_DATA=%s | mocks applied to app.main.init_db "
    "and app.main.initialize_mock_data",
    config.USE_MOCK_DATA,
)


# ---------------------------------------------------------------------------
# Event loop (required by pytest-asyncio)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Single event loop shared across the whole test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Database initialisation (session-scoped, runs once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def initialize_database() -> None:
    """
    Set up the database schema and seed it with canonical mock data.

    Skipped entirely when USE_MOCK_DATA=true.
    """
    if config.USE_MOCK_DATA:
        logger.info("USE_MOCK_DATA=true — database initialisation skipped")
        return

    from app.db import engine
    from app.data import initialize_mock_data

    logger.info("Initialising test database …")
    logger.info("  DATABASE_URL : %s", config.DATABASE_URL)
    logger.info("  SCHEMA       : %s", config.POSTGRES_SCHEMA)

    try:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        initialize_mock_data()
        logger.info("Test database ready (schema created + mock data seeded)")
    except Exception as exc:
        logger.error("Database initialisation failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Per-test synchronous DB session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def db_session() -> Generator:
    """
    Yield a live synchronous SQLAlchemy session.

    Yields None when USE_MOCK_DATA=true so that dependent fixtures can
    detect the mock-data mode without failing on a missing connection.
    """
    if config.USE_MOCK_DATA:
        yield None
        return

    from app.db import get_db

    db = get_db()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


# ---------------------------------------------------------------------------
# Per-test asynchronous DB session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
async def async_db_session() -> AsyncGenerator:
    """Yield a live async SQLAlchemy session."""
    from app.db import get_async_db

    async with get_async_db() as session:
        try:
            yield session
        finally:
            await session.rollback()


# ---------------------------------------------------------------------------
# Per-test database cleanup + re-seed  (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function", autouse=True)
def cleanup_database(db_session) -> None:
    """
    Truncate all tables after each test and re-seed with mock data.

    This ensures every test starts with exactly the canonical data set.
    No-op when USE_MOCK_DATA=true (db_session is None).
    """
    yield  # ← test runs here

    if db_session is None:
        # USE_MOCK_DATA=true — nothing to clean up in the database.
        return

    try:
        logger.info("Post-test database cleanup …")

        # Disable FK triggers for this transaction only (SET LOCAL reverts on commit)
        db_session.execute(text("SET LOCAL session_replication_role = 'replica'"))

        tables = list(reversed(Base.metadata.sorted_tables))
        logger.debug("Tables to truncate: %s", [t.name for t in tables])

        for table in tables:
            schema_table = f'"{table.schema}"."{table.name}"'
            try:
                db_session.execute(text(f"TRUNCATE TABLE {schema_table} CASCADE"))
            except Exception as exc:
                logger.warning(
                    "TRUNCATE failed for %s (%s) — trying DELETE", table.name, exc
                )
                try:
                    db_session.execute(text(f"DELETE FROM {schema_table}"))
                except Exception as exc2:
                    logger.error(
                        "DELETE also failed for %s: %s", table.name, exc2
                    )

        # Commit releases the transaction; SET LOCAL is automatically reverted,
        # so the connection returns to the pool with session_replication_role = 'origin'
        db_session.commit()
        logger.info("Tables truncated")

        # Re-seed canonical mock data so the next test starts clean
        from app.data import initialize_mock_data

        initialize_mock_data()
        logger.info("Mock data re-seeded after test cleanup")

    except Exception as exc:
        logger.error("Error during post-test cleanup: %s", exc)
        db_session.rollback()
