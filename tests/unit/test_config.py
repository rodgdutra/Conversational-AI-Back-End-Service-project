"""
Unit tests for app.config.Config.

All tests instantiate a fresh Config() and exercise the computed
properties/class-level attributes without requiring a running database
or real API keys.
"""
import os
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config():
    """Return a fresh Config instance (avoids stale module-level singleton)."""
    from app.config import Config
    return Config()


# ---------------------------------------------------------------------------
# Class-level attribute defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Verify sensible defaults when no environment variables are set."""

    def test_openrouter_base_url_default(self):
        cfg = _config()
        assert "openrouter.ai" in cfg.OPENROUTER_BASE_URL

    def test_postgres_port_is_int(self):
        cfg = _config()
        assert isinstance(cfg.POSTGRES_PORT, int)

    def test_postgres_port_default_value(self):
        cfg = _config()
        # May be overridden by .env, but must at least be a valid port
        assert 1 <= cfg.POSTGRES_PORT <= 65535

    def test_log_level_is_valid(self):
        cfg = _config()
        assert cfg.LOG_LEVEL in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    def test_use_mock_data_is_bool(self):
        cfg = _config()
        assert isinstance(cfg.USE_MOCK_DATA, bool)


# ---------------------------------------------------------------------------
# DATABASE_URL property
# ---------------------------------------------------------------------------


class TestDatabaseURLProperty:
    """Config.DATABASE_URL reads os.environ at call time."""

    def test_returns_env_var_when_set(self):
        cfg = _config()
        expected = "postgresql+psycopg2://user:pass@db_host:5432/testdb"
        with patch.dict(os.environ, {"DATABASE_URL": expected}):
            assert cfg.DATABASE_URL == expected

    def test_builds_url_from_parts_when_env_var_absent(self):
        cfg = _config()
        # Remove DATABASE_URL so the property falls back to building the URL
        env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
        with patch.dict(os.environ, env, clear=True):
            url = cfg.DATABASE_URL
        assert url.startswith("postgresql+psycopg2://")
        # The URL must contain the configured host and database
        assert cfg.POSTGRES_HOST in url
        assert cfg.POSTGRES_DB in url

    def test_url_contains_user_and_password(self):
        cfg = _config()
        env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
        with patch.dict(os.environ, env, clear=True):
            url = cfg.DATABASE_URL
        assert cfg.POSTGRES_USER in url


# ---------------------------------------------------------------------------
# ASYNC_DATABASE_URL property
# ---------------------------------------------------------------------------


class TestAsyncDatabaseURLProperty:
    """Config.ASYNC_DATABASE_URL must always use the asyncpg driver."""

    def test_returns_async_env_var_when_set(self):
        cfg = _config()
        expected = "postgresql+asyncpg://user:pass@db_host:5432/testdb"
        with patch.dict(os.environ, {"ASYNC_DATABASE_URL": expected}):
            assert cfg.ASYNC_DATABASE_URL == expected

    def test_converts_psycopg2_url_to_asyncpg(self):
        cfg = _config()
        sync_url = "postgresql+psycopg2://user:pass@host/db"
        env = {"DATABASE_URL": sync_url}
        env_without_async = {k: v for k, v in os.environ.items() if k != "ASYNC_DATABASE_URL"}
        with patch.dict(os.environ, {**env_without_async, **env}, clear=True):
            url = cfg.ASYNC_DATABASE_URL
        assert "asyncpg" in url
        assert "psycopg2" not in url

    def test_url_always_contains_asyncpg_driver(self):
        """Whatever the source, the resulting URL must use asyncpg."""
        cfg = _config()
        url = cfg.ASYNC_DATABASE_URL
        assert "asyncpg" in url


# ---------------------------------------------------------------------------
# USE_OLLAMA property
# ---------------------------------------------------------------------------


class TestUseOllamaProperty:
    """Config.USE_OLLAMA should reflect whether OLLAMA_MODEL is configured."""

    def test_false_when_ollama_model_is_empty(self):
        cfg = _config()
        cfg.OLLAMA_MODEL = ""
        assert cfg.USE_OLLAMA is False

    def test_true_when_ollama_model_is_set(self):
        cfg = _config()
        cfg.OLLAMA_MODEL = "llama3"
        assert cfg.USE_OLLAMA is True


# ---------------------------------------------------------------------------
# model_name property
# ---------------------------------------------------------------------------


class TestModelNameProperty:
    """Config.model_name picks the right backend model."""

    def test_returns_ollama_model_when_use_ollama_is_true(self):
        cfg = _config()
        cfg.OLLAMA_MODEL = "llama3"
        cfg.OPENROUTER_MODEL = "gpt-4"
        assert cfg.model_name == "llama3"

    def test_returns_openrouter_model_when_ollama_not_configured(self):
        cfg = _config()
        cfg.OLLAMA_MODEL = ""
        cfg.OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
        assert cfg.model_name == "meta-llama/llama-3.1-8b-instruct:free"
