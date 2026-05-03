"""
Unit tests for app.logger.get_logger.

Verifies that the factory returns a properly configured Logger instance.
"""
import logging
import pytest


from app.logger import get_logger


class TestGetLogger:
    """Tests for the get_logger() factory function."""

    def test_returns_logger_instance(self):
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches_argument(self):
        logger = get_logger("app.some.module")
        assert logger.name == "app.some.module"

    def test_dunder_name_usage(self):
        """Typical usage: get_logger(__name__) inside a module."""
        logger = get_logger(__name__)
        assert isinstance(logger, logging.Logger)
        assert logger.name == __name__

    def test_same_name_returns_same_instance(self):
        """Python's logging module caches loggers by name."""
        l1 = get_logger("shared.logger")
        l2 = get_logger("shared.logger")
        assert l1 is l2

    def test_different_names_return_different_instances(self):
        l1 = get_logger("module.a")
        l2 = get_logger("module.b")
        assert l1 is not l2
        assert l1.name != l2.name

    def test_root_handler_is_configured(self):
        """The root logger must have at least one handler after import."""
        root = logging.getLogger()
        assert len(root.handlers) > 0

    def test_logger_can_emit_messages_without_raising(self):
        """Calling log methods must not raise any exceptions."""
        logger = get_logger("test.emit")
        # These should not raise regardless of the log level
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warn message")
        logger.error("error message")
