"""
Centralised logging configuration for the Appointment Assistant service.

Usage (in any module):
    from app.logger import get_logger
    logger = get_logger(__name__)

Log level is controlled via the LOG_LEVEL environment variable
(default: INFO).  Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL.
"""

import logging
import os
import sys
from app.config import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_LEVEL_ENV = config.LOG_LEVEL
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_ENV, logging.INFO)

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Root handler — configure once at import time
# ---------------------------------------------------------------------------

def _setup_root_logger() -> None:
    """Configure the root logger with a single StreamHandler to stdout."""
    root = logging.getLogger()

    # Avoid adding duplicate handlers if the module is re-imported
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(_LOG_LEVEL)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root.setLevel(_LOG_LEVEL)
    root.addHandler(handler)

    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langsmith").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


_setup_root_logger()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
