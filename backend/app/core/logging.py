"""
Centralized logging infrastructure with colored formatter.

Provides unified logging setup that can be passed to uvicorn via log_config.
"""

import logging
import sys
from datetime import datetime, timezone
from typing import Optional

_console_handler: Optional[logging.StreamHandler] = None


def _get_log_level() -> int:
    """Get log level from environment or default to DEBUG."""
    import os
    level = os.environ.get("LOG_LEVEL", "DEBUG").upper()
    return getattr(logging, level, logging.DEBUG)


class ColoredFormatter(logging.Formatter):
    """Formatter with timestamp, level, source, and color coding."""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[41m", # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        source = record.name
        message = record.getMessage()

        color = self.COLORS.get(level, self.RESET)
        return f"{timestamp}: {color}{level}{self.RESET}: {source}: {message}"


def get_log_config() -> dict:
    """
    Returns a logging configuration dict suitable for uvicorn's log_config parameter.
    Uses ColoredFormatter for all handlers.
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "custom": {
                "()": f"{__name__}.{ColoredFormatter.__name__}",
            },
            "default": {
                "()": f"{__name__}.{ColoredFormatter.__name__}",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": _get_log_level(),
                "formatter": "default",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn": {"level": _get_log_level(), "handlers": ["console"], "propagate": False},
            "uvicorn.error": {"level": _get_log_level(), "handlers": ["console"], "propagate": False},
            "uvicorn.access": {"level": _get_log_level(), "handlers": ["console"], "propagate": False},
        },
    }


def init_logging() -> None:
    """Initialize the logging system with colored formatter (for non-uvicorn use)."""
    global _console_handler

    if _console_handler is not None:
        return

    _console_handler = logging.StreamHandler(sys.stderr)
    _console_handler.setLevel(_get_log_level())
    _console_handler.setFormatter(ColoredFormatter())

    root_logger = logging.getLogger()
    root_logger.addHandler(_console_handler)
    root_logger.setLevel(_get_log_level())


_NOISY_LOGGERS = [
    "LiteLLM",
    "litellm",
    "litellm.utils",
    "litellm.main",
    "openai",
    "openai._base_client",
    "anthropic",
    "anthropic._base_client",
    "chromadb",
    "watchfiles",
]

_VERY_NOISY_LOGGERS = [
    "aiosqlite",
    "sqlalchemy.engine",
    "sqlalchemy.engine.Engine",
    "sqlalchemy.pool",
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpx",
    "urllib3",
    "urllib3.connectionpool",
]


def ensure_logging() -> None:
    """Guarantee root logger has exactly our handler — safe to call repeatedly.

    Replaces all existing root handlers (e.g. alembic's fileConfig leftovers) so
    there is exactly one handler with ColoredFormatter. Call from lifespan startup
    after run_migrations().
    """
    global _console_handler
    root = logging.getLogger()

    # Replace whatever handlers fileConfig or other libraries left on root
    for h in root.handlers[:]:
        root.removeHandler(h)
    _console_handler = logging.StreamHandler(sys.stderr)
    _console_handler.setLevel(_get_log_level())
    _console_handler.setFormatter(ColoredFormatter())
    root.addHandler(_console_handler)
    root.setLevel(_get_log_level())

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO)
    for name in _VERY_NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Suppress uvicorn's built-in access log — the HTTP middleware already covers it
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(component: str) -> logging.Logger:
    return logging.getLogger(component)