"""Structured logging setup.

Console output stays human-readable; the file sink is JSON lines so that log
analysis (and, later, the observability dashboard) can query it without
regex-scraping. Configure once at process start via :func:`setup_logging`.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from typing import Any

from app.utils.paths import ensure_dir, logs_dir

_CONFIGURED = False

# Attributes present on every LogRecord; anything else was attached by the
# caller via `extra=` and belongs in the structured payload.
_STANDARD_ATTRS = frozenset(
    ["args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName", "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name", "pathname", "process", "processName", "relativeCreated", "stack_info", "thread", "threadName", "taskName"]
)


class JsonFormatter(logging.Formatter):
    """Render a record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = _jsonable(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


class ConsoleFormatter(logging.Formatter):
    """Compact, aligned console output."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)-32s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def setup_logging(
    level: str | int = "INFO",
    *,
    log_file: str = "platform.jsonl",
    to_file: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Configure root logging. Idempotent unless ``force`` is set.

    Args:
        level: console log level.
        log_file: filename inside the logs directory.
        to_file: write the JSON-lines sink. Disable in tests.
        force: reconfigure even if already configured.

    Returns:
        The configured root logger.
    """
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED and not force:
        return root

    for handler in list(root.handlers):
        root.removeHandler(handler)

    # Root passes everything through; individual handlers do the filtering.
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level if isinstance(level, int) else level.upper())
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)

    if to_file:
        path = ensure_dir(logs_dir()) / log_file
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Module-level logger. Use ``get_logger(__name__)``."""
    return logging.getLogger(name)
