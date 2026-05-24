"""Structured logging configuration using structlog."""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def setup_logging(level: str = "INFO") -> None:
    structlog.configure(
        processors=[structlog.stdlib.add_log_level, structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=getattr(logging, level.upper())
    )


def get_logger(**ctx: Any) -> structlog.stdlib.BoundLogger:
    from typing import cast

    return cast(structlog.stdlib.BoundLogger, structlog.get_logger().bind(**ctx))
