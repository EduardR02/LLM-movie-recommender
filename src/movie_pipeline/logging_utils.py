"""Logging configuration for the movie pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .paths import PATHS


def configure_logging(verbosity: int = 0) -> None:
    """Configure root logger with stdout handler and file mirror."""
    level = logging.INFO
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    handlers.append(stream_handler)

    log_file = Path(PATHS.logs) / "pipeline.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers)
