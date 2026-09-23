"""Logging setup for the whole package.

Two sinks with different verbosity: a rotating file that keeps the full INFO
trace of a run, and a console handler that stays quiet so long training runs do
not drown the terminal. Progress meant for the user goes through ``print`` or
tqdm, not through the logger.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from topiclens.config import LoggingConfig

LOGGER_NAME = "topiclens"

_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONSOLE_FORMAT = "%(levelname)s: %(message)s"

_MAX_BYTES = 2_000_000
_BACKUP_COUNT = 3


def setup_logging(config: LoggingConfig | None = None, *, force: bool = False) -> logging.Logger:
    """Configure and return the package logger.

    Calling this twice is a no-op unless ``force`` is set, so importing a module
    that logs never duplicates handlers.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers and not force:
        return logger
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    config = config or LoggingConfig()
    file_level = logging.getLevelName(config.level)
    console_level = logging.getLevelName(config.console_level)

    logger.setLevel(min(file_level, console_level))
    # Messages are emitted by our handlers only; the root logger stays untouched
    # so that Streamlit and third-party libraries keep their own configuration.
    logger.propagate = False

    log_path = config.file_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger, e.g. ``get_logger(__name__)``."""
    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    suffix = name.split(".", 1)[-1] if name.startswith(f"{LOGGER_NAME}.") else name
    return logging.getLogger(f"{LOGGER_NAME}.{suffix}")
