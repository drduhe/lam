"""Colored stderr logging helpers used by video and IO utilities."""

from __future__ import annotations

import logging
import os

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class ColoredFormatter(logging.Formatter):
    """ANSI-colored formatter for terminal log lines (one color per level)."""

    def __init__(self) -> None:
        super().__init__()
        reset = "\033[0m"
        colors = {
            logging.DEBUG: f"{reset}\033[36m",
            logging.INFO: f"{reset}\033[32m",
            logging.WARNING: f"{reset}\033[33m",
            logging.ERROR: f"{reset}\033[31m",
            logging.CRITICAL: f"{reset}\033[35m",
        }
        fmt_str = "{color}%(levelname)s %(asctime)s %(process)d %(filename)s:%(lineno)4d:{reset} %(message)s"
        self.formatters = {
            level: logging.Formatter(fmt_str.format(color=color, reset=reset)) for level, color in colors.items()
        }
        self.default_formatter = self.formatters[logging.INFO]

    def format(self, record: logging.LogRecord) -> str:
        formatter = self.formatters.get(record.levelno, self.default_formatter)
        return formatter.format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module logger with a single colored stream handler.

    If the environment variable ``LOG_LEVEL`` is set to a known level name
    (``DEBUG``, ``INFO``, etc.), that level overrides ``level``.

    Args:
        name: Logger name (typically ``__name__`` of the caller).
        level: Default minimum level when ``LOG_LEVEL`` is unset.

    Returns:
        Configured logger with propagation disabled.

    Raises:
        AssertionError: If ``LOG_LEVEL`` is set to an unknown string.
    """
    if "LOG_LEVEL" in os.environ:
        level_name = os.environ["LOG_LEVEL"].upper()
        assert level_name in LOG_LEVELS, f"Invalid LOG_LEVEL: {level_name}, must be one of {list(LOG_LEVELS.keys())}"
        level = LOG_LEVELS[level_name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(ColoredFormatter())
    logger.addHandler(ch)
    return logger
