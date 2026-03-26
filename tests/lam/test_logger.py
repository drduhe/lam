"""Tests for ``lam.logger`` (real logging records, no mocked formatters)."""

from __future__ import annotations

import logging

import pytest

from lam.logger import ColoredFormatter, LOG_LEVELS, get_logger


def test_colored_formatter_emits_message():
    fmt = ColoredFormatter()
    record = logging.LogRecord("n", logging.WARNING, __file__, 1, "hello", (), None)
    line = fmt.format(record)
    assert "WARNING" in line
    assert "hello" in line


def test_get_logger_respects_log_level_env(monkeypatch, clear_log_level_env):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    log = get_logger("lam_tests_logger_env", level=logging.ERROR)
    assert log.level == logging.DEBUG


def test_get_logger_invalid_log_level_raises(monkeypatch, clear_log_level_env):
    monkeypatch.setenv("LOG_LEVEL", "not_a_level")
    with pytest.raises(AssertionError, match="Invalid LOG_LEVEL"):
        get_logger("lam_tests_logger_bad")


def test_log_levels_keys():
    assert set(LOG_LEVELS) == {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
