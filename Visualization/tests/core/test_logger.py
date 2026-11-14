"""Verify structured logger configuration and handler idempotency.

This module tests `get_logger` behavior for `SMARTCROP_LOG_LEVEL` parsing,
invalid level fallbacks, and handler deduplication.
"""

import logging

import pytest

from src.logger import get_logger


class TestLogger:
    """Test suite for `get_logger` configuration behavior."""

    def test_get_logger_honors_debug_level(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify `SMARTCROP_LOG_LEVEL=DEBUG` configures level and propagation.

        Args:
            monkeypatch: Pytest environment patch fixture.
        """
        monkeypatch.setenv("SMARTCROP_LOG_LEVEL", "DEBUG")
        logger = get_logger("smartcrop.test.debug")
        assert logger.level == logging.DEBUG
        assert logger.propagate is False
        assert len(logger.handlers) == 1

    def test_get_logger_falls_back_to_info_on_unknown_level(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify unrecognized log level names fall back to `INFO`.

        Args:
            monkeypatch: Pytest environment patch fixture.
        """
        monkeypatch.setenv("SMARTCROP_LOG_LEVEL", "NOT_A_LEVEL")
        logger = get_logger("smartcrop.test.fallback")
        assert logger.level == logging.INFO

    def test_get_logger_does_not_duplicate_handlers(self) -> None:
        """Verify repeated calls for the same logger name reuse one handler."""
        name = "smartcrop.test.idempotent"
        first = get_logger(name)
        second = get_logger(name)
        assert first is second
        assert len(second.handlers) == 1
