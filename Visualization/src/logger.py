"""Provide structured logging configuration for SmartCropAdvisor.

This module exposes `get_logger` to construct stream-backed loggers honoring
the `SMARTCROP_LOG_LEVEL` environment variable without duplicating handlers.
"""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger honoring `SMARTCROP_LOG_LEVEL`.

    Attaches a single formatted `StreamHandler` if none exist and disables
    log propagation to prevent duplicate root-logger records.

    Args:
        name: Logger name, typically `__name__` of the calling module.

    Returns:
        Configured `logging.Logger` instance.
    """
    level_name = os.getenv("SMARTCROP_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
