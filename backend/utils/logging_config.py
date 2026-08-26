"""
Centralized logging configuration.

Call ``configure_logging()`` once, early in application startup (e.g. in
``main.py``), and every module can then simply do:

    import logging
    logger = logging.getLogger(__name__)

and get consistently formatted output without needing to know about
handlers, formatters, or levels.
"""

from __future__ import annotations

import logging
import sys


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once for the whole application.

    Args:
        level: Logging level name, e.g. "DEBUG", "INFO", "WARNING".
    """
    global _CONFIGURED
    if _CONFIGURED:
        # Avoid attaching duplicate handlers if called more than once
        # (e.g. once from main.py and once from a test module).
        logging.getLogger().setLevel(level.upper())
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Quiet down noisy third-party libraries unless we're in debug mode.
    if level.upper() != "DEBUG":
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _CONFIGURED = True
