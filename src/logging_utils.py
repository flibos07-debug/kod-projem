"""Centralised logging configuration.

A single :func:`setup_logging` call configures the root logger with a console
handler and an optional rotating file handler. Modules obtain loggers via
:func:`get_logger` (a thin wrapper over :func:`logging.getLogger`) so that the
formatting/handlers are defined in exactly one place.

The configuration is idempotent: calling :func:`setup_logging` more than once
replaces the handlers instead of stacking duplicates.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def setup_logging(
    level: str | int = "INFO",
    log_dir: str | Path | None = None,
    *,
    log_filename: str = "app.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger and return it.

    Parameters
    ----------
    level:
        Logging level name (e.g. ``"INFO"``) or numeric level.
    log_dir:
        If given, a rotating file handler writing to ``log_dir/log_filename``
        is added. The directory is created if necessary.
    force:
        Reconfigure even if logging was already set up in this process.
    """
    global _CONFIGURED

    root = logging.getLogger()
    if _CONFIGURED and not force:
        return root

    numeric_level = logging.getLevelName(level) if isinstance(level, str) else level
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level!r}")

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Remove any previously installed handlers so repeated calls don't duplicate.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            directory / log_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(numeric_level)
    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Ensures a basic console configuration exists even if :func:`setup_logging`
    was never called, so library code can log without a hard dependency on the
    application having configured logging first.
    """
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
