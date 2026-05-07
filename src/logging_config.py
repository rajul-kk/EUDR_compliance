"""
Central logging configuration for the EUDR pipeline.

Usage in every module:
    import logging
    logger = logging.getLogger(__name__)

Call setup_pipeline_logging() once at the entry point (main_audit.py).
All other modules just get a logger by name — they inherit the root config.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def setup_pipeline_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = "reports/pipeline.log",
) -> None:
    """Configure root logger for the full pipeline run.

    Args:
        level:    Logging level (default INFO). Pass logging.DEBUG for verbose output.
        log_file: Path to write a persistent log file alongside console output.
                  Pass None to disable file logging.
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)

    # Quieten noisy third-party loggers
    for noisy in ("urllib3", "requests", "fiona", "rasterio", "pyproj"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper — equivalent to logging.getLogger(name)."""
    return logging.getLogger(name)
