"""
logger.py
─────────
Centralised logger with automatic daily file rotation.
- Console output at INFO level.
- One log file per day: backend/logs/YYYY-MM-DD.log
  (rotation happens at midnight; keeps the last 30 days).
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from backend.config import LOGS_DIR

os.makedirs(LOGS_DIR, exist_ok=True)


def _daily_namer(default_name: str) -> str:
    """
    Rename rotated files to YYYY-MM-DD.log format.
    TimedRotatingFileHandler appends the date suffix after a dot by default;
    this namer moves the date to the start of the filename.
    """
    # default_name is like: /path/to/logs/2026-03-09.log.2026-03-08
    # We want:               /path/to/logs/2026-03-08.log
    base_dir = os.path.dirname(default_name)
    # The suffix appended by the handler is the date part after the last dot
    suffix = default_name.rsplit(".", 1)[-1]
    return os.path.join(base_dir, f"{suffix}.log")


def get_logger(name: str = "TumorAI") -> logging.Logger:
    """
    Return a named logger, configuring it only on first call.
    Subsequent calls return the cached instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured — skip re-init

    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent duplicate output in parent loggers

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ── Daily rotating file handler ───────────────────────────────────────────
    # Base filename uses today's date; at midnight Python renames it and starts
    # a new file for the new day.
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOGS_DIR, f"{today}.log")

    file_handler = TimedRotatingFileHandler(
        log_path,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.namer = _daily_namer  # produce YYYY-MM-DD.log on rotation
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# Module-level convenience instance
logger = get_logger()
