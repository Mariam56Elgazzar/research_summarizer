"""
app/core/logging_config.py

Central logging configuration for the application.

Features:
  - Coloured console output (INFO and above)
  - Rotating file log  → logs/app.log  (DEBUG and above)
  - Single call: setup_logging() — call once at app startup
  - Respects LOG_LEVEL from .env / Settings
  - Suppresses noisy third-party loggers (fitz, PIL, httpx, urllib3)
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path


# ── ANSI colour codes ─────────────────────────────────────────────────────────
_RESET  = "\x1b[0m"
_BOLD   = "\x1b[1m"
_GREY   = "\x1b[38;5;240m"
_CYAN   = "\x1b[36m"
_YELLOW = "\x1b[33m"
_RED    = "\x1b[31m"
_BRED   = "\x1b[1;31m"

_LEVEL_COLOURS = {
    logging.DEBUG:    _GREY,
    logging.INFO:     _CYAN,
    logging.WARNING:  _YELLOW,
    logging.ERROR:    _RED,
    logging.CRITICAL: _BRED,
}


class _ColouredFormatter(logging.Formatter):
    """Formatter that adds ANSI colours to level name and module path."""

    FMT = "{colour}{level:<8}{reset} {grey}{name}{reset}  {msg}"

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelno, "")
        level  = record.levelname
        name   = record.name          # e.g. app.services.pdf_parser
        msg    = record.getMessage()

        # Format exception info if present
        exc = ""
        if record.exc_info:
            exc = "\n" + self.formatException(record.exc_info)

        return (
            self.FMT.format(
                colour=colour, level=level, reset=_RESET,
                grey=_GREY, name=name, msg=msg,
            )
            + exc
        )


class _PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no ANSI codes)."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# ── Noisy third-party loggers to silence ─────────────────────────────────────
_QUIET_LOGGERS = [
    "fitz",           # PyMuPDF: very verbose
    "PIL",            # Pillow
    "httpx",          # Groq SDK uses httpx
    "httpcore",
    "urllib3",
    "urllib3.connectionpool",
    "google",
    "google.auth",
    "marker",         # marker-pdf internal steps
    "transformers",   # HuggingFace (if marker uses it)
    "torch",
    "filelock",
    "matplotlib",
]


def setup_logging(
    log_level: str = "INFO",
    log_dir:   str = "logs",
    log_file:  str = "app.log",
    enable_file_log: bool = True,
) -> None:
    """
    Configure application-wide logging.

    Call once at startup (app.py or streamlit_ui.py entry point).

    Args:
        log_level:       Root log level string ("DEBUG", "INFO", "WARNING", …).
        log_dir:         Directory for log files (created if missing).
        log_file:        Log filename inside log_dir.
        enable_file_log: Whether to write logs to a rotating file.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # ── Root logger ───────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # capture everything; handlers filter

    # Clear any handlers already attached (Streamlit re-runs call setup again)
    if root.handlers:
        root.handlers.clear()

    # ── Console handler (coloured, INFO+) ─────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(_ColouredFormatter())
    root.addHandler(console)

    # ── File handler (plain, DEBUG+, rotating 5 MB × 3 backups) ──────
    if enable_file_log:
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_path / log_file,
                maxBytes=5 * 1024 * 1024,   # 5 MB
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(_PlainFormatter())
            root.addHandler(file_handler)
        except PermissionError:
            # Read-only filesystem (e.g. some cloud deployments) — skip file log
            logging.getLogger(__name__).warning(
                "Cannot write log file to '%s' — file logging disabled.", log_dir
            )

    # ── Silence noisy third-party loggers ─────────────────────────────
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # ── Confirm setup ─────────────────────────────────────────────────
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured: level=%s, file=%s",
        log_level.upper(),
        str(Path(log_dir) / log_file) if enable_file_log else "disabled",
    )