"""
app.py — entry point. Keep this tiny.
"""
from app.core.config import get_settings
from app.core.logger import setup_logging

# Initialise logging before anything else is imported
_settings = get_settings()
setup_logging(
    log_level=_settings.log_level,   # from .env LOG_LEVEL, default "INFO"
    log_dir="logs",
    enable_file_log=True,
)

from app.ui.streamlit_ui import render_app  # noqa: E402 (import after logging setup)

render_app()