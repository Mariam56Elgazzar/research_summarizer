"""
app/utils/files.py
File system helpers for temporary file management.
"""
import logging
import os
import tempfile

logger = logging.getLogger(__name__)


def write_temp_pdf(pdf_bytes: bytes) -> str:
    """Write PDF bytes to a named temporary file and return the file path.

    Caller is responsible for cleanup (use safe_unlink).
    """
    fd, path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(pdf_bytes)
    except Exception:
        os.close(fd)
        raise
    logger.debug("Wrote temp PDF to %s (%d bytes)", path, len(pdf_bytes))
    return path


def safe_unlink(path: str) -> None:
    """Delete a file, silently ignoring errors if it doesn't exist."""
    try:
        os.unlink(path)
        logger.debug("Deleted temp file: %s", path)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("Could not delete temp file %s: %s", path, exc)
