"""
app/utils/text.py
Text manipulation helpers used across the app.
"""
import re


def strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from LLM output.

    Handles:
        ```json ... ```
        ``` ... ```
        plain text (unchanged)
    """
    text = text.strip()
    # Remove opening fence (```json, ```python, ``` etc.)
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    # Remove closing fence
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def safe_html(text: str) -> str:
    """Escape HTML special characters to prevent injection / rendering issues."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def markdown_bold_to_html(text: str) -> str:
    """Convert **bold** markdown syntax to <b>bold</b> HTML tags."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending an ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated for processing ...]"
