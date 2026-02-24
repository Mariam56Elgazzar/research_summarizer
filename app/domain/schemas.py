"""
app/domain/schemas.py
Pydantic schemas for API I/O validation (e.g. Streamlit inputs, pipeline results).
"""
from pydantic import BaseModel, field_validator


class UploadInput(BaseModel):
    """Validated input for a PDF upload + API key."""

    filename: str
    pdf_bytes: bytes
    api_key: str

    @field_validator("api_key")
    @classmethod
    def api_key_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("API key must not be empty")
        return v.strip()

    @field_validator("pdf_bytes")
    @classmethod
    def pdf_must_not_be_empty(cls, v: bytes) -> bytes:
        if not v:
            raise ValueError("PDF bytes must not be empty")
        return v

    model_config = {"arbitrary_types_allowed": True}


class PipelineSummary(BaseModel):
    """Lightweight summary of pipeline output for display purposes."""

    title: str
    authors: str
    summary_markdown: str
    has_report: bool
