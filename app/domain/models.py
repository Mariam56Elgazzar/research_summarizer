"""
app/domain/models.py
Pydantic models representing core domain objects.
"""
from __future__ import annotations
from pydantic import BaseModel, Field

NOT_FOUND = "Not found"


class ExtractedEquation(BaseModel):
    page_number: int
    latex: str
    description: str


class ExtractedFigure(BaseModel):
    """A figure extracted from the PDF — image is a CROPPED region, not a full page."""
    page_number: int
    caption: str        # e.g. "Figure 2: Scaled Dot-Product Attention"
    description: str    # vision model's description of what the figure shows
    png_b64: str = ""   # base64 PNG of the cropped figure region only


class PaperSections(BaseModel):
    title:        str = Field(default=NOT_FOUND)
    authors:      str = Field(default=NOT_FOUND)
    abstract:     str = Field(default=NOT_FOUND)
    introduction: str = Field(default=NOT_FOUND)
    methodology:  str = Field(default=NOT_FOUND)
    results:      str = Field(default=NOT_FOUND)
    conclusion:   str = Field(default=NOT_FOUND)
    limitations:  str = Field(default=NOT_FOUND)
    future_work:  str = Field(default=NOT_FOUND)

    equations:     list[ExtractedEquation] = Field(default_factory=list)
    figures:       list[ExtractedFigure]   = Field(default_factory=list)
    page_summaries: dict[int, str]         = Field(default_factory=dict)

    @classmethod
    def empty(cls) -> "PaperSections":
        return cls()

    @classmethod
    def from_dict(cls, data: dict) -> "PaperSections":
        text_fields = {
            "title", "authors", "abstract", "introduction",
            "methodology", "results", "conclusion", "limitations", "future_work",
        }
        cleaned = {
            f: (data.get(f, "") if isinstance(data.get(f), str) and data.get(f, "").strip()
                else NOT_FOUND)
            for f in text_fields
        }
        return cls(**cleaned)


class PipelineResult(BaseModel):
    sections:         PaperSections
    summary_markdown: str
    report_pdf_bytes: bytes

    model_config = {"arbitrary_types_allowed": True}
