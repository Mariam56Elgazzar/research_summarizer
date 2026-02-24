"""
app/services/vision_service.py

Vision analysis of PDF page images.

Flow:
  - Full page images  → sent to LLM for equation + figure metadata extraction
  - Cropped figures   → stored separately from pdf_parser (actual diagram crops)
  - We match LLM-described figures to cropped figures by page number + caption
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from groq import Groq

from app.core.config import Settings
from app.services.pdf_parser import PageData, CroppedFigure
from app.utils.text import strip_code_fences

logger = logging.getLogger(__name__)

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


@dataclass
class ExtractedFigure:
    page_number: int
    caption:     str   # e.g. "Figure 2: Scaled Dot-Product Attention"
    description: str   # what the figure shows (1–2 sentences)
    png_b64:     str   # CROPPED figure image (not full page)


@dataclass
class ExtractedEquation:
    page_number: int
    latex:       str
    description: str


@dataclass
class VisionAnalysis:
    figures:       list[ExtractedFigure]   = field(default_factory=list)
    equations:     list[ExtractedEquation] = field(default_factory=list)
    page_summaries: dict[int, str]         = field(default_factory=dict)


# ── Prompts ──────────────────────────────────────────────────────────────────

VISION_SYSTEM_PROMPT = (
    "You are an expert academic document analyst. "
    "Analyze research paper pages and extract structured information. "
    "Always respond with valid JSON only. No markdown, no explanation, no code fences."
)


def _build_vision_user_prompt(page_number: int) -> str:
    return f"""Analyze this research paper page (page {page_number}).

Extract:
1. All mathematical equations → write in LaTeX
2. Any figures/diagrams/graphs → provide caption (exactly as written in the paper) and a 1-sentence description of what it shows
3. A 1–2 sentence page summary

Return ONLY this JSON (no other text):
{{
  "equations": [
    {{"latex": "E = mc^2", "description": "Mass-energy equivalence"}}
  ],
  "figures": [
    {{"caption": "Figure 2: Scaled Dot-Product Attention", "description": "Architecture diagram showing Q, K, V inputs through matmul, scale, softmax, and output"}}
  ],
  "page_summary": "This page introduces scaled dot-product attention..."
}}

Rules:
- equations: LaTeX only, no surrounding $
- figures.caption: copy the exact figure label from the page (e.g. "Figure 1: ...")
- figures.description: 1 sentence — what a reader SEES, not just the label
- Return empty lists [] if nothing found"""


# ── Internal ─────────────────────────────────────────────────────────────────

def _analyze_page(client: Groq, page: PageData, settings: Settings) -> dict:
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{page.png_b64}"},
                        },
                        {"type": "text", "text": _build_vision_user_prompt(page.page_number)},
                    ],
                },
            ],
            max_tokens=1200,
            temperature=0.1,
        )
        raw     = response.choices[0].message.content or ""
        cleaned = strip_code_fences(raw)
        return json.loads(cleaned)

    except json.JSONDecodeError as exc:
        logger.warning("Vision JSON parse failed for page %d: %s", page.page_number, exc)
        return {"equations": [], "figures": [], "page_summary": ""}
    except Exception as exc:
        logger.warning("Vision call failed for page %d: %s", page.page_number, exc)
        return {"equations": [], "figures": [], "page_summary": ""}


def _match_cropped(
    caption: str,
    page_number: int,
    cropped_figures: list[CroppedFigure],
) -> str:
    """
    Find the best cropped figure PNG for an LLM-described figure.

    Strategy:
      1. Same page + caption keyword overlap  (best)
      2. Same page, any cropped figure        (fallback)
      3. Empty string                         (no match)
    """
    caption_lower = caption.lower()
    same_page = [c for c in cropped_figures if c.page_number == page_number]

    # Try caption keyword match
    for crop in same_page:
        crop_cap = crop.caption.lower()
        # Check if they share the figure number  e.g. "figure 2" in both
        import re
        nums_llm  = set(re.findall(r'\d+', caption_lower))
        nums_crop = set(re.findall(r'\d+', crop_cap))
        if nums_llm & nums_crop:   # shared figure number
            return crop.png_b64

    # Fallback: first on same page
    if same_page:
        return same_page[0].png_b64

    return ""


# ── Public API ───────────────────────────────────────────────────────────────

def analyze_key_pages(
    key_pages: list[PageData],
    all_cropped_figures: list[CroppedFigure],
    client: Groq,
    settings: Settings,
) -> VisionAnalysis:
    """Run vision analysis and attach CROPPED figure images to results.

    Args:
        key_pages:           Pages with full-page PNGs for the vision model.
        all_cropped_figures: Cropped figure images from the PDF parser.
        client:              Authenticated Groq client.
        settings:            App settings.
    """
    result = VisionAnalysis()

    for page in key_pages:
        if not page.png_b64:
            continue

        logger.info("Vision: analysing page %d ...", page.page_number)
        data = _analyze_page(client, page, settings)

        # Equations
        for eq in data.get("equations", []):
            latex = eq.get("latex", "").strip()
            if latex:
                result.equations.append(ExtractedEquation(
                    page_number=page.page_number,
                    latex=latex,
                    description=eq.get("description", "").strip(),
                ))

        # Figures — attach cropped image
        for fig in data.get("figures", []):
            caption = fig.get("caption", "").strip()
            desc    = fig.get("description", "").strip()
            if not (caption or desc):
                continue
            png = _match_cropped(caption, page.page_number, all_cropped_figures)
            result.figures.append(ExtractedFigure(
                page_number=page.page_number,
                caption=caption,
                description=desc,
                png_b64=png,
            ))

        summary = data.get("page_summary", "").strip()
        if summary:
            result.page_summaries[page.page_number] = summary

    logger.info(
        "Vision done: %d equations, %d figures (%d with cropped images).",
        len(result.equations),
        len(result.figures),
        sum(1 for f in result.figures if f.png_b64),
    )
    return result
