"""
app/services/gemini_vision_service.py

Vision analysis using Google Gemini 1.5 Flash.

Why Gemini 1.5 Flash instead of llama-4-scout:
  - 1M token context window (can see the WHOLE paper at once)
  - Significantly better at reading academic figures and diagrams
  - Substantially better LaTeX equation transcription from images
  - Free tier: 15 requests/minute, 1500 requests/day
  - No GPU needed — it's an API call

Setup:
  - Get a free API key at https://aistudio.google.com/app/apikey
  - Add GEMINI_API_KEY to your .env file

Falls back gracefully to the Groq vision model if no Gemini key is set.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

GEMINI_MODEL    = "gemini-1.5-flash"
GEMINI_API_URL  = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


# ── Data structures (same as vision_service for compatibility) ───────────────

@dataclass
class ExtractedFigure:
    page_number: int
    caption:     str
    description: str
    png_b64:     str   # cropped figure PNG


@dataclass
class ExtractedEquation:
    page_number: int
    latex:       str
    description: str


@dataclass
class VisionAnalysis:
    figures:        list[ExtractedFigure]   = field(default_factory=list)
    equations:      list[ExtractedEquation] = field(default_factory=list)
    page_summaries: dict[int, str]          = field(default_factory=dict)


# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert academic document analyst specialising in mathematics, machine learning, \
and scientific visualisation. You extract precise structured information from research paper pages.
Always respond with valid JSON only. No markdown fences, no commentary."""


def _page_prompt(page_number: int) -> str:
    return f"""\
Analyse this research paper page (page {page_number}) with high precision.

Extract:
1. EQUATIONS — transcribe every visible mathematical expression in LaTeX. \
Be precise: include superscripts, subscripts, fractions, summations, Greek letters. \
Do NOT paraphrase; write the exact LaTeX.
2. FIGURES — for each diagram, chart, or graph: copy its exact caption text, \
then write one sentence describing what a reader visually sees.
3. PAGE SUMMARY — 1–2 sentences on what this page covers.

Return ONLY this JSON (no other text):
{{
  "equations": [
    {{"latex": "\\\\text{{Attention}}(Q,K,V)=\\\\text{{softmax}}\\\\!\\\\left(\\\\frac{{QK^\\\\top}}{{\\\\sqrt{{d_k}}}}\\\\right)V",
      "description": "Scaled dot-product attention formula"}}
  ],
  "figures": [
    {{"caption": "Figure 2: Scaled Dot-Product Attention",
      "description": "Block diagram showing Q, K, V inputs flowing through MatMul, Scale, Mask, SoftMax, then another MatMul to output"}}
  ],
  "page_summary": "Introduces the scaled dot-product attention mechanism and compares it to additive attention."
}}

Rules:
- equations.latex: raw LaTeX, no surrounding $ signs
- figures.caption: copy EXACTLY as printed (e.g. "Figure 2: ...")
- figures.description: describe what you SEE, not just what the caption says
- Return empty [] if nothing found for that category"""


# ── HTTP call (no SDK needed — just requests) ─────────────────────────────────

def _call_gemini(api_key: str, page_number: int, png_b64: str) -> dict:
    """Call Gemini 1.5 Flash with a page image and return parsed JSON."""
    import urllib.request
    import urllib.error

    url = (GEMINI_API_URL.format(model=GEMINI_MODEL)
           + f"?key={api_key}")

    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": png_b64,
                    }
                },
                {"text": _page_prompt(page_number)},
            ]
        }],
        "generationConfig": {
            "temperature":    0.1,
            "maxOutputTokens": 2048,
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()
        logger.warning("Gemini HTTP %d for page %d: %s", e.code, page_number, body_text[:300])
        return {"equations": [], "figures": [], "page_summary": ""}
    except Exception as exc:
        logger.warning("Gemini call failed for page %d: %s", page_number, exc)
        return {"equations": [], "figures": [], "page_summary": ""}

    # Extract text from Gemini response structure
    try:
        text = raw["candidates"][0]["content"]["parts"][0]["text"]
        # Strip any accidental markdown fences
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:])
        if text.endswith("```"):
            text = "\n".join(text.splitlines()[:-1])
        return json.loads(text.strip())
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.warning("Gemini response parse failed for page %d: %s", page_number, exc)
        return {"equations": [], "figures": [], "page_summary": ""}


# ── Cropped figure matching (same logic as vision_service) ───────────────────

def _match_cropped(
    caption: str,
    page_number: int,
    cropped_figures: list,
) -> str:
    import re
    caption_lower = caption.lower()
    same_page     = [c for c in cropped_figures if c.page_number == page_number]

    for crop in same_page:
        nums_llm  = set(re.findall(r'\d+', caption_lower))
        nums_crop = set(re.findall(r'\d+', crop.caption.lower()))
        if nums_llm & nums_crop:
            return crop.png_b64

    return same_page[0].png_b64 if same_page else ""


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_key_pages_gemini(
    key_pages: list,
    all_cropped_figures: list,
    gemini_api_key: str,
) -> VisionAnalysis:
    """
    Run Gemini 1.5 Flash vision analysis on key pages.

    Args:
        key_pages:           PageData list (with full-page png_b64).
        all_cropped_figures: CroppedFigure list from pdf_parser.
        gemini_api_key:      Google AI Studio API key.

    Returns:
        VisionAnalysis with equations, figures (with cropped PNGs), page summaries.
    """
    result = VisionAnalysis()

    for page in key_pages:
        if not page.png_b64:
            continue

        logger.info("Gemini vision: page %d ...", page.page_number)
        data = _call_gemini(gemini_api_key, page.page_number, page.png_b64)

        for eq in data.get("equations", []):
            latex = eq.get("latex", "").strip()
            if latex and len(latex) > 2:
                result.equations.append(ExtractedEquation(
                    page_number=page.page_number,
                    latex=latex,
                    description=eq.get("description", "").strip(),
                ))

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
        "Gemini vision done: %d equations, %d figures (%d with images).",
        len(result.equations),
        len(result.figures),
        sum(1 for f in result.figures if f.png_b64),
    )
    return result
