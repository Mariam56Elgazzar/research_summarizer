"""
app/services/marker_service.py

PDF → structured markdown using the `marker` library (by VikParuchuri).

Why marker instead of raw PyMuPDF text:
  - Preserves document structure: headings, paragraphs, lists
  - Converts inline and block equations to proper LaTeX  (e.g. $$\\text{Attention}(Q,K,V)=..$$)
  - Handles multi-column layouts correctly
  - Identifies and labels figure/table regions
  - Works on CPU, no GPU needed
  - Completely free / open source

marker output is a single markdown string.  We post-process it to:
  1. Extract LaTeX equations ($$...$$) before they get mangled downstream
  2. Split into logical sections by heading detection
  3. Return a MarkerResult the pipeline can use directly
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── regex helpers ─────────────────────────────────────────────────────────────
_BLOCK_EQ_RE  = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
_INLINE_EQ_RE = re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)')
_HEADING_RE   = re.compile(r'^#{1,4}\s+(.+)$', re.MULTILINE)

# Map common section heading words to our canonical field names
_SECTION_MAP = {
    "abstract":       "abstract",
    "introduction":   "introduction",
    "related work":   "introduction",   # fold into intro
    "background":     "introduction",
    "method":         "methodology",
    "methodology":    "methodology",
    "approach":       "methodology",
    "model":          "methodology",
    "experiment":     "results",
    "result":         "results",
    "evaluation":     "results",
    "discussion":     "results",
    "conclusion":     "conclusion",
    "concluding":     "conclusion",
    "limitation":     "limitations",
    "future":         "future_work",
}


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class ExtractedLatexEquation:
    latex: str
    is_block: bool       # True = $$...$$, False = inline $...$
    context: str = ""    # surrounding text snippet for description later


@dataclass
class MarkerResult:
    """Structured output from marker PDF conversion."""
    full_markdown: str                    # complete marker output
    sections: dict[str, str]             # canonical section name → text
    equations: list[ExtractedLatexEquation] = field(default_factory=list)
    title: str = ""
    authors: str = ""

    @property
    def full_text_for_llm(self) -> str:
        """Clean text to feed to the LLM — equations preserved as LaTeX."""
        return self.full_markdown


# ── internal helpers ──────────────────────────────────────────────────────────

def _extract_equations(markdown: str) -> list[ExtractedLatexEquation]:
    """Pull all LaTeX equations from marker output."""
    equations: list[ExtractedLatexEquation] = []

    # Block equations  $$...$$
    for m in _BLOCK_EQ_RE.finditer(markdown):
        latex = m.group(1).strip()
        if latex and len(latex) > 2:
            # Grab ≤80 chars of surrounding text as context
            start = max(0, m.start() - 80)
            ctx   = markdown[start:m.start()].strip().replace("\n", " ")[-60:]
            equations.append(ExtractedLatexEquation(
                latex=latex, is_block=True, context=ctx
            ))

    # Inline equations $...$ (only those with math-like content)
    for m in _INLINE_EQ_RE.finditer(markdown):
        latex = m.group(1).strip()
        # Filter: must contain a math operator or Greek letter to avoid false positives
        if latex and re.search(r'[=+\-*/^_\\{}]|\\[a-zA-Z]', latex):
            if len(latex) > 2:
                equations.append(ExtractedLatexEquation(
                    latex=latex, is_block=False
                ))

    return equations


def _split_into_sections(markdown: str) -> dict[str, str]:
    """
    Split marker markdown into canonical section buckets.

    Strategy:
      - Find all headings with their positions
      - Assign the text between heading[i] and heading[i+1] to that section
      - Map heading text to canonical names via _SECTION_MAP
    """
    sections: dict[str, str] = {
        "abstract": "", "introduction": "", "methodology": "",
        "results": "", "conclusion": "", "limitations": "", "future_work": "",
    }

    # Find all headings and their byte offsets
    headings: list[tuple[int, int, str]] = []  # (start, end, text)
    for m in _HEADING_RE.finditer(markdown):
        headings.append((m.start(), m.end(), m.group(1).strip()))

    if not headings:
        # No headings found — treat everything as introduction
        sections["introduction"] = markdown[:8000]
        return sections

    # Add a sentinel at the end
    headings.append((len(markdown), len(markdown), ""))

    for i, (hstart, hend, htext) in enumerate(headings[:-1]):
        next_hstart = headings[i + 1][0]
        body = markdown[hend:next_hstart].strip()

        # Map heading to canonical section
        key = _canonical_section(htext)
        if key and body:
            # Append (multiple headings can map to same section)
            existing = sections.get(key, "")
            sections[key] = (existing + "\n\n" + body).strip() if existing else body

    return sections


def _canonical_section(heading: str) -> str | None:
    """Map a heading string to a canonical section name, or None if irrelevant."""
    h = heading.lower().strip()
    for keyword, canonical in _SECTION_MAP.items():
        if keyword in h:
            return canonical
    return None


def _extract_title_authors(markdown: str) -> tuple[str, str]:
    """Heuristically extract title and authors from the start of marker output."""
    lines = [l.strip() for l in markdown.splitlines() if l.strip()]

    title   = ""
    authors = ""

    for i, line in enumerate(lines[:15]):
        # First H1 heading is usually the title
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        # A line with multiple comma-separated proper nouns near the top = authors
        elif (not authors and i < 10 and "," in line
              and not line.startswith("#")
              and len(line) < 300
              and not any(w in line.lower() for w in ["abstract", "doi", "http", "arxiv"])):
            authors = line

    return title, authors


# ── public API ────────────────────────────────────────────────────────────────

def convert_pdf(pdf_bytes: bytes) -> MarkerResult:
    """
    Convert a PDF to structured markdown using marker.

    Falls back to a plain PyMuPDF extraction if marker is not installed
    or fails, so the pipeline never breaks.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        MarkerResult with full markdown, sections dict, and extracted equations.
    """
    try:
        return _convert_with_marker(pdf_bytes)
    except ImportError:
        logger.warning("marker not installed — falling back to PyMuPDF text extraction. "
                       "Run: pip install marker-pdf")
        return _convert_with_pymupdf(pdf_bytes)
    except Exception as exc:
        logger.warning("marker conversion failed (%s) — falling back to PyMuPDF.", exc)
        return _convert_with_pymupdf(pdf_bytes)


def _convert_with_marker(pdf_bytes: bytes) -> MarkerResult:
    """Use marker-pdf library for high-quality conversion."""
    import tempfile, os
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models

    models = load_all_models()

    # marker needs a file path
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp_path = f.name

    try:
        full_markdown, _, _ = convert_single_pdf(tmp_path, models)
    finally:
        os.unlink(tmp_path)

    equations          = _extract_equations(full_markdown)
    sections           = _split_into_sections(full_markdown)
    title, authors     = _extract_title_authors(full_markdown)

    logger.info(
        "marker: %d chars, %d equations, sections: %s",
        len(full_markdown),
        len(equations),
        [k for k, v in sections.items() if v],
    )
    return MarkerResult(
        full_markdown=full_markdown,
        sections=sections,
        equations=equations,
        title=title,
        authors=authors,
    )


def _convert_with_pymupdf(pdf_bytes: bytes) -> MarkerResult:
    """Fallback: plain PyMuPDF text extraction."""
    import fitz
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp_path = f.name

    try:
        doc  = fitz.open(tmp_path)
        text = "\n\n".join(page.get_text("text") for page in doc)
        doc.close()
    finally:
        os.unlink(tmp_path)

    equations      = _extract_equations(text)   # won't find much without marker
    sections       = _split_into_sections(text)
    title, authors = _extract_title_authors(text)

    logger.info("PyMuPDF fallback: %d chars extracted.", len(text))
    return MarkerResult(
        full_markdown=text,
        sections=sections,
        equations=equations,
        title=title,
        authors=authors,
    )
