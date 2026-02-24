"""
app/services/pdf_parser.py

PDF parser with robust figure extraction.

ROOT CAUSE OF THE BUG:
  The Attention Is All You Need paper (and most academic PDFs) use VECTOR
  graphics (PDF drawing commands / paths), not embedded raster images.
  The old code only looked for raster image blocks (type == 1) and gave up
  immediately when none were found — which is why figures were missing.

NEW STRATEGY — caption-first detection:
  1. Find every "Figure N:" caption on the page (text search)
  2. For each caption, determine whether the figure is ABOVE or BELOW it
  3. Get all vector drawing paths on the page via page.get_drawings()
     (this captures boxes, arrows, flow charts, attention diagrams, etc.)
  4. Build a bounding box covering ALL drawings in the figure's column/region
  5. Also handle raster images (type == 1 blocks) as before
  6. Render the combined bbox → PNG crop

This works for:
  - Vector diagrams (flow charts, architecture diagrams) ← the main fix
  - Embedded raster images (photos, plots saved as PNG/JPG)
  - Mixed pages (some vector, some raster)
"""
from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from app.core.exceptions import PDFExtractionError
from app.utils.files import safe_unlink, write_temp_pdf

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FIGURE_CAPTION_RE = re.compile(
    r"^\s*(figure|fig\.?)\s*\d", re.IGNORECASE
)
MATH_SYMBOLS  = set("∑∏∫∂∇αβγδεζηθλμνπρσφψωΩΓΔΘΛΞΠΣΦΨ≈≠≤≥←→⇒⇔∈∉∅∞±×÷√")
MAX_VISION_PAGES = 8
FULL_PAGE_DPI    = 150
CROP_DPI         = 200   # higher DPI for crisp vector figure crops
CROP_PAD         = 8     # pts of padding around crop rect

# Minimum drawing element size to count as part of a figure
# (filters out page borders, underlines, horizontal rules)
MIN_DRAWING_W = 10
MIN_DRAWING_H = 10


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CroppedFigure:
    page_number: int
    caption:     str    # "Figure 2: Scaled Dot-Product Attention"
    png_b64:     str    # base64 PNG of the cropped figure region


@dataclass
class PageData:
    page_number:     int
    text:            str
    has_figure_keyword: bool
    math_density:    float
    image_count:     int
    is_key_page:     bool = False
    png_b64:         str  = ""
    cropped_figures: list[CroppedFigure] = field(default_factory=list)


@dataclass
class ParsedPDF:
    pages:     list[PageData] = field(default_factory=list)
    full_text: str            = ""
    key_pages: list[PageData] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def all_cropped_figures(self) -> list[CroppedFigure]:
        out = []
        for p in self.pages:
            out.extend(p.cropped_figures)
        return out


# ── Render helpers ────────────────────────────────────────────────────────────

def _render_rect(page: fitz.Page, rect: fitz.Rect, dpi: int) -> str:
    """Render a rect region of a page → base64 PNG."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


def _render_full_page(page: fitz.Page, dpi: int = FULL_PAGE_DPI) -> str:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


# ── Core figure extraction ────────────────────────────────────────────────────

def _collect_text_blocks(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    """Return (bbox, text) for every non-empty text block on the page."""
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
    result: list[tuple[fitz.Rect, str]] = []
    for blk in blocks:
        if blk.get("type") != 0:
            continue
        text = " ".join(
            span["text"]
            for line in blk.get("lines", [])
            for span in line.get("spans", [])
        ).strip()
        if text:
            result.append((fitz.Rect(blk["bbox"]), text))
    return result


def _collect_raster_blocks(page: fitz.Page) -> list[fitz.Rect]:
    """Bounding boxes of embedded raster images (type == 1 blocks)."""
    blocks = page.get_text("dict").get("blocks", [])
    rects: list[fitz.Rect] = []
    for blk in blocks:
        if blk.get("type") == 1:
            r = fitz.Rect(blk["bbox"])
            if r.width > MIN_DRAWING_W and r.height > MIN_DRAWING_H:
                rects.append(r)
    return rects


def _collect_drawing_rects(page: fitz.Page) -> list[fitz.Rect]:
    """
    Bounding boxes of all vector drawing elements on the page.

    page.get_drawings() returns every PDF path: rectangles, lines,
    curves, beziers — the building blocks of flow charts, architecture
    diagrams, attention visualisations, etc.
    """
    rects: list[fitz.Rect] = []
    try:
        for d in page.get_drawings():
            r = fitz.Rect(d["rect"])
            if r.width > MIN_DRAWING_W and r.height > MIN_DRAWING_H:
                rects.append(r)
    except Exception:
        pass
    return rects


def _union_rects(rects: list[fitz.Rect]) -> fitz.Rect | None:
    """Union a list of rects into one bounding box."""
    if not rects:
        return None
    result = rects[0]
    for r in rects[1:]:
        result = result | r
    return result


def _figure_is_above_caption(
    caption_rect: fitz.Rect,
    page_height:  float,
) -> bool:
    """
    Heuristic: is the figure drawn above its caption?
    In most academic papers, yes. Captions near the bottom of their
    region suggest the figure is above.
    """
    # If caption is in the top 30% of the page, figure is probably below
    return caption_rect.y0 > page_height * 0.30


def _crop_for_caption(
    caption_rect:  fitz.Rect,
    caption_text:  str,
    page:          fitz.Page,
    drawing_rects: list[fitz.Rect],
    raster_rects:  list[fitz.Rect],
    text_blocks:   list[tuple[fitz.Rect, str]],
) -> fitz.Rect | None:
    """
    Given a caption rect, find the figure region that belongs to it.

    Strategy:
      1. Determine search zone: above OR below the caption
      2. Collect all drawing/raster rects in that zone (same horizontal band)
      3. Union them into one crop rect
      4. Fall back to a fixed-height region above caption if nothing found
    """
    page_rect = page.rect
    fig_above = _figure_is_above_caption(caption_rect, page_rect.height)

    # Horizontal band: allow full page width (handles two-column layouts)
    x0 = page_rect.x0
    x1 = page_rect.x1

    if fig_above:
        # Search from page top (or previous caption bottom) to caption top
        search_zone = fitz.Rect(x0, page_rect.y0, x1, caption_rect.y0)
    else:
        # Search from caption bottom to next content
        search_zone = fitz.Rect(x0, caption_rect.y1, x1, page_rect.y1)

    # Collect all drawing elements in the search zone
    zone_rects: list[fitz.Rect] = []
    for r in drawing_rects + raster_rects:
        inter = r & search_zone
        if not inter.is_empty and inter.width > MIN_DRAWING_W and inter.height > MIN_DRAWING_H:
            zone_rects.append(r)

    if zone_rects:
        combined = _union_rects(zone_rects)
        if combined and combined.width > 20 and combined.height > 20:
            # Include the caption itself
            combined = combined | caption_rect
            return combined

    # ── Fallback: fixed region above/below the caption ────────────────
    # Use when the figure has no detectable drawing elements
    # (e.g. purely text-based tables, or very sparse vector art)
    fallback_h = min(200, caption_rect.y0 - page_rect.y0) if fig_above else 200
    if fig_above and fallback_h > 40:
        return fitz.Rect(
            caption_rect.x0 - 20,
            caption_rect.y0 - fallback_h,
            caption_rect.x1 + 20,
            caption_rect.y1,
        )

    return None


def _extract_cropped_figures(page: fitz.Page, page_number: int) -> list[CroppedFigure]:
    """
    Extract figure crops from a single page.

    1. Find all "Figure N:" captions
    2. For each caption, determine the figure region (vector or raster)
    3. Render the region to a cropped PNG
    """
    figures:      list[CroppedFigure] = []
    page_rect     = page.rect
    text_blocks   = _collect_text_blocks(page)
    drawing_rects = _collect_drawing_rects(page)
    raster_rects  = _collect_raster_blocks(page)

    # Find caption text blocks
    captions: list[tuple[fitz.Rect, str]] = [
        (r, t) for r, t in text_blocks
        if FIGURE_CAPTION_RE.match(t)
    ]

    if not captions:
        logger.debug("Page %d: no figure captions found.", page_number)
        return figures

    logger.debug(
        "Page %d: %d caption(s), %d drawings, %d rasters.",
        page_number, len(captions), len(drawing_rects), len(raster_rects),
    )

    for cap_rect, cap_text in captions:
        crop_rect = _crop_for_caption(
            caption_rect=cap_rect,
            caption_text=cap_text,
            page=page,
            drawing_rects=drawing_rects,
            raster_rects=raster_rects,
            text_blocks=text_blocks,
        )

        if crop_rect is None:
            logger.debug("Page %d: could not determine crop for '%s'", page_number, cap_text[:40])
            continue

        # Clamp to page, add padding
        crop_rect = fitz.Rect(
            max(crop_rect.x0 - CROP_PAD, page_rect.x0),
            max(crop_rect.y0 - CROP_PAD, page_rect.y0),
            min(crop_rect.x1 + CROP_PAD, page_rect.x1),
            min(crop_rect.y1 + CROP_PAD, page_rect.y1),
        )

        if crop_rect.width < 20 or crop_rect.height < 20:
            continue

        png_b64 = _render_rect(page, crop_rect, dpi=CROP_DPI)
        figures.append(CroppedFigure(
            page_number=page_number,
            caption=cap_text.strip(),
            png_b64=png_b64,
        ))
        logger.debug("Page %d: cropped figure '%s' (%.0fx%.0f pt)",
                     page_number, cap_text[:30], crop_rect.width, crop_rect.height)

    return figures


# ── Scoring for key page selection ────────────────────────────────────────────

def _math_density(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if ch in MATH_SYMBOLS) / len(text)


def _score_page(pd: PageData) -> float:
    score = 0.0
    if pd.has_figure_keyword:
        score += 4.0
    if pd.cropped_figures:                      # found actual figures
        score += 3.0 * len(pd.cropped_figures)
    if pd.image_count > 0:
        score += 2.0 * min(pd.image_count, 3)
    score += _math_density(pd.text) * 20.0
    return score


# ── Public API ────────────────────────────────────────────────────────────────

def parse_pdf(pdf_bytes: bytes) -> ParsedPDF:
    """Parse PDF: full text + cropped figure images + key pages for vision.

    Returns:
        ParsedPDF with all per-page data, figures cropped from vector/raster
        graphics, and key pages rendered for the vision model.
    """
    tmp_path = write_temp_pdf(pdf_bytes)
    try:
        doc   = fitz.open(tmp_path)
        pages: list[PageData] = []

        for i, fitz_page in enumerate(doc):
            page_num = i + 1
            text     = fitz_page.get_text("text")
            img_list = fitz_page.get_images(full=False)
            cropped  = _extract_cropped_figures(fitz_page, page_num)

            pd = PageData(
                page_number=page_num,
                text=text,
                has_figure_keyword=bool(
                    re.search(r"\b(figure|fig\.|table|equation|eq\.)\b", text, re.I)
                ),
                math_density=_math_density(text),
                image_count=len(img_list),
                cropped_figures=cropped,
            )
            pages.append(pd)

        if not pages:
            raise PDFExtractionError("PDF appears to have no pages.")

        full_text = "\n\n".join(p.text for p in pages).strip()
        if len(full_text) < 100:
            raise PDFExtractionError(
                "Very little text extracted — PDF may be scanned or image-only."
            )

        # ── Key page selection for vision model ───────────────────────
        key_pages_set: set[int] = {1}
        if len(pages) > 1:
            key_pages_set.add(len(pages))
        for pd in sorted(pages, key=_score_page, reverse=True)[:MAX_VISION_PAGES]:
            if _score_page(pd) > 0.5:
                key_pages_set.add(pd.page_number)

        key_page_list: list[PageData] = []
        for pd in pages:
            if pd.page_number in key_pages_set:
                pd.is_key_page = True
                pd.png_b64     = _render_full_page(doc[pd.page_number - 1])
                key_page_list.append(pd)

        doc.close()

        n_figs = sum(len(p.cropped_figures) for p in pages)
        logger.info(
            "Parsed PDF: %d pages, %d chars, %d figures cropped, %d key pages.",
            len(pages), len(full_text), n_figs, len(key_page_list),
        )

        return ParsedPDF(
            pages=pages,
            full_text=full_text,
            key_pages=sorted(key_page_list, key=lambda p: p.page_number),
        )

    except PDFExtractionError:
        raise
    except Exception as exc:
        raise PDFExtractionError("Failed to parse PDF.", original=exc) from exc
    finally:
        safe_unlink(tmp_path)
