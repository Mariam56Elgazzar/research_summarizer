"""
app/services/report_service.py

Clean, well-organised PDF report generator.

Structure:
  ┌─────────────────────────────┐
  │  COVER PAGE                 │  title, authors, date, stats
  ├─────────────────────────────┤
  │  AT-A-GLANCE                │  abstract + key facts table
  ├─────────────────────────────┤
  │  OVERVIEW SECTIONS          │  summary markdown, rendered inline
  │  (equations as images)      │
  ├─────────────────────────────┤
  │  FIGURES REFERENCE          │  page images + captions
  ├─────────────────────────────┤
  │  EQUATIONS REFERENCE        │  rendered equation images + descriptions
  └─────────────────────────────┘
"""
from __future__ import annotations

import base64
import io
import logging
import re
from datetime import datetime
from typing import Callable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    HRFlowable,
    Table,
    TableStyle,
)

from app.core.config import get_settings
from app.core.exceptions import ReportGenerationError
from app.domain.models import ExtractedEquation, ExtractedFigure, PaperSections
from app.utils.equation_renderer import latex_to_png
from app.utils.text import markdown_bold_to_html, safe_html

logger = logging.getLogger(__name__)

# ── Page geometry ────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = LETTER
MARGIN        = 0.85 * inch
CONTENT_W     = PAGE_W - 2 * MARGIN

# ── Brand colours ────────────────────────────────────────────────────────────
C_NAVY   = colors.HexColor("#0f2557")
C_BLUE   = colors.HexColor("#2563eb")
C_LBLUE  = colors.HexColor("#dbeafe")
C_STEEL  = colors.HexColor("#64748b")
C_RULE   = colors.HexColor("#e2e8f0")
C_EQBG   = colors.HexColor("#f8faff")
C_TEXT   = colors.HexColor("#1e293b")
C_WHITE  = colors.white


# ════════════════════════════════════════════════════════════════════════════
# STYLES
# ════════════════════════════════════════════════════════════════════════════

def _styles() -> dict:
    base = getSampleStyleSheet()

    def _p(name, **kw) -> ParagraphStyle:
        parent = kw.pop("parent", base["Normal"])
        return ParagraphStyle(name, parent=parent, **kw)

    return {
        # Cover
        "cover_title":   _p("CoverTitle",   fontSize=24, textColor=C_NAVY,
                              fontName="Helvetica-Bold", leading=30,
                              alignment=TA_CENTER, spaceAfter=8),
        "cover_authors": _p("CoverAuthors", fontSize=11, textColor=C_STEEL,
                              alignment=TA_CENTER, spaceAfter=4),
        "cover_meta":    _p("CoverMeta",    fontSize=9,  textColor=C_STEEL,
                              alignment=TA_CENTER, spaceAfter=2),
        "cover_stat":    _p("CoverStat",    fontSize=10, textColor=C_NAVY,
                              alignment=TA_CENTER, fontName="Helvetica-Bold"),

        # Section headings
        "h1": _p("H1", fontSize=15, textColor=C_NAVY, fontName="Helvetica-Bold",
                  spaceBefore=18, spaceAfter=6, leading=20),
        "h2": _p("H2", fontSize=12, textColor=C_BLUE, fontName="Helvetica-Bold",
                  spaceBefore=12, spaceAfter=4, leading=16),
        "h3": _p("H3", fontSize=10, textColor=C_NAVY, fontName="Helvetica-Bold",
                  spaceBefore=8,  spaceAfter=3, leading=14),

        # Body
        "body":   _p("Body",   fontSize=10, leading=16, spaceAfter=6,
                      textColor=C_TEXT, alignment=TA_JUSTIFY),
        "bullet": _p("Bullet", fontSize=10, leading=15, spaceAfter=4,
                      textColor=C_TEXT, leftIndent=16, firstLineIndent=0),
        "italic": _p("Italic", fontSize=9,  leading=13, spaceAfter=4,
                      textColor=C_STEEL, fontName="Helvetica-Oblique"),

        # Equation caption
        "eq_label": _p("EqLabel", fontSize=9, textColor=C_BLUE,
                        fontName="Helvetica-Bold", spaceAfter=2),
        "eq_where": _p("EqWhere", fontSize=9, leading=13, spaceAfter=8,
                        textColor=C_STEEL, leftIndent=12,
                        fontName="Helvetica-Oblique"),

        # Figure caption
        "fig_caption": _p("FigCaption", fontSize=9, leading=13,
                           alignment=TA_CENTER, textColor=C_STEEL,
                           fontName="Helvetica-Oblique", spaceAfter=14),

        # Abstract box
        "abstract": _p("Abstract", fontSize=10, leading=16, spaceAfter=0,
                        textColor=C_TEXT, alignment=TA_JUSTIFY),

        # Footer
        "footer": _p("Footer", fontSize=8, textColor=C_STEEL, alignment=TA_CENTER),
    }


# ════════════════════════════════════════════════════════════════════════════
# CANVAS CALLBACKS  (header bar + footer)
# ════════════════════════════════════════════════════════════════════════════

def _draw_header_bar(canvas, doc, title: str) -> None:
    """Thin navy bar across the top of every page except the cover."""
    canvas.saveState()
    bar_h = 0.25 * inch
    canvas.setFillColor(C_NAVY)
    canvas.rect(0, PAGE_H - bar_h, PAGE_W, bar_h, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(C_WHITE)
    canvas.drawString(MARGIN, PAGE_H - bar_h + 5, title[:90])
    canvas.restoreState()


def _draw_footer(canvas, doc, author: str) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(C_STEEL)
    ts = datetime.now().strftime("%Y-%m-%d")
    canvas.drawString(MARGIN, 0.4 * inch, f"AI-generated overview  ·  {author}  ·  {ts}")
    canvas.drawRightString(PAGE_W - MARGIN, 0.4 * inch, f"Page {doc.page}")
    canvas.restoreState()


# ════════════════════════════════════════════════════════════════════════════
# COVER PAGE
# ════════════════════════════════════════════════════════════════════════════

def _cover_page(sections: PaperSections, styles: dict) -> list:
    story = []

    # Top spacer  (push content to vertical center)
    story.append(Spacer(1, 1.8 * inch))

    # Navy accent bar above title
    story.append(HRFlowable(width="100%", thickness=4, color=C_NAVY, spaceAfter=16))

    story.append(Paragraph(safe_html(sections.title), styles["cover_title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(safe_html(sections.authors), styles["cover_authors"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"AI Research Overview  ·  Generated {datetime.now().strftime('%B %d, %Y')}",
        styles["cover_meta"],
    ))

    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE, spaceAfter=20))

    # Stats table  (equations | figures | pages)
    n_eq  = len(sections.equations)
    n_fig = len(sections.figures)
    rows = [[
        Paragraph(f"<b>{n_eq}</b><br/>Equations", styles["cover_stat"]),
        Paragraph(f"<b>{n_fig}</b><br/>Figures",   styles["cover_stat"]),
    ]]
    tbl = Table(rows, colWidths=[CONTENT_W / 2] * 2)
    tbl.setStyle(TableStyle([
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LINEAFTER",   (0, 0), (0, -1), 0.5, C_RULE),
    ]))
    story.append(tbl)

    story.append(PageBreak())
    return story


# ════════════════════════════════════════════════════════════════════════════
# AT-A-GLANCE  (abstract box)
# ════════════════════════════════════════════════════════════════════════════

def _at_a_glance(sections: PaperSections, styles: dict) -> list:
    story = []
    story.append(Paragraph("At a Glance", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY, spaceAfter=10))

    abstract_text = safe_html(sections.abstract) if sections.abstract != "Not found" else ""
    if abstract_text:
        # Light-blue framed abstract box using a 1-cell table
        cell = Paragraph(abstract_text, styles["abstract"])
        box = Table([[cell]], colWidths=[CONTENT_W])
        box.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), C_LBLUE),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 14),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
            ("LINEABOVE",     (0, 0), (-1, 0),  3, C_BLUE),
            ("BOX",           (0, 0), (-1, -1), 0.5, C_RULE),
        ]))
        story.append(box)
        story.append(Spacer(1, 14))

    return story


# ════════════════════════════════════════════════════════════════════════════
# MARKDOWN → FLOWABLES
# ════════════════════════════════════════════════════════════════════════════

_BLOCK_EQ_RE = re.compile(r'^\$\$(.+?)\$\$$', re.DOTALL)
_INLINE_EQ_RE = re.compile(r'\$([^$\n]+)\$')


def _render_eq_image(latex: str, fontsize: int = 14) -> Image | None:
    """Return a ReportLab Image of a rendered equation, or None on failure."""
    png = latex_to_png(latex, fontsize=fontsize)
    if not png:
        return None
    buf = io.BytesIO(png)
    # Measure natural size then cap width
    img = Image(buf)
    nat_w, nat_h = img.drawWidth, img.drawHeight
    max_w = CONTENT_W - 0.4 * inch   # leave indent
    if nat_w > max_w:
        scale = max_w / nat_w
        img.drawWidth  = max_w
        img.drawHeight = nat_h * scale
    return img


def _inline_eq(text: str) -> str:
    """Replace $...$ with styled monospace spans for ReportLab."""
    return _INLINE_EQ_RE.sub(
        lambda m: (
            f'<font name="Courier" color="#0f2557">'
            f'{safe_html(m.group(1))}'
            f'</font>'
        ),
        text,
    )


def _parse_markdown(summary: str, styles: dict) -> list:
    """Convert markdown summary → ReportLab flowables.

    Handles: ## h2, ### h3, - bullets, $$block eq$$, $inline$, **bold**, plain body.
    Block equations are rendered as images (via matplotlib); inline as Courier spans.
    """
    story = []
    lines = summary.splitlines()
    i = 0
    while i < len(lines):
        raw  = lines[i]
        line = raw.strip()
        i   += 1

        # ── blank ──────────────────────────────────────────────────
        if not line:
            story.append(Spacer(1, 4))
            continue

        # ── Section heading h2 ──────────────────────────────────────
        if line.startswith("## "):
            story.append(Spacer(1, 6))
            story.append(Paragraph(safe_html(line[3:]), styles["h1"]))
            story.append(HRFlowable(width="100%", thickness=1,
                                    color=C_RULE, spaceAfter=6))
            continue

        # ── Sub-heading h3 ──────────────────────────────────────────
        if line.startswith("### "):
            story.append(Paragraph(safe_html(line[4:]), styles["h2"]))
            continue

        # ── Block equation  $$...$$  (may span multiple lines) ──────
        if line.startswith("$$"):
            latex_lines = [line[2:]]
            while i < len(lines):
                nxt = lines[i].strip()
                i  += 1
                if nxt.endswith("$$"):
                    latex_lines.append(nxt[:-2])
                    break
                latex_lines.append(nxt)
            latex = " ".join(latex_lines).strip()
            eq_img = _render_eq_image(latex, fontsize=14)
            if eq_img:
                eq_img.hAlign = "CENTER"
                block = Table(
                    [[eq_img]],
                    colWidths=[CONTENT_W],
                )
                block.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, -1), C_EQBG),
                    ("TOPPADDING",    (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                    ("BOX",           (0, 0), (-1, -1), 0.5, C_RULE),
                ]))
                story.append(block)
            else:
                # Fallback: styled text
                story.append(Paragraph(
                    f'<font name="Courier" color="#0f2557">{safe_html(latex)}</font>',
                    styles["body"],
                ))
            story.append(Spacer(1, 4))
            continue

        # ── Bullet point ─────────────────────────────────────────────
        if line.startswith(("- ", "* ")):
            content = line[2:]
            content = markdown_bold_to_html(_inline_eq(safe_html(content)))
            story.append(Paragraph(f"• {content}", styles["bullet"]))
            continue

        # ── Italic line (starts/ends with *) ─────────────────────────
        if line.startswith("*") and line.endswith("*") and not line.startswith("**"):
            inner = safe_html(line.strip("*"))
            story.append(Paragraph(f"<i>{inner}</i>", styles["italic"]))
            continue

        # ── Plain body text ──────────────────────────────────────────
        content = markdown_bold_to_html(_inline_eq(safe_html(line)))
        story.append(Paragraph(content, styles["body"]))

    return story


# ════════════════════════════════════════════════════════════════════════════
# FIGURES REFERENCE SECTION
# ════════════════════════════════════════════════════════════════════════════

def _figures_section(figures: list[ExtractedFigure], styles: dict) -> list:
    """Inline figures — only renders figures that have an actual cropped image."""
    visible = [f for f in figures if f.png_b64]
    if not visible:
        return []

    story = [
        Spacer(1, 8),
        Paragraph("📈 Visual Evidence", styles["h1"]),
        HRFlowable(width="100%", thickness=1, color=C_RULE, spaceAfter=12),
    ]

    MAX_FIG_W = CONTENT_W
    MAX_FIG_H = 3.2 * inch

    for idx, fig in enumerate(visible, 1):
        elements: list = []
        try:
            img_bytes = base64.b64decode(fig.png_b64)
            img = Image(io.BytesIO(img_bytes))
            nat_w, nat_h = img.drawWidth, img.drawHeight
            scale = min(MAX_FIG_W / nat_w, MAX_FIG_H / nat_h, 1.0)
            img.drawWidth  = nat_w * scale
            img.drawHeight = nat_h * scale
            img.hAlign     = "CENTER"
            box = Table([[img]], colWidths=[CONTENT_W])
            box.setStyle(TableStyle([
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING",    (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f8faff")),
                ("BOX",           (0, 0), (-1, -1), 0.5, C_RULE),
                ("LINEABOVE",     (0, 0), (-1,  0), 2,   C_BLUE),
            ]))
            elements.append(box)
        except Exception as exc:
            logger.warning("Could not embed figure %d: %s", idx, exc)
            continue
        caption = safe_html(fig.caption or f"Figure {idx}")
        desc    = safe_html(fig.description or "")
        cap_text = f"<b>{caption}</b>"
        if desc:
            cap_text += f"<br/><i>{desc}</i>"
        elements.append(Paragraph(cap_text, styles["fig_caption"]))
        elements.append(Spacer(1, 16))
        story.append(KeepTogether(elements))

    return story


# ════════════════════════════════════════════════════════════════════════════
# EQUATIONS REFERENCE SECTION
# ════════════════════════════════════════════════════════════════════════════

def _equations_section(equations: list[ExtractedEquation], styles: dict) -> list:
    if not equations:
        return []

    story = [
        PageBreak(),
        Paragraph("Equations Reference", styles["h1"]),
        HRFlowable(width="100%", thickness=1.5, color=C_NAVY, spaceAfter=14),
    ]

    for idx, eq in enumerate(equations, 1):
        elements: list = []

        # Label
        label = f"Eq. {idx}  (page {eq.page_number})"
        elements.append(Paragraph(label, styles["eq_label"]))

        # Rendered equation image inside a shaded box
        eq_img = _render_eq_image(eq.latex, fontsize=15)
        if eq_img:
            eq_img.hAlign = "CENTER"
            box = Table([[eq_img]], colWidths=[CONTENT_W])
            box.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), C_EQBG),
                ("TOPPADDING",    (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                ("BOX",           (0, 0), (-1, -1), 0.5, C_RULE),
                ("LINEABOVE",     (0, 0), (-1, 0),  2,   C_BLUE),
            ]))
            elements.append(box)
        else:
            # Fallback: styled Courier text
            elements.append(Paragraph(
                f'<font name="Courier" color="#0f2557" size="10">'
                f'{safe_html(eq.latex)}</font>',
                styles["body"],
            ))

        # Description
        if eq.description:
            elements.append(Paragraph(
                f"<i>{safe_html(eq.description)}</i>",
                styles["eq_where"],
            ))

        elements.append(Spacer(1, 10))
        story.append(KeepTogether(elements))

    return story


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def build_pdf(summary: str, sections: PaperSections) -> bytes:
    """Build a clean, well-organised PDF overview report.

    Layout:
      Cover → At-a-Glance → Overview sections → Figures ref → Equations ref

    Args:
        summary: Markdown string from the LLM.
        sections: Enriched PaperSections (may include equations/figures).

    Returns:
        PDF as raw bytes.
    """
    settings = get_settings()
    st = _styles()
    buf = io.BytesIO()

    # Short title for header bar
    short_title = (sections.title[:80] + "…") \
        if len(sections.title) > 80 else sections.title

    def on_cover(canvas, doc):
        # Plain white cover — no header bar
        _draw_footer(canvas, doc, settings.report_author)

    def on_content(canvas, doc):
        _draw_header_bar(canvas, doc, short_title)
        _draw_footer(canvas, doc, settings.report_author)

    try:
        doc = SimpleDocTemplate(
            buf,
            pagesize=LETTER,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN + 0.2 * inch,  # room for header bar
            bottomMargin=MARGIN,
        )

        story: list = []

        # ── 1. Cover ──────────────────────────────────────────────────
        story.extend(_cover_page(sections, st))

        # ── 2. At-a-Glance ────────────────────────────────────────────
        story.extend(_at_a_glance(sections, st))

        # ── 3. Overview sections (from LLM markdown) ──────────────────
        story.extend(_parse_markdown(summary, st))

        # ── 4. Figures — inline after summary (only if images exist) ──
        story.extend(_figures_section(sections.figures, st))

        # ── 5. Equations reference ────────────────────────────────────
        if sections.equations:
            story.append(PageBreak())
        story.extend(_equations_section(sections.equations, st))

        # ── Disclaimer ────────────────────────────────────────────────
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=C_RULE, spaceAfter=6))
        story.append(Paragraph(
            "This overview was generated automatically by AI. "
            "Always verify against the original publication.",
            st["footer"],
        ))

        doc.build(story, onFirstPage=on_cover, onLaterPages=on_content)

    except Exception as exc:
        raise ReportGenerationError(
            "Failed to build PDF report.", original=exc
        ) from exc

    result = buf.getvalue()
    logger.info("PDF report built: %d bytes.", len(result))
    return result
