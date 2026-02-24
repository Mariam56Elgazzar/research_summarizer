"""
app/utils/equation_renderer.py

Renders LaTeX equation strings to PNG bytes using matplotlib mathtext.
Produces clean, properly-sized images suitable for embedding in PDF reports.
"""
from __future__ import annotations

import io
import logging
import re

logger = logging.getLogger(__name__)

# Colours that match the report's design language
BG_COLOR  = "#f8faff"   # very light blue-white
EQ_COLOR  = "#0f2557"   # deep navy


def _clean_latex(latex: str) -> str:
    """
    Sanitise a raw LaTeX string so matplotlib mathtext can render it.

    Mathtext supports a large subset of LaTeX math, but NOT:
      - begin/end environments (stripped, content kept)
      - Double-dollar or single-dollar wrappers (we add our own)
    """
    s = latex.strip()

    # Strip outer $…$  or  $$…$$  wrappers (we'll add our own)
    s = re.sub(r'^\$\$?|\$\$?$', '', s).strip()

    # Strip \begin{…}…\end{…} align/equation environments, keep content
    s = re.sub(r'\\begin\{[^}]+\}', '', s)
    s = re.sub(r'\\end\{[^}]+\}', '', s)

    # Replace \nonumber, \label{...}, \tag{...} — not supported
    s = re.sub(r'\\(nonumber|label|tag)\{[^}]*\}', '', s)

    # \operatorname{foo} → \mathrm{foo}
    s = s.replace(r'\operatorname', r'\mathrm')

    # \DeclareMathOperator and similar
    s = re.sub(r'\\DeclareMathOperator\{[^}]*\}\{[^}]*\}', '', s)

    # Remove \; \, \! \quad \qquad spacing that confuses mathtext
    # (keep \, which mathtext handles, remove others)
    s = re.sub(r'\\(quad|qquad|;|!)', ' ', s)

    # Collapse multiple spaces / newlines
    s = re.sub(r'\s+', ' ', s).strip()

    return s


def latex_to_png(latex: str, fontsize: int = 14, dpi: int = 150) -> bytes | None:
    """
    Render a LaTeX math string to PNG bytes.

    Returns None if rendering fails (caller should fall back to text).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cleaned = _clean_latex(latex)
        if not cleaned:
            return None

        # Wrap in $…$ for mathtext
        display_str = f"${cleaned}$"

        fig = plt.figure(figsize=(1, 1))   # will be resized by tight layout
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        fig.patch.set_facecolor(BG_COLOR)

        txt = ax.text(
            0.5, 0.5, display_str,
            ha="center", va="center",
            fontsize=fontsize,
            color=EQ_COLOR,
            transform=ax.transAxes,
        )

        # Measure the text and resize figure to fit tightly
        fig.canvas.draw()
        bbox = txt.get_window_extent(renderer=fig.canvas.get_renderer())
        pad_x, pad_y = 20, 10   # pixels of padding
        fig.set_size_inches(
            (bbox.width + pad_x * 2) / dpi,
            (bbox.height + pad_y * 2) / dpi,
        )

        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", dpi=dpi,
            bbox_inches="tight",
            facecolor=BG_COLOR,
            edgecolor="none",
        )
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception as exc:
        logger.warning("Equation render failed for '%s…': %s", latex[:40], exc)
        return None
