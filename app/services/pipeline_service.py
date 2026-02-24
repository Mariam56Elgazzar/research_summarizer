"""
app/services/pipeline_service.py

Upgraded pipeline:
  1. marker_service  → PDF → structured markdown + LaTeX equations
  2. pdf_parser      → cropped figure images + key page renders
  3. gemini_vision   → accurate figure/equation analysis (Gemini 1.5 Flash)
     OR groq_vision  → fallback if no Gemini key
  4. summarizer      → chunked full-paper section extraction (no truncation)
  5. report_service  → PDF report with rendered equations + cropped figures
"""
from __future__ import annotations

import logging

from app.core.config import get_settings
from app.domain.models import PipelineResult
from app.services import llm_service, report_service, summarizer_service
from app.services.marker_service import convert_pdf as marker_convert
from app.services.pdf_parser import parse_pdf

logger = logging.getLogger(__name__)


def run_pipeline(
    pdf_bytes:  bytes,
    api_key:    str,
    model:      str | None = None,
    use_vision: bool = True,
    gemini_key: str = "",
) -> PipelineResult:
    """
    Full accuracy-maximised pipeline.

    Args:
        pdf_bytes:  Raw PDF bytes.
        api_key:    Groq API key (text LLM).
        model:      Optional Groq model override.
        use_vision: Whether to run vision analysis on figures/equations.
        gemini_key: Google Gemini API key (preferred vision backend).
                    Falls back to Groq llama-4-scout if empty.
    """
    logger.info("Pipeline started (%d bytes)", len(pdf_bytes))

    settings = get_settings()
    if model:
        settings.groq_model = model
    # Allow key override from UI even if not in .env
    if gemini_key:
        settings.gemini_api_key = gemini_key

    client = llm_service.create_groq_client(api_key)

    # ── Step 1: marker conversion ──────────────────────────────────────
    # Gives us: structured markdown, LaTeX equations, section hints
    logger.info("Step 1/6: Converting PDF with marker...")
    marker_result = marker_convert(pdf_bytes)
    logger.info(
        "  marker: %d chars, %d equations found in text",
        len(marker_result.full_markdown),
        len(marker_result.equations),
    )

    # ── Step 2: PDF parsing for figure crops + vision key pages ───────
    logger.info("Step 2/6: Parsing PDF for figures and key pages...")
    parsed = parse_pdf(pdf_bytes)
    logger.info(
        "  %d pages, %d cropped figures, %d key pages",
        parsed.page_count,
        len(parsed.all_cropped_figures),
        len(parsed.key_pages),
    )

    # ── Step 3: Vision analysis ────────────────────────────────────────
    vision = _run_vision(parsed, settings, use_vision)

    # ── Step 4: Chunked section extraction from full paper text ────────
    logger.info("Step 4/6: Extracting sections (chunked, full paper)...")
    sections = summarizer_service.extract_sections(
        full_text=marker_result.full_text_for_llm,
        client=client,
        settings=settings,
        marker_sections=marker_result.sections,
    )

    # Override title/authors from marker if LLM missed them
    if sections.title == "Not found" and marker_result.title:
        sections = sections.model_copy(update={"title": marker_result.title})
    if sections.authors == "Not found" and marker_result.authors:
        sections = sections.model_copy(update={"authors": marker_result.authors})

    # ── Step 5: Merge vision + marker equations into sections ──────────
    logger.info("Step 5/6: Merging vision + marker data...")
    sections = summarizer_service.enrich_sections_with_vision(sections, vision)
    sections = summarizer_service.enrich_sections_with_marker(
        sections, marker_result.equations
    )

    # ── Step 6: Generate summary ───────────────────────────────────────
    logger.info("Step 6/6: Generating enriched summary...")
    summary_markdown = summarizer_service.generate_summary(sections, client, settings)

    # ── Build report ───────────────────────────────────────────────────
    logger.info("Building PDF report...")
    report_pdf_bytes = report_service.build_pdf(summary_markdown, sections)

    logger.info("Pipeline complete.")
    return PipelineResult(
        sections=sections,
        summary_markdown=summary_markdown,
        report_pdf_bytes=report_pdf_bytes,
    )


def _run_vision(parsed, settings, use_vision: bool):
    """Select and run the best available vision backend."""
    from app.services.vision_service import VisionAnalysis as GroqVisionAnalysis

    if not use_vision or not parsed.key_pages:
        logger.info("Step 3/6: Vision skipped.")
        return GroqVisionAnalysis()

    # Prefer Gemini if key is available
    if settings.has_gemini:
        logger.info(
            "Step 3/6: Vision with Gemini 1.5 Flash on %d pages...",
            len(parsed.key_pages),
        )
        try:
            from app.services.gemini_vision_service import analyze_key_pages_gemini
            vision = analyze_key_pages_gemini(
                key_pages=parsed.key_pages,
                all_cropped_figures=parsed.all_cropped_figures,
                gemini_api_key=settings.gemini_api_key,
            )
            # Convert to Groq-compatible VisionAnalysis for downstream compatibility
            return _bridge_vision(vision, GroqVisionAnalysis)
        except Exception as exc:
            logger.warning("Gemini vision failed (%s) — falling back to Groq.", exc)

    # Fallback: Groq llama-4-scout
    logger.info(
        "Step 3/6: Vision with Groq llama-4-scout on %d pages...",
        len(parsed.key_pages),
    )
    from app.services.vision_service import analyze_key_pages
    client = llm_service.create_groq_client(settings.groq_api_key)
    return analyze_key_pages(
        key_pages=parsed.key_pages,
        all_cropped_figures=parsed.all_cropped_figures,
        client=client,
        settings=settings,
    )


def _bridge_vision(src, TargetClass):
    """
    Convert between Gemini and Groq VisionAnalysis dataclasses.
    Both have identical field names so we can copy by attribute.
    """
    result = TargetClass()
    result.figures        = src.figures
    result.equations      = src.equations
    result.page_summaries = src.page_summaries
    return result
