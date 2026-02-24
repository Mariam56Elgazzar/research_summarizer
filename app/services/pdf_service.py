"""
app/services/pipeline_service.py
Orchestrates the full vision-enriched pipeline.
"""
import logging

from app.core.config import get_settings
from app.domain.models import PipelineResult
from app.services import llm_service, report_service, summarizer_service
from app.services.pdf_parser import parse_pdf
from app.services.vision_service import analyze_key_pages, VisionAnalysis

logger = logging.getLogger(__name__)


def run_pipeline(
    pdf_bytes: bytes,
    api_key: str,
    model: str | None = None,
    use_vision: bool = True,
) -> PipelineResult:
    """Full pipeline: parse → vision → extract → summarise → report."""
    logger.info("Pipeline started (%d bytes, vision=%s)", len(pdf_bytes), use_vision)

    settings = get_settings()
    if model:
        settings.groq_model = model

    client = llm_service.create_groq_client(api_key)

    # 1. Parse PDF — text + cropped figures + key page renders
    logger.info("Step 1/5: Parsing PDF...")
    parsed = parse_pdf(pdf_bytes)
    logger.info("  %d pages, %d cropped figures, %d key pages",
                parsed.page_count,
                len(parsed.all_cropped_figures),
                len(parsed.key_pages))

    # 2. Vision analysis on key pages
    if use_vision and parsed.key_pages:
        logger.info("Step 2/5: Vision analysis on %d pages...", len(parsed.key_pages))
        vision = analyze_key_pages(
            key_pages=parsed.key_pages,
            all_cropped_figures=parsed.all_cropped_figures,
            client=client,
            settings=settings,
        )
    else:
        vision = VisionAnalysis()
        logger.info("Step 2/5: Vision skipped.")

    # 3. Text section extraction
    logger.info("Step 3/5: Extracting sections from text...")
    sections = summarizer_service.extract_sections(parsed.full_text, client, settings)

    # 4. Merge vision data + cropped figures into sections
    logger.info("Step 4/5: Merging vision data...")
    sections = summarizer_service.enrich_sections_with_vision(sections, vision)

    # 5. Generate summary
    logger.info("Step 5/5: Generating summary...")
    summary_markdown = summarizer_service.generate_summary(sections, client, settings)

    # 6. Build report
    logger.info("Building PDF report...")
    report_pdf_bytes = report_service.build_pdf(summary_markdown, sections)

    logger.info("Pipeline complete.")
    return PipelineResult(
        sections=sections,
        summary_markdown=summary_markdown,
        report_pdf_bytes=report_pdf_bytes,
    )
