"""
app/ui/streamlit_ui.py
Streamlit UI — upgraded for marker + Gemini + chunked extraction.
"""
import base64
import logging

import streamlit as st

from app.core.config import get_settings
from app.core.exceptions import AppError, PDFExtractionError, LLMServiceError, ReportGenerationError
from app.services.pipeline_service import run_pipeline

logger = logging.getLogger(__name__)

STEPS = [
    "📄 marker parse",
    "🔍 Figure crop",
    "🔭 Vision",
    "🧩 Chunk extract",
    "✍️ Summary",
    "📑 Report",
]

CUSTOM_CSS = """
<style>
    .main-title { font-size:2.2rem; font-weight:800; color:#0f2557; margin-bottom:0.2rem; }
    .subtitle   { color:#64748b; font-size:1rem; margin-bottom:1.5rem; }
    .step-done  { color:#22c55e; font-weight:600; }
    .step-active{ color:#3b82f6; font-weight:600; }
    .step-wait  { color:#9ca3af; }
    .error-box  { background:#fee2e2; border-left:4px solid #ef4444;
                  padding:1rem; border-radius:4px; color:#7f1d1d; }
    .badge      { display:inline-block; border-radius:999px; padding:2px 10px;
                  margin:2px; font-size:0.8rem; font-weight:600; }
    .badge-blue { background:#dbeafe; color:#1d4ed8; }
    .badge-green{ background:#dcfce7; color:#15803d; }
    .badge-gold { background:#fef9c3; color:#92400e; }
</style>
"""


def _render_steps(placeholders, current_step, done=False):
    for i, (ph, label) in enumerate(zip(placeholders, STEPS)):
        if done or i < current_step:
            ph.markdown(f'<span class="step-done">✅ {label}</span>', unsafe_allow_html=True)
        elif i == current_step:
            ph.markdown(f'<span class="step-active">⏳ {label}…</span>', unsafe_allow_html=True)
        else:
            ph.markdown(f'<span class="step-wait">○ {label}</span>', unsafe_allow_html=True)


def _render_figures_tab(sections):
    visible = [f for f in sections.figures if f.png_b64]
    if not visible:
        if sections.figures:
            st.info("Figures were detected but images could not be cropped from this PDF.")
        else:
            st.info("No figures extracted.")
        return
    st.markdown(f"**{len(visible)} figure(s) extracted**")
    for fig in visible:
        try:
            st.image(base64.b64decode(fig.png_b64), use_container_width=True)
        except Exception:
            st.caption("_(image unavailable)_")
        caption = fig.caption or f"Page {fig.page_number}"
        st.markdown(f"**{caption}**")
        if fig.description:
            st.caption(fig.description)
        st.divider()


def _render_equations_tab(sections):
    if not sections.equations:
        st.info("No equations extracted.")
        return
    st.markdown(f"**{len(sections.equations)} equation(s) extracted**")
    for i, eq in enumerate(sections.equations, 1):
        label = eq.description or eq.latex[:60]
        with st.expander(f"Eq. {i} (p.{eq.page_number}) — {label}"):
            try:
                st.latex(eq.latex)
            except Exception:
                st.code(eq.latex)
            if eq.description:
                st.caption(eq.description)


def render_app():
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon=settings.app_icon,
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="main-title">{settings.app_title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{settings.app_subtitle}</div>', unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        # ── Groq ──────────────────────────────────────────────────────
        st.subheader("Groq  (text LLM)")
        api_key = st.text_input(
            "API Key", key="groq_key", type="password", placeholder="gsk_…",
            help="Free at https://console.groq.com",
        )
        model = st.selectbox("Model", [
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "llama3-8b-8192",
        ])

        st.divider()

        # ── Gemini ────────────────────────────────────────────────────
        st.subheader("Gemini 1.5 Flash  (vision)")
        gemini_key = st.text_input(
            "API Key", key="gemini_key", type="password", placeholder="AIza…",
            help="Free at https://aistudio.google.com/app/apikey · 15 req/min",
        )
        if gemini_key:
            st.success("✅ Gemini vision active")
        else:
            st.warning("⚠️ No Gemini key — using Groq vision (less accurate)")

        st.divider()

        # ── Options ───────────────────────────────────────────────────
        st.subheader("Options")
        use_vision = st.toggle("🔭 Vision analysis", value=True,
            help="Analyse figure/equation images. Disable to speed up.")
        use_marker = st.toggle("📄 marker PDF parser", value=True,
            help="Use marker for accurate LaTeX extraction. Install: pip install marker-pdf")

        st.divider()

        # ── Pipeline overview ─────────────────────────────────────────
        st.markdown("**Accuracy stack:**")
        st.markdown(
            "1. 📄 **marker** → structured markdown + LaTeX\n"
            "2. 🔍 PyMuPDF → cropped figure images\n"
            "3. 🔭 **Gemini 1.5 Flash** → figure & equation vision\n"
            "4. 🧩 **Chunked extraction** → full paper, no cutoff\n"
            "5. ✍️ llama-3.3-70b → enriched summary\n"
            "6. 📑 ReportLab → PDF report"
        )

    # ── Upload ────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload a research paper (PDF)",
        type=["pdf"],
        help="Text-based PDFs. marker handles equations; Gemini handles figures.",
    )

    if not uploaded_file:
        st.info("⬆️ Upload a PDF to get started.")
        return

    if not api_key:
        st.warning("⚠️ Enter your Groq API key in the sidebar.")
        return

    if st.button("🚀 Analyze Paper", type="primary", use_container_width=True):
        pdf_bytes = uploaded_file.read()

        st.markdown("---")
        step_cols = st.columns(len(STEPS))
        placeholders = [col.empty() for col in step_cols]
        _render_steps(placeholders, 0)
        progress = st.progress(0, "Starting…")

        try:
            result = _run_with_progress(
                pdf_bytes=pdf_bytes,
                api_key=api_key,
                gemini_key=gemini_key,
                model=model,
                use_vision=use_vision,
                placeholders=placeholders,
                progress=progress,
            )

        except PDFExtractionError as exc:
            progress.empty()
            st.markdown(f'<div class="error-box">📄 <b>PDF Error:</b> {exc.message}</div>',
                        unsafe_allow_html=True)
            return
        except LLMServiceError as exc:
            progress.empty()
            st.markdown(f'<div class="error-box">🤖 <b>AI Error:</b> {exc.message}</div>',
                        unsafe_allow_html=True)
            if exc.original:
                with st.expander("🔍 Full error details"):
                    st.code(f"{type(exc.original).__name__}: {exc.original}")
            return
        except ReportGenerationError as exc:
            progress.empty()
            st.markdown(f'<div class="error-box">📑 <b>Report Error:</b> {exc.message}</div>',
                        unsafe_allow_html=True)
            return
        except AppError as exc:
            progress.empty()
            st.markdown(f'<div class="error-box">⚠️ <b>Error:</b> {exc.message}</div>',
                        unsafe_allow_html=True)
            return

        progress.progress(100, "Done!")
        _render_steps(placeholders, len(STEPS), done=True)
        st.success("✅ Analysis complete!")

        # Badges
        badges = ""
        if result.sections.equations:
            badges += f'<span class="badge badge-blue">🧮 {len(result.sections.equations)} equations</span>'
        if result.sections.figures:
            badges += f'<span class="badge badge-green">📊 {len(result.sections.figures)} figures</span>'
        if gemini_key:
            badges += '<span class="badge badge-gold">⚡ Gemini vision</span>'
        if badges:
            st.markdown(badges, unsafe_allow_html=True)
        st.markdown("")

        # Tabs
        tab_summary, tab_figures, tab_equations, tab_download = st.tabs([
            "✍️ Summary", "📊 Figures", "🧮 Equations", "📥 Download",
        ])
        with tab_summary:
            st.markdown(result.summary_markdown)
        with tab_figures:
            _render_figures_tab(result.sections)
        with tab_equations:
            _render_equations_tab(result.sections)
        with tab_download:
            st.download_button(
                "⬇️ Download PDF Report",
                data=result.report_pdf_bytes,
                file_name=f"summary_{uploaded_file.name}",
                mime="application/pdf",
                use_container_width=True,
            )
            st.caption(f"**{result.sections.title}**  ·  {result.sections.authors}")
            st.caption(
                f"{len(result.sections.equations)} equations  ·  "
                f"{len(result.sections.figures)} figures"
            )


def _run_with_progress(
    pdf_bytes, api_key, gemini_key, model,
    use_vision, placeholders, progress,
):
    _render_steps(placeholders, 1)
    progress.progress(10, "marker: converting PDF…")

    _render_steps(placeholders, 2)
    progress.progress(25, "Cropping figures…")

    _render_steps(placeholders, 3)
    vision_label = "Gemini vision…" if gemini_key else "Groq vision…"
    progress.progress(40, vision_label if use_vision else "Vision skipped…")

    result = run_pipeline(
        pdf_bytes=pdf_bytes,
        api_key=api_key,
        model=model,
        use_vision=use_vision,
        gemini_key=gemini_key,
    )

    _render_steps(placeholders, 5)
    progress.progress(95, "Building report…")
    return result
