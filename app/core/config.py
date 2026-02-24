"""
app/core/config.py
Central configuration. All tuneable constants live here.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # ── Groq (text LLM) ───────────────────────────────────────────────
    groq_api_key:  str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model:    str = "llama-3.3-70b-versatile"

    # ── Gemini (vision) ───────────────────────────────────────────────
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # ── LLM token budgets ─────────────────────────────────────────────
    max_tokens_section_extraction: int = 2048
    max_tokens_summary:            int = 3000   # more room for detailed overview

    # ── Chunking ──────────────────────────────────────────────────────
    # (actual chunk sizes live in chunker.py — these are informational)
    max_section_input_chars: int = 6_000   # kept for backward compat, unused now

    # ── Retry ─────────────────────────────────────────────────────────
    llm_max_retries: int   = 3
    llm_retry_delay: float = 1.5

    # ── App branding ──────────────────────────────────────────────────
    app_title:    str = "📄 Research Paper Summarizer"
    app_subtitle: str = "Powered by Groq + Gemini · marker-enhanced"
    app_icon:     str = "📄"
    report_title: str = "Research Paper Summary Report"
    report_author: str = "AI Research Assistant"

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.strip())


def get_settings() -> Settings:
    return Settings()
