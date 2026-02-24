"""
app/prompts/summary_generation.py
"""
from app.domain.models import PaperSections


def build_summary_system_prompt() -> str:
    return """\
You are a scientific editor producing a concise, reader-friendly overview of a research paper.

RULES:
- Write for a reader who wants the key ideas in 3-5 minutes.
- Each section must be self-contained. No padding or repetition.
- Use LaTeX for every equation (inline: $...$, block: $$...$$).
- After each block equation add: *Where: X = ..., Y = ...*
- Keep bullet points to 1 sentence each.
- Do NOT include a "Visual Evidence" or "Figures" section — figures are handled separately.
- Output clean markdown only. No preamble, no commentary."""


def build_summary_user_prompt(sections: PaperSections) -> str:
    eq_ctx = ""
    if sections.equations:
        lines = [f"  Eq{i} (p.{eq.page_number}): {eq.latex}  →  {eq.description}"
                 for i, eq in enumerate(sections.equations, 1)]
        eq_ctx = "EQUATIONS FOUND IN PAPER:\n" + "\n".join(lines)

    return f"""\
Produce a structured overview. Follow the exact section order below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAPER DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Title:   {sections.title}
Authors: {sections.authors}

Abstract:      {sections.abstract}
Introduction:  {sections.introduction}
Methodology:   {sections.methodology}
Results:       {sections.results}
Conclusion:    {sections.conclusion}
Limitations:   {sections.limitations}
Future Work:   {sections.future_work}

{eq_ctx}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (use exactly these headings, no others)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🎯 What This Paper Does
One paragraph, 3-5 sentences. Problem, solution, core claim.

## 🔬 How It Works
Technical approach as bullet points. Embed key equations as $$...$$  with *Where:* line after each.

## 📊 Key Results
2-4 bullets. Each: one concrete finding + its significance.

## 🧮 Core Equations
For each important equation:
**Equation name** — one sentence on what it computes.
$$latex here$$
*Where: symbol = meaning*

## 💡 Key Takeaways
3-5 bullets a researcher would remember.

## ⚠️ Limitations & Future Work
2-3 bullets on limitations, 1-2 on future directions.
"""
