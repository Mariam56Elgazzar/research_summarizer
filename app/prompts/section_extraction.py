"""
app/prompts/section_extraction.py
Section extraction prompt — now accepts full chunk text (no truncation)
and an optional marker pre-seeding hint.
"""


def build_section_extraction_system_prompt() -> str:
    return """\
You are an expert academic paper parser with deep knowledge of research paper structure.
Extract the requested sections accurately from the provided text chunk.
A paper may be split across multiple chunks — extract whatever is present in THIS chunk.
Always respond with valid JSON only. No markdown, no explanation, no code fences.
If a section is not in this chunk, use "Not found"."""


def build_section_extraction_user_prompt(text: str, hint: str = "") -> str:
    hint_block = f"\n{hint}\n" if hint else ""

    return f"""Extract these sections from the paper text chunk below.
{hint_block}
Return a JSON object with exactly these keys:
  title, authors, abstract, introduction, methodology, results, conclusion, limitations, future_work

Rules:
- Each value: 3–6 sentences summarising that section. Include key technical details.
- Preserve any LaTeX equations exactly as written (e.g. $$\\frac{{QK^T}}{{\\sqrt{{d_k}}}}$$)
- "Not found" if the section is not present in this chunk.
- Return ONLY the JSON object.

Paper text:
\"\"\"
{text}
\"\"\"
"""
