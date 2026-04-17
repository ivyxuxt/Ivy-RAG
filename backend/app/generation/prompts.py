"""
Prompt templates for each intent class.

All templates share a system prompt enforcing:
- Answer only from the provided context
- Cite every claim with inline [S1], [S2] tags
- Return "Insufficient evidence" if context is weak
- No outside knowledge
"""
from typing import List

# ── System prompt (shared across all intents) ──────────────────────────────────

SYSTEM_PROMPT = """You are a document-grounded assistant.

Rules:
1. Answer ONLY using the context provided below. Do not use outside knowledge.
2. Cite every factual claim using inline source tags: [S1], [S2], etc.
3. If the context does not clearly support the answer, say exactly:
   "Insufficient evidence in the uploaded documents to answer this question."
4. Do not speculate. Do not add information not present in the context.
5. Be concise and factual."""


# ── Context block builder ──────────────────────────────────────────────────────

def build_context_block(chunks: List[dict]) -> str:
    """Format retrieved chunks as numbered [S1]...[Sk] context blocks."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        doc_name = chunk.get("doc_name", "unknown")
        p_start = chunk.get("page_start")
        p_end = chunk.get("page_end")
        section = chunk.get("section_title")

        # Location descriptor
        if p_start and p_end and p_start != p_end:
            location = f"pages {p_start}–{p_end}"
        elif p_start:
            location = f"page {p_start}"
        else:
            location = "location unknown"

        if section:
            loc_str = f"{doc_name}, {location}, {section}"
        else:
            loc_str = f"{doc_name}, {location}"

        lines.append(f"[S{i}] ({loc_str})")
        lines.append(chunk["text"])
        lines.append("")

    return "\n".join(lines)


# ── Per-intent user prompts ────────────────────────────────────────────────────

def prose_prompt(question: str, context: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer with inline citations like [S1]. "
        f"If the context does not support an answer, say \"Insufficient evidence in the uploaded documents.\""
    )


def list_prompt(question: str, context: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Format your answer as a bulleted list. Cite each bullet with [S1], [S2], etc. "
        f"If the context does not support an answer, say \"Insufficient evidence in the uploaded documents.\""
    )


def table_prompt(question: str, context: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Format your answer as a Markdown table with inline [S1], [S2] citations in cells. "
        f"After the table, add 3–5 bullet points on the key takeaways, each cited with [S1] etc. "
        f"Do NOT add a separate Citations section at the end. "
        f"If the context does not support an answer, say \"Insufficient evidence in the uploaded documents.\""
    )


def summary_prompt(question: str, context: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Provide a 2–3 sentence summary, then an Evidence section with cited bullets. "
        f"Use [S1], [S2], etc. for citations. "
        f"If the context does not support an answer, say \"Insufficient evidence in the uploaded documents.\""
    )


def get_user_prompt(intent: str, question: str, context: str) -> str:
    """Select the correct template by intent."""
    if intent == "list_request":
        return list_prompt(question, context)
    elif intent == "table_request":
        return table_prompt(question, context)
    elif intent == "summary_request":
        return summary_prompt(question, context)
    else:
        return prose_prompt(question, context)
