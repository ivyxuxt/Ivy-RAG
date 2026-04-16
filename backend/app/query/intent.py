"""
Query intent classifier — 6 classes.

Classes:
    chitchat         → no retrieval, friendly direct reply
    knowledge_lookup → full RAG pipeline
    list_request     → RAG + bullet list template
    table_request    → RAG + markdown table template
    summary_request  → RAG + summary template
    unsafe           → immediate refusal

Classifier logic:
    1. Rule layer (fast, deterministic) using curated signal phrases
    2. If ambiguous, fall back to a single Mistral chat call
"""
import re
import time
from typing import Literal

import httpx

from app.config import settings

IntentLabel = Literal[
    "chitchat",
    "knowledge_lookup",
    "list_request",
    "table_request",
    "summary_request",
    "unsafe",
]

# ── Signal phrase lists (from .rules.md) ──────────────────────────────────────

_RAG_TRIGGER_PHRASES = [
    "according to", "based on the document", "based on the file",
    "in the uploaded", "from the pdf", "what does it say",
    "per the policy", "as stated in", "the document mentions",
    "what does the report", "in section", "on page",
    "what are the", "list all", "list the", "compare",
    "how many", "when did", "where is", "who is responsible",
    "summarize", "overview", "brief",
]

_NO_RETRIEVAL_PHRASES = [
    "hello", "hi", "hey", "thanks", "thank you", "good morning",
    "good afternoon", "good evening", "who are you", "what can you do",
    "how are you", "tell me a joke", "what's up", "nice to meet",
    "goodbye", "bye", "see you",
]

_LIST_SIGNALS = [
    "list", "enumerate", "what are all the", "give me all",
    "what are the", "show all", "tell me all",
]

_TABLE_SIGNALS = [
    "table", "compare", "comparison", "vs ", " vs", "versus",
    "difference between", "differences between",
    "side by side", "breakdown",
]

_SUMMARY_SIGNALS = [
    "summarize", "summary", "overview", "brief", "give me a brief",
    "in a nutshell", "tldr", "tl;dr", "key points", "main points",
]

_UNSAFE_SIGNALS = [
    "hack", "exploit", "attack", "malware", "ransomware",
    "how to make a bomb", "how to make poison",
    "jailbreak", "bypass safety",
]

# WH-question words that suggest a knowledge query
_WH_WORDS = {"what", "how", "why", "when", "where", "who", "which", "whose", "whom"}


def classify(query: str) -> IntentLabel:
    """Return the intent label for a query."""
    q_lower = query.lower().strip()
    tokens = set(re.findall(r"\b\w+\b", q_lower))

    # Unsafe check first
    for phrase in _UNSAFE_SIGNALS:
        if phrase in q_lower:
            return "unsafe"

    # Chitchat: short query + no RAG trigger + matches no-retrieval phrases
    if any(q_lower.startswith(p) or q_lower == p for p in _NO_RETRIEVAL_PHRASES):
        return "chitchat"
    if len(tokens) <= 4 and not (tokens & _WH_WORDS) and not any(p in q_lower for p in _RAG_TRIGGER_PHRASES):
        return "chitchat"

    # Table request (check before list — "compare" appears in both)
    if any(p in q_lower for p in _TABLE_SIGNALS):
        return "table_request"

    # List request
    if any(p in q_lower for p in _LIST_SIGNALS):
        return "list_request"

    # Summary request
    if any(p in q_lower for p in _SUMMARY_SIGNALS):
        return "summary_request"

    # Explicit RAG trigger phrase
    if any(p in q_lower for p in _RAG_TRIGGER_PHRASES):
        return "knowledge_lookup"

    # WH-question with content words → likely knowledge lookup
    if tokens & _WH_WORDS and len(tokens) > 3:
        return "knowledge_lookup"

    # Ambiguous — ask Mistral
    return _mistral_classify(query)


def _mistral_classify(query: str) -> IntentLabel:
    """
    Fallback classifier using a single Mistral chat call.
    Returns one of the 6 intent labels.
    """
    prompt = (
        "Classify the following user query into exactly one category. "
        "Return ONLY the category name, nothing else.\n\n"
        "Categories:\n"
        "- chitchat (greetings, thanks, meta questions about you)\n"
        "- knowledge_lookup (asking about document content)\n"
        "- list_request (asking to list or enumerate items from documents)\n"
        "- table_request (asking for a comparison table from documents)\n"
        "- summary_request (asking for a summary of document content)\n"
        "- unsafe (harmful, illegal, or security-threatening requests)\n\n"
        f"Query: {query}\n\nCategory:"
    )

    valid = {"chitchat", "knowledge_lookup", "list_request", "table_request", "summary_request", "unsafe"}

    for attempt in range(2):
        try:
            response = httpx.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.MISTRAL_CHAT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 20,
                },
                timeout=15.0,
            )
            response.raise_for_status()
            label = response.json()["choices"][0]["message"]["content"].strip().lower()
            if label in valid:
                return label  # type: ignore
        except Exception:
            if attempt == 0:
                time.sleep(2)

    # Safe default
    return "knowledge_lookup"
