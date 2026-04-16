"""
Pre-retrieval policy filter.
Runs before intent detection on every query.

Checks:
1. PII in the query itself → immediate refusal
2. Medical/legal advice patterns → allow retrieval but flag for disclaimer
"""
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class PolicyDecision:
    refuse: bool
    reason: Optional[str]
    disclaimer: Optional[str]   # prepend to answer if not None


# ── PII patterns ────────────────────────────────────────────────────────────────
# These patterns indicate the user is sending or asking to extract PII

_PII_PATTERNS = [
    (r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", "SSN-like pattern"),
    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "credit card-like pattern"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email address in query"),
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "phone number-like pattern"),
]

# Queries that request PII extraction from documents
_PII_EXTRACTION_PHRASES = [
    "extract.*email", "list.*email", "show.*email",
    "extract.*phone", "list.*phone", "extract.*ssn",
    "extract.*social security", "list.*personal.*data",
    "extract.*password", "show.*password", "list.*password",
    "extract.*credit card", "find.*credit card",
    "extract.*address", "list.*home address",
    "get.*personal.*information", "extract.*pii",
    "ignore.*instructions", "ignore previous",  # prompt injection
    "forget.*instructions", "you are now",
    "new personality", "act as",
]

# ── Medical / legal patterns ────────────────────────────────────────────────────

_MEDICAL_PHRASES = [
    "should i take", "can i take", "is it safe to take",
    "what dosage", "side effects of", "drug interaction",
    "medical advice", "diagnose me", "do i have",
    "symptoms of", "treat my", "cure for",
]

_LEGAL_PHRASES = [
    "is it legal for me", "am i liable", "will i be sued",
    "legal advice", "can i sue", "my legal rights",
    "should i sign", "is this contract valid",
]

_MEDICAL_DISCLAIMER = (
    "⚠️ Note: I can summarize what the uploaded documents say, "
    "but this is not medical advice. Please consult a qualified healthcare professional."
)

_LEGAL_DISCLAIMER = (
    "⚠️ Note: I can summarize what the uploaded documents say, "
    "but this is not legal advice. Please consult a qualified legal professional."
)


def check(query: str) -> PolicyDecision:
    """
    Evaluate a query against policy rules.

    Returns a PolicyDecision:
    - refuse=True → do not proceed, return the reason as the answer
    - refuse=False, disclaimer set → proceed but prepend disclaimer
    - refuse=False, disclaimer None → proceed normally
    """
    q_lower = query.lower()

    # Check PII in the query text itself
    for pattern, label in _PII_PATTERNS:
        if re.search(pattern, query):
            return PolicyDecision(
                refuse=True,
                reason=f"Query contains sensitive data ({label}). I cannot process queries containing personal information.",
                disclaimer=None,
            )

    # Check PII extraction requests
    for phrase in _PII_EXTRACTION_PHRASES:
        if re.search(phrase, q_lower):
            return PolicyDecision(
                refuse=True,
                reason="I cannot extract or list personal information from documents.",
                disclaimer=None,
            )

    # Medical advice
    for phrase in _MEDICAL_PHRASES:
        if phrase in q_lower:
            return PolicyDecision(
                refuse=False,
                reason=None,
                disclaimer=_MEDICAL_DISCLAIMER,
            )

    # Legal advice
    for phrase in _LEGAL_PHRASES:
        if phrase in q_lower:
            return PolicyDecision(
                refuse=False,
                reason=None,
                disclaimer=_LEGAL_DISCLAIMER,
            )

    return PolicyDecision(refuse=False, reason=None, disclaimer=None)
