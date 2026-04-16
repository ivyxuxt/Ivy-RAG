"""
PDF text extraction via pypdf.
Returns per-page text with quality check.
OCR fallback is logged but not implemented here (Mistral OCR is a config-gated upgrade).
"""
import re
from typing import List, Tuple

import pypdf


def extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extract text from each page of a PDF.

    Returns:
        List of (page_number, cleaned_text) tuples. Page numbers are 1-indexed.
        Pages with very little text are flagged in logs — potential scanned content.
    """
    results: List[Tuple[int, str]] = []
    low_quality_pages = 0

    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        n_pages = len(reader.pages)

        for i, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _clean_text(raw)

            if len(cleaned) < 100:
                low_quality_pages += 1

            results.append((i, cleaned))

    # Log warning if many pages are low quality (likely scanned)
    if n_pages > 0 and low_quality_pages / n_pages > 0.3:
        import logging
        logging.getLogger(__name__).warning(
            f"{pdf_path}: {low_quality_pages}/{n_pages} pages have <100 chars. "
            "PDF may be scanned — consider OCR fallback."
        )

    return results


def _clean_text(text: str) -> str:
    """
    Normalize extracted PDF text:
    - De-hyphenate line-break hyphens (foo-\nbar → foobar)
    - Collapse repeated blank lines to one
    - Normalize unicode whitespace
    - Strip leading/trailing whitespace
    """
    # De-hyphenate: word-\nnextword → wordnextword
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize whitespace characters (tabs, non-breaking spaces, etc.)
    text = text.replace("\t", " ").replace("\xa0", " ")

    # Collapse 3+ newlines to 2 (preserves paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces to one
    text = re.sub(r" {2,}", " ", text)

    return text.strip()
