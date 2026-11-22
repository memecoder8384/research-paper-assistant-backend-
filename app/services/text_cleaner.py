import re
from typing import Any, Dict, List


def clean_text(text: str) -> str:
    """Clean extracted text while preserving paragraph boundaries, punctuation, equations, and citations.

    - Normalizes line endings (\r\n -> \n)
    - Fixes hyphenated line breaks (e.g., 'transfor-\nmation' -> 'transformation')
    - Preserves meaningful paragraph boundaries (\n\n)
    - Unwraps soft line breaks within sentences while preserving list items and formulas
    - Removes redundant horizontal whitespace without destroying mathematical expressions or citations
    """
    if not text:
        return ""

    # 1. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Reconnect hyphenated words split across line breaks (common in multi-column research papers)
    text = re.sub(r"(\b[A-Za-z]+)-\n([A-Za-z]+\b)", r"\1\2", text)

    # 3. Split by paragraphs (separated by two or more newlines)
    raw_paragraphs = re.split(r"\n\s*\n+", text)
    cleaned_paragraphs: List[str] = []

    for para in raw_paragraphs:
        lines = para.split("\n")
        processed_lines: List[str] = []

        for line in lines:
            # Replace multiple consecutive spaces or tabs with a single space
            cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
            if cleaned_line:
                processed_lines.append(cleaned_line)

        if not processed_lines:
            continue

        # Check if lines inside paragraph look like bullet points or numbered lists
        # If so, keep them on separate lines; otherwise, combine them with a space
        is_list = any(re.match(r"^(\*|-|•|\d+\.|\([a-zA-Z0-9]+\))\s+", l) for l in processed_lines)
        if is_list:
            joined_para = "\n".join(processed_lines)
        else:
            joined_para = " ".join(processed_lines)

        cleaned_paragraphs.append(joined_para)

    return "\n\n".join(cleaned_paragraphs)


def clean_extracted_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean text on each extracted page while preserving page numbers and page boundaries."""
    cleaned_pages: List[Dict[str, Any]] = []

    for p in pages:
        cleaned = clean_text(p.get("text", ""))
        cleaned_pages.append(
            {
                "page_number": p["page_number"],
                "text": cleaned,
                "char_count": len(cleaned),
                "is_empty": (len(cleaned) == 0),
            }
        )

    return cleaned_pages

