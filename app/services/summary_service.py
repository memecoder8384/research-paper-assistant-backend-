import logging
from typing import Any, Dict
from app.agents.summarizer import SummarizerAgent
from app.db.database import get_document, get_document_chunks
from app.schemas.summary import SummarySections
from app.services.citation_service import extract_citations

logger = logging.getLogger(__name__)

VALID_SUMMARY_TYPES = {"short", "detailed", "academic", "key_points"}


def generate_paper_summary(document_id: str, summary_type: str = "detailed") -> Dict[str, Any]:
    """Generate a structured research summary for a specific uploaded paper."""
    if not document_id or not document_id.strip():
        raise ValueError("Document ID cannot be empty.")

    doc_id = document_id.strip()
    doc = get_document(doc_id)
    if not doc:
        raise KeyError(f"Document with ID '{doc_id}' not found.")

    if doc.get("is_scanned"):
        raise ValueError("Cannot summarize a scanned or image-only document without extractable text.")

    stype = summary_type.lower().strip() if summary_type else "detailed"
    if stype not in VALID_SUMMARY_TYPES:
        raise ValueError(f"Invalid summary type '{summary_type}'. Allowed: {sorted(VALID_SUMMARY_TYPES)}")

    chunks = get_document_chunks(doc_id)
    if not chunks:
        raise ValueError(f"No extracted text chunks found for document '{doc_id}'.")

    # Generate structured summary via agent
    result = SummarizerAgent.summarize_paper(
        filename=doc["filename"],
        chunks=chunks,
        summary_type=stype,
    )

    # Format sections safely
    raw_sections = result.get("sections", {})
    sections = SummarySections(
        objective=raw_sections.get("objective", "Not explicitly discussed in the paper."),
        problem_statement=raw_sections.get("problem_statement", "Not explicitly discussed in the paper."),
        methodology=raw_sections.get("methodology", "Not explicitly discussed in the paper."),
        dataset=raw_sections.get("dataset", "Not explicitly discussed in the paper."),
        results=raw_sections.get("results", "Not explicitly discussed in the paper."),
        conclusion=raw_sections.get("conclusion", "Not explicitly discussed in the paper."),
        limitations=raw_sections.get("limitations", "Not explicitly discussed in the paper."),
        future_work=raw_sections.get("future_work", "Not explicitly discussed in the paper."),
    )

    citations = extract_citations(chunks)

    return {
        "document_id": doc_id,
        "filename": doc["filename"],
        "summary_type": stype,
        "summary": result.get("summary", ""),
        "sections": sections,
        "sources": citations,
        "status": "success",
    }
