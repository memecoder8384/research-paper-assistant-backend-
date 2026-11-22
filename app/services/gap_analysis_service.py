import logging
from typing import Any, Dict, List
from app.agents.gap_analyzer import GapAnalyzerAgent
from app.db.database import get_document, get_document_chunks
from app.schemas.chat import SourceCitation
from app.schemas.research_gaps import ResearchGapItem
from app.services.citation_service import extract_citations

logger = logging.getLogger(__name__)


def analyze_research_gaps(document_ids: List[str]) -> Dict[str, Any]:
    """Extract explicit limitations, author future work, and inferred research gaps across papers."""
    if not document_ids:
        raise ValueError("Document IDs list cannot be empty.")

    # Deduplicate while preserving order
    unique_ids: List[str] = []
    for d in document_ids:
        did = d.strip() if d else ""
        if did and did not in unique_ids:
            unique_ids.append(did)

    if not unique_ids:
        raise ValueError("At least one valid document ID is required.")
    if len(unique_ids) > 5:
        raise ValueError("Gap analysis supports at most 5 papers simultaneously.")

    papers_data: List[Dict[str, Any]] = []
    missing_ids: List[str] = []
    all_chunks: List[Dict[str, Any]] = []

    for did in unique_ids:
        doc = get_document(did)
        if not doc:
            missing_ids.append(did)
            continue

        if doc.get("is_scanned"):
            raise ValueError(f"Document '{doc['filename']}' ({did}) is scanned/image-only and cannot be analyzed.")

        chunks = get_document_chunks(did)
        if not chunks:
            raise ValueError(f"Document '{doc['filename']}' ({did}) contains no text chunks for gap analysis.")

        papers_data.append(
            {
                "document_id": did,
                "filename": doc["filename"],
                "chunks": chunks,
            }
        )
        all_chunks.extend(chunks)

    if missing_ids:
        raise KeyError(f"The following document IDs were not found: {', '.join(missing_ids)}")

    # Execute gap analysis agent
    raw_res = GapAnalyzerAgent.analyze_gaps(papers_data)

    all_citations = extract_citations(all_chunks)
    source_citation_models = [
        SourceCitation(
            document_id=c["document_id"],
            filename=c["filename"],
            page_number=c["page_number"],
            start_page=c["start_page"],
            end_page=c["end_page"],
            chunk_id=c["chunk_id"],
        )
        for c in all_citations
    ]

    # Format research gap items
    raw_gaps = raw_res.get("potential_research_gaps", [])
    potential_gaps: List[ResearchGapItem] = []
    for g in raw_gaps:
        # Match citations relevant to this gap if available
        g_type = g.get("gap_type", "inferred_potential_gap")
        if g_type not in ("explicit_limitation", "inferred_potential_gap"):
            g_type = "inferred_potential_gap"

        potential_gaps.append(
            ResearchGapItem(
                gap_type=g_type,
                title=g.get("title", "Research Gap Opportunity"),
                description=g.get("description", ""),
                why_identified=g.get("why_identified", ""),
                supporting_evidence=g.get("supporting_evidence", ""),
                suggested_future_direction=g.get("suggested_future_direction", ""),
                supporting_sources=source_citation_models[:2],  # Associate top sources
            )
        )

    return {
        "document_ids": unique_ids,
        "explicit_limitations": raw_res.get("explicit_limitations", []),
        "author_suggested_future_work": raw_res.get("author_suggested_future_work", []),
        "potential_research_gaps": potential_gaps,
        "future_research_directions": raw_res.get("future_research_directions", []),
        "sources": source_citation_models,
        "status": "success",
    }
