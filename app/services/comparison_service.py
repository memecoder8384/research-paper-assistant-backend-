import logging
from typing import Any, Dict, List
from app.agents.comparator import ComparatorAgent
from app.db.database import get_document, get_document_chunks
from app.schemas.comparison import PaperComparisonDetail
from app.services.citation_service import extract_citations

logger = logging.getLogger(__name__)


def compare_research_papers(document_ids: List[str]) -> Dict[str, Any]:
    """Perform comparative analysis between two or more uploaded research papers."""
    if not document_ids:
        raise ValueError("Document IDs list cannot be empty.")

    # Deduplicate while preserving order
    unique_ids: List[str] = []
    for d in document_ids:
        did = d.strip() if d else ""
        if did and did not in unique_ids:
            unique_ids.append(did)

    if len(unique_ids) < 2:
        raise ValueError("Comparison requires at least two distinct research paper IDs.")
    if len(unique_ids) > 5:
        raise ValueError("Comparison supports at most 5 papers simultaneously to maintain context fidelity.")

    # Fetch and validate all documents
    papers_data: List[Dict[str, Any]] = []
    missing_ids: List[str] = []
    all_chunks: List[Dict[str, Any]] = []

    for did in unique_ids:
        doc = get_document(did)
        if not doc:
            missing_ids.append(did)
            continue

        if doc.get("is_scanned"):
            raise ValueError(f"Document '{doc['filename']}' ({did}) is scanned/image-only and cannot be compared.")

        chunks = get_document_chunks(did)
        if not chunks:
            raise ValueError(f"Document '{doc['filename']}' ({did}) contains no text chunks for comparison.")

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

    # Execute agent comparison
    raw_res = ComparatorAgent.compare_papers(papers_data)

    # Format individual papers cleanly
    raw_individual = raw_res.get("individual_papers", [])
    individual_papers: List[PaperComparisonDetail] = []
    for p in raw_individual:
        individual_papers.append(
            PaperComparisonDetail(
                document_id=p.get("document_id", ""),
                filename=p.get("filename", ""),
                objective=p.get("objective", "Not explicitly discussed."),
                methodology=p.get("methodology", "Not explicitly discussed."),
                dataset=p.get("dataset", "Not explicitly discussed."),
                algorithms_or_models=p.get("algorithms_or_models", "Not explicitly discussed."),
                evaluation_metrics=p.get("evaluation_metrics", "Not explicitly discussed."),
                main_findings=p.get("main_findings", "Not explicitly discussed."),
                limitations=p.get("limitations", "Not explicitly discussed."),
                future_work=p.get("future_work", "Not explicitly discussed."),
            )
        )

    sources = extract_citations(all_chunks)

    return {
        "documents_compared": [p["filename"] for p in papers_data],
        "criteria": raw_res.get(
            "criteria",
            [
                "Research Objective",
                "Methodology",
                "Dataset",
                "Algorithms / Models",
                "Evaluation Metrics",
                "Main Findings",
                "Limitations",
                "Future Work",
            ],
        ),
        "individual_papers": individual_papers,
        "similarities": raw_res.get("similarities", []),
        "differences": raw_res.get("differences", []),
        "methodological_differences": raw_res.get("methodological_differences", ""),
        "overall_analysis": raw_res.get("overall_analysis", ""),
        "sources": sources,
        "status": "success",
    }
