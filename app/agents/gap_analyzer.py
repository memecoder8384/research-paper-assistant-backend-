import logging
from typing import Any, Dict, List
from app.agents.orchestrator import ResearchOrchestrator
from app.services.context_builder import build_rag_context

logger = logging.getLogger(__name__)

GAP_ANALYZER_SYSTEM_PROMPT = """You are a senior academic research advisor specializing in research gap discovery and literature analysis.
Your job is to identify explicit limitations and inferred research gaps across the provided research paper(s).

CRITICAL ACADEMIC RULES:
1. Strict Distinction:
   - "explicit_limitation": Limitations or shortcomings explicitly acknowledged by the authors in the paper text.
   - "inferred_potential_gap": Underexplored topics, unanswered questions, or methodological differences discovered by comparing the papers. Clearly mark these as potential opportunities, NEVER as established universal facts.
2. Scope Discipline: Do not claim an idea or topic is absent from the entire global scientific literature simply because it was not covered in these specific uploaded documents.
3. Evidence Grounding: Every identified gap must cite concrete supporting evidence (quotation or paraphrased fact) from the uploaded papers.
4. Format: Return valid JSON ONLY with this exact structure:
{
    "explicit_limitations": [
        "...",
        "..."
    ],
    "author_suggested_future_work": [
        "...",
        "..."
    ],
    "potential_research_gaps": [
        {
            "gap_type": "explicit_limitation",
            "title": "...",
            "description": "...",
            "why_identified": "...",
            "supporting_evidence": "...",
            "suggested_future_direction": "..."
        },
        {
            "gap_type": "inferred_potential_gap",
            "title": "...",
            "description": "...",
            "why_identified": "...",
            "supporting_evidence": "...",
            "suggested_future_direction": "..."
        }
    ],
    "future_research_directions": [
        "...",
        "..."
    ]
}
"""


class GapAnalyzerAgent:
    """Agent that identifies author-stated limitations, suggested future work, and inferred research gaps."""

    @classmethod
    def analyze_gaps(
        cls,
        papers: List[Dict[str, Any]],  # [{"document_id": ..., "filename": ..., "chunks": [...]}, ...]
    ) -> Dict[str, Any]:
        """Analyze one or multiple research papers for academic limitations and research gaps."""
        if not papers:
            raise ValueError("At least one research paper must be provided for gap analysis.")

        paper_blocks: List[str] = []
        doc_meta_list: List[Dict[str, str]] = []

        for p in papers:
            doc_id = p["document_id"]
            fname = p["filename"]
            chunks = p["chunks"]
            doc_meta_list.append({"document_id": doc_id, "filename": fname})

            context = build_rag_context(chunks, max_length=6000)
            paper_blocks.append(
                f"=========================================\n"
                f"DOCUMENT ID: {doc_id}\n"
                f"FILENAME: {fname}\n"
                f"=========================================\n"
                f"{context}\n"
            )

        all_papers_text = "\n\n".join(paper_blocks)
        user_prompt = (
            f"Papers to analyze for research gaps:\n{doc_meta_list}\n\n"
            f"Full Document Content:\n{all_papers_text}\n\n"
            f"Perform research gap analysis and return the structured JSON result."
        )

        return ResearchOrchestrator.call_structured_llm(
            system_prompt=GAP_ANALYZER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=3000,
        )
