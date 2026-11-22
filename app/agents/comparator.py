import logging
from typing import Any, Dict, List
from app.agents.orchestrator import ResearchOrchestrator
from app.services.context_builder import build_rag_context

logger = logging.getLogger(__name__)

COMPARATOR_SYSTEM_PROMPT = """You are an expert academic research evaluator specializing in comparative paper analysis.
Your job is to compare two or more uploaded research papers across rigorous academic criteria.

Strict Operational Guidelines:
1. Isolation: Analyze each paper on its own merits using only its provided text. Do not mix findings or attribute one paper's dataset/results to another.
2. Objectivity: Do not make unsupported claims about which research paper is "superior" or "the best". Compare them objectively based on empirical evidence and design trade-offs.
3. Unavailable Information: If a paper does not mention a criterion (e.g. dataset, limitations), mark it as "Not explicitly discussed in the paper."
4. Format: Return valid JSON ONLY with this exact structure:
{
    "criteria": [
        "Research Objective",
        "Methodology",
        "Dataset",
        "Algorithms / Models",
        "Evaluation Metrics",
        "Main Findings",
        "Limitations",
        "Future Work"
    ],
    "individual_papers": [
        {
            "document_id": "...",
            "filename": "...",
            "objective": "...",
            "methodology": "...",
            "dataset": "...",
            "algorithms_or_models": "...",
            "evaluation_metrics": "...",
            "main_findings": "...",
            "limitations": "...",
            "future_work": "..."
        }
    ],
    "similarities": ["...", "..."],
    "differences": ["...", "..."],
    "methodological_differences": "Detailed explanation of key methodological contrasts...",
    "overall_analysis": "Comprehensive comparative synthesis..."
}
"""


class ComparatorAgent:
    """Agent that performs multi-paper comparative analysis."""

    @classmethod
    def compare_papers(
        cls,
        papers: List[Dict[str, Any]],  # [{"document_id": ..., "filename": ..., "chunks": [...]}, ...]
    ) -> Dict[str, Any]:
        """Perform cross-paper comparative analysis across standardized academic criteria."""
        if len(papers) < 2:
            raise ValueError("Comparison requires at least two research papers.")

        # Build separated context blocks for each paper
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
            f"Papers to compare:\n{doc_meta_list}\n\n"
            f"Full Document Content:\n{all_papers_text}\n\n"
            f"Produce the complete structured comparative analysis in valid JSON."
        )

        return ResearchOrchestrator.call_structured_llm(
            system_prompt=COMPARATOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=3000,
        )
