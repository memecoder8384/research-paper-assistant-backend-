"""Pydantic schemas for research paper comparison."""
from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.chat import SourceCitation


class PaperComparisonDetail(BaseModel):
    """Comparative profile of an individual research paper."""

    document_id: str
    filename: str
    objective: str
    methodology: str
    dataset: Optional[str] = "Not explicitly discussed."
    algorithms_or_models: Optional[str] = "Not explicitly discussed."
    evaluation_metrics: Optional[str] = "Not explicitly discussed."
    main_findings: str
    limitations: Optional[str] = "Not explicitly discussed."
    future_work: Optional[str] = "Not explicitly discussed."


class ComparisonRequest(BaseModel):
    """Request payload for comparing multiple research papers."""

    document_ids: List[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="List of 2 to 5 document IDs to compare",
    )


class ComparisonResponse(BaseModel):
    """Structured comparison response across multiple research papers."""

    documents_compared: List[str]
    criteria: List[str]
    individual_papers: List[PaperComparisonDetail]
    similarities: List[str]
    differences: List[str]
    methodological_differences: str
    overall_analysis: str
    sources: List[SourceCitation]
    status: str = "success"
