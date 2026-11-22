"""Pydantic schemas for research gap analysis."""
from typing import List
from pydantic import BaseModel, Field
from app.schemas.chat import SourceCitation


class ResearchGapItem(BaseModel):
    """An individual research gap with supporting evidence and categorization."""

    gap_type: str = Field(
        description="Category: 'explicit_limitation' (author-stated) or 'inferred_potential_gap' (derived opportunity)"
    )
    title: str = Field(description="Concise descriptive title of the gap")
    description: str = Field(description="Detailed explanation of the research gap")
    why_identified: str = Field(description="Reasoning behind why this was identified as a gap")
    supporting_evidence: str = Field(description="Direct quotation or evidence from the uploaded paper(s)")
    suggested_future_direction: str = Field(description="Concrete recommendation for future research")
    supporting_sources: List[SourceCitation] = Field(default_factory=list, description="Citations supporting this gap")


class GapAnalysisRequest(BaseModel):
    """Request payload for discovering research gaps across papers."""

    document_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="List of 1 to 5 document IDs to analyze for research gaps",
    )


class GapAnalysisResponse(BaseModel):
    """Structured response detailing explicit limitations, author future work, and potential gaps."""

    document_ids: List[str]
    explicit_limitations: List[str]
    author_suggested_future_work: List[str]
    potential_research_gaps: List[ResearchGapItem]
    future_research_directions: List[str]
    sources: List[SourceCitation]
    status: str = "success"
