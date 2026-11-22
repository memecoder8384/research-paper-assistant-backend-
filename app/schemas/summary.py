"""Pydantic schemas for research paper summarization."""
from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.chat import SourceCitation


class SummarySections(BaseModel):
    """Detailed sections extracted from the research paper."""

    objective: str = Field(description="Core research objective or hypothesis")
    problem_statement: Optional[str] = Field(default="Not explicitly discussed in the paper.", description="The specific problem addressed")
    methodology: str = Field(description="Techniques, architectures, or empirical methods used")
    dataset: Optional[str] = Field(default="Not explicitly discussed in the paper.", description="Datasets, benchmarks, or corpora evaluated")
    results: str = Field(description="Key empirical results, accuracy scores, or findings")
    conclusion: str = Field(description="Overall conclusion and implications")
    limitations: Optional[str] = Field(default="Not explicitly discussed in the paper.", description="Author-stated limitations or scope boundaries")
    future_work: Optional[str] = Field(default="Not explicitly discussed in the paper.", description="Future research directions suggested by authors")


class SummaryRequest(BaseModel):
    """Request payload for summarizing a research paper."""

    document_id: str = Field(..., description="Unique ID of the uploaded document to summarize")
    summary_type: str = Field(
        default="detailed",
        description="Format of the summary: 'short', 'detailed', 'academic', or 'key_points'",
    )


class SummaryResponse(BaseModel):
    """Structured response containing the paper summary and section breakdown."""

    document_id: str
    filename: str
    summary_type: str
    summary: str
    sections: SummarySections
    sources: List[SourceCitation]
    status: str = "success"
