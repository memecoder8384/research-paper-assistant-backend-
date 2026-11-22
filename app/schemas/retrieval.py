"""Pydantic schemas for semantic retrieval and vector search."""
from typing import List, Optional
from pydantic import BaseModel, Field


class RetrievalQueryRequest(BaseModel):
    """Request payload for semantic document search."""

    query: str = Field(..., min_length=1, description="Natural-language search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of relevant chunks to retrieve")
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific document IDs to filter the search",
    )


class ChunkResult(BaseModel):
    """Metadata and content for a retrieved text chunk."""

    chunk_id: str
    filename: str
    document_id: str
    page_number: int
    start_page: int
    end_page: int
    chunk_index: int
    text: str
    distance: float


class RetrievalQueryResponse(BaseModel):
    """Response payload containing semantic retrieval results."""

    query: str
    total_results: int
    results: List[ChunkResult]


class RetrievalStatsResponse(BaseModel):
    """ChromaDB collection statistics."""

    collection_name: str
    total_chunks: int
    total_documents: int
    embedding_model: str

