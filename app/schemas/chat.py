"""Pydantic schemas for chat and question answering."""
from typing import List, Optional
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Citation metadata identifying the source document and page."""

    document_id: str
    filename: str
    page_number: int
    start_page: int
    end_page: int
    chunk_id: str


class ChatRequest(BaseModel):
    """Request payload for answering a research question."""

    question: Optional[str] = Field(default=None, max_length=1000, description="Research question to answer")
    message: Optional[str] = Field(default=None, description="Alias for question from frontend")
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific document IDs to restrict the answer to",
    )
    selectedPaperIds: Optional[List[str]] = Field(
        default=None,
        description="Alias for document_ids from frontend",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of relevant chunks to retrieve")
    history: Optional[List[dict]] = Field(default=None, description="Chat history")


class ChatResponse(BaseModel):
    """Response payload containing the grounded answer and source citations."""

    question: str
    answer: str
    sources: List[SourceCitation]
    retrieved_chunks: int
    conversation_id: str
    status: str
    id: Optional[str] = None
    sender: Optional[str] = "assistant"
    content: Optional[str] = None
    citations: Optional[List[dict]] = None
    groundedPapers: Optional[List[str]] = None


class ChatMessage(BaseModel):
    """Individual message in conversation history."""

    role: str
    content: str


class ConversationHistoryResponse(BaseModel):
    """Conversation history response payload."""

    conversation_id: str
    total_messages: int
    messages: List[ChatMessage]
