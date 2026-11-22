from fastapi import APIRouter, HTTPException, status
from app.retrieval.retriever import retrieve_relevant_chunks
from app.retrieval.vector_store import get_collection_stats
from app.schemas.retrieval import (
    ChunkResult,
    RetrievalQueryRequest,
    RetrievalQueryResponse,
    RetrievalStatsResponse,
)

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


@router.post(
    "/search",
    response_model=RetrievalQueryResponse,
    summary="Semantic document chunk search",
)
async def semantic_search(payload: RetrievalQueryRequest):
    """Retrieve semantically relevant chunks for a natural-language question.

    - Computes query embedding using Sentence Transformers
    - Searches ChromaDB collection using cosine similarity
    - Preserves source citations (filename, page numbers, start/end pages)
    - Returns structured results without exposing internal server paths
    """
    try:
        raw_results = retrieve_relevant_chunks(
            query=payload.query,
            top_k=payload.top_k,
            document_ids=payload.document_ids,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval service encountered an internal error. Please try again.",
        )

    chunk_results = [
        ChunkResult(
            chunk_id=item["chunk_id"],
            filename=item["filename"],
            document_id=item["document_id"],
            page_number=item["page_number"],
            start_page=item["start_page"],
            end_page=item["end_page"],
            chunk_index=item["chunk_index"],
            text=item["text"],
            distance=item["distance"],
        )
        for item in raw_results
    ]

    return RetrievalQueryResponse(
        query=payload.query,
        total_results=len(chunk_results),
        results=chunk_results,
    )


@router.get(
    "/stats",
    response_model=RetrievalStatsResponse,
    summary="Vector database statistics",
)
async def retrieval_stats():
    """Retrieve vector store inspection statistics."""
    try:
        stats = get_collection_stats()
        return RetrievalStatsResponse(
            collection_name=stats["collection_name"],
            total_chunks=stats["total_chunks"],
            total_documents=stats["total_documents"],
            embedding_model=stats["embedding_model"],
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve vector store statistics.",
        )

