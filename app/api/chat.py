from fastapi import APIRouter, HTTPException, status
from app.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationHistoryResponse,
    SourceCitation,
)
from app.services.rag_service import answer_research_question
from app.services.session_store import SessionStore

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask a research question (new or one-off conversation)",
)
async def chat_new_question(payload: ChatRequest):
    """Answer a research question using RAG over indexed papers."""
    question = (payload.question or payload.message or "").strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty.",
        )
    document_ids = payload.document_ids or payload.selectedPaperIds

    try:
        result = answer_research_question(
            question=question,
            conversation_id=None,
            document_ids=document_ids,
            top_k=payload.top_k,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except RuntimeError as run_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(run_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the answer. Please try again.",
        )

    sources = [
        SourceCitation(
            document_id=s["document_id"],
            filename=s["filename"],
            page_number=s["page_number"],
            start_page=s["start_page"],
            end_page=s["end_page"],
            chunk_id=s["chunk_id"],
        )
        for s in result["sources"]
    ]

    citations_data = [
        {
            "id": f"cite-{s['chunk_id']}",
            "paperId": s["document_id"],
            "paperTitle": s["filename"],
            "authorsYear": s["filename"],
            "page": f"p. {s['page_number'] or s['start_page']}",
            "quote": f"Passage from {s['filename']} (p. {s['page_number'] or s['start_page']})",
        }
        for s in result["sources"]
    ]
    grounded_papers = list({s["document_id"] for s in result["sources"]})

    return ChatResponse(
        question=result["question"],
        answer=result["answer"],
        sources=sources,
        retrieved_chunks=result["retrieved_chunks"],
        conversation_id=result["conversation_id"],
        status=result["status"],
        id=f"msg-{result['conversation_id']}",
        sender="assistant",
        content=result["answer"],
        citations=citations_data,
        groundedPapers=grounded_papers,
    )


@router.post(
    "/{conversation_id}",
    response_model=ChatResponse,
    summary="Continue a conversation with a follow-up question",
)
async def chat_continue_conversation(conversation_id: str, payload: ChatRequest):
    """Answer a follow-up research question within an existing conversation session."""
    if not conversation_id or not conversation_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conversation ID cannot be empty.",
        )

    question = (payload.question or payload.message or "").strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty.",
        )
    document_ids = payload.document_ids or payload.selectedPaperIds

    try:
        result = answer_research_question(
            question=question,
            conversation_id=conversation_id.strip(),
            document_ids=document_ids,
            top_k=payload.top_k,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except RuntimeError as run_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(run_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the answer. Please try again.",
        )

    sources = [
        SourceCitation(
            document_id=s["document_id"],
            filename=s["filename"],
            page_number=s["page_number"],
            start_page=s["start_page"],
            end_page=s["end_page"],
            chunk_id=s["chunk_id"],
        )
        for s in result["sources"]
    ]

    citations_data = [
        {
            "id": f"cite-{s['chunk_id']}",
            "paperId": s["document_id"],
            "paperTitle": s["filename"],
            "authorsYear": s["filename"],
            "page": f"p. {s['page_number'] or s['start_page']}",
            "quote": f"Passage from {s['filename']} (p. {s['page_number'] or s['start_page']})",
        }
        for s in result["sources"]
    ]
    grounded_papers = list({s["document_id"] for s in result["sources"]})

    return ChatResponse(
        question=result["question"],
        answer=result["answer"],
        sources=sources,
        retrieved_chunks=result["retrieved_chunks"],
        conversation_id=result["conversation_id"],
        status=result["status"],
        id=f"msg-{result['conversation_id']}",
        sender="assistant",
        content=result["answer"],
        citations=citations_data,
        groundedPapers=grounded_papers,
    )


@router.get(
    "/{conversation_id}/history",
    response_model=ConversationHistoryResponse,
    summary="Get conversation history",
)
async def get_conversation_history(conversation_id: str):
    """Retrieve message history for an active conversation session."""
    history = SessionStore.get_history(conversation_id)
    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in history]

    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        total_messages=len(messages),
        messages=messages,
    )


@router.delete(
    "/{conversation_id}",
    summary="Clear conversation history",
)
async def clear_conversation(conversation_id: str):
    """Delete conversation history for the specified session."""
    cleared = SessionStore.clear_history(conversation_id)
    return {
        "success": True,
        "message": "Conversation history cleared successfully.",
        "conversation_id": conversation_id,
        "had_prior_history": cleared,
    }
