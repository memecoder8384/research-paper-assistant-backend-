import logging
from typing import Any, Dict, List, Optional
import uuid
from app.retrieval.retriever import retrieve_relevant_chunks
from app.services.citation_service import extract_citations
from app.services.context_builder import SYSTEM_RAG_PROMPT, build_rag_context
from app.services.llm_service import LLMService
from app.services.session_store import SessionStore

logger = logging.getLogger(__name__)


def answer_research_question(
    question: str,
    conversation_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Orchestrate Retrieval-Augmented Generation (RAG) question answering.

    Workflow:
    1. Validate input question
    2. Retrieve top-k relevant document chunks via ChromaDB semantic search
    3. Extract and deduplicate source citations (preserving filenames and page numbers)
    4. Handle empty retrieval cleanly (return grounded fallback without LLM call)
    5. Construct structured academic context
    6. Attach conversation history
    7. Generate grounded answer via Groq LLM
    8. Record turn in session history and return structured response
    """
    clean_question = question.strip() if question else ""
    if not clean_question:
        raise ValueError("Question cannot be empty or whitespace only.")
    if len(clean_question) > 1000:
        raise ValueError("Question exceeds maximum allowed length of 1000 characters.")

    session_id = conversation_id.strip() if (conversation_id and conversation_id.strip()) else str(uuid.uuid4())

    # Step 1: Semantic retrieval
    retrieved_chunks = retrieve_relevant_chunks(
        query=clean_question,
        top_k=top_k,
        document_ids=document_ids,
    )

    # Step 2: Extract citations from retrieved chunks
    citations = extract_citations(retrieved_chunks)

    # Step 3: Handle empty retrieval (no indexed docs or no results)
    if not retrieved_chunks:
        fallback_msg = (
            "The provided research paper(s) do not contain sufficient information to answer this question. "
            "No relevant document sections were found in the database."
        )
        SessionStore.add_turn(session_id, clean_question, fallback_msg)
        return {
            "question": clean_question,
            "answer": fallback_msg,
            "sources": [],
            "retrieved_chunks": 0,
            "conversation_id": session_id,
            "status": "success",
        }

    # Step 4: Build context block
    context_str = build_rag_context(retrieved_chunks)

    # Step 5: Prepare messages with conversation history
    history = SessionStore.get_history(session_id)
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_RAG_PROMPT}]

    # Include recent dialogue turns
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add the current prompt containing the retrieved paper context and question
    user_prompt = (
        f"Context from uploaded research papers:\n\n"
        f"{context_str}\n\n"
        f"User Question:\n{clean_question}"
    )
    messages.append({"role": "user", "content": user_prompt})

    # Step 6: Call Groq LLM
    answer = LLMService.generate_chat_completion(messages)

    # Step 7: Update conversation history
    SessionStore.add_turn(session_id, clean_question, answer)

    return {
        "question": clean_question,
        "answer": answer,
        "sources": citations,
        "retrieved_chunks": len(retrieved_chunks),
        "conversation_id": session_id,
        "status": "success",
    }
