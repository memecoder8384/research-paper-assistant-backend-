import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import query_similar_chunks

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(
    query: str,
    top_k: Optional[int] = None,
    document_ids: Optional[List[str]] = None,
    persist_dir: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve semantically relevant chunks from ChromaDB for a natural-language query.

    1. Validates the query string
    2. Generates query embedding via Sentence Transformers
    3. Searches ChromaDB collection using cosine similarity
    4. Filters by document IDs if specified
    5. Preserves all citation metadata (filename, page numbers, start/end pages)
    """
    if not query or not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty or whitespace only.")

    k = top_k if (top_k is not None and top_k > 0) else settings.default_top_k
    k = min(k, 50)  # Bound top_k to reasonable upper limit

    # Clean document_ids list (ignore None or empty strings)
    valid_doc_ids = [did.strip() for did in document_ids if did and did.strip()] if document_ids else None

    # Generate query embedding vector
    query_embedding = EmbeddingService.generate_query_embedding(query)

    # Query persistent ChromaDB
    chunks = query_similar_chunks(
        query_embedding=query_embedding,
        top_k=k,
        document_ids=valid_doc_ids,
        persist_dir=persist_dir,
    )

    return chunks

