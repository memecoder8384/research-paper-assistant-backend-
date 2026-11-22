import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.db.database import get_document, save_document, update_document_status
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import add_document_chunks

logger = logging.getLogger(__name__)


def index_document(
    document_id: str,
    filename: str,
    chunks: List[Dict[str, Any]],
    db_path: Optional[str | Path] = None,
    chroma_persist_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Connect Phase 2 chunking with embedding generation and ChromaDB indexing.

    Sequence:
    1. Validate chunk data.
    2. Generate dense vector embeddings using Sentence Transformers.
    3. Insert chunks, embeddings, and scalar metadata into persistent ChromaDB.
    4. Update SQLite document status to 'indexed'.
    5. On error, record 'indexing_failed' status and error message in SQLite.
    """
    if not chunks:
        logger.warning("No chunks provided for indexing document %s", document_id)
        return {"status": "no_chunks", "indexed_chunks": 0}

    # Extract chunk texts
    texts = [c["text"] for c in chunks]

    try:
        # 1. Generate embeddings (using batch encoding)
        logger.info("Generating embeddings for %d chunks of document %s", len(texts), document_id)
        embeddings = EmbeddingService.generate_embeddings(texts)

        # 2. Insert into ChromaDB collection
        indexed_count = add_document_chunks(
            document_id=document_id,
            filename=filename,
            chunks=chunks,
            embeddings=embeddings,
            persist_dir=chroma_persist_dir,
        )

        # 3. Update persistent SQLite document status to 'indexed'
        update_document_status(
            document_id=document_id,
            status="indexed",
            message="Document processed, chunked, and indexed in ChromaDB successfully.",
            custom_path=db_path,
        )

        return {
            "status": "indexed",
            "indexed_chunks": indexed_count,
            "document_id": document_id,
        }

    except Exception as exc:
        logger.error("Failed indexing document %s: %s", document_id, exc)
        # Update SQLite document status to 'indexing_failed'
        try:
            update_document_status(
                document_id=document_id,
                status="indexing_failed",
                message=f"Vector indexing failed: {str(exc)}",
                custom_path=db_path,
            )
        except Exception:
            pass

        raise RuntimeError(f"Vector indexing failed for document {document_id}: {str(exc)}") from exc

