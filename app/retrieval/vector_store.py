import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from app.core.config import settings

logger = logging.getLogger(__name__)

# Cache client instances by path to avoid duplicate client initialization
_clients: Dict[str, ClientAPI] = {}


def get_chroma_client(persist_dir: Optional[str | Path] = None) -> ClientAPI:
    """Initialize or reuse a persistent ChromaDB client for the specified directory."""
    path_str = str(Path(persist_dir) if persist_dir else settings.chroma_dir)
    if path_str not in _clients:
        Path(path_str).mkdir(parents=True, exist_ok=True)
        _clients[path_str] = chromadb.PersistentClient(path=path_str)
    return _clients[path_str]


def get_collection(
    client: Optional[ClientAPI] = None,
    collection_name: Optional[str] = None,
    persist_dir: Optional[str | Path] = None,
) -> Collection:
    """Get or create the target ChromaDB collection configured with cosine distance."""
    cli = client or get_chroma_client(persist_dir)
    target_name = collection_name or settings.chroma_collection_name
    return cli.get_or_create_collection(
        name=target_name,
        metadata={"hnsw:space": "cosine"},
    )


def normalize_metadata(meta: Dict[str, Any]) -> Dict[str, str | int | float | bool]:
    """Ensure all metadata values are valid ChromaDB scalar types (str, int, float, bool)."""
    normalized = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            normalized[k] = v
        elif v is None:
            normalized[k] = ""
        else:
            normalized[k] = str(v)
    return normalized


def add_document_chunks(
    document_id: str,
    filename: str,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    collection: Optional[Collection] = None,
    persist_dir: Optional[str | Path] = None,
) -> int:
    """Insert text chunks and embeddings into ChromaDB with duplicate protection.

    - Removes any pre-existing chunks for this document_id first (idempotent ingestion)
    - Validates scalar metadata
    - Uses stable chunk IDs (e.g. '{doc_id}#chunk_{index}')
    """
    if not chunks:
        return 0

    col = collection or get_collection(persist_dir=persist_dir)

    # 1. Prevent duplicate ingestion by deleting prior records for this document
    delete_document_chunks(document_id, collection=col)

    # 2. Build records
    chunk_ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for chunk in chunks:
        idx = chunk.get("chunk_index", 0)
        c_id = f"{document_id}#chunk_{idx}"
        chunk_ids.append(c_id)
        documents.append(chunk.get("text", ""))

        raw_meta = {
            "document_id": document_id,
            "filename": filename,
            "chunk_id": c_id,
            "chunk_index": int(idx),
            "page_number": int(chunk.get("page_number", chunk.get("start_page", 1))),
            "start_page": int(chunk.get("start_page", 1)),
            "end_page": int(chunk.get("end_page", 1)),
        }
        metadatas.append(normalize_metadata(raw_meta))

    col.add(
        ids=chunk_ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info("Indexed %d chunks for document %s in ChromaDB", len(chunks), document_id)
    return len(chunks)


def delete_document_chunks(
    document_id: str,
    collection: Optional[Collection] = None,
    persist_dir: Optional[str | Path] = None,
) -> int:
    """Delete all chunks belonging to a document from ChromaDB."""
    col = collection or get_collection(persist_dir=persist_dir)
    try:
        col.delete(where={"document_id": document_id})
        logger.info("Deleted chunks for document %s from ChromaDB", document_id)
        return 1
    except Exception as exc:
        logger.warning("Error deleting chunks for document %s: %s", document_id, exc)
        return 0


def query_similar_chunks(
    query_embedding: List[float],
    top_k: int = 5,
    document_ids: Optional[List[str]] = None,
    collection: Optional[Collection] = None,
    persist_dir: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Query ChromaDB for semantically similar chunks.

    Returns structured chunk records including source metadata and raw cosine distance.
    """
    col = collection or get_collection(persist_dir=persist_dir)
    total_count = col.count()
    if total_count == 0:
        return []

    # Configure filtering by document_ids if provided
    where_clause = None
    if document_ids and len(document_ids) > 0:
        if len(document_ids) == 1:
            where_clause = {"document_id": document_ids[0]}
        else:
            where_clause = {"document_id": {"$in": document_ids}}

    k = min(top_k, total_count)
    if k <= 0:
        return []

    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_clause:
        query_kwargs["where"] = where_clause

    try:
        results = col.query(**query_kwargs)
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        return []

    retrieved: List[Dict[str, Any]] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        raw_dist = distances[i] if i < len(distances) else 0.0

        retrieved.append(
            {
                "chunk_id": ids[i],
                "text": docs[i] if i < len(docs) else "",
                "document_id": meta.get("document_id", ""),
                "filename": meta.get("filename", ""),
                "page_number": int(meta.get("page_number", meta.get("start_page", 1))),
                "start_page": int(meta.get("start_page", 1)),
                "end_page": int(meta.get("end_page", 1)),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "distance": round(float(raw_dist), 4),
            }
        )

    return retrieved


def get_collection_stats(
    collection: Optional[Collection] = None,
    persist_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Retrieve collection inspection statistics."""
    col = collection or get_collection(persist_dir=persist_dir)
    total_chunks = col.count()

    total_docs = 0
    if total_chunks > 0:
        # Sample or query all document_ids in metadata
        all_meta = col.get(include=["metadatas"])
        metas = all_meta.get("metadatas", [])
        unique_docs = {m.get("document_id") for m in metas if m and "document_id" in m}
        total_docs = len(unique_docs)

    return {
        "collection_name": col.name,
        "total_chunks": total_chunks,
        "total_documents": total_docs,
        "embedding_model": settings.embedding_model_name,
    }


def clear_collection(
    client: Optional[ClientAPI] = None,
    collection_name: Optional[str] = None,
    persist_dir: Optional[str | Path] = None,
) -> None:
    """Clear all records in the collection only when explicitly requested."""
    cli = client or get_chroma_client(persist_dir)
    target_name = collection_name or settings.chroma_collection_name
    try:
        cli.delete_collection(target_name)
    except Exception:
        pass
    cli.get_or_create_collection(name=target_name, metadata={"hnsw:space": "cosine"})

