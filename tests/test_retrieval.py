import io
from pathlib import Path
import pytest
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.indexer import index_document
from app.retrieval.retriever import retrieve_relevant_chunks
from app.retrieval.vector_store import (
    add_document_chunks,
    clear_collection,
    delete_document_chunks,
    get_chroma_client,
    get_collection,
    get_collection_stats,
    query_similar_chunks,
)
from tests.conftest import make_pdf_bytes


# 1. Embedding generation test
def test_embedding_generation():
    texts = [
        "Attention mechanisms in transformer networks.",
        "Convolutional neural networks for image recognition.",
    ]
    embeddings = EmbeddingService.generate_embeddings(texts)
    assert len(embeddings) == 2
    assert isinstance(embeddings[0], list)
    assert isinstance(embeddings[0][0], float)


# 2. Query embedding generation test
def test_query_embedding_generation():
    query = "What is self-attention?"
    emb = EmbeddingService.generate_query_embedding(query)
    assert isinstance(emb, list)
    assert len(emb) == 384


# 3. Embedding dimensions test
def test_embedding_dimensions():
    texts = ["Sample sentence for dimension check."]
    embs = EmbeddingService.generate_embeddings(texts)
    assert len(embs[0]) == 384


# 4. Empty text validation test
def test_empty_text_validation():
    with pytest.raises(ValueError, match="cannot be empty or whitespace only"):
        EmbeddingService.generate_embeddings(["Valid text", "   "])

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        EmbeddingService.generate_query_embedding("   ")


# 5. ChromaDB insertion test
def test_chromadb_insertion(temp_storage):
    col = get_collection(persist_dir=temp_storage / "chroma")
    chunks = [
        {
            "chunk_index": 0,
            "text": "First chunk on neural networks.",
            "page_number": 1,
            "start_page": 1,
            "end_page": 1,
        },
        {
            "chunk_index": 1,
            "text": "Second chunk on loss functions.",
            "page_number": 2,
            "start_page": 2,
            "end_page": 2,
        },
    ]
    embeddings = EmbeddingService.generate_embeddings([c["text"] for c in chunks])

    added = add_document_chunks(
        document_id="doc-001",
        filename="nn_paper.pdf",
        chunks=chunks,
        embeddings=embeddings,
        collection=col,
    )
    assert added == 2
    assert col.count() == 2


# 6. Persistent database reopening test
def test_persistent_database_reopening(temp_storage):
    chroma_path = temp_storage / "chroma"
    col1 = get_collection(persist_dir=chroma_path)
    chunks = [
        {"chunk_index": 0, "text": "Persistent vector data.", "page_number": 1, "start_page": 1, "end_page": 1}
    ]
    embeddings = EmbeddingService.generate_embeddings([c["text"] for c in chunks])
    add_document_chunks("doc-p", "p.pdf", chunks, embeddings, collection=col1)
    assert col1.count() == 1

    # Reopen using a new client pointing to the same persistent path
    client2 = get_chroma_client(persist_dir=chroma_path)
    col2 = get_collection(client=client2)
    assert col2.count() == 1


# 7. Duplicate indexing prevention test
def test_duplicate_indexing_prevention(temp_storage):
    col = get_collection(persist_dir=temp_storage / "chroma")
    chunks = [
        {"chunk_index": 0, "text": "Chunk text version 1.", "page_number": 1, "start_page": 1, "end_page": 1}
    ]
    embs = EmbeddingService.generate_embeddings([c["text"] for c in chunks])

    # Index once
    add_document_chunks("doc-dup", "paper.pdf", chunks, embs, collection=col)
    assert col.count() == 1

    # Index again with updated text
    chunks_v2 = [
        {"chunk_index": 0, "text": "Chunk text version 2.", "page_number": 1, "start_page": 1, "end_page": 1}
    ]
    embs_v2 = EmbeddingService.generate_embeddings([c["text"] for c in chunks_v2])
    add_document_chunks("doc-dup", "paper.pdf", chunks_v2, embs_v2, collection=col)

    # Count must remain 1 (no duplicate chunks)
    assert col.count() == 1


# 8. Semantic retrieval test
def test_semantic_retrieval(temp_storage):
    chroma_path = temp_storage / "chroma"
    chunks = [
        {"chunk_index": 0, "text": "Optimization algorithms like Adam and SGD.", "page_number": 1, "start_page": 1, "end_page": 1},
        {"chunk_index": 1, "text": "Reinforcement learning and Markov Decision Processes.", "page_number": 2, "start_page": 2, "end_page": 2},
    ]
    embs = EmbeddingService.generate_embeddings([c["text"] for c in chunks])
    add_document_chunks("doc-rl", "ml_guide.pdf", chunks, embs, persist_dir=chroma_path)

    results = retrieve_relevant_chunks(
        query="Tell me about Adam and gradient descent",
        top_k=2,
        persist_dir=chroma_path,
    )
    assert len(results) > 0
    assert "Optimization algorithms" in results[0]["text"]


# 9. Top-k result limits test
def test_top_k_result_limits(temp_storage):
    chroma_path = temp_storage / "chroma"
    chunks = [
        {"chunk_index": i, "text": f"Machine learning concept number {i}.", "page_number": i + 1, "start_page": i + 1, "end_page": i + 1}
        for i in range(10)
    ]
    embs = EmbeddingService.generate_embeddings([c["text"] for c in chunks])
    add_document_chunks("doc-many", "ml.pdf", chunks, embs, persist_dir=chroma_path)

    results = retrieve_relevant_chunks(query="Machine learning", top_k=3, persist_dir=chroma_path)
    assert len(results) == 3


# 10. Document-specific filtering test
def test_document_specific_filtering(temp_storage):
    chroma_path = temp_storage / "chroma"
    # Document A
    chunks_a = [{"chunk_index": 0, "text": "Alpha document discussion.", "page_number": 1, "start_page": 1, "end_page": 1}]
    embs_a = EmbeddingService.generate_embeddings([c["text"] for c in chunks_a])
    add_document_chunks("doc-A", "alpha.pdf", chunks_a, embs_a, persist_dir=chroma_path)

    # Document B
    chunks_b = [{"chunk_index": 0, "text": "Beta document discussion.", "page_number": 1, "start_page": 1, "end_page": 1}]
    embs_b = EmbeddingService.generate_embeddings([c["text"] for c in chunks_b])
    add_document_chunks("doc-B", "beta.pdf", chunks_b, embs_b, persist_dir=chroma_path)

    # Query filtered only to doc-B
    results = retrieve_relevant_chunks(
        query="document discussion",
        top_k=5,
        document_ids=["doc-B"],
        persist_dir=chroma_path,
    )
    assert len(results) == 1
    assert results[0]["document_id"] == "doc-B"
    assert results[0]["filename"] == "beta.pdf"


# 11. Metadata preservation test
def test_metadata_preservation(temp_storage):
    chroma_path = temp_storage / "chroma"
    chunks = [
        {
            "chunk_index": 4,
            "text": "Cross-page chunk spanning abstract and introduction.",
            "page_number": 2,
            "start_page": 2,
            "end_page": 3,
        }
    ]
    embs = EmbeddingService.generate_embeddings([c["text"] for c in chunks])
    add_document_chunks("doc-meta", "preservation.pdf", chunks, embs, persist_dir=chroma_path)

    results = retrieve_relevant_chunks(query="Cross-page chunk", top_k=1, persist_dir=chroma_path)
    assert len(results) == 1
    res = results[0]
    assert res["document_id"] == "doc-meta"
    assert res["filename"] == "preservation.pdf"
    assert res["page_number"] == 2
    assert res["start_page"] == 2
    assert res["end_page"] == 3
    assert res["chunk_index"] == 4
    assert isinstance(res["distance"], float)


# 12. Document deletion test
def test_document_deletion_removes_from_chroma(temp_storage):
    chroma_path = temp_storage / "chroma"
    col = get_collection(persist_dir=chroma_path)
    chunks = [{"chunk_index": 0, "text": "To be deleted.", "page_number": 1, "start_page": 1, "end_page": 1}]
    embs = EmbeddingService.generate_embeddings([c["text"] for c in chunks])
    add_document_chunks("doc-del", "del.pdf", chunks, embs, collection=col)
    assert col.count() == 1

    delete_document_chunks("doc-del", collection=col)
    assert col.count() == 0


# 13. Empty collection behaviour test
def test_empty_collection_behaviour(temp_storage):
    chroma_path = temp_storage / "chroma"
    clear_collection(persist_dir=chroma_path)

    results = retrieve_relevant_chunks(query="Any search query", top_k=5, persist_dir=chroma_path)
    assert results == []


# 14. Invalid query handling test
def test_invalid_query_handling():
    with pytest.raises(ValueError, match="cannot be empty"):
        retrieve_relevant_chunks(query="")

    with pytest.raises(ValueError, match="cannot be empty"):
        retrieve_relevant_chunks(query="   ")


# 15. Retrieval API endpoints test
def test_retrieval_api_endpoints(client):
    # 1. Upload a PDF to trigger indexing
    pdf_bytes = make_pdf_bytes(["A study on deep reinforcement learning algorithms and Q-networks."])
    upload_res = client.post(
        "/documents/upload",
        files={"file": ("rl_paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert upload_res.status_code == 201
    upload_data = upload_res.json()
    assert upload_data["status"] == "indexed"
    doc_id = upload_data["document_id"]

    # 2. Test GET /retrieval/stats
    stats_res = client.get("/retrieval/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["collection_name"] == "research_papers"
    assert stats["total_chunks"] >= 1
    assert stats["total_documents"] >= 1

    # 3. Test POST /retrieval/search
    search_payload = {
        "query": "What algorithms were studied?",
        "top_k": 3,
        "document_ids": [doc_id],
    }
    search_res = client.post("/retrieval/search", json=search_payload)
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data["query"] == "What algorithms were studied?"
    assert search_data["total_results"] >= 1
    first_result = search_data["results"][0]
    assert first_result["document_id"] == doc_id
    assert first_result["filename"] == "rl_paper.pdf"
    assert "deep reinforcement learning" in first_result["text"]
    assert "distance" in first_result

    # 4. Test POST /retrieval/search with invalid empty query
    bad_search = client.post("/retrieval/search", json={"query": "", "top_k": 3})
    assert bad_search.status_code == 422 or bad_search.status_code == 400

