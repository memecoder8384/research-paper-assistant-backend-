import io
from unittest.mock import MagicMock
import pytest
from app.core.config import settings
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import add_document_chunks, clear_collection
from app.services.citation_service import extract_citations
from app.services.context_builder import build_rag_context
from app.services.llm_service import LLMService
from app.services.rag_service import answer_research_question
from app.services.session_store import SessionStore
from tests.conftest import make_pdf_bytes


class MockGeminiModels:
    def __init__(self, response_text: str = "The transformer model relies on self-attention [attention.pdf, p. 1]."):
        self.response_text = response_text
        self.last_call_kwargs = None
        self.raise_exc = None

    def generate_content(self, model, contents, config=None):
        temp = getattr(config, "temperature", None) if config else None
        tokens = getattr(config, "max_output_tokens", None) if config else None
        self.last_call_kwargs = {
            "model": model,
            "contents": contents,
            "config": config,
            "temperature": temp,
            "max_tokens": tokens,
        }
        if self.raise_exc:
            raise self.raise_exc
        mock_resp = MagicMock()
        mock_resp.text = self.response_text
        return mock_resp


class MockGeminiClient:
    def __init__(self, response_text: str = "The transformer model relies on self-attention [attention.pdf, p. 1]."):
        self.models = MockGeminiModels(response_text)
        self.completions = self.models  # For compatibility with legacy test assertions


# Backwards compatibility alias
MockGroqClient = MockGeminiClient
MockGroqChatCompletions = MockGeminiModels


@pytest.fixture(autouse=True)
def setup_mock_groq():
    """Inject mock Gemini client and clear session store before each test."""
    mock_client = MockGeminiClient()
    LLMService.set_client(mock_client)
    SessionStore.reset_all()
    yield mock_client
    LLMService.reset_client()
    SessionStore.reset_all()


# 1. Gemini client configuration test
def test_groq_client_configuration(setup_mock_groq):
    messages = [{"role": "user", "content": "Hello"}]
    LLMService.generate_chat_completion(messages, model="custom-model", temperature=0.5, max_tokens=256)
    call_kwargs = setup_mock_groq.models.last_call_kwargs
    assert call_kwargs["model"] == "custom-model"
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["max_tokens"] == 256


# 2. Missing API key test
def test_missing_api_key(monkeypatch):
    LLMService.reset_client()
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "google_api_key", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        LLMService.get_client()


# 3. Context construction test
def test_context_construction():
    chunks = [
        {
            "chunk_id": "c1",
            "filename": "resnet.pdf",
            "document_id": "doc1",
            "page_number": 3,
            "start_page": 3,
            "end_page": 3,
            "text": "Residual connections allow training of extremely deep networks.",
        },
        {
            "chunk_id": "c2",
            "filename": "resnet.pdf",
            "document_id": "doc1",
            "page_number": 4,
            "start_page": 4,
            "end_page": 5,
            "text": "Experimental evaluation on ImageNet demonstrates 3.57% error.",
        },
    ]
    context = build_rag_context(chunks, max_length=5000)
    assert "resnet.pdf" in context
    assert "Page 3" in context
    assert "Pages 4-5" in context
    assert "Residual connections" in context


# 4. Empty retrieval results test
def test_empty_retrieval_results(temp_storage):
    clear_collection(persist_dir=temp_storage / "chroma")
    res = answer_research_question("What optimizer was used in the study?")
    assert "do not contain sufficient information" in res["answer"]
    assert res["sources"] == []
    assert res["retrieved_chunks"] == 0


# 5. RAG orchestration test
def test_rag_orchestration(temp_storage, setup_mock_groq):
    chroma_path = temp_storage / "chroma"
    chunks = [
        {"chunk_index": 0, "text": "We trained with the Adam optimizer.", "page_number": 2, "start_page": 2, "end_page": 2}
    ]
    embs = EmbeddingService.generate_embeddings([c["text"] for c in chunks])
    add_document_chunks("doc-adam", "adam_paper.pdf", chunks, embs, persist_dir=chroma_path)

    res = answer_research_question("What optimizer was used?")
    assert res["status"] == "success"
    assert res["retrieved_chunks"] >= 1
    assert len(res["sources"]) >= 1
    assert "conversation_id" in res


# 6. Grounded answer generation test
def test_grounded_answer_generation(temp_storage, setup_mock_groq):
    chroma_path = temp_storage / "chroma"
    chunks = [{"chunk_index": 0, "text": "Accuracy reached 98.4% on CIFAR-10.", "page_number": 5, "start_page": 5, "end_page": 5}]
    embs = EmbeddingService.generate_embeddings([c["text"] for c in chunks])
    add_document_chunks("doc-acc", "cifar.pdf", chunks, embs, persist_dir=chroma_path)

    setup_mock_groq.completions.response_text = "The model achieved 98.4% accuracy on CIFAR-10 [cifar.pdf, p. 5]."
    res = answer_research_question("What was the accuracy on CIFAR-10?")
    assert "98.4%" in res["answer"]
    assert res["sources"][0]["filename"] == "cifar.pdf"
    assert res["sources"][0]["page_number"] == 5


# 7. Unsupported question handling test
def test_unsupported_question_handling(temp_storage):
    clear_collection(persist_dir=temp_storage / "chroma")
    res = answer_research_question("What is the recipe for chocolate cake?")
    assert "do not contain sufficient information" in res["answer"]
    assert res["sources"] == []


# 8. Citation preservation test
def test_citation_preservation():
    chunks = [
        {
            "document_id": "doc-citation",
            "filename": "vit.pdf",
            "page_number": 4,
            "start_page": 4,
            "end_page": 5,
            "chunk_id": "vit#chunk_1",
        }
    ]
    citations = extract_citations(chunks)
    assert len(citations) == 1
    c = citations[0]
    assert c["document_id"] == "doc-citation"
    assert c["filename"] == "vit.pdf"
    assert c["page_number"] == 4
    assert c["start_page"] == 4
    assert c["end_page"] == 5
    assert c["chunk_id"] == "vit#chunk_1"


# 9. Duplicate citation removal test
def test_duplicate_citation_removal():
    chunks = [
        {"document_id": "doc1", "filename": "paper.pdf", "page_number": 3, "start_page": 3, "end_page": 3, "chunk_id": "c1"},
        {"document_id": "doc1", "filename": "paper.pdf", "page_number": 3, "start_page": 3, "end_page": 3, "chunk_id": "c2"},
        {"document_id": "doc1", "filename": "paper.pdf", "page_number": 4, "start_page": 4, "end_page": 4, "chunk_id": "c3"},
    ]
    citations = extract_citations(chunks)
    assert len(citations) == 2  # Only page 3 and page 4
    pages = [c["page_number"] for c in citations]
    assert pages == [3, 4]


# 10. Invalid user questions test
def test_invalid_user_questions():
    with pytest.raises(ValueError, match="cannot be empty"):
        answer_research_question("")

    with pytest.raises(ValueError, match="cannot be empty"):
        answer_research_question("   ")

    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        answer_research_question("a" * 1001)


# 11. Conversation history test
def test_conversation_history(temp_storage):
    chroma_path = temp_storage / "chroma"
    chunks = [{"chunk_index": 0, "text": "First chunk context.", "page_number": 1, "start_page": 1, "end_page": 1}]
    embs = EmbeddingService.generate_embeddings([c["text"] for c in chunks])
    add_document_chunks("doc-h", "h.pdf", chunks, embs, persist_dir=chroma_path)

    cid = "test-session-123"
    res1 = answer_research_question("Question 1", conversation_id=cid)
    res2 = answer_research_question("Question 2", conversation_id=cid)

    history = SessionStore.get_history(cid)
    assert len(history) == 4  # 2 user messages, 2 assistant responses
    assert history[0]["content"] == "Question 1"
    assert history[2]["content"] == "Question 2"

    SessionStore.clear_history(cid)
    assert SessionStore.get_history(cid) == []


# 12. Chat API endpoints test
def test_chat_api_endpoints(client, temp_storage):
    # Upload PDF
    pdf_bytes = make_pdf_bytes(["A study on transformer self-attention projections and multi-head scaling."])
    client.post("/documents/upload", files={"file": ("attention.pdf", io.BytesIO(pdf_bytes), "application/pdf")})

    # Test POST /chat
    req_body = {"question": "What is self-attention?", "top_k": 3}
    chat_res = client.post("/chat", json=req_body)
    assert chat_res.status_code == 200
    data = chat_res.json()
    assert data["status"] == "success"
    assert "answer" in data
    assert len(data["sources"]) >= 1
    cid = data["conversation_id"]

    # Test POST /chat/{conversation_id} (continuation)
    follow_up_res = client.post(f"/chat/{cid}", json={"question": "Can you elaborate on scaling?"})
    assert follow_up_res.status_code == 200
    assert follow_up_res.json()["conversation_id"] == cid

    # Test GET /chat/{conversation_id}/history
    hist_res = client.get(f"/chat/{cid}/history")
    assert hist_res.status_code == 200
    assert hist_res.json()["total_messages"] == 4

    # Test DELETE /chat/{conversation_id}
    del_res = client.delete(f"/chat/{cid}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # Confirm history is now empty
    hist_after = client.get(f"/chat/{cid}/history").json()
    assert hist_after["total_messages"] == 0


# 13. Mocked Gemini API failure test
def test_mocked_groq_api_failures(setup_mock_groq, temp_storage, client):
    # Upload doc so retrieval succeeds and calls LLM
    pdf_bytes = make_pdf_bytes(["Test failure scenario."])
    client.post("/documents/upload", files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")})

    # Inject timeout exception into mock Gemini
    setup_mock_groq.models.raise_exc = TimeoutError("Gemini API request timed out")

    res = client.post("/chat", json={"question": "What happens on timeout?"})
    assert res.status_code == 503
    assert "timed out" in res.json()["error"]["message"]


# 14. Rate-limit handling test
def test_rate_limit_handling(setup_mock_groq, temp_storage, client):
    # Upload doc
    pdf_bytes = make_pdf_bytes(["Test rate limit scenario."])
    client.post("/documents/upload", files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")})

    # Inject RateLimit into mock Gemini
    class MockRateLimit(Exception):
        pass

    setup_mock_groq.models.raise_exc = MockRateLimit("Gemini API rate limit exceeded (429)")

    res = client.post("/chat", json={"question": "Are we rate limited?"})
    assert res.status_code == 503
    assert "rate limit exceeded" in res.json()["error"]["message"].lower()
