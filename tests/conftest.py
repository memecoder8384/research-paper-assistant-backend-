import hashlib
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
import pymupdf as fitz
from app.core.config import settings
from app.db.database import init_db
from app.retrieval.embeddings import EmbeddingService
from main import app


class MockSentenceTransformer:
    """Deterministic mock embedding model producing 384-dimensional unit vectors."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    def encode(
        self,
        sentences,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        single_input = isinstance(sentences, str)
        items = [sentences] if single_input else sentences

        vectors = []
        for s in items:
            h = hashlib.sha256(s.encode("utf-8")).digest()
            floats = []
            for i in range(384):
                byte_val = h[i % 32]
                floats.append(float(byte_val - 128) / 128.0)
            vec = np.array(floats, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)

        arr = np.array(vectors, dtype=np.float32)
        return arr[0] if single_input else arr


@pytest.fixture(autouse=True)
def setup_mock_model():
    """Ensure mock embedding model is injected for unit testing without downloading model files."""
    mock = MockSentenceTransformer()
    EmbeddingService.set_model(mock, model_name="mock-all-MiniLM-L6-v2")
    yield mock
    EmbeddingService.reset_model()


def make_pdf_bytes(pages_text: list[str]) -> bytes:
    """Create a minimal real PDF in memory with specified text per page."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_image_only_pdf() -> bytes:
    """Create a PDF with blank pages (no text layer) to simulate a scanned document."""
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(50, 50, 200, 200), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def temp_storage(tmp_path: Path, monkeypatch):
    """Isolate storage, database, and ChromaDB vector store to a temporary directory for each test."""
    temp_dir = tmp_path / "test_storage"
    temp_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = temp_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir = temp_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    db_file = temp_dir / "test_metadata.db"

    monkeypatch.setattr(settings, "storage_dir", str(temp_dir))
    monkeypatch.setattr(settings, "chroma_persist_dir", str(chroma_dir))
    init_db(str(db_file))

    # Clear cached Chroma clients to guarantee fresh isolated instances
    from app.retrieval import vector_store

    vector_store._clients.clear()

    yield temp_dir


@pytest.fixture
def client(temp_storage):
    """Test client configured to use the isolated temporary environment."""
    with TestClient(app) as test_client:
        yield test_client
