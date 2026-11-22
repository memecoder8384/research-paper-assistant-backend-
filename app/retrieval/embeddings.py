import logging
import threading
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Thread-safe singleton service for generating text and query embeddings using Sentence Transformers."""

    _model: Optional[SentenceTransformer] = None
    _lock = threading.Lock()
    _model_name: Optional[str] = None

    @classmethod
    def get_model(cls, model_name: Optional[str] = None) -> SentenceTransformer:
        """Load and return the shared SentenceTransformer model instance.

        Loads the model once and reuses it for all subsequent document and query embeddings.
        """
        if cls._model is not None and (model_name is None or model_name == cls._model_name):
            return cls._model

        target_model = model_name or settings.embedding_model_name
        if cls._model is None or cls._model_name != target_model:
            with cls._lock:
                if cls._model is None or cls._model_name != target_model:
                    logger.info("Loading embedding model: %s", target_model)
                    cls._model = SentenceTransformer(target_model)
                    cls._model_name = target_model
                    logger.info("Embedding model loaded successfully: %s", target_model)
        return cls._model

    @classmethod
    def set_model(cls, model: Any, model_name: Optional[str] = None) -> None:
        """Inject a preloaded model or mock instance (useful for unit testing without downloading)."""
        with cls._lock:
            cls._model = model
            cls._model_name = model_name or settings.embedding_model_name

    @classmethod
    def reset_model(cls) -> None:
        """Reset the cached model instance."""
        with cls._lock:
            cls._model = None
            cls._model_name = None

    @classmethod
    def generate_embeddings(
        cls,
        texts: List[str],
        batch_size: int = 32,
        model_name: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate normalized numerical embedding vectors for a list of document chunks.

        - Validates input texts
        - Supports configurable batch sizes
        - Returns list of float vectors
        """
        if not texts:
            return []

        for idx, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid text at index {idx}: Text cannot be empty or whitespace only.")

        model = cls.get_model(model_name)
        try:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.tolist()
        except Exception as exc:
            logger.error("Failed to generate document embeddings: %s", exc)
            raise RuntimeError(f"Embedding generation failed: {str(exc)}") from exc

    @classmethod
    def generate_query_embedding(
        cls,
        query: str,
        model_name: Optional[str] = None,
    ) -> List[float]:
        """Generate a single normalized numerical embedding vector for a search query."""
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace only.")

        model = cls.get_model(model_name)
        try:
            embedding = model.encode(
                query.strip(),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embedding.tolist()
        except Exception as exc:
            logger.error("Failed to generate query embedding: %s", exc)
            raise RuntimeError(f"Query embedding generation failed: {str(exc)}") from exc
