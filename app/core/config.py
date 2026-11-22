from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables."""

    app_name: str = "Research Paper Assistant API"
    app_env: str = "development"
    debug: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    version: str = "0.5.0"

    # Document Storage & Processing Settings
    storage_dir: str = "./storage"
    max_upload_size_mb: int = 20
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ChromaDB & Embedding Configuration
    chroma_persist_dir: str = "./storage/chroma"
    chroma_collection_name: str = "research_papers"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    default_top_k: int = 5

    # Gemini LLM Settings
    gemini_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    max_context_length_chars: int = 12000

    @property
    def groq_api_key(self) -> Optional[str]:
        return self.gemini_api_key or self.google_api_key

    @groq_api_key.setter
    def groq_api_key(self, val: Optional[str]):
        self.gemini_api_key = val

    @property
    def groq_model(self) -> str:
        return self.gemini_model

    @groq_model.setter
    def groq_model(self, val: str):
        self.gemini_model = val

    @property
    def uploads_dir(self) -> Path:
        """Absolute or relative path to stored PDF files."""
        p = Path(self.storage_dir) / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        """Path to SQLite database file."""
        p = Path(self.storage_dir) / "metadata.db"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chroma_dir(self) -> Path:
        """Path to ChromaDB persistent vector storage."""
        p = Path(self.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum file size allowed in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
