import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.core.config import settings


def get_db_path(custom_path: Optional[str | Path] = None) -> Path:
    """Resolve database path, ensuring parent directory exists."""
    path = Path(custom_path) if custom_path else settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(custom_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """Open SQLite connection with row factory enabled."""
    db_path = get_db_path(custom_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(custom_path: Optional[str | Path] = None) -> None:
    """Initialize database tables for document metadata and chunk storage."""
    with get_connection(custom_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                upload_timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                total_pages INTEGER NOT NULL DEFAULT 0,
                extracted_text_length INTEGER NOT NULL DEFAULT 0,
                total_chunks INTEGER NOT NULL DEFAULT 0,
                is_scanned INTEGER NOT NULL DEFAULT 0,
                message TEXT
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                start_page INTEGER NOT NULL,
                end_page INTEGER NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(document_id);
            """
        )
        conn.commit()


def save_document(doc_data: Dict[str, Any], custom_path: Optional[str | Path] = None) -> None:
    """Insert or update a document metadata entry without triggering ON DELETE CASCADE on chunks."""
    with get_connection(custom_path) as conn:
        cursor = conn.execute("SELECT id FROM documents WHERE id = ?;", (doc_data["id"],))
        if cursor.fetchone():
            conn.execute(
                """
                UPDATE documents SET
                    filename = :filename,
                    file_path = :file_path,
                    file_size = :file_size,
                    upload_timestamp = :upload_timestamp,
                    status = :status,
                    total_pages = :total_pages,
                    extracted_text_length = :extracted_text_length,
                    total_chunks = :total_chunks,
                    is_scanned = :is_scanned,
                    message = :message
                WHERE id = :id;
                """,
                doc_data,
            )
        else:
            conn.execute(
                """
                INSERT INTO documents (
                    id, filename, file_path, file_size, upload_timestamp,
                    status, total_pages, extracted_text_length, total_chunks,
                    is_scanned, message
                ) VALUES (
                    :id, :filename, :file_path, :file_size, :upload_timestamp,
                    :status, :total_pages, :extracted_text_length, :total_chunks,
                    :is_scanned, :message
                );
                """,
                doc_data,
            )
        conn.commit()


def update_document_status(
    document_id: str,
    status: str,
    message: str,
    custom_path: Optional[str | Path] = None,
) -> None:
    """Update only status and message for a document."""
    with get_connection(custom_path) as conn:
        conn.execute(
            "UPDATE documents SET status = ?, message = ? WHERE id = ?;",
            (status, message, document_id),
        )
        conn.commit()


def save_chunks(chunks: List[Dict[str, Any]], custom_path: Optional[str | Path] = None) -> None:
    """Save processed text chunks for a document."""
    if not chunks:
        return
    with get_connection(custom_path) as conn:
        conn.executemany(
            """
            INSERT INTO document_chunks (
                document_id, chunk_index, text, page_number, start_page, end_page
            ) VALUES (
                :document_id, :chunk_index, :text, :page_number, :start_page, :end_page
            );
            """,
            chunks,
        )
        conn.commit()


def get_document(document_id: str, custom_path: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve metadata for a single document by its ID."""
    with get_connection(custom_path) as conn:
        cursor = conn.execute("SELECT * FROM documents WHERE id = ?;", (document_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)


def list_documents(custom_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """List all documents ordered by upload timestamp descending."""
    with get_connection(custom_path) as conn:
        cursor = conn.execute("SELECT * FROM documents ORDER BY upload_timestamp DESC;")
        return [dict(row) for row in cursor.fetchall()]


def delete_document(document_id: str, custom_path: Optional[str | Path] = None) -> bool:
    """Delete a document record and cascade deletion of its chunks."""
    with get_connection(custom_path) as conn:
        cursor = conn.execute("DELETE FROM documents WHERE id = ?;", (document_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_document_chunks(document_id: str, custom_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all chunks for a document ordered by chunk_index."""
    with get_connection(custom_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM document_chunks WHERE document_id = ? ORDER BY chunk_index ASC;",
            (document_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
