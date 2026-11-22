from datetime import datetime, timezone
import logging
from pathlib import Path
import re
import uuid
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from app.core.config import settings
from app.db.database import (
    delete_document as db_delete_document,
    get_document as db_get_document,
    list_documents as db_list_documents,
    save_chunks as db_save_chunks,
    save_document as db_save_document,
)
from app.retrieval.indexer import index_document
from app.retrieval.vector_store import delete_document_chunks
from app.services.pdf_loader import extract_pages_from_pdf, validate_pdf_content
from app.services.text_cleaner import clean_extracted_pages
from app.services.text_chunker import chunk_document_pages

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


def sanitize_filename(filename: str) -> str:
    """Sanitize the incoming filename to prevent path traversal and unsafe characters."""
    raw_name = Path(filename).name
    sanitized = re.sub(r"[^\w\.-]", "_", raw_name)
    if not sanitized or sanitized.startswith("."):
        sanitized = f"document_{uuid.uuid4().hex[:6]}.pdf"
    return sanitized


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload, process, and index a research paper PDF",
)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, extract its text, clean it, chunk it, and index it in ChromaDB."""
    raw_filename = file.filename or "unknown.pdf"
    filename = sanitize_filename(raw_filename)

    # 1. Enforce size limits while reading bytes safely in chunks
    contents = bytearray()
    max_size = settings.max_upload_size_bytes

    while True:
        chunk = await file.read(1024 * 1024)  # 1MB chunks
        if not chunk:
            break
        contents.extend(chunk)
        if len(contents) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File exceeds the maximum allowed size of {settings.max_upload_size_mb} MB.",
            )

    file_bytes = bytes(contents)
    file_size = len(file_bytes)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    # 2. Validate PDF structure and integrity
    try:
        validate_pdf_content(file_bytes, filename)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )

    # 3. Generate unique document ID and save file to local storage safely
    document_id = str(uuid.uuid4())
    stored_file_path = settings.uploads_dir / f"{document_id}_{filename}"

    with open(stored_file_path, "wb") as f:
        f.write(file_bytes)

    # 4. Extract text page by page with PyMuPDF
    try:
        extraction = extract_pages_from_pdf(file_bytes)
    except Exception as exc:
        stored_file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to extract text from the uploaded PDF.",
        )

    total_pages = extraction["total_pages"]
    is_scanned = extraction["is_scanned"]
    upload_timestamp = datetime.now(timezone.utc).isoformat()

    # 5. Handle Scanned / Image-Only PDFs
    if is_scanned:
        status_label = "scanned_pdf"
        message = extraction["message"]

        db_save_document(
            {
                "id": document_id,
                "filename": filename,
                "file_path": str(stored_file_path),
                "file_size": file_size,
                "upload_timestamp": upload_timestamp,
                "status": status_label,
                "total_pages": total_pages,
                "extracted_text_length": 0,
                "total_chunks": 0,
                "is_scanned": 1,
                "message": message,
            }
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "status": status_label,
            "total_pages": total_pages,
            "extracted_text_length": 0,
            "total_chunks": 0,
            "message": message,
        }

    # 6. Clean extracted text
    cleaned_pages = clean_extracted_pages(extraction["pages"])
    extracted_text_length = sum(p["char_count"] for p in cleaned_pages)

    # 7. Chunk document text
    chunks = chunk_document_pages(
        pages=cleaned_pages,
        document_id=document_id,
        filename=filename,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    total_chunks = len(chunks)

    # 8. Persist initial metadata and chunks to SQLite
    status_label = "processing"
    message = "Document processed and chunked. Generating vector embeddings..."

    db_save_document(
        {
            "id": document_id,
            "filename": filename,
            "file_path": str(stored_file_path),
            "file_size": file_size,
            "upload_timestamp": upload_timestamp,
            "status": status_label,
            "total_pages": total_pages,
            "extracted_text_length": extracted_text_length,
            "total_chunks": total_chunks,
            "is_scanned": 0,
            "message": message,
        }
    )

    db_save_chunks(chunks)

    # 9. Phase 3: Index in ChromaDB
    try:
        index_document(document_id=document_id, filename=filename, chunks=chunks)
        status_label = "indexed"
        message = "Document processed, chunked, and indexed in ChromaDB successfully."
    except Exception as exc:
        logger.error("ChromaDB indexing error for document %s: %s", document_id, exc)
        status_label = "indexing_failed"
        message = f"Document chunked, but vector indexing failed: {str(exc)}"

    return {
        "document_id": document_id,
        "filename": filename,
        "status": status_label,
        "total_pages": total_pages,
        "extracted_text_length": extracted_text_length,
        "total_chunks": total_chunks,
        "message": message,
    }


@router.get("", summary="List all uploaded documents")
async def list_documents():
    """Retrieve metadata for all uploaded documents from persistent storage."""
    docs = db_list_documents()

    safe_docs = []
    for d in docs:
        item = {
            "id": d["id"],
            "filename": d["filename"],
            "file_size": d["file_size"],
            "upload_timestamp": d["upload_timestamp"],
            "status": d["status"],
            "total_pages": d["total_pages"],
            "extracted_text_length": d["extracted_text_length"],
            "total_chunks": d["total_chunks"],
            "is_scanned": bool(d["is_scanned"]),
            "message": d["message"],
        }
        safe_docs.append(item)

    return {
        "total": len(safe_docs),
        "documents": safe_docs,
    }


@router.delete("/{document_id}", summary="Delete an uploaded document")
async def delete_document(document_id: str):
    """Delete a document by its ID, removing its file, chunks, SQLite metadata, and ChromaDB records."""
    doc = db_get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    # 1. Delete physical file from storage safely
    file_path = Path(doc["file_path"])
    try:
        file_path.unlink(missing_ok=True)
    except Exception:
        pass

    # 2. Delete document and chunks from SQLite
    db_delete_document(document_id)

    # 3. Delete chunks from ChromaDB vector store
    delete_document_chunks(document_id)

    return {
        "success": True,
        "message": "Document deleted successfully.",
        "document_id": document_id,
    }
