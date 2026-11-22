"""Script to ingest real research papers into the backend pipeline.

Downloads seminal open-access arXiv research papers and runs them
through the complete PDF extraction, cleaning, chunking, SQLite
persistence, and ChromaDB vector indexing pipeline.
"""
import sys
import os
from pathlib import Path
import urllib.request
import uuid

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.db.database import (
    init_db,
    save_document as db_save_document,
    save_chunks as db_save_chunks,
    list_documents as db_list_documents,
)
from app.services.pdf_loader import extract_pages_from_pdf, validate_pdf_content
from app.services.text_cleaner import clean_extracted_pages
from app.services.text_chunker import chunk_document_pages
from app.retrieval.indexer import index_document

REAL_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "filename": "Attention_Is_All_You_Need_Vaswani_2017.pdf",
        "url": "https://arxiv.org/pdf/1706.03762.pdf",
    },
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "filename": "Retrieval_Augmented_Generation_Lewis_2020.pdf",
        "url": "https://arxiv.org/pdf/2005.11401.pdf",
    },
]


def ingest_pdf_bytes(file_bytes: bytes, filename: str) -> dict:
    """Ingest a PDF from raw bytes through the full pipeline."""
    validate_pdf_content(file_bytes, filename)
    doc_id = f"doc_{uuid.uuid4().hex[:10]}"
    stored_path = settings.uploads_dir / f"{doc_id}_{filename}"

    with open(stored_path, "wb") as f:
        f.write(file_bytes)

    extraction = extract_pages_from_pdf(file_bytes)
    total_pages = extraction["total_pages"]
    cleaned_pages = clean_extracted_pages(extraction["pages"])
    extracted_text_length = sum(p["char_count"] for p in cleaned_pages)

    chunks = chunk_document_pages(
        pages=cleaned_pages,
        document_id=doc_id,
        filename=filename,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    db_save_document(
        {
            "id": doc_id,
            "filename": filename,
            "file_path": str(stored_path),
            "file_size": len(file_bytes),
            "upload_timestamp": "2026-11-23T01:00:00Z",
            "status": "indexed",
            "total_pages": total_pages,
            "extracted_text_length": extracted_text_length,
            "total_chunks": len(chunks),
            "is_scanned": 0,
            "message": "Real research paper successfully parsed, chunked, and indexed in ChromaDB.",
        }
    )

    db_save_chunks(chunks)
    index_document(document_id=doc_id, filename=filename, chunks=chunks)

    return {
        "document_id": doc_id,
        "filename": filename,
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "status": "indexed",
    }


def main():
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    init_db()

    existing = db_list_documents()
    existing_filenames = {d["filename"] for d in existing}
    print(f"Current documents in database: {len(existing)}")

    for paper in REAL_PAPERS:
        if paper["filename"] in existing_filenames:
            print(f"Already ingested: {paper['filename']}")
            continue

        print(f"\nDownloading real paper: {paper['title']}...")
        headers = {"User-Agent": "ResearchPaperAssistant/1.0"}
        req = urllib.request.Request(paper["url"], headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            pdf_bytes = resp.read()

        print(f"Downloaded {len(pdf_bytes)} bytes. Ingesting into vector store & SQLite...")
        result = ingest_pdf_bytes(pdf_bytes, paper["filename"])
        print(f"Success! Document ID: {result['document_id']} | Pages: {result['total_pages']} | Chunks: {result['total_chunks']}")

    print("\nIngestion complete! All real documents are now active in the backend.")


if __name__ == "__main__":
    main()
