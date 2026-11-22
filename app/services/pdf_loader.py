from typing import Any, Dict, List
import pymupdf as fitz



def validate_pdf_content(file_bytes: bytes, filename: str) -> None:
    """Validate that the file has a .pdf extension, valid PDF magic header, and can be opened safely."""
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Invalid file extension: Only PDF files (.pdf) are accepted.")

    # PDF magic byte check: A valid PDF file begins with %PDF- (0x25 0x50 0x44 0x46 0x2D)
    if not file_bytes.startswith(b"%PDF-"):
        raise ValueError("Invalid PDF format: File is missing the standard '%PDF-' magic header.")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Corrupted or unreadable PDF: Unable to parse document structure. ({str(exc)})")

    try:
        if doc.is_encrypted:
            raise ValueError("Unsupported PDF: Password-protected or encrypted PDFs cannot be processed.")
        if len(doc) == 0:
            raise ValueError("Empty PDF: The document contains 0 pages.")
    finally:
        doc.close()


def extract_pages_from_pdf(file_bytes: bytes) -> Dict[str, Any]:
    """Safely extracts text page by page from PDF bytes using PyMuPDF.

    Preserves 1-based page numbers, handles empty pages, and detects scanned/image-only PDFs.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = len(doc)
    pages: List[Dict[str, Any]] = []
    total_extracted_chars = 0

    try:
        for page_index in range(total_pages):
            page = doc[page_index]
            page_num = page_index + 1  # 1-based page indexing
            raw_text = page.get_text("text") or ""
            stripped_text = raw_text.strip()
            char_count = len(stripped_text)
            total_extracted_chars += char_count

            pages.append(
                {
                    "page_number": page_num,
                    "text": raw_text,
                    "char_count": char_count,
                    "is_empty": (char_count == 0),
                }
            )

        # Scanned / Image-only detection:
        # If the document has pages but zero (or fewer than 20) extractable characters total,
        # it is a scanned or image-based document without a digital text layer.
        is_scanned = total_pages > 0 and total_extracted_chars < 20

        if is_scanned:
            message = (
                "OCR support is not implemented yet. The uploaded document contains scanned "
                "or image-only pages with no extractable text layer."
            )
        else:
            message = "Text extracted successfully."

        return {
            "total_pages": total_pages,
            "is_scanned": is_scanned,
            "extracted_text_length": total_extracted_chars,
            "pages": pages,
            "message": message,
        }
    finally:
        doc.close()
