import pytest
from app.services.pdf_loader import extract_pages_from_pdf, validate_pdf_content
from app.services.text_cleaner import clean_extracted_pages, clean_text
from app.services.text_chunker import chunk_document_pages
from tests.conftest import make_image_only_pdf, make_pdf_bytes


def test_validate_pdf_content_valid():
    pdf_bytes = make_pdf_bytes(["Test content for validation."])
    # Should not raise any error
    validate_pdf_content(pdf_bytes, "paper.pdf")


def test_validate_pdf_content_invalid_extension():
    pdf_bytes = make_pdf_bytes(["Test content"])
    with pytest.raises(ValueError, match="Only PDF files"):
        validate_pdf_content(pdf_bytes, "paper.txt")


def test_validate_pdf_content_invalid_magic_header():
    fake_bytes = b"NOT_A_PDF_CONTENT"
    with pytest.raises(ValueError, match="missing the standard '%PDF-'"):
        validate_pdf_content(fake_bytes, "paper.pdf")


def test_validate_pdf_content_corrupted():
    corrupted_bytes = b"%PDF-1.4\nCorrupted binary content that cannot be opened"
    with pytest.raises(ValueError, match="Corrupted or unreadable PDF"):
        validate_pdf_content(corrupted_bytes, "paper.pdf")


def test_text_extraction_and_page_number_preservation():
    page_texts = [
        "First page content with intro to attention mechanisms.",
        "Second page content explaining transformers and multi-head projection.",
        "Third page content with experimental results and benchmark numbers.",
    ]
    pdf_bytes = make_pdf_bytes(page_texts)

    result = extract_pages_from_pdf(pdf_bytes)

    assert result["total_pages"] == 3
    assert not result["is_scanned"]
    assert len(result["pages"]) == 3

    # Check 1-based page numbering preservation
    assert result["pages"][0]["page_number"] == 1
    assert "attention mechanisms" in result["pages"][0]["text"]
    assert result["pages"][1]["page_number"] == 2
    assert "transformers" in result["pages"][1]["text"]
    assert result["pages"][2]["page_number"] == 3
    assert "experimental results" in result["pages"][2]["text"]


def test_empty_page_handling():
    # Page 1 has text, Page 2 is empty, Page 3 has text
    page_texts = ["Non-empty introductory text.", "", "Concluding observations."]
    pdf_bytes = make_pdf_bytes(page_texts)

    result = extract_pages_from_pdf(pdf_bytes)

    assert result["total_pages"] == 3
    assert result["pages"][0]["is_empty"] is False
    assert result["pages"][1]["is_empty"] is True
    assert result["pages"][2]["is_empty"] is False


def test_scanned_pdf_detection():
    pdf_bytes = make_image_only_pdf()
    result = extract_pages_from_pdf(pdf_bytes)

    assert result["total_pages"] == 1
    assert result["is_scanned"] is True
    assert "OCR support is not implemented yet" in result["message"]


def test_text_cleaning():
    raw_text = (
        "This is an introduc-\ntion to neural net-\nworks.\n\n"
        "Here is   excessive    spacing   and a citation [1].\n"
        "Also equation E = mc^2 must remain intact.\n\n"
        "New paragraph with trailing whitespace.   \n"
    )

    cleaned = clean_text(raw_text)

    # Reconnected hyphenated words
    assert "introduction" in cleaned
    assert "networks" in cleaned
    # Preserved paragraph breaks
    assert "\n\n" in cleaned
    # Normalized spacing
    assert "excessive spacing and a citation [1]" in cleaned
    # Preserved math & citations
    assert "[1]" in cleaned
    assert "E = mc^2" in cleaned


def test_clean_extracted_pages():
    pages = [
        {"page_number": 1, "text": "Page one  with extra   spaces and soft\nbreak."},
        {"page_number": 2, "text": "   "},
    ]
    cleaned = clean_extracted_pages(pages)

    assert len(cleaned) == 2
    assert cleaned[0]["page_number"] == 1
    assert cleaned[0]["text"] == "Page one with extra spaces and soft break."
    assert cleaned[0]["is_empty"] is False
    assert cleaned[1]["page_number"] == 2
    assert cleaned[1]["is_empty"] is True


def test_chunking_and_metadata_preservation():
    pages = [
        {
            "page_number": 1,
            "text": "Deep residual learning frameworks ease the training of networks. " * 8,
        },
        {
            "page_number": 2,
            "text": "We evaluate the representations on ImageNet classification. " * 8,
        },
    ]

    chunks = chunk_document_pages(
        pages=pages,
        document_id="doc-123",
        filename="resnet.pdf",
        chunk_size=400,
        chunk_overlap=50,
    )

    assert len(chunks) > 1

    # Verify every chunk has required metadata and is not empty
    for i, chunk in enumerate(chunks):
        assert chunk["document_id"] == "doc-123"
        assert chunk["filename"] == "resnet.pdf"
        assert chunk["chunk_index"] == i
        assert len(chunk["text"].strip()) > 0
        assert "start_page" in chunk
        assert "end_page" in chunk
        assert "page_number" in chunk
        assert chunk["start_page"] <= chunk["end_page"]

