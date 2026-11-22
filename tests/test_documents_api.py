import io
from tests.conftest import make_image_only_pdf, make_pdf_bytes


def test_upload_valid_pdf(client):
    pdf_bytes = make_pdf_bytes(["A novel architecture that eschews recurrence and instead relies entirely on attention."])
    response = client.post(
        "/documents/upload",
        files={"file": ("attention.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "attention.pdf"
    assert data["status"] in ("completed", "indexed")
    assert data["total_pages"] == 1
    assert data["extracted_text_length"] > 0
    assert data["total_chunks"] >= 1
    assert "Document processed" in data["message"]
    # Ensure full extracted text is NOT leaked in the response
    assert "text" not in data
    assert "pages" not in data


def test_upload_invalid_extension(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(b"Hello world"), "text/plain")},
    )

    assert response.status_code == 400
    assert "Only PDF files (.pdf) are accepted" in response.json()["error"]["message"]


def test_upload_corrupted_pdf(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("corrupted.pdf", io.BytesIO(b"%PDF-corrupted-bytes-stream"), "application/pdf")},
    )

    assert response.status_code == 400
    assert "Corrupted or unreadable PDF" in response.json()["error"]["message"]


def test_upload_empty_bytes(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )

    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["error"]["message"]


def test_upload_scanned_pdf(client):
    pdf_bytes = make_image_only_pdf()
    response = client.post(
        "/documents/upload",
        files={"file": ("scanned.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "scanned_pdf"
    assert data["total_pages"] == 1
    assert data["extracted_text_length"] == 0
    assert data["total_chunks"] == 0
    assert "OCR support is not implemented yet" in data["message"]


def test_upload_oversized_file(client, monkeypatch):
    # Set limit to 0 MB to trigger oversized rejection
    from app.core.config import settings

    monkeypatch.setattr(settings, "max_upload_size_mb", 0)

    pdf_bytes = make_pdf_bytes(["Test oversize check"])
    response = client.post(
        "/documents/upload",
        files={"file": ("oversized.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 413
    assert "exceeds the maximum allowed size" in response.json()["error"]["message"]


def test_list_and_delete_documents(client):
    # 1. Upload two documents
    pdf1 = make_pdf_bytes(["Doc one body text."])
    pdf2 = make_pdf_bytes(["Doc two body text."])

    r1 = client.post(
        "/documents/upload",
        files={"file": ("paper1.pdf", io.BytesIO(pdf1), "application/pdf")},
    )
    r2 = client.post(
        "/documents/upload",
        files={"file": ("paper2.pdf", io.BytesIO(pdf2), "application/pdf")},
    )
    id1 = r1.json()["document_id"]
    id2 = r2.json()["document_id"]

    # 2. List documents
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 2
    ids = [d["id"] for d in list_data["documents"]]
    assert id1 in ids
    assert id2 in ids
    # Verify internal server path is not exposed
    for d in list_data["documents"]:
        assert "file_path" not in d

    # 3. Delete document 1
    del_res = client.delete(f"/documents/{id1}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True
    assert del_res.json()["document_id"] == id1

    # 4. Confirm document 1 is gone from listing
    list_after = client.get("/documents").json()
    ids_after = [d["id"] for d in list_after["documents"]]
    assert id1 not in ids_after
    assert id2 in ids_after

    # 5. Deleting a non-existent document returns 404
    del_non_existent = client.delete("/documents/non-existent-id")
    assert del_non_existent.status_code == 404

