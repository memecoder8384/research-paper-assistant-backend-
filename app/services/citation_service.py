from typing import Any, Dict, List


def extract_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract and deduplicate source citations from retrieved chunk metadata.

    - Preserves document_id, filename, page_number, start_page, end_page, chunk_id
    - Avoids duplicate source entries (e.g. multiple chunks from the same page)
    - Never fabricates metadata
    """
    seen_keys = set()
    citations: List[Dict[str, Any]] = []

    for c in chunks:
        doc_id = str(c.get("document_id") or "unknown_doc")
        filename = str(c.get("filename") or "unknown_file.pdf")
        page_num = int(c.get("page_number", c.get("start_page", 1)))
        start_page = int(c.get("start_page", page_num))
        end_page = int(c.get("end_page", page_num))
        chunk_id = str(c.get("chunk_id") or "")

        # Unique key to avoid duplicate citations for identical document & page ranges
        key = (doc_id, page_num, start_page, end_page)
        if key not in seen_keys:
            seen_keys.add(key)
            citations.append(
                {
                    "document_id": doc_id,
                    "filename": filename,
                    "page_number": page_num,
                    "start_page": start_page,
                    "end_page": end_page,
                    "chunk_id": chunk_id,
                }
            )

    return citations
