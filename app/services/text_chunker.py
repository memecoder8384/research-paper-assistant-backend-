import re
from typing import Any, Dict, List


def split_text_into_units(text: str) -> List[str]:
    """Split text into sentences or paragraphs to serve as natural chunk boundaries."""
    # Split on paragraph boundaries first
    paragraphs = text.split("\n\n")
    units: List[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Split paragraph into sentences preserving ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", para)
        for s in sentences:
            s_clean = s.strip()
            if s_clean:
                units.append(s_clean)

    return units


def chunk_document_pages(
    pages: List[Dict[str, Any]],
    document_id: str,
    filename: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    """Split extracted page text into manageable chunks.

    - Uses configurable chunk size and overlap.
    - Avoids creating empty chunks.
    - Preserves metadata: document_id, filename, chunk_index, page_number.
    - If a chunk spans multiple pages, records start_page and end_page accurately.
    """
    if chunk_size <= 0:
        chunk_size = 800
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        chunk_overlap = 150

    # Collect units with their source page number
    page_units: List[Dict[str, Any]] = []
    for page in pages:
        page_num = page["page_number"]
        page_text = page.get("text", "").strip()
        if not page_text:
            continue

        units = split_text_into_units(page_text)
        if not units:
            # Fallback if no sentence boundary found
            units = [page_text]

        for u in units:
            page_units.append({"text": u, "page_number": page_num})

    if not page_units:
        return []

    chunks: List[Dict[str, Any]] = []
    chunk_index = 0

    current_units: List[Dict[str, Any]] = []
    current_length = 0

    for unit in page_units:
        unit_text = unit["text"]
        unit_len = len(unit_text)

        # If a single unit exceeds chunk_size, slice it directly
        if unit_len > chunk_size and not current_units:
            start_pos = 0
            while start_pos < unit_len:
                sub_text = unit_text[start_pos : start_pos + chunk_size].strip()
                if sub_text:
                    chunks.append(
                        {
                            "document_id": document_id,
                            "filename": filename,
                            "chunk_index": chunk_index,
                            "text": sub_text,
                            "page_number": unit["page_number"],
                            "start_page": unit["page_number"],
                            "end_page": unit["page_number"],
                        }
                    )
                    chunk_index += 1
                start_pos += chunk_size - chunk_overlap
            continue

        # If adding this unit exceeds chunk_size, emit current chunk
        if current_length + unit_len + 1 > chunk_size and current_units:
            chunk_text = " ".join(u["text"] for u in current_units).strip()
            if chunk_text:
                pages_in_chunk = [u["page_number"] for u in current_units]
                start_page = min(pages_in_chunk)
                end_page = max(pages_in_chunk)

                chunks.append(
                    {
                        "document_id": document_id,
                        "filename": filename,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "page_number": start_page,
                        "start_page": start_page,
                        "end_page": end_page,
                    }
                )
                chunk_index += 1

            # Retain overlap units from end of current chunk
            overlap_units: List[Dict[str, Any]] = []
            overlap_length = 0
            for u in reversed(current_units):
                if overlap_length + len(u["text"]) <= chunk_overlap:
                    overlap_units.insert(0, u)
                    overlap_length += len(u["text"]) + 1
                else:
                    break

            current_units = overlap_units
            current_length = sum(len(u["text"]) for u in current_units) + max(0, len(current_units) - 1)

        current_units.append(unit)
        current_length += unit_len + 1

    # Emit final remaining chunk
    if current_units:
        chunk_text = " ".join(u["text"] for u in current_units).strip()
        if chunk_text:
            pages_in_chunk = [u["page_number"] for u in current_units]
            start_page = min(pages_in_chunk)
            end_page = max(pages_in_chunk)

            chunks.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "page_number": start_page,
                    "start_page": start_page,
                    "end_page": end_page,
                }
            )

    return chunks

