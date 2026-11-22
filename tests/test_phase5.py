import io
import json
from unittest.mock import MagicMock
import pytest
from app.agents.orchestrator import extract_json_from_llm
from app.db.database import save_document
from app.services.comparison_service import compare_research_papers
from app.services.gap_analysis_service import analyze_research_gaps
from app.services.llm_service import LLMService
from app.services.summary_service import generate_paper_summary
from tests.conftest import make_pdf_bytes
from tests.test_chat import MockGroqClient


SAMPLE_SUMMARY_JSON = json.dumps(
    {
        "summary": "This research paper introduces the Transformer architecture relying on self-attention.",
        "sections": {
            "objective": "Eliminate recurrence and convolutions in sequence modeling.",
            "problem_statement": "Sequential computation prevents parallelization.",
            "methodology": "Multi-head scaled dot-product attention.",
            "dataset": "WMT 2014 English-to-German.",
            "results": "Achieved 28.4 BLEU, establishing a new state of the art.",
            "conclusion": "Transformers train faster and achieve superior translation quality.",
            "limitations": "Quadratic memory scaling with sequence length.",
            "future_work": "Apply attention to image and audio domains.",
        },
    }
)

SAMPLE_COMPARISON_JSON = json.dumps(
    {
        "criteria": ["Objective", "Methodology", "Dataset", "Results", "Limitations"],
        "individual_papers": [
            {
                "document_id": "doc1",
                "filename": "paper1.pdf",
                "objective": "Attention mechanisms",
                "methodology": "Self-attention",
                "dataset": "WMT 2014",
                "algorithms_or_models": "Transformer",
                "evaluation_metrics": "BLEU",
                "main_findings": "28.4 BLEU",
                "limitations": "Quadratic memory",
                "future_work": "Vision models",
            },
            {
                "document_id": "doc2",
                "filename": "paper2.pdf",
                "objective": "Deep residual learning",
                "methodology": "Residual shortcuts",
                "dataset": "ImageNet",
                "algorithms_or_models": "ResNet",
                "evaluation_metrics": "Top-5 error",
                "main_findings": "3.57% error",
                "limitations": "Not explicitly discussed in the paper.",
                "future_work": "Video networks",
            },
        ],
        "similarities": ["Both propose deep learning architectures to improve representation learning."],
        "differences": ["Paper 1 operates on NLP sequence data while Paper 2 targets computer vision."],
        "methodological_differences": "Attention layers allow global token interaction whereas residual blocks utilize local convolutions.",
        "overall_analysis": "Both architectures form modern deep learning cornerstones.",
    }
)

SAMPLE_GAP_JSON = json.dumps(
    {
        "explicit_limitations": ["Quadratic memory complexity with long sequences."],
        "author_suggested_future_work": ["Investigate attention mechanisms for vision and speech."],
        "potential_research_gaps": [
            {
                "gap_type": "explicit_limitation",
                "title": "Quadratic Memory Scaling",
                "description": "Full attention matrix computation requires O(N^2) memory.",
                "why_identified": "Explicitly stated by authors in model analysis section.",
                "supporting_evidence": "Attention maps require O(n^2) computational resources.",
                "suggested_future_direction": "Explore sparse or linear attention approximations.",
            },
            {
                "gap_type": "inferred_potential_gap",
                "title": "Cross-Modality Transferability",
                "description": "Evaluation was limited to text translation datasets.",
                "why_identified": "No benchmarks on multimodal or low-resource domains.",
                "supporting_evidence": "Trained only on WMT 2014 English-German dataset.",
                "suggested_future_direction": "Evaluate self-attention on multimodal and low-resource tasks.",
            },
        ],
        "future_research_directions": ["Linear-complexity self-attention", "Multimodal transformer adaptation"],
    }
)


@pytest.fixture
def mock_phase5_groq():
    """Inject mock Groq client configured with valid structured JSON responses."""
    mock_client = MockGroqClient(SAMPLE_SUMMARY_JSON)
    LLMService.set_client(mock_client)
    yield mock_client
    LLMService.reset_client()


# =========================================================================
# FEATURE 1: SUMMARIZATION TESTS
# =========================================================================

def test_valid_summary_request(client, mock_phase5_groq):
    # Upload document
    pdf_bytes = make_pdf_bytes(["Attention mechanisms in Transformers.", "Empirical evaluation on WMT 2014."])
    up_res = client.post("/documents/upload", files={"file": ("transformer.pdf", io.BytesIO(pdf_bytes), "application/pdf")})
    doc_id = up_res.json()["document_id"]

    res = client.post("/summarize", json={"document_id": doc_id, "summary_type": "detailed"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["document_id"] == doc_id
    assert data["filename"] == "transformer.pdf"
    assert "Transformer architecture" in data["summary"]
    assert "objective" in data["sections"]
    assert len(data["sources"]) >= 1


def test_summary_types(client, mock_phase5_groq):
    pdf_bytes = make_pdf_bytes(["Test short and key_points summaries."])
    up_res = client.post("/documents/upload", files={"file": ("paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")})
    doc_id = up_res.json()["document_id"]

    for stype in ["short", "key_points", "academic"]:
        res = client.post("/summarize", json={"document_id": doc_id, "summary_type": stype})
        assert res.status_code == 200
        assert res.json()["summary_type"] == stype


def test_summary_missing_document(client):
    res = client.post("/summarize", json={"document_id": "non-existent-id", "summary_type": "detailed"})
    assert res.status_code == 404
    assert "not found" in res.json()["error"]["message"]


def test_summary_scanned_document(client, temp_storage):
    # Create scanned doc entry in SQLite
    save_document(
        {
            "id": "doc-scanned",
            "filename": "scanned.pdf",
            "file_path": str(temp_storage / "uploads" / "scanned.pdf"),
            "file_size": 1000,
            "upload_timestamp": "2026-11-22T00:00:00Z",
            "status": "scanned_pdf",
            "total_pages": 1,
            "extracted_text_length": 0,
            "total_chunks": 0,
            "is_scanned": 1,
            "message": "Scanned document",
        }
    )
    res = client.post("/summarize", json={"document_id": "doc-scanned"})
    assert res.status_code == 400
    assert "scanned" in res.json()["error"]["message"].lower()


def test_long_document_map_reduce(mock_phase5_groq):
    # Simulate long paper with 12 chunks
    chunks = [
        {"chunk_index": i, "text": f"Section content paragraph {i} explaining complex deep learning mechanics." * 10, "page_number": i + 1, "start_page": i + 1, "end_page": i + 1}
        for i in range(12)
    ]
    from app.agents.summarizer import SummarizerAgent

    res = SummarizerAgent.summarize_paper("long_paper.pdf", chunks, summary_type="academic")
    assert "summary" in res
    assert "sections" in res


# =========================================================================
# FEATURE 2: COMPARISON TESTS
# =========================================================================

def test_compare_two_papers(client, mock_phase5_groq):
    mock_phase5_groq.completions.response_text = SAMPLE_COMPARISON_JSON

    p1 = make_pdf_bytes(["Paper 1 on attention."])
    p2 = make_pdf_bytes(["Paper 2 on residual connections."])
    id1 = client.post("/documents/upload", files={"file": ("p1.pdf", io.BytesIO(p1), "application/pdf")}).json()["document_id"]
    id2 = client.post("/documents/upload", files={"file": ("p2.pdf", io.BytesIO(p2), "application/pdf")}).json()["document_id"]

    res = client.post("/compare", json={"document_ids": [id1, id2]})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert len(data["documents_compared"]) == 2
    assert len(data["individual_papers"]) == 2
    assert len(data["similarities"]) >= 1
    assert len(data["differences"]) >= 1
    assert "methodological_differences" in data
    assert len(data["sources"]) >= 2


def test_compare_invalid_or_missing_ids(client):
    res = client.post("/compare", json={"document_ids": ["doc-valid", "doc-invalid"]})
    assert res.status_code in (400, 404)

    # Fewer than 2 docs
    res_few = client.post("/compare", json={"document_ids": ["single-doc"]})
    assert res_few.status_code == 422 or res_few.status_code == 400


def test_compare_duplicate_ids(client, mock_phase5_groq):
    p1 = make_pdf_bytes(["Only paper."])
    id1 = client.post("/documents/upload", files={"file": ("single.pdf", io.BytesIO(p1), "application/pdf")}).json()["document_id"]

    # Passing the exact same ID twice should be caught as < 2 distinct papers
    res = client.post("/compare", json={"document_ids": [id1, id1]})
    assert res.status_code == 400
    assert "at least two distinct" in res.json()["error"]["message"]


# =========================================================================
# FEATURE 3: RESEARCH GAP ANALYSIS TESTS
# =========================================================================

def test_research_gaps_analysis(client, mock_phase5_groq):
    mock_phase5_groq.completions.response_text = SAMPLE_GAP_JSON

    pdf1 = make_pdf_bytes(["Transformer study stating memory limitation."])
    id1 = client.post("/documents/upload", files={"file": ("transformer.pdf", io.BytesIO(pdf1), "application/pdf")}).json()["document_id"]

    res = client.post("/research-gaps", json={"document_ids": [id1]})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert len(data["explicit_limitations"]) >= 1
    assert len(data["author_suggested_future_work"]) >= 1
    assert len(data["potential_research_gaps"]) >= 1

    gap_item = data["potential_research_gaps"][0]
    assert gap_item["gap_type"] in ("explicit_limitation", "inferred_potential_gap")
    assert "why_identified" in gap_item
    assert "supporting_evidence" in gap_item
    assert "suggested_future_direction" in gap_item


# =========================================================================
# ORCHESTRATION & ERROR HANDLING TESTS
# =========================================================================

def test_extract_json_from_llm():
    # Markdown wrapped
    raw_md = "```json\n{\"test\": 123}\n```"
    parsed = extract_json_from_llm(raw_md)
    assert parsed["test"] == 123

    # Direct raw
    assert extract_json_from_llm("{\"value\": true}")["value"] is True

    # Bad JSON raises ValueError
    with pytest.raises(ValueError, match="Could not parse valid JSON"):
        extract_json_from_llm("Not a json string at all")
