from fastapi import APIRouter, HTTPException, status
from app.schemas.summary import SummaryRequest, SummaryResponse
from app.services.summary_service import generate_paper_summary

router = APIRouter(tags=["Summarization"])


@router.post(
    "/summarize",
    response_model=SummaryResponse,
    summary="Generate structured research paper summary",
)
async def summarize_document(payload: SummaryRequest):
    """Generate a structured academic summary for an uploaded research paper.

    - Supports 'short', 'detailed', 'academic', and 'key_points' formats
    - Breaks down into objective, methodology, dataset, results, limitations, and future work
    - Uses hierarchical processing for long papers
    - Preserves source citations
    """
    try:
        result = generate_paper_summary(
            document_id=payload.document_id,
            summary_type=payload.summary_type,
        )
        return SummaryResponse(**result)

    except KeyError as key_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(key_err),
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except RuntimeError as run_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(run_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during summarization. Please try again.",
        )
