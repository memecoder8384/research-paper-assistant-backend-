from fastapi import APIRouter, HTTPException, status
from app.schemas.comparison import ComparisonRequest, ComparisonResponse
from app.services.comparison_service import compare_research_papers

router = APIRouter(tags=["Comparison"])


@router.post(
    "/compare",
    response_model=ComparisonResponse,
    summary="Compare two or more research papers",
)
async def compare_documents(payload: ComparisonRequest):
    """Compare 2 to 5 research papers across standardized academic dimensions.

    - Analyzes objectives, methodologies, datasets, models, metrics, findings, and limitations
    - Identifies commonalities, differences, and methodological trade-offs
    - Preserves isolated source citations per paper
    """
    try:
        result = compare_research_papers(document_ids=payload.document_ids)
        return ComparisonResponse(**result)

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
            detail="An unexpected error occurred during paper comparison. Please try again.",
        )
