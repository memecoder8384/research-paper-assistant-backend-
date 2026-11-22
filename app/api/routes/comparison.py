from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status
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


@router.get(
    "/comparison",
    response_model=ComparisonResponse,
    summary="Compare two or more research papers via GET",
)
@router.get(
    "/compare",
    response_model=ComparisonResponse,
    summary="Compare two or more research papers via GET (alias)",
)
async def compare_documents_get(
    paperA: Optional[str] = Query(default=None),
    paperB: Optional[str] = Query(default=None),
    paperC: Optional[str] = Query(default=None),
    doc_ids: Optional[str] = Query(default=None),
):
    """Compare papers using query parameters."""
    ids = []
    if doc_ids:
        ids.extend([d.strip() for d in doc_ids.split(",") if d.strip()])
    else:
        if paperA:
            ids.append(paperA)
        if paperB:
            ids.append(paperB)
        if paperC:
            ids.append(paperC)

    if len(ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 paper IDs are required for comparison.",
        )

    try:
        result = compare_research_papers(document_ids=ids[:5])
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
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during paper comparison. Please try again.",
        )
