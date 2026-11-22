from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status
from app.schemas.research_gaps import GapAnalysisRequest, GapAnalysisResponse
from app.services.gap_analysis_service import analyze_research_gaps

router = APIRouter(tags=["Research Gaps"])


@router.post(
    "/research-gaps",
    response_model=GapAnalysisResponse,
    summary="Discover research gaps and future directions",
)
async def analyze_gaps(payload: GapAnalysisRequest):
    """Analyze uploaded papers to surface explicit author limitations and inferred potential research gaps.

    - Strictly categorizes explicit author limitations vs inferred opportunities
    - Grounded in cited evidence from uploaded documents
    - Provides concrete suggested future directions
    """
    try:
        result = analyze_research_gaps(document_ids=payload.document_ids)
        return GapAnalysisResponse(**result)

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
            detail="An unexpected error occurred during research gap analysis. Please try again.",
        )


@router.get(
    "/gaps",
    response_model=GapAnalysisResponse,
    summary="Discover research gaps and future directions via GET",
)
@router.get(
    "/research-gaps",
    response_model=GapAnalysisResponse,
    summary="Discover research gaps and future directions via GET (alias)",
)
async def analyze_gaps_get(doc_ids: Optional[str] = Query(default=None)):
    """Analyze research gaps across papers via GET."""
    ids = [d.strip() for d in doc_ids.split(",") if d.strip()] if doc_ids else None
    try:
        result = analyze_research_gaps(document_ids=ids)
        return GapAnalysisResponse(**result)
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
            detail="An unexpected error occurred during research gap analysis. Please try again.",
        )
