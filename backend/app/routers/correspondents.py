from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from app.database import get_db
from app.models import SavedAnalysis
from app.services.paperless import PaperlessClient
from app.services.similarity import SimilarityService
from app.services.merge import MergeService
from app.services.correspondents import CorrespondentsService
from dishka.integrations.fastapi import inject
from dishka import FromDishka

router = APIRouter()
ENTITY_TYPE = "correspondents"


class AnalyzeRequest(BaseModel):
    """Request to analyze correspondents for duplicates."""
    batch_size: int = 200  # Max items per batch


class MergeRequest(BaseModel):
    """Request to merge correspondents."""
    target_id: int
    target_name: str
    source_ids: List[int]


class MergeGroup(BaseModel):
    """A group of similar correspondents."""
    suggested_name: str
    confidence: float
    members: List[dict]
    reasoning: str


@router.get("/")
@inject
async def list_correspondents(client: FromDishka[PaperlessClient] = None):
    """List all correspondents with document counts."""
    return await client.get_correspondents_with_counts()


@router.get("/estimate")
@inject
async def estimate_correspondents(
    correspondents_service: FromDishka[CorrespondentsService] = None,
):
    """Estimate tokens needed for analysis."""
    return await correspondents_service.estimate_correspondents()


@router.get("/saved-analysis")
async def get_saved_analysis(db: AsyncSession = Depends(get_db)):
    """Check if there's a saved analysis."""
    result = await db.execute(
        select(SavedAnalysis)
        .where(SavedAnalysis.entity_type == ENTITY_TYPE)
        .order_by(SavedAnalysis.created_at.desc())
        .limit(1)
    )
    saved = result.scalar_one_or_none()

    if saved:
        return {
            "exists": True,
            "id": saved.id,
            "created_at": saved.created_at.isoformat() if saved.created_at else None,
            "items_count": saved.items_count,
            "groups_count": saved.groups_count,
            "processed_groups": saved.processed_groups or []
        }
    return {"exists": False}


@router.get("/saved-analysis/load")
async def load_saved_analysis(db: AsyncSession = Depends(get_db)):
    """Load the saved analysis results."""
    result = await db.execute(
        select(SavedAnalysis)
        .where(SavedAnalysis.entity_type == ENTITY_TYPE)
        .order_by(SavedAnalysis.created_at.desc())
        .limit(1)
    )
    saved = result.scalar_one_or_none()

    if not saved:
        raise HTTPException(status_code=404, detail="Keine gespeicherte Analyse gefunden")

    return {
        "groups": saved.groups,
        "stats": saved.stats,
        "created_at": saved.created_at.isoformat() if saved.created_at else None,
        "processed_groups": saved.processed_groups or []
    }


@router.delete("/saved-analysis")
async def delete_saved_analysis(db: AsyncSession = Depends(get_db)):
    """Delete saved analysis."""
    await db.execute(delete(SavedAnalysis).where(SavedAnalysis.entity_type == ENTITY_TYPE))
    await db.commit()
    return {"success": True}


@router.post("/saved-analysis/mark-processed")
async def mark_group_processed(
    group_index: int,
    db: AsyncSession = Depends(get_db)
):
    """Mark a group as processed (merged or dismissed)."""
    result = await db.execute(
        select(SavedAnalysis)
        .where(SavedAnalysis.entity_type == ENTITY_TYPE)
        .order_by(SavedAnalysis.created_at.desc())
        .limit(1)
    )
    saved = result.scalar_one_or_none()

    if saved:
        processed = saved.processed_groups or []
        if group_index not in processed:
            processed.append(group_index)
            saved.processed_groups = processed
            await db.commit()

    return {"success": True}


@router.post("/analyze")
@inject
async def analyze_correspondents(
    request: AnalyzeRequest = None,
    similarity_service: FromDishka[SimilarityService] = None,
    correspondents_service: FromDishka[CorrespondentsService] = None,
):
    """Analyze correspondents and find similar groups using AI."""
    batch_size = request.batch_size if request else 200
    result = await similarity_service.find_similar_correspondents(batch_size=batch_size)
    await correspondents_service.save_similarity_analysis(result)
    return result


@router.post("/merge")
@inject
async def merge_correspondents(
    request: MergeRequest,
    merge_service: FromDishka[MergeService] = None
):
    """Merge multiple correspondents into one."""
    result = await merge_service.merge_correspondents(
        target_id=request.target_id,
        target_name=request.target_name,
        source_ids=request.source_ids
    )
    return result


@router.get("/history")
@inject
async def get_merge_history(
    merge_service: FromDishka[MergeService] = None
):
    """Get merge history for correspondents."""
    return await merge_service.get_history("correspondents")


@router.get("/empty")
@inject
async def get_empty_correspondents(
    client: FromDishka[PaperlessClient] = None
):
    """Get correspondents with 0 documents."""
    correspondents = await client.get_correspondents_with_counts()
    empty = [c for c in correspondents if c.get("document_count", 0) == 0]
    return {
        "count": len(empty),
        "items": empty
    }


@router.delete("/empty")
@inject
async def delete_empty_correspondents(
    correspondents_service: FromDishka[CorrespondentsService] = None,
):
    """Delete all correspondents with 0 documents - PARALLEL for speed."""
    return await correspondents_service.delete_empty_correspondents()


@router.delete("/{correspondent_id}")
@inject
async def delete_correspondent(
    correspondent_id: int,
    client: FromDishka[PaperlessClient] = None
):
    """Delete a single correspondent."""
    try:
        await client.delete_correspondent(correspondent_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
