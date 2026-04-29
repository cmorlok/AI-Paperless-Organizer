from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from app.database import get_db
from app.models import SavedAnalysis
from app.services.paperless.protocol import PaperlessClient
from app.services.similarity.protocol import SimilarityService
from app.services.merge.protocol import MergeService
from app.services.statistics.protocol import StatisticsService
from app.services.llm.protocol import LLMService as LLMProviderService
from app.models.settings_model import LLM_KEY_CLASSIFIER_PROVIDER, LLM_KEY_CLASSIFIER_MODEL
from app.routers.settings import get_setting
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
    client: FromDishka[PaperlessClient] = None,
    llm: FromDishka[LLMProviderService] = None,
    db: AsyncSession = Depends(get_db)
):
    """Estimate tokens needed for analysis."""
    correspondents = await client.get_correspondents_with_counts()
    items_count = len(correspondents)
    # Rough estimate: prompt template ~500 chars + ~30 chars per item name
    avg_name_length = sum(len(c.get("name", "")) for c in correspondents) / max(items_count, 1)
    estimated_input = 500 + int(items_count * (avg_name_length + 10))
    estimated_tokens = estimated_input // 4
    
    # Default token limit for batching decisions
    provider = await get_setting(LLM_KEY_CLASSIFIER_PROVIDER, db) or ""
    model = await get_setting(LLM_KEY_CLASSIFIER_MODEL, db) or ""
    token_limit = await llm.get_token_limit(provider, model)
    is_cloud = not llm.is_local_provider(provider)
    safe_limit = int(token_limit * 0.8)
    needs_batching = estimated_tokens > safe_limit
    recommended_batches = max(1, (estimated_tokens + safe_limit - 1) // safe_limit) if needs_batching else 1

    return {
        "items_info": f"{items_count} Korrespondenten",
        "estimated_tokens": estimated_tokens,
        "token_limit": token_limit,
        "is_cloud": is_cloud,
        "recommended_batches": recommended_batches,
        "warning": f"~{estimated_tokens:,} Tokens > {safe_limit:,} Limit. Wird in {recommended_batches} Batches aufgeteilt." if needs_batching else None
    }


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
    db: AsyncSession = Depends(get_db)
):
    """Analyze correspondents and find similar groups using AI."""
    batch_size = request.batch_size if request else 200
    result = await similarity_service.find_similar_correspondents(batch_size=batch_size)
    
    # Save the analysis result
    groups = result.get("groups", [])
    stats = result.get("stats", {})
    
    # Delete old analysis
    await db.execute(delete(SavedAnalysis).where(SavedAnalysis.entity_type == ENTITY_TYPE))
    
    # Save new analysis
    saved = SavedAnalysis(
        entity_type=ENTITY_TYPE,
        analysis_type="similarity",
        groups=groups,
        stats=stats,
        items_count=stats.get("items_count", 0),
        groups_count=len(groups),
        processed_groups=[]
    )
    db.add(saved)
    await db.commit()
    
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
    client: FromDishka[PaperlessClient] = None,
    stats_service: FromDishka[StatisticsService] = None
):
    """Delete all correspondents with 0 documents - PARALLEL for speed."""
    correspondents = await client.get_correspondents_with_counts()
    empty = [c for c in correspondents if c.get("document_count", 0) == 0]
    
    if not empty:
        return {"deleted": 0, "total": 0, "errors": None}
    
    # Parallel deletion for speed (batch of 10 at a time)
    errors = []
    deleted = 0
    batch_size = 10
    
    async def delete_one(item):
        try:
            await client.delete_correspondent(item["id"])
            return True, None
        except Exception as e:
            return False, f"{item['name']}: {str(e)}"
    
    for i in range(0, len(empty), batch_size):
        batch = empty[i:i + batch_size]
        results = await asyncio.gather(*[delete_one(c) for c in batch])
        for success, error in results:
            if success:
                deleted += 1
            elif error:
                errors.append(error)
    
    # Record statistics
    if deleted > 0:
        await stats_service.record_operation(
            entity_type="correspondents",
            operation="deleted",
            items_affected=deleted,
            documents_affected=0,
            items_before=len(correspondents),
            items_after=len(correspondents) - deleted
        )
    
    return {
        "deleted": deleted,
        "total": len(empty),
        "errors": errors if errors else None
    }


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

