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
from app.services.tags import TagsService
from dishka.integrations.fastapi import inject
from dishka import FromDishka

router = APIRouter()
ENTITY_TYPE = "tags"


class MergeRequest(BaseModel):
    """Request to merge tags."""
    target_id: int
    target_name: str
    source_ids: List[int]


class AnalyzeRequest(BaseModel):
    """Request to analyze tags for duplicates."""
    batch_size: int = 200


class BulkDeleteRequest(BaseModel):
    tag_ids: List[int]


class RemoveFromAnalysesRequest(BaseModel):
    tag_ids: List[int]


# ============ LIST / ESTIMATE ============

@router.get("/")
@inject
async def list_tags(client: FromDishka[PaperlessClient] = None):
    assert client is not None
    """List all tags with document counts."""
    return await client.get_tags_with_counts()


@router.get("/estimate")
@inject
async def estimate_tags(
    analysis_type: str = "nonsense",
    tags_service: FromDishka[TagsService] = None,
):
    assert tags_service is not None
    """Estimate tokens needed for specific analysis type.

    analysis_type can be: nonsense, correspondent, doctype, similar
    """
    return await tags_service.estimate_tags(analysis_type)


# ============ SAVED ANALYSIS CRUD ============

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


# ============ ANALYZE ============

@router.post("/analyze")
@inject
async def analyze_tags(
    request: AnalyzeRequest | None = None,
    similarity_service: FromDishka[SimilarityService] = None,
    tags_service: FromDishka[TagsService] = None,
):
    assert similarity_service is not None
    assert tags_service is not None
    """Analyze tags and find similar groups using AI."""
    batch_size = request.batch_size if request else 200
    result = await similarity_service.find_similar_tags(batch_size=batch_size)
    await getattr(tags_service, "save_similarity_analysis")(result)
    return result


# ============ MERGE ============

@router.post("/merge")
@inject
async def merge_tags(
    request: MergeRequest,
    merge_service: FromDishka[MergeService] = None
):
    assert merge_service is not None
    """Merge multiple tags into one."""
    return await merge_service.merge_tags(
        target_id=request.target_id,
        target_name=request.target_name,
        source_ids=request.source_ids
    )


@router.get("/history")
@inject
async def get_merge_history(
    merge_service: FromDishka[MergeService] = None
):
    assert merge_service is not None
    """Get merge history for tags."""
    return await merge_service.get_history("tags")


# ============ EMPTY TAGS ============

@router.get("/empty")
@inject
async def get_empty_tags(
    client: FromDishka[PaperlessClient] = None
):
    assert client is not None
    """Get tags with 0 documents."""
    tags = await client.get_tags_with_counts()
    empty = [t for t in tags if t.get("document_count", 0) == 0]
    return {
        "count": len(empty),
        "items": empty
    }


@router.delete("/empty")
@inject
async def delete_empty_tags(
    tags_service: FromDishka[TagsService] = None,
):
    assert tags_service is not None
    """Delete all tags with 0 documents - PARALLEL for speed."""
    return await tags_service.delete_empty_tags()


# ============ BULK DELETE ============

@router.post("/bulk-delete")
@inject
async def bulk_delete_tags(
    request: BulkDeleteRequest,
    tags_service: FromDishka[TagsService] = None,
):
    assert tags_service is not None
    """Delete multiple tags in parallel and keep DB cache in sync."""
    try:
        return await tags_service.bulk_delete_tags(request.tag_ids)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============ REMOVE FROM SAVED ANALYSES ============

@router.post("/saved-analyses/remove-tags")
@inject
async def remove_tags_from_saved_analyses(
    request: RemoveFromAnalysesRequest,
    tags_service: FromDishka[TagsService] = None,
):
    assert tags_service is not None
    """Remove deleted tag IDs from ALL saved analyses so they don't reappear."""
    return await tags_service.remove_tags_from_saved_analyses(request.tag_ids)


# ============ SINGLE DELETE ============

@router.delete("/{tag_id}")
@inject
async def delete_tag(
    tag_id: int,
    client: FromDishka[PaperlessClient] = None
):
    assert client is not None
    """Delete a single tag by ID."""
    try:
        await client.delete_tag(tag_id)
        return {"success": True, "message": f"Tag {tag_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============ NONSENSE TAGS ============

@router.get("/saved-nonsense")
async def get_saved_nonsense_analysis(db: AsyncSession = Depends(get_db)):
    """Check if there's a saved nonsense analysis."""
    result = await db.execute(
        select(SavedAnalysis)
        .where(SavedAnalysis.entity_type == "tags_nonsense")
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
            "groups_count": saved.groups_count
        }
    return {"exists": False}


@router.get("/saved-nonsense/load")
async def load_saved_nonsense_analysis(db: AsyncSession = Depends(get_db)):
    """Load the saved nonsense analysis results."""
    result = await db.execute(
        select(SavedAnalysis)
        .where(SavedAnalysis.entity_type == "tags_nonsense")
        .order_by(SavedAnalysis.created_at.desc())
        .limit(1)
    )
    saved = result.scalar_one_or_none()

    if not saved:
        return {"exists": False, "nonsense_tags": []}

    return {
        "exists": True,
        "nonsense_tags": saved.groups,
        "stats": saved.stats,
        "created_at": saved.created_at.isoformat() if saved.created_at else None
    }


@router.delete("/saved-nonsense")
async def delete_saved_nonsense_analysis(db: AsyncSession = Depends(get_db)):
    """Delete saved nonsense analysis."""
    await db.execute(delete(SavedAnalysis).where(SavedAnalysis.entity_type == "tags_nonsense"))
    await db.commit()
    return {"success": True}


@router.post("/analyze-nonsense")
@inject
async def analyze_nonsense_tags(
    similarity_service: FromDishka[SimilarityService] = None,
    tags_service: FromDishka[TagsService] = None,
):
    assert similarity_service is not None
    assert tags_service is not None
    """Analyze tags to find nonsensical/useless tags using AI and SAVE results."""
    result = await similarity_service.find_nonsense_tags()
    await getattr(tags_service, "save_nonsense_analysis")(result)
    return result


# ============ CORRESPONDENT TAGS ============

@router.get("/saved-correspondent-matches")
async def get_saved_correspondent_analysis(db: AsyncSession = Depends(get_db)):
    """Check if there's a saved correspondent matches analysis."""
    result = await db.execute(
        select(SavedAnalysis)
        .where(SavedAnalysis.entity_type == "tags_correspondents")
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
            "groups_count": saved.groups_count
        }
    return {"exists": False}


@router.get("/saved-correspondent-matches/load")
async def load_saved_correspondent_analysis(db: AsyncSession = Depends(get_db)):
    """Load the saved correspondent matches analysis results."""
    result = await db.execute(
        select(SavedAnalysis)
        .where(SavedAnalysis.entity_type == "tags_correspondents")
        .order_by(SavedAnalysis.created_at.desc())
        .limit(1)
    )
    saved = result.scalar_one_or_none()

    if not saved:
        return {"exists": False, "correspondent_tags": []}

    return {
        "exists": True,
        "correspondent_tags": saved.groups,
        "stats": saved.stats,
        "created_at": saved.created_at.isoformat() if saved.created_at else None
    }


@router.delete("/saved-correspondent-matches")
async def delete_saved_correspondent_analysis(db: AsyncSession = Depends(get_db)):
    """Delete saved correspondent matches analysis."""
    await db.execute(delete(SavedAnalysis).where(SavedAnalysis.entity_type == "tags_correspondents"))
    await db.commit()
    return {"success": True}


@router.post("/analyze-correspondent-matches")
@inject
async def analyze_correspondent_matches(
    similarity_service: FromDishka[SimilarityService] = None,
    tags_service: FromDishka[TagsService] = None,
):
    assert similarity_service is not None
    assert tags_service is not None
    """Analyze tags that should be correspondents using AI and SAVE results."""
    result = await similarity_service.find_tags_that_are_correspondents()
    await getattr(tags_service, "save_correspondent_matches")(result)
    return result


# ============ DOCTYPE TAGS ============

@router.get("/saved-doctype-matches")
async def get_saved_doctype_analysis(db: AsyncSession = Depends(get_db)):
    """Check if there's a saved doctype matches analysis."""
    result = await db.execute(
        select(SavedAnalysis)
        .where(SavedAnalysis.entity_type == "tags_doctypes")
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
            "groups_count": saved.groups_count
        }
    return {"exists": False}


@router.get("/saved-doctype-matches/load")
async def load_saved_doctype_analysis(db: AsyncSession = Depends(get_db)):
    """Load the saved doctype matches analysis results."""
    result = await db.execute(
        select(SavedAnalysis)
        .where(SavedAnalysis.entity_type == "tags_doctypes")
        .order_by(SavedAnalysis.created_at.desc())
        .limit(1)
    )
    saved = result.scalar_one_or_none()

    if not saved:
        return {"exists": False, "doctype_tags": []}

    return {
        "exists": True,
        "doctype_tags": saved.groups,
        "stats": saved.stats,
        "created_at": saved.created_at.isoformat() if saved.created_at else None
    }


@router.delete("/saved-doctype-matches")
async def delete_saved_doctype_analysis(db: AsyncSession = Depends(get_db)):
    """Delete saved doctype matches analysis."""
    await db.execute(delete(SavedAnalysis).where(SavedAnalysis.entity_type == "tags_doctypes"))
    await db.commit()
    return {"success": True}


@router.post("/analyze-doctype-matches")
@inject
async def analyze_doctype_matches(
    similarity_service: FromDishka[SimilarityService] = None,
    tags_service: FromDishka[TagsService] = None,
):
    assert similarity_service is not None
    assert tags_service is not None
    """Analyze tags that should be document types using AI and SAVE results."""
    result = await similarity_service.find_tags_that_are_document_types()
    await getattr(tags_service, "save_doctype_matches")(result)
    return result
