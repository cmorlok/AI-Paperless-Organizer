"""Correspondents Router — thin endpoints only."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

logger = logging.getLogger(__name__)
from app.services.paperless import PaperlessClient
from app.services.similarity import SimilarityService
from app.services.merge import MergeService
from app.services.correspondents import CorrespondentsService
from dishka.integrations.fastapi import inject
from dishka import FromDishka

router = APIRouter()


class AnalyzeRequest(BaseModel):
    batch_size: int = 200


class MergeRequest(BaseModel):
    target_id: int
    target_name: str
    source_ids: List[int]


@router.get("/")
@inject
async def list_correspondents(client: FromDishka[PaperlessClient] = None):
    assert client is not None
    return await client.get_correspondents_with_counts()


@router.get("/estimate")
@inject
async def estimate_correspondents(correspondents_service: FromDishka[CorrespondentsService] = None):
    assert correspondents_service is not None
    return await correspondents_service.estimate_correspondents()


@router.get("/saved-analysis")
@inject
async def get_saved_analysis(correspondents_service: FromDishka[CorrespondentsService] = None):
    assert correspondents_service is not None
    return await correspondents_service.get_saved_analysis()


@router.get("/saved-analysis/load")
@inject
async def load_saved_analysis(correspondents_service: FromDishka[CorrespondentsService] = None):
    assert correspondents_service is not None
    result = await correspondents_service.load_saved_analysis()
    if not result:
        raise HTTPException(status_code=404, detail="Keine gespeicherte Analyse gefunden")
    return result


@router.delete("/saved-analysis")
@inject
async def delete_saved_analysis(correspondents_service: FromDishka[CorrespondentsService] = None):
    assert correspondents_service is not None
    await correspondents_service.delete_saved_analysis()
    return {"success": True}


@router.post("/saved-analysis/mark-processed")
@inject
async def mark_group_processed(group_index: int, correspondents_service: FromDishka[CorrespondentsService] = None):
    assert correspondents_service is not None
    await correspondents_service.mark_group_processed(group_index)
    return {"success": True}


@router.post("/analyze")
@inject
async def analyze_correspondents(
    request: AnalyzeRequest | None = None,
    similarity_service: FromDishka[SimilarityService] = None,
    correspondents_service: FromDishka[CorrespondentsService] = None,
):
    assert similarity_service is not None
    assert correspondents_service is not None
    batch_size = request.batch_size if request else 200
    result = await similarity_service.find_similar_correspondents(batch_size=batch_size)
    await correspondents_service.save_similarity_analysis(result)
    return result


@router.post("/merge")
@inject
async def merge_correspondents(request: MergeRequest, merge_service: FromDishka[MergeService] = None):
    assert merge_service is not None
    return await merge_service.merge_correspondents(
        target_id=request.target_id,
        target_name=request.target_name,
        source_ids=request.source_ids,
    )


@router.get("/history")
@inject
async def get_merge_history(merge_service: FromDishka[MergeService] = None):
    assert merge_service is not None
    return await merge_service.get_history("correspondents")


@router.get("/empty")
@inject
async def get_empty_correspondents(client: FromDishka[PaperlessClient] = None):
    assert client is not None
    correspondents = await client.get_correspondents_with_counts()
    empty = [c for c in correspondents if c.get("document_count", 0) == 0]
    return {"count": len(empty), "items": empty}


@router.delete("/empty")
@inject
async def delete_empty_correspondents(correspondents_service: FromDishka[CorrespondentsService] = None):
    assert correspondents_service is not None
    return await correspondents_service.delete_empty_correspondents()


@router.delete("/{correspondent_id}")
@inject
async def delete_correspondent(correspondent_id: int, client: FromDishka[PaperlessClient] = None):
    assert client is not None
    try:
        await client.delete_correspondent(correspondent_id)
        return {"success": True}
    except Exception as e:
        logger.error("Delete correspondent failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")
