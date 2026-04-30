"""Document Types Router — thin endpoints only."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

logger = logging.getLogger(__name__)
from app.services.paperless import PaperlessClient
from app.services.similarity import SimilarityService
from app.services.merge import MergeService
from app.services.document_types import DocumentTypesService
from dishka.integrations.fastapi import inject
from dishka import FromDishka

router = APIRouter()


class MergeRequest(BaseModel):
    target_id: int
    target_name: str
    source_ids: List[int]


class AnalyzeRequest(BaseModel):
    batch_size: int = 200


@router.get("/")
@inject
async def list_document_types(client: FromDishka[PaperlessClient] = None):
    assert client is not None
    return await client.get_document_types_with_counts()


@router.get("/estimate")
@inject
async def estimate_document_types(document_types_service: FromDishka[DocumentTypesService] = None):
    assert document_types_service is not None
    return await document_types_service.estimate_document_types()


@router.get("/saved-analysis")
@inject
async def get_saved_analysis(document_types_service: FromDishka[DocumentTypesService] = None):
    assert document_types_service is not None
    return await document_types_service.get_saved_analysis()


@router.get("/saved-analysis/load")
@inject
async def load_saved_analysis(document_types_service: FromDishka[DocumentTypesService] = None):
    assert document_types_service is not None
    result = await document_types_service.load_saved_analysis()
    if not result:
        raise HTTPException(status_code=404, detail="Keine gespeicherte Analyse gefunden")
    return result


@router.delete("/saved-analysis")
@inject
async def delete_saved_analysis(document_types_service: FromDishka[DocumentTypesService] = None):
    assert document_types_service is not None
    await document_types_service.delete_saved_analysis()
    return {"success": True}


@router.post("/saved-analysis/mark-processed")
@inject
async def mark_group_processed(group_index: int, document_types_service: FromDishka[DocumentTypesService] = None):
    assert document_types_service is not None
    await document_types_service.mark_group_processed(group_index)
    return {"success": True}


@router.post("/analyze")
@inject
async def analyze_document_types(
    request: AnalyzeRequest | None = None,
    similarity_service: FromDishka[SimilarityService] = None,
    document_types_service: FromDishka[DocumentTypesService] = None,
):
    assert similarity_service is not None
    assert document_types_service is not None
    batch_size = request.batch_size if request else 200
    result = await similarity_service.find_similar_document_types(batch_size=batch_size)
    await document_types_service.save_similarity_analysis(result)
    return result


@router.post("/merge")
@inject
async def merge_document_types(request: MergeRequest, merge_service: FromDishka[MergeService] = None):
    assert merge_service is not None
    return await merge_service.merge_document_types(
        target_id=request.target_id,
        target_name=request.target_name,
        source_ids=request.source_ids,
    )


@router.get("/history")
@inject
async def get_merge_history(merge_service: FromDishka[MergeService] = None):
    assert merge_service is not None
    return await merge_service.get_history("document_types")


@router.get("/empty")
@inject
async def get_empty_document_types(client: FromDishka[PaperlessClient] = None):
    assert client is not None
    doc_types = await client.get_document_types_with_counts()
    empty = [dt for dt in doc_types if dt.get("document_count", 0) == 0]
    return {"count": len(empty), "items": empty}


@router.delete("/empty")
@inject
async def delete_empty_document_types(document_types_service: FromDishka[DocumentTypesService] = None):
    assert document_types_service is not None
    return await document_types_service.delete_empty_document_types()


@router.delete("/{document_type_id}")
@inject
async def delete_document_type_by_id(document_type_id: int, client: FromDishka[PaperlessClient] = None):
    assert client is not None
    try:
        await client.delete_document_type(document_type_id)
        return {"success": True}
    except Exception as e:
        logger.error("Delete document type failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")
