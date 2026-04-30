"""Tags Router — thin endpoints only: validate input, call service, return response."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.paperless import PaperlessClient
from app.services.similarity import SimilarityService
from app.services.merge import MergeService
from app.services.tags import TagsService
from dishka.integrations.fastapi import inject
from dishka import FromDishka

router = APIRouter()


class MergeRequest(BaseModel):
    target_id: int
    target_name: str
    source_ids: List[int]


class AnalyzeRequest(BaseModel):
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
    return await client.get_tags_with_counts()


@router.get("/estimate")
@inject
async def estimate_tags(tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    return await tags_service.estimate_tags("nonsense")


# ============ SAVED ANALYSIS CRUD ============

@router.get("/saved-analysis")
@inject
async def get_saved_analysis(tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    return await tags_service.get_saved_analysis("tags")


@router.get("/saved-analysis/load")
@inject
async def load_saved_analysis(tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    result = await tags_service.load_saved_analysis("tags")
    if not result:
        raise HTTPException(status_code=404, detail="Keine gespeicherte Analyse gefunden")
    return result


@router.delete("/saved-analysis")
@inject
async def delete_saved_analysis(tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    await tags_service.delete_saved_analysis("tags")
    return {"success": True}


@router.post("/saved-analysis/mark-processed")
@inject
async def mark_group_processed(group_index: int, tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    await tags_service.mark_group_processed("tags", group_index)
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
    batch_size = request.batch_size if request else 200
    result = await similarity_service.find_similar_tags(batch_size=batch_size)
    await tags_service.save_similarity_analysis(result)
    return result


# ============ MERGE ============

@router.post("/merge")
@inject
async def merge_tags(request: MergeRequest, merge_service: FromDishka[MergeService] = None):
    assert merge_service is not None
    return await merge_service.merge_tags(
        target_id=request.target_id,
        target_name=request.target_name,
        source_ids=request.source_ids,
    )


@router.get("/history")
@inject
async def get_merge_history(merge_service: FromDishka[MergeService] = None):
    assert merge_service is not None
    return await merge_service.get_history("tags")


# ============ EMPTY TAGS ============

@router.get("/empty")
@inject
async def get_empty_tags(client: FromDishka[PaperlessClient] = None):
    assert client is not None
    tags = await client.get_tags_with_counts()
    empty = [t for t in tags if t.get("document_count", 0) == 0]
    return {"count": len(empty), "items": empty}


@router.delete("/empty")
@inject
async def delete_empty_tags(tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    return await tags_service.delete_empty_tags()


# ============ BULK DELETE ============

@router.post("/bulk-delete")
@inject
async def bulk_delete_tags(request: BulkDeleteRequest, tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    try:
        return await tags_service.bulk_delete_tags(request.tag_ids)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============ REMOVE FROM SAVED ANALYSES ============

@router.post("/saved-analyses/remove-tags")
@inject
async def remove_tags_from_saved_analyses(
    request: RemoveFromAnalysesRequest, tags_service: FromDishka[TagsService] = None,
):
    assert tags_service is not None
    return await tags_service.remove_tags_from_saved_analyses(request.tag_ids)


# ============ SINGLE DELETE ============

@router.delete("/{tag_id}")
@inject
async def delete_tag(tag_id: int, client: FromDishka[PaperlessClient] = None):
    assert client is not None
    try:
        await client.delete_tag(tag_id)
        return {"success": True, "message": f"Tag {tag_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============ NONSENSE TAGS ============

@router.get("/saved-nonsense")
@inject
async def get_saved_nonsense_analysis(tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    return await tags_service.get_saved_nonsense_analysis()


@router.get("/saved-nonsense/load")
@inject
async def load_saved_nonsense_analysis(tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    return await tags_service.load_saved_nonsense_analysis()


@router.delete("/saved-nonsense")
@inject
async def delete_saved_nonsense_analysis(tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    await tags_service.delete_saved_nonsense_analysis()
    return {"success": True}


@router.post("/analyze-nonsense")
@inject
async def analyze_nonsense_tags(
    similarity_service: FromDishka[SimilarityService] = None,
    tags_service: FromDishka[TagsService] = None,
):
    assert similarity_service is not None
    assert tags_service is not None
    result = await similarity_service.find_nonsense_tags()
    await tags_service.save_nonsense_analysis(result)
    return result


# ============ CORRESPONDENT TAGS ============

@router.get("/saved-correspondent-matches")
@inject
async def get_saved_correspondent_analysis(tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    return await tags_service.get_saved_correspondent_matches()


@router.get("/saved-correspondent-matches/load")
@inject
async def load_saved_correspondent_analysis(tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    return await tags_service.load_saved_correspondent_matches()


@router.delete("/saved-correspondent-matches")
@inject
async def delete_saved_correspondent_analysis(tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    await tags_service.delete_saved_correspondent_matches()
    return {"success": True}


@router.post("/analyze-correspondent-matches")
@inject
async def analyze_correspondent_matches(
    similarity_service: FromDishka[SimilarityService] = None,
    tags_service: FromDishka[TagsService] = None,
):
    assert similarity_service is not None
    assert tags_service is not None
    result = await similarity_service.find_tags_that_are_correspondents()
    await tags_service.save_correspondent_matches(result)
    return result


# ============ DOCTYPE TAGS ============

@router.get("/saved-doctype-matches")
@inject
async def get_saved_doctype_analysis(tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    return await tags_service.get_saved_doctype_matches()


@router.get("/saved-doctype-matches/load")
@inject
async def load_saved_doctype_analysis(tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    return await tags_service.load_saved_doctype_matches()


@router.delete("/saved-doctype-matches")
@inject
async def delete_saved_doctype_analysis(tags_service: FromDishka[TagsService] = None):
    assert tags_service is not None
    await tags_service.delete_saved_doctype_matches()
    return {"success": True}


@router.post("/analyze-doctype-matches")
@inject
async def analyze_doctype_matches(
    similarity_service: FromDishka[SimilarityService] = None,
    tags_service: FromDishka[TagsService] = None,
):
    assert similarity_service is not None
    assert tags_service is not None
    result = await similarity_service.find_tags_that_are_document_types()
    await tags_service.save_doctype_matches(result)
    return result
