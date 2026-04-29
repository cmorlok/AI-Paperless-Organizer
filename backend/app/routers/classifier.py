"""API Router for the KI-Klassifizierer feature."""
import asyncio
import logging
from fastapi import APIRouter, HTTPException
from typing import List
from dataclasses import asdict
from dishka.integrations.fastapi import inject
from dishka import FromDishka
from app.container import container as di_container
from app.services.paperless import PaperlessClient
from app.services.classifier import DocumentClassifierService, AutoClassifyState, auto_classify_loop
from app.models.settings_model import LLM_KEY_CLASSIFIER_MODEL
from app.services.config import ConfigService
from app.services.llm import LLMService
from app.routers.classifier_schemas import ClassifierConfigUpdate, StoragePathProfileUpdate, CustomFieldMappingUpdate, ApplyRequest, BenchmarkRequest, TagIdeaApproveRequest

logger = logging.getLogger(__name__)
router = APIRouter()

async def _persist_auto_classify_enabled(enabled: bool, config_svc: ConfigService) -> None:
    await config_svc.set("auto_classify_enabled", "true" if enabled else "false", "bool")

@router.get("/config")
@inject
async def get_config(service: FromDishka[DocumentClassifierService] = None, config_svc: FromDishka[ConfigService] = None):
    return await service.get_config_response(await config_svc.get("classifier_provider") or "", await config_svc.get(LLM_KEY_CLASSIFIER_MODEL) or "")

@router.put("/config")
@inject
async def update_config(data: ClassifierConfigUpdate, service: FromDishka[DocumentClassifierService] = None):
    update = {k: v for k, v in data.model_dump().items() if v is not None}
    config = await service.save_config(update)
    return {"status": "ok", "active_provider": config.active_provider}

@router.get("/prompt-defaults")
async def get_prompt_defaults():
    from app.services.classifier import FIELD_DEFAULTS
    return FIELD_DEFAULTS

@router.get("/stats")
@inject
async def get_classifier_stats(service: FromDishka[DocumentClassifierService] = None, client: FromDishka[PaperlessClient] = None):
    return await service.get_stats(client)

@router.get("/next-unclassified")
@inject
async def get_next_unclassified(after_id: int = 0, service: FromDishka[DocumentClassifierService] = None, client: FromDishka[PaperlessClient] = None):
    return await service.get_next_unclassified(after_id, client)

@router.post("/refresh-cache")
@inject
async def refresh_paperless_cache(client: FromDishka[PaperlessClient] = None):
    from app.services.cache import get_cache
    await get_cache().clear("paperless:")
    tags, correspondents, doc_types, paths = await asyncio.gather(
        client.get_tags(use_cache=False), client.get_correspondents(use_cache=False),
        client.get_document_types(use_cache=False), client.get_storage_paths(use_cache=False),
    )
    return {"refreshed": True, "tags": len(tags), "correspondents": len(correspondents), "document_types": len(doc_types), "storage_paths": len(paths)}

@router.get("/tags")
@inject
async def get_tags(client: FromDishka[PaperlessClient] = None):
    return await client.get_tags(use_cache=False)

@router.get("/correspondents")
@inject
async def get_correspondents(client: FromDishka[PaperlessClient] = None):
    return await client.get_correspondents(use_cache=False)

@router.get("/document-types")
@inject
async def get_document_types(client: FromDishka[PaperlessClient] = None):
    return await client.get_document_types(use_cache=False)

@router.get("/storage-paths")
@inject
async def get_storage_paths(client: FromDishka[PaperlessClient] = None):
    return await client.get_storage_paths(use_cache=False)

@router.get("/storage-path-profiles")
@inject
async def get_storage_path_profiles(service: FromDishka[DocumentClassifierService] = None, client: FromDishka[PaperlessClient] = None):
    return await service.get_storage_path_profiles_merged(client)

@router.put("/storage-path-profiles")
@inject
async def save_storage_path_profiles(profiles: List[StoragePathProfileUpdate], service: FromDishka[DocumentClassifierService] = None):
    saved = [await service.save_storage_profile(p.model_dump()) for p in profiles]
    return {"status": "ok", "saved_count": len(saved)}

@router.get("/custom-fields")
@inject
async def get_custom_fields(client: FromDishka[PaperlessClient] = None):
    return await client.get_custom_fields(use_cache=False)

@router.get("/custom-field-mappings")
@inject
async def get_custom_field_mappings(service: FromDishka[DocumentClassifierService] = None, client: FromDishka[PaperlessClient] = None):
    return await service.get_custom_field_mappings_merged(client)

@router.put("/custom-field-mappings")
@inject
async def save_custom_field_mappings(mappings: List[CustomFieldMappingUpdate], service: FromDishka[DocumentClassifierService] = None):
    saved = [await service.save_custom_field_mapping(m.model_dump()) for m in mappings]
    return {"status": "ok", "saved_count": len(saved)}

@router.get("/document/{document_id}/thumb")
@inject
async def get_document_thumbnail(document_id: int, client: FromDishka[PaperlessClient] = None):
    from fastapi.responses import Response
    try:
        return Response(content=await client.get_document_thumbnail_bytes(document_id), media_type="image/webp")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Thumbnail not available: {e}")

@router.get("/document/{document_id}/preview")
@inject
async def get_document_preview(document_id: int, client: FromDishka[PaperlessClient] = None):
    from fastapi.responses import Response
    try:
        pdf_bytes = await client.get_document_preview_image(document_id)
        mt = {b'\x89PNG': "image/png", b'\xff\xd8': "image/jpeg", b'RIFF': "image/webp"}.get(pdf_bytes[:4], "application/pdf")
        return Response(content=pdf_bytes, media_type=mt, headers={"Content-Disposition": "inline", "X-Content-Type-Options": "nosniff"})
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Preview not available: {e}")

@router.post("/analyze")
@inject
async def analyze_document(document_id: int, service: FromDishka[DocumentClassifierService] = None):
    try:
        result = await service.classify_document(document_id)
        if result.error:
            raise HTTPException(status_code=500, detail=result.error)
        return asdict(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Analyze failed for document_id=%s: %s", document_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/benchmark")
@inject
async def benchmark_document(req: BenchmarkRequest, service: FromDishka[DocumentClassifierService] = None):
    return await service.benchmark_document(req.document_id, [(s.provider, s.model or None) for s in req.slots])

@router.post("/apply")
@inject
async def apply_classification(req: ApplyRequest, service: FromDishka[DocumentClassifierService] = None):
    return await service.apply_classification(req.document_id, req.classification)

@router.get("/history")
@inject
async def get_history(limit: int = 50, service: FromDishka[DocumentClassifierService] = None):
    return await service.get_history(limit)

@router.get("/tag-stats")
@inject
async def get_tag_stats(service: FromDishka[DocumentClassifierService] = None):
    return await service.get_tag_stats()

@router.post("/auto-classify/start")
@inject
async def start_auto_classify(state: FromDishka[AutoClassifyState] = None, config_svc: FromDishka[ConfigService] = None):
    if state.enabled:
        return {"status": "already_running"}
    state.enabled, state.processed, state.errors, state.reviewed = True, 0, 0, 0
    state._task = asyncio.create_task(auto_classify_loop(di_container))
    try:
        await _persist_auto_classify_enabled(True, config_svc)
    except Exception:
        pass
    return {"status": "started"}

@router.post("/auto-classify/stop")
@inject
async def stop_auto_classify(state: FromDishka[AutoClassifyState] = None, config_svc: FromDishka[ConfigService] = None):
    state.enabled = False
    if state._task and not state._task.done():
        state._task.cancel()
    state.running, state.current_doc = False, None
    try:
        await _persist_auto_classify_enabled(False, config_svc)
    except Exception:
        pass
    return {"status": "stopped"}

@router.get("/auto-classify/status")
@inject
async def get_auto_classify_status(state: FromDishka[AutoClassifyState] = None, llm_service: FromDishka[LLMService] = None):
    lock = llm_service.get_lock_status() if llm_service else {}
    waiting = next((p for p, s in lock.items() if s["locked"] and p != "classifier"), None) if state.enabled else None
    return {"enabled": state.enabled, "running": state.running, "processed": state.processed, "errors": state.errors, "reviewed": state.reviewed, "current_doc": state.current_doc, "last_run": state.last_run, "waiting_for": waiting}

@router.get("/review-queue")
@inject
async def get_review_queue(service: FromDishka[DocumentClassifierService] = None):
    return await service.get_review_queue()

@router.post("/review-queue/{entry_id}/approve")
@inject
async def approve_review_entry(entry_id: int, req: ApplyRequest, service: FromDishka[DocumentClassifierService] = None):
    try:
        return await service.approve_review(entry_id, req.document_id, req.classification)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/review-queue/{entry_id}/dismiss")
@inject
async def dismiss_review_entry(entry_id: int, service: FromDishka[DocumentClassifierService] = None):
    try:
        return await service.dismiss_review(entry_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/tag-ideas")
@inject
async def get_tag_ideas(service: FromDishka[DocumentClassifierService] = None):
    return await service.get_tag_ideas()

@router.get("/tag-ideas/stats")
@inject
async def get_tag_ideas_stats(service: FromDishka[DocumentClassifierService] = None):
    return await service.get_tag_ideas_stats()

@router.post("/tag-ideas/{entry_id}/approve")
@inject
async def approve_tag_idea(entry_id: int, req: TagIdeaApproveRequest, service: FromDishka[DocumentClassifierService] = None, client: FromDishka[PaperlessClient] = None):
    try:
        return await service.approve_tag_idea(entry_id, req.tag_name, client)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/tag-ideas/{entry_id}/dismiss")
@inject
async def dismiss_tag_idea(entry_id: int, req: TagIdeaApproveRequest, service: FromDishka[DocumentClassifierService] = None):
    try:
        return await service.dismiss_tag_idea(entry_id, req.tag_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/tag-ideas/{entry_id}/approve-all")
@inject
async def approve_all_tag_ideas(entry_id: int, service: FromDishka[DocumentClassifierService] = None, client: FromDishka[PaperlessClient] = None):
    try:
        return await service.approve_all_tag_ideas(entry_id, client)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/tag-ideas/bulk-approve")
@inject
async def bulk_approve_tag_idea(req: TagIdeaApproveRequest, service: FromDishka[DocumentClassifierService] = None, client: FromDishka[PaperlessClient] = None):
    try:
        return await service.bulk_approve_tag_idea(req.tag_name, client)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tag-ideas/bulk-dismiss")
@inject
async def bulk_dismiss_tag_idea(req: TagIdeaApproveRequest, service: FromDishka[DocumentClassifierService] = None):
    return await service.bulk_dismiss_tag_idea(req.tag_name)

@router.post("/tag-ideas/{entry_id}/assign-existing")
@inject
async def assign_existing_tag(entry_id: int, req: TagIdeaApproveRequest, service: FromDishka[DocumentClassifierService] = None, client: FromDishka[PaperlessClient] = None):
    try:
        return await service.assign_existing_tag(entry_id, req.tag_name, client)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
