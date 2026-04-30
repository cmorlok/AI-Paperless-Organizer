"""OCR Router - Thin endpoints for vision OCR."""

import logging
import asyncio
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List

from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from dishka.integrations.fastapi import inject
from dishka import FromDishka

from app.services.paperless import PaperlessClient
from app.services.config import ConfigService
from app.services.ocr import (
    OcrService,
    OcrCompareState,
    OcrCompareSlot,
    DEFAULT_OCR_MODEL,
    load_review_queue,
    load_ocr_ignore_list,
)
from app.services.llm import LLMService

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Pydantic Models ---

class OcrSettingsRequest(BaseModel):
    model: str = DEFAULT_OCR_MODEL
    max_image_size: int = 1344
    smart_skip_enabled: bool = True


class OcrApplyRequest(BaseModel):
    content: str
    set_finish_tag: bool = True


class BatchOcrRequest(BaseModel):
    mode: str = "all"
    document_ids: Optional[List[int]] = None
    set_finish_tag: bool = True
    remove_runocr_tag: bool = True


class OcrCompareRequest(BaseModel):
    document_id: int
    slots: List[OcrCompareSlot]
    page: int = 1


class OcrEvaluateRequest(BaseModel):
    document_title: str
    results: List[dict]
    evaluation_model: Optional[str] = None


class ProcessorSettingsRequest(BaseModel):
    enabled: bool
    interval_minutes: int = 5


# --- Settings Endpoints ---

@router.get("/settings")
@inject
async def get_ocr_settings(service: FromDishka[OcrService] = None):
    assert service is not None
    return service.get_ocr_settings_with_state()


@router.post("/settings")
@inject
async def save_ocr_settings_endpoint(
    request: OcrSettingsRequest,
    service: FromDishka[OcrService] = None,
    config_svc: FromDishka[ConfigService] = None,
):
    assert service is not None
    assert config_svc is not None
    return await service.save_ocr_settings(
        request.model, request.max_image_size, request.smart_skip_enabled, config_svc,
    )


# --- Processor Endpoints ---

@router.get("/processor/status")
@inject
async def get_processor_status(service: FromDishka[OcrService] = None):
    assert service is not None
    return service.get_processor_status_dict()


@router.post("/processor/settings")
@inject
async def set_processor_settings(
    request: ProcessorSettingsRequest,
    config_svc: FromDishka[ConfigService] = None,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    assert config_svc is not None
    assert client is not None
    assert service is not None
    return await service.configure_processor(
        request.enabled, request.interval_minutes, config_svc, client,
    )


# --- Batch Control Endpoints ---

@router.post("/batch/pause")
@inject
async def pause_batch_ocr(service: FromDishka[OcrService] = None):
    assert service is not None
    return service.pause_batch()


@router.post("/batch/resume")
@inject
async def resume_batch_ocr(service: FromDishka[OcrService] = None):
    assert service is not None
    return service.resume_batch()


# --- Tag Management ---

@router.get("/tags/ensure")
@inject
async def ensure_ocr_tags(client: FromDishka[PaperlessClient] = None):
    assert client is not None
    try:
        runocr_tag = await client.get_or_create_tag("runocr")
        ocrfinish_tag = await client.get_or_create_tag("ocrfinish")
        return {
            "runocr": {"id": runocr_tag.get("id"), "name": "runocr"},
            "ocrfinish": {"id": ocrfinish_tag.get("id"), "name": "ocrfinish"},
        }
    except Exception as e:
        logger.error("Tag-Fehler: %s", e)
        raise HTTPException(status_code=500, detail="Interner Serverfehler")


@router.post("/test-connection")
@inject
async def test_ocr_connection(service: FromDishka[OcrService] = None):
    assert service is not None
    return await service.test_connection()


@router.get("/stats")
@inject
async def get_ocr_stats(service: FromDishka[OcrService] = None):
    assert service is not None
    return service.get_stats()


@router.get("/status")
@inject
async def get_ocr_status(
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    assert client is not None
    assert service is not None
    try:
        return await service.get_ocr_status(client)
    except Exception as e:
        logger.error("OCR-Status-Fehler: %s", e)
        raise HTTPException(status_code=500, detail="Interner Serverfehler")


# --- Single Document OCR ---

@router.post("/single/{document_id}")
@inject
async def ocr_single_document(
    document_id: int,
    force: bool = False,
    client: FromDishka[PaperlessClient] = None,
    db: AsyncSession = Depends(get_db),
    service: FromDishka[OcrService] = None,
):
    assert client is not None
    assert service is not None
    try:
        return await service.ocr_single_document_safe(client, document_id, force, db)
    except ValueError as e:
        error_msg = str(e)
        if "nicht gefunden" in error_msg and f"Dokument {document_id}" in error_msg:
            raise HTTPException(status_code=404, detail=error_msg)
        raise HTTPException(status_code=422, detail=f"OCR Verarbeitungsfehler: {error_msg}")
    except Exception as e:
        logger.error("OCR-Fehler: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Interner Serverfehler")


@router.get("/progress/{document_id}")
@inject
async def get_ocr_progress(document_id: int, service: FromDishka[OcrService] = None):
    assert service is not None
    return service.get_progress_dict(document_id)


@router.post("/apply/{document_id}")
@inject
async def apply_ocr_result(
    document_id: int,
    request: OcrApplyRequest,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    assert client is not None
    assert service is not None
    logger.info("OCR apply result requested", extra={"document_id": document_id})
    asyncio.create_task(
        service.apply_ocr_result_background(client, document_id, request.content, request.set_finish_tag)
    )
    return {"success": True, "document_id": document_id, "status": "saving"}


# --- Batch OCR ---

@router.post("/batch/start")
@inject
async def start_batch_ocr(
    request: BatchOcrRequest,
    background_tasks: BackgroundTasks,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    assert client is not None
    assert service is not None
    try:
        service.check_batch_not_running()
    except ValueError:
        raise HTTPException(status_code=409, detail="Ein Batch-OCR-Job läuft bereits")

    background_tasks.add_task(
        service.batch_ocr,
        client,
        request.mode,
        request.document_ids or [],
        request.set_finish_tag,
        request.remove_runocr_tag,
    )
    return {"started": True, "mode": request.mode}


@router.get("/batch/status")
@inject
async def get_batch_status(
    service: FromDishka[OcrService] = None,
    llm_service: FromDishka[LLMService] = None,
):
    assert service is not None
    assert llm_service is not None
    return service.get_batch_status_dict(llm_service)


@router.post("/batch/stop")
@inject
async def stop_batch_ocr(service: FromDishka[OcrService] = None):
    assert service is not None
    return service.stop_batch()


# --- Review Queue ---

@router.get("/review/queue")
async def get_review_queue():
    queue = load_review_queue()
    return {"items": queue, "count": len(queue)}


@router.post("/review/apply/{document_id}")
@inject
async def apply_review_item(
    document_id: int,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    assert client is not None
    assert service is not None
    try:
        return await service.apply_review_item(document_id, client)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Review-Apply-Fehler: %s", e)
        raise HTTPException(status_code=500, detail="Interner Serverfehler")


@router.post("/review/dismiss/{document_id}")
@inject
async def dismiss_review_item(document_id: int, service: FromDishka[OcrService] = None):
    assert service is not None
    try:
        return service.dismiss_review_item_from_queue(document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/review/reset-all")
@inject
async def reset_all_review_items(
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    assert client is not None
    assert service is not None
    try:
        return await service.reset_all_review_items(client)
    except Exception as e:
        logger.error("Tag-Lookup fehlgeschlagen: %s", e)
        raise HTTPException(status_code=500, detail="Interner Serverfehler")


@router.post("/review/keep-all-originals")
@inject
async def keep_all_originals(
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    assert client is not None
    assert service is not None
    try:
        return await service.keep_all_originals(client)
    except Exception as e:
        logger.error("Tag-Lookup fehlgeschlagen: %s", e)
        raise HTTPException(status_code=500, detail="Interner Serverfehler")


@router.post("/review/ignore/{document_id}")
@inject
async def ignore_review_item(document_id: int, service: FromDishka[OcrService] = None):
    assert service is not None
    return service.ignore_review_item_permanently(document_id)


# --- OCR Ignore List ---

@router.get("/ignore/list")
async def get_ocr_ignore_list():
    ignore_list = load_ocr_ignore_list()
    return {"items": ignore_list, "count": len(ignore_list)}


@router.post("/ignore/add/{document_id}")
@inject
async def add_to_ocr_ignore_list(
    document_id: int,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    assert client is not None
    assert service is not None
    return await service.add_to_ignore_list(document_id, client)


@router.delete("/ignore/remove/{document_id}")
@inject
async def remove_from_ocr_ignore_list(document_id: int, service: FromDishka[OcrService] = None):
    assert service is not None
    try:
        service.remove_from_ocr_ignore_list(document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"removed": True, "document_id": document_id}


# --- OCR Error List ---

@router.get("/errors/list")
@inject
async def get_ocr_errors(service: FromDishka[OcrService] = None):
    assert service is not None
    return service.get_error_list_with_counts()


@router.delete("/errors/remove/{document_id}")
@inject
async def remove_from_ocr_error_list(
    document_id: int,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    assert client is not None
    assert service is not None
    return await service.remove_from_error_list(document_id, client)


@router.post("/errors/clear")
@inject
async def clear_ocr_error_list(service: FromDishka[OcrService] = None):
    assert service is not None
    return service.clear_all_errors()


# --- Document Preview Proxy ---

@router.get("/preview/{document_id}")
@inject
async def get_document_preview(
    document_id: int,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    assert client is not None
    assert service is not None
    try:
        return await service.get_preview_response(client, document_id)
    except Exception as e:
        logger.error(f"Error getting preview for {document_id}: {e}")
        raise HTTPException(status_code=404, detail="Preview not found")


@router.get("/thumbnail/{document_id}")
@inject
async def get_document_thumbnail(
    document_id: int,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    assert client is not None
    assert service is not None
    try:
        return await service.get_thumbnail_response(client, document_id)
    except Exception as e:
        logger.error(f"Error getting thumbnail for {document_id}: {e}")
        raise HTTPException(status_code=404, detail="Thumbnail not found")


# --- Compare ---

@router.post("/compare")
@inject
async def start_compare(
    request: OcrCompareRequest,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
    compare_state: FromDishka[OcrCompareState] = None,
):
    assert client is not None
    assert service is not None
    assert compare_state is not None
    try:
        return await service.validate_and_start_compare(
            client, request.document_id, request.slots, request.page, compare_state,
        )
    except ValueError as e:
        msg = str(e)
        if msg == "already_running":
            raise HTTPException(status_code=409, detail="Ein Vergleich läuft bereits")
        raise HTTPException(status_code=400, detail=msg)


@router.get("/compare/status")
@inject
async def get_compare_status(compare_state: FromDishka[OcrCompareState] = None):
    assert compare_state is not None
    return {
        "running": compare_state.running,
        "phase": compare_state.phase,
        "current_model": compare_state.current_model,
        "current_model_index": compare_state.current_model_index,
        "total_models": compare_state.total_models,
        "current_page": compare_state.current_page,
        "total_pages": compare_state.total_pages,
        "models": compare_state.models,
        "document_id": compare_state.document_id,
        "title": compare_state.title,
        "old_content": compare_state.old_content,
        "compared_page": compare_state.compared_page,
        "results": compare_state.results,
        "error": compare_state.error,
        "elapsed_seconds": compare_state.elapsed_seconds,
    }


@router.post("/compare/evaluate")
@inject
async def evaluate_ocr_results(
    request: OcrEvaluateRequest,
    llm_service: FromDishka[LLMService] = None,
    config_svc: FromDishka[ConfigService] = None,
    service: FromDishka[OcrService] = None,
):
    assert llm_service is not None
    assert config_svc is not None
    assert service is not None
    from app.models.settings_model import LLM_KEY_CLASSIFIER_PROVIDER
    eval_provider = await config_svc.get(LLM_KEY_CLASSIFIER_PROVIDER)
    if not eval_provider:
        raise HTTPException(
            status_code=400,
            detail="Kein LLM-Provider konfiguriert. Bitte zuerst unter Einstellungen einen Provider einrichten.",
        )
    if not request.results or len(request.results) < 1:
        raise HTTPException(status_code=400, detail="Keine OCR-Ergebnisse zum Auswerten")
    try:
        return await service.evaluate_ocr_results(
            document_title=request.document_title,
            results=request.results,
            eval_provider=eval_provider,
            eval_model=request.evaluation_model,
        )
    except Exception as e:
        logger.error("LLM-Auswertung fehlgeschlagen: %s", e)
        raise HTTPException(status_code=500, detail="Interner Serverfehler")
