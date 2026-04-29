"""OCR Router - Endpoints for vision OCR."""

import asyncio
import logging
import time
import traceback
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List

from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from dishka.integrations.fastapi import inject
from dishka import FromDishka

from app.services.paperless import PaperlessClient
from app.models.settings_model import LLM_KEY_CLASSIFIER_PROVIDER
from app.services.config import ConfigService
from app.services.ocr import (
    OcrService,
    OcrState,
    OcrCompareState,
    OcrCompareSlot,
    DEFAULT_OCR_MODEL,
    load_review_queue,
    save_review_queue,
    load_ocr_ignore_list,
    save_ocr_ignore_list,
    load_ocr_error_list,
    save_ocr_error_list,
    load_ocr_error_counts,
    save_ocr_error_counts,
)
from app.services.llm import LLMService

logger = logging.getLogger(__name__)

router = APIRouter()


def load_ocr_settings() -> dict:
    """Load OCR settings defaults."""
    return {
        "model": DEFAULT_OCR_MODEL,
        "max_image_size": 1344,
        "smart_skip_enabled": True
    }


# --- Pydantic Models ---

class OcrSettingsRequest(BaseModel):
    model: str = DEFAULT_OCR_MODEL
    max_image_size: int = 1344
    smart_skip_enabled: bool = True


class OcrApplyRequest(BaseModel):
    content: str
    set_finish_tag: bool = True


class BatchOcrRequest(BaseModel):
    mode: str = "all"  # "all", "tagged", "manual"
    document_ids: Optional[List[int]] = None
    set_finish_tag: bool = True
    remove_runocr_tag: bool = True


class OcrCompareRequest(BaseModel):
    document_id: int
    slots: List[OcrCompareSlot]
    page: int = 1  # Which page to compare (1-based, 0 = all pages)


class OcrEvaluateRequest(BaseModel):
    document_title: str
    results: List[dict]  # [{model, text, chars, duration_seconds}]
    evaluation_model: Optional[str] = None  # Override: e.g. "gpt-4.1", "o3", "gpt-4o"


# --- Helper ---

# --- Settings Endpoints ---

@router.get("/settings")
@inject
async def get_ocr_settings(state: FromDishka[OcrState] = None):
    """Get current OCR settings."""
    settings = load_ocr_settings()
    settings["processor_enabled"] = state.processor.enabled
    settings["processor_interval"] = state.processor.interval_minutes
    return settings


@router.post("/settings")
@inject
async def save_ocr_settings_endpoint(request: OcrSettingsRequest, config_svc: FromDishka[ConfigService] = None):
    """Save OCR settings to KV store."""
    await config_svc.set("ocr_model", request.model, "str")
    await config_svc.set("max_image_size", str(request.max_image_size), "int")
    await config_svc.set("smart_skip_enabled", str(request.smart_skip_enabled).lower(), "bool")

    return {"success": True, "model": request.model, "max_image_size": request.max_image_size, "smart_skip_enabled": request.smart_skip_enabled}

# --- Processor Endpoints ---

# Persist enabled flag to AppSettings KV store

async def _persist_ocr_processor_enabled(enabled: bool, config_svc: ConfigService) -> None:
    """Persist ocr_processor_enabled flag to AppSettings."""
    await config_svc.set(
        "ocr_processor_enabled",
        "true" if enabled else "false",
        "bool",
    )


class ProcessorSettingsRequest(BaseModel):
    enabled: bool
    interval_minutes: int = 5

@router.get("/processor/status")
@inject
async def get_processor_status(state: FromDishka[OcrState] = None):
    """Get processor status."""
    return {
        "enabled": state.processor.enabled,
        "running": state.processor.running,
        "interval_minutes": state.processor.interval_minutes,
        "last_run": state.processor.last_run
    }

@router.post("/processor/settings")
@inject
async def set_processor_settings(
    request: ProcessorSettingsRequest,
    background_tasks: BackgroundTasks,
    config_svc: FromDishka[ConfigService] = None,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
    state: FromDishka[OcrState] = None,
):
    """Enable/Disable processor and set interval."""
    state.processor.interval_minutes = max(1, request.interval_minutes)

    # Persist to AppSettings KV store
    try:
        await _persist_ocr_processor_enabled(request.enabled, config_svc)
    except Exception as e:
        logger.warning(f"Could not persist ocr_processor_enabled to KV: {e}")

    if request.enabled and not state.processor.enabled:
        # Start processor
        state.processor.enabled = True
        loop = asyncio.get_running_loop()
        state.processor.task = loop.create_task(service.processor_loop(client))

    elif not request.enabled and state.processor.enabled:
        # Stop processor
        state.processor.enabled = False
        # Task will exit on next loop

    return get_processor_status()


# --- Batch Control Endpoints ---

@router.post("/batch/pause")
@inject
async def pause_batch_ocr(state: FromDishka[OcrState] = None):
    """Pause the running batch OCR job."""
    if not state.batch.running:
        return {"success": False, "message": "Kein Batch-Job aktiv"}
    
    state.batch.paused = True
    return {"success": True, "message": "Batch-Job pausiert", "paused": True}

@router.post("/batch/resume")
@inject
async def resume_batch_ocr(state: FromDishka[OcrState] = None):
    """Resume the paused batch OCR job."""
    if not state.batch.running:
        return {"success": False, "message": "Kein Batch-Job aktiv"}
    
    state.batch.paused = False
    return {"success": True, "message": "Batch-Job fortgesetzt", "paused": False}

# ... (Watchdog auto-start is handled in main.py lifespan)

# --- Tag Management ---

@router.get("/tags/ensure")
@inject
async def ensure_ocr_tags(
    client: FromDishka[PaperlessClient] = None
):
    """Ensure runocr and ocrfinish tags exist in Paperless."""
    try:
        runocr_tag = await client.get_or_create_tag("runocr")
        ocrfinish_tag = await client.get_or_create_tag("ocrfinish")
        return {
            "runocr": {"id": runocr_tag.get("id"), "name": "runocr"},
            "ocrfinish": {"id": ocrfinish_tag.get("id"), "name": "ocrfinish"}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tag-Fehler: {str(e)}")


@router.post("/test-connection")
@inject
async def test_ocr_connection(service: FromDishka[OcrService] = None):
    """Test connection to OCR provider."""
    return await service.test_connection()


@router.get("/stats")
@inject
async def get_ocr_stats(service: FromDishka[OcrService] = None):
    """Get OCR statistics."""
    return service.get_stats()


@router.get("/status")
@inject
async def get_ocr_status(
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    """Get overall OCR status - total docs, finished docs, percentage."""
    try:
        return await service.get_ocr_status(client)
    except Exception as e:
        logger.error(f"Error getting OCR status: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Abrufen des OCR-Status: {str(e)}")


# --- Single Document OCR ---

@router.post("/single/{document_id}")
@inject
async def ocr_single_document(
    document_id: int,
    force: bool = False,
    client: FromDishka[PaperlessClient] = None,
    db: AsyncSession = Depends(get_db),
    service: FromDishka[OcrService] = None,
    state: FromDishka[OcrState] = None,
):
    """Run OCR on a single document with page-level persistence and resume support."""
    try:
        state.acquire_lock("single")
        result = await service.ocr_document(client, document_id, force=force, db_session=db)
        return result
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"OCR ValueError for doc {document_id}: {error_msg}")
        if "nicht gefunden" in error_msg and f"Dokument {document_id}" in error_msg:
            raise HTTPException(status_code=404, detail=error_msg)
        raise HTTPException(status_code=422, detail=f"OCR Verarbeitungsfehler: {error_msg}")
    except Exception as e:
        logger.error(f"OCR single document error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"OCR Fehler: {str(e)}")
    finally:
        state.release_lock()
        state.page_progress.pop(document_id, None)


@router.get("/progress/{document_id}")
@inject
async def get_ocr_progress(document_id: int, state: FromDishka[OcrState] = None):
    """Get live page-level progress for an ongoing OCR job."""
    progress = state.page_progress.get(document_id)
    if not progress:
        return {"active": False, "document_id": document_id}
    elapsed = time.time() - progress.get("started_at", time.time())
    return {
        "active": True,
        "document_id": document_id,
        "status": progress.get("status", "unknown"),
        "total_pages": progress.get("total_pages", 0),
        "done": progress.get("done", 0),
        "errors": progress.get("errors", 0),
        "current_page": progress.get("current_page", 0),
        "elapsed_seconds": round(elapsed, 1),
        "pages": progress.get("pages", []),
    }


@router.post("/apply/{document_id}")
@inject
async def apply_ocr_result(
    document_id: int,
    request: OcrApplyRequest,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    """Apply new OCR content to a document.

    Fires off the Paperless update as async task for instant response.
    The PATCH to Paperless can take 20-30s due to full-text re-indexing,
    so we don't make the user wait.
    """
    logger.info("OCR apply result requested", extra={"document_id": document_id})

    async def _apply_in_background():
        try:
            await service.apply_ocr_result(
                client, document_id, request.content, request.set_finish_tag
            )
            logger.info("OCR result applied successfully", extra={"document_id": document_id})
        except Exception as e:
            logger.error("OCR result apply failed", extra={"document_id": document_id, "error": str(e)})
            logger.error(f"Background apply error: {e}")

    # Fire and forget: don't wait for Paperless re-indexing
    asyncio.create_task(_apply_in_background())
    return {"success": True, "document_id": document_id, "status": "saving"}


# --- Batch OCR ---

@router.post("/batch/start")
@inject
async def start_batch_ocr(
    request: BatchOcrRequest,
    background_tasks: BackgroundTasks,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
    state: FromDishka[OcrState] = None,
):
    """Start batch OCR processing in the background."""
    if state.batch.running:
        raise HTTPException(status_code=409, detail="Ein Batch-OCR-Job läuft bereits")

    # Run batch OCR as background task
    background_tasks.add_task(
        service.batch_ocr,
        client,
        request.mode,
        request.document_ids,
        request.set_finish_tag,
        request.remove_runocr_tag
    )
    
    return {"started": True, "mode": request.mode}


@router.get("/batch/status")
@inject
async def get_batch_status(
    state: FromDishka[OcrState] = None,
    llm_service: FromDishka[LLMService] = None,
):
    """Get current batch OCR job status, including page-level progress for current document."""
    current_doc = state.batch.current_document
    current_doc_id = current_doc.get("id") if isinstance(current_doc, dict) else None

    # Include live page progress for the currently processing document
    page_progress = None
    if current_doc_id and current_doc_id in state.page_progress:
        pp = state.page_progress[current_doc_id]
        page_progress = {
            "document_id": current_doc_id,
            "total_pages": pp.get("total_pages", 0),
            "done": pp.get("done", 0),
            "errors": pp.get("errors", 0),
            "current_page": pp.get("current_page", 0),
            "status": pp.get("status", "unknown"),
            "pages": pp.get("pages", []),
        }

    # Use LLMService lock status instead of direct lock.py imports
    lock_status = llm_service.get_lock_status() if llm_service else {}
    waiting = next((p for p, s in lock_status.items() if s["locked"]), None) if not state.batch.running else None

    return {
        "running": state.batch.running,
        "total": state.batch.total,
        "processed": state.batch.processed,
        "current_document": current_doc,
        "current_page_progress": page_progress,
        "errors_count": len(state.batch.errors),
        "log": state.batch.log[-50:],
        "mode": state.batch.mode,
        "paused": state.batch.paused,
        "waiting_for": waiting,
    }


@router.post("/batch/stop")
@inject
async def stop_batch_ocr(state: FromDishka[OcrState] = None):
    """Stop the running batch OCR job."""
    if not state.batch.running:
        return {"stopped": False, "message": "Kein Batch-Job aktiv"}
    
    state.batch.should_stop = True
    return {"stopped": True, "message": "Batch-Job wird gestoppt..."}


# --- Review Queue ---

@router.get("/review/queue")
async def get_review_queue():
    """Get all documents in the OCR review queue."""
    queue = load_review_queue()
    return {"items": queue, "count": len(queue)}


@router.post("/review/apply/{document_id}")
@inject
async def apply_review_item(
    document_id: int,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    """Apply review queue item (accept the new OCR text)."""
    try:
        return await service.apply_review_item(document_id, client)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/review/dismiss/{document_id}")
async def dismiss_review_item(document_id: int):
    """Dismiss review queue item (discard the new OCR text)."""
    queue = load_review_queue()
    new_queue = [q for q in queue if q["document_id"] != document_id]
    if len(new_queue) == len(queue):
        raise HTTPException(status_code=404, detail="Dokument nicht in Review Queue")
    save_review_queue(new_queue)
    return {"dismissed": True, "document_id": document_id}


@router.post("/review/reset-all")
@inject
async def reset_all_review_items(
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    """Reset all review queue items: remove ocrpruefen tag so batch OCR re-processes them."""
    try:
        return await service.reset_all_review_items(client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tag-Lookup fehlgeschlagen: {e}")


@router.post("/review/keep-all-originals")
@inject
async def keep_all_originals(
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    """Keep all original contents: set ocrfinish on all review items without changing content."""
    try:
        return await service.keep_all_originals(client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tag-Lookup fehlgeschlagen: {e}")


@router.post("/review/ignore/{document_id}")
async def ignore_review_item(document_id: int):
    """Ignore document permanently: remove from review queue and add to OCR ignore list."""
    # Remove from review queue
    queue = load_review_queue()
    item = next((q for q in queue if q["document_id"] == document_id), None)
    title = item["title"] if item else f"Dokument {document_id}"
    new_queue = [q for q in queue if q["document_id"] != document_id]
    save_review_queue(new_queue)
    
    # Add to ignore list (avoid duplicates)
    ignore_list = load_ocr_ignore_list()
    if not any(entry["document_id"] == document_id for entry in ignore_list):
        ignore_list.append({
            "document_id": document_id,
            "title": title,
            "reason": "Original besser als OCR",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        })
        save_ocr_ignore_list(ignore_list)
    
    return {"ignored": True, "document_id": document_id, "title": title}


# --- OCR Ignore List ---

@router.get("/ignore/list")
async def get_ocr_ignore_list():
    """Get all documents on the OCR ignore list."""
    ignore_list = load_ocr_ignore_list()
    return {"items": ignore_list, "count": len(ignore_list)}


@router.post("/ignore/add/{document_id}")
@inject
async def add_to_ocr_ignore_list(
    document_id: int,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    """Add a document to the OCR ignore list."""
    return await service.add_to_ignore_list(document_id, client)


@router.delete("/ignore/remove/{document_id}")
async def remove_from_ocr_ignore_list(document_id: int):
    """Remove a document from the OCR ignore list."""
    ignore_list = load_ocr_ignore_list()
    new_list = [entry for entry in ignore_list if entry["document_id"] != document_id]
    if len(new_list) == len(ignore_list):
        raise HTTPException(status_code=404, detail="Dokument nicht in der Ignore-Liste")
    save_ocr_ignore_list(new_list)
    return {"removed": True, "document_id": document_id}


# --- OCR Error List ---

@router.get("/errors/list")
async def get_ocr_errors():
    """Get all documents on the OCR error list (permanently failed)."""
    error_list = load_ocr_error_list()
    error_counts = load_ocr_error_counts()
    return {"items": error_list, "count": len(error_list), "pending_errors": error_counts}


@router.delete("/errors/remove/{document_id}")
@inject
async def remove_from_ocr_error_list(
    document_id: int,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
):
    """Remove a document from the error list and remove its ocrfehler tag so it can be retried."""
    return await service.remove_from_error_list(document_id, client)


@router.post("/errors/clear")
async def clear_ocr_error_list():
    """Clear the entire error list and error counts."""
    save_ocr_error_list([])
    save_ocr_error_counts({})
    return {"cleared": True}


# --- Document Preview Proxy ---

@router.get("/preview/{document_id}")
@inject
async def get_document_preview(
    document_id: int,
    client: FromDishka[PaperlessClient] = None
):
    """Proxy document preview from Paperless. Auto-detects PDF vs image."""
    try:
        file_bytes = await client.get_document_preview_image(document_id)

        if file_bytes[:4] == b'%PDF':
            media_type = "application/pdf"
        elif file_bytes[:4] == b'\x89PNG':
            media_type = "image/png"
        elif file_bytes[:2] == b'\xff\xd8':
            media_type = "image/jpeg"
        elif file_bytes[:4] == b'RIFF':
            media_type = "image/webp"
        else:
            media_type = "application/pdf"
        return Response(
            content=file_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": "inline",
                "X-Content-Type-Options": "nosniff",
            },
        )
    except Exception as e:
        logger.error(f"Error getting preview for {document_id}: {e}")
        raise HTTPException(status_code=404, detail="Preview not found")


@router.get("/thumbnail/{document_id}")
@inject
async def get_document_thumbnail(
    document_id: int,
    client: FromDishka[PaperlessClient] = None
):
    """Proxy document thumbnail from Paperless (small image, handles auth)."""
    try:
        image_bytes = await client.get_document_thumbnail_bytes(document_id)
        if image_bytes[:4] == b'\x89PNG':
            return Response(content=image_bytes, media_type="image/png")
        return Response(content=image_bytes, media_type="image/webp")
    except Exception as e:
        logger.error(f"Error getting thumbnail for {document_id}: {e}")
        raise HTTPException(status_code=404, detail="Thumbnail not found")


@router.post("/compare")
@inject
async def start_compare(
    request: OcrCompareRequest,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[OcrService] = None,
    compare_state: FromDishka[OcrCompareState] = None,
):
    """Start OCR model comparison as background task."""
    if compare_state.running:
        raise HTTPException(status_code=409, detail="Ein Vergleich läuft bereits")

    slots = request.slots
    if not slots or len(slots) == 0:
        raise HTTPException(status_code=400, detail="Mindestens ein Modell auswählen")
    if len(slots) > 5:
        raise HTTPException(status_code=400, detail="Maximal 5 Modelle gleichzeitig")

    compare_state.reset()
    compare_state.running = True
    compare_state.document_id = request.document_id
    compare_state.models = [s.model for s in slots]
    compare_state.total_models = len(slots)
    compare_state.phase = "starting"

    asyncio.create_task(service.run_compare_job(client, request.document_id, slots, request.page, compare_state))

    return {"started": True, "models": len(slots)}


@router.get("/compare/status")
@inject
async def get_compare_status(
    compare_state: FromDishka[OcrCompareState] = None,
):
    """Get current compare job status (for polling)."""
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
    """Send OCR comparison results to an external LLM for quality evaluation.

    WARNING: This sends document text to a cloud API (OpenAI, Anthropic, etc.)!
    """
    eval_provider = await config_svc.get(LLM_KEY_CLASSIFIER_PROVIDER)
    if not eval_provider:
        raise HTTPException(
            status_code=400,
            detail="Kein LLM-Provider konfiguriert. Bitte zuerst unter Einstellungen einen Provider (z.B. OpenAI) einrichten."
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
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM-Auswertung fehlgeschlagen: {str(e)}")

