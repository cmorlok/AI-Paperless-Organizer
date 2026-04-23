"""OCR auto-trigger loop (extracted from watchdog_loop)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.ocr.service import OcrService

logger = logging.getLogger(__name__)


async def ocr_auto_trigger_loop(
    ocr_service: "OcrService",
    paperless_client,
) -> None:
    """Continuous background loop to check for new documents.

    Extracted from watchdog_loop (D-13, D-14). Runs as a standalone async function
    receiving the service instance and client as parameters.
    """
    from app.services.ocr.state import (
        batch_state,
        single_ocr_running,
        watchdog_state,
        TAG_OCR_FINISH,
        TAG_OCR_REVIEW,
        TAG_OCR_ERROR,
    )
    from app.services.llm import is_locked as ollama_is_locked, current_holder as ollama_holder

    logger.info("Watchdog started")
    print("[OCR] Watchdog started")

    # Cache tag objects so we don't query Paperless on every cycle
    _ocrfinish_tag = None
    _ocrpruefen_tag = None
    _ocrerror_tag = None

    async def _get_exclude_tag_ids():
        nonlocal _ocrfinish_tag, _ocrpruefen_tag, _ocrerror_tag
        try:
            if _ocrfinish_tag is None:
                _ocrfinish_tag = await paperless_client.get_or_create_tag(TAG_OCR_FINISH)
            if _ocrpruefen_tag is None:
                _ocrpruefen_tag = await paperless_client.get_or_create_tag(TAG_OCR_REVIEW)
            if _ocrerror_tag is None:
                _ocrerror_tag = await paperless_client.get_or_create_tag(TAG_OCR_ERROR)
            return [
                t["id"] for t in [_ocrfinish_tag, _ocrpruefen_tag, _ocrerror_tag]
                if t and t.get("id")
            ]
        except Exception:
            return []

    while watchdog_state["enabled"]:
        try:
            watchdog_state["running"] = True

            if batch_state["running"] or single_ocr_running["running"] or ollama_is_locked():
                reason = "Batch" if batch_state["running"] else "Single-OCR" if single_ocr_running["running"] else f"Ollama belegt ({ollama_holder()})"
                logger.info(f"Watchdog: {reason} aktiv, ueberspringe diesen Zyklus")
            else:
                logger.info("Watchdog checking for new documents...")
                print(f"[OCR] Watchdog check at {datetime.now().isoformat()}")

                # --- Smart pre-check: only start batch when there's something to do ---
                should_run = True
                try:
                    exclude_ids = await _get_exclude_tag_ids()
                    if exclude_ids:
                        pending_count = await paperless_client.get_document_count(
                            tags_id_none=exclude_ids
                        )
                        if pending_count == 0:
                            logger.info("Watchdog: Keine neuen Dokumente – überspringe diesen Zyklus")
                            should_run = False
                        else:
                            logger.info(f"Watchdog: ~{pending_count} Dokument(e) ohne OCR gefunden, starte Batch...")
                except Exception as check_err:
                    logger.warning(f"Watchdog: Pre-Check fehlgeschlagen, starte Batch trotzdem: {check_err}")

                if should_run:
                    await ocr_service.batch_ocr(
                        paperless_client,
                        mode="all",
                        set_finish_tag=True,
                        remove_runocr_tag=True
                    )

            watchdog_state["last_run"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Watchdog error: {e}")
            print(f"[OCR] Watchdog error: {e}")

        # Idle between cycles – not "running" during the wait
        watchdog_state["running"] = False
        interval_min = watchdog_state.get("interval_minutes", 1)
        for _ in range(interval_min * 60):
            if not watchdog_state["enabled"]:
                break
            await asyncio.sleep(1)

    watchdog_state["running"] = False
    logger.info("Watchdog stopped")
    print("[OCR] Watchdog stopped")
