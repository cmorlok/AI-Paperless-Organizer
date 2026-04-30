"""Duplicates Router — thin endpoints only."""

import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.duplicate import DuplicateService, DuplicateScanState
from dishka.integrations.fastapi import inject
from dishka import FromDishka

router = APIRouter()
logger = logging.getLogger(__name__)


class ScanRequest(BaseModel):
    modes: List[str] = ["exact", "similar", "invoices"]
    similarity_threshold: float = 0.92


class IgnoreRequest(BaseModel):
    doc_ids: List[int]


@router.post("/scan")
@inject
async def start_scan(
    body: ScanRequest,
    service: FromDishka[DuplicateService] = None,
    state: FromDishka[DuplicateScanState] = None,
):
    assert service is not None
    assert state is not None
    import asyncio
    if state.running:
        raise HTTPException(status_code=409, detail="Scan läuft bereits")
    asyncio.create_task(service.scan_all(modes=body.modes, similarity_threshold=body.similarity_threshold))
    return {"status": "started", "modes": body.modes}


@router.get("/status")
@inject
async def scan_status(state: FromDishka[DuplicateScanState] = None):
    assert state is not None
    return {
        "running": state.running,
        "phase": state.phase,
        "progress": state.progress,
        "total": state.total,
        "error": state.error,
    }


@router.post("/stop")
@inject
async def stop_scan(state: FromDishka[DuplicateScanState] = None):
    assert state is not None
    if not state.running:
        return {"status": "not_running"}
    state.request_cancel()
    logger.info("Duplicate scan stop requested")
    return {"status": "stopping"}


@router.get("/results")
@inject
async def scan_results(state: FromDishka[DuplicateScanState] = None):
    assert state is not None
    if state.running:
        raise HTTPException(status_code=409, detail="Scan läuft noch")
    return {"groups": state.results}


@router.post("/ignore")
@inject
async def ignore_group(body: IgnoreRequest, service: FromDishka[DuplicateService] = None):
    assert service is not None
    try:
        return await service.ignore_group(body.doc_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/ignored")
@inject
async def list_ignored(service: FromDishka[DuplicateService] = None):
    assert service is not None
    return await service.list_ignored()


@router.delete("/ignore/{doc_id_a}/{doc_id_b}")
@inject
async def remove_ignore(doc_id_a: int, doc_id_b: int, service: FromDishka[DuplicateService] = None):
    assert service is not None
    try:
        await service.remove_ignore(doc_id_a, doc_id_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"removed": True}
