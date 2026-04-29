import asyncio
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, and_

from app.database import get_db
from app.models.duplicates import DuplicateIgnore
from app.services.duplicate import DuplicateService, DuplicateScanState
from dishka.integrations.fastapi import inject
from dishka import FromDishka

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    modes: List[str] = ["exact", "similar", "invoices"]
    similarity_threshold: float = 0.92


class IgnoreRequest(BaseModel):
    doc_ids: List[int]


# ── Scan endpoints ───────────────────────────────────────────────────────────

@router.post("/scan")
@inject
async def start_scan(
    body: ScanRequest,
    service: FromDishka[DuplicateService] = None,
    state: FromDishka[DuplicateScanState] = None,
):
    assert service is not None
    assert state is not None
    """Startet einen Duplikat-Scan im Hintergrund."""
    if state.running:
        raise HTTPException(status_code=409, detail="Scan läuft bereits")

    asyncio.create_task(
        service.scan_all(
            modes=body.modes,
            similarity_threshold=body.similarity_threshold,
        )
    )

    return {"status": "started", "modes": body.modes}


@router.get("/status")
@inject
async def scan_status(state: FromDishka[DuplicateScanState] = None):
    assert state is not None
    """Polling-Endpoint für den Scan-Fortschritt."""
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
    """Stoppt den laufenden Scan."""
    if not state.running:
        return {"status": "not_running"}
    state.request_cancel()
    logger.info("Duplicate scan stop requested")
    return {"status": "stopping"}


@router.get("/results")
@inject
async def scan_results(state: FromDishka[DuplicateScanState] = None):
    assert state is not None
    """Gibt die Ergebnis-Gruppen des letzten Scans zurück."""
    if state.running:
        raise HTTPException(status_code=409, detail="Scan läuft noch")
    return {"groups": state.results}


# ── Ignore-Liste ─────────────────────────────────────────────────────────────

@router.post("/ignore")
async def ignore_group(body: IgnoreRequest, db: AsyncSession = Depends(get_db)):
    """Markiert eine Gruppe von Dokumenten als 'kein Duplikat'."""
    if len(body.doc_ids) < 2:
        raise HTTPException(status_code=400, detail="Mindestens 2 Dokument-IDs erforderlich")

    # Alle Paare speichern (sortiert, um Duplikate zu vermeiden)
    added = 0
    for i in range(len(body.doc_ids)):
        for j in range(i + 1, len(body.doc_ids)):
            a, b = sorted([body.doc_ids[i], body.doc_ids[j]])
            # Prüfen ob Paar schon existiert
            existing = await db.execute(
                select(DuplicateIgnore).where(
                    and_(
                        DuplicateIgnore.doc_id_a == a,
                        DuplicateIgnore.doc_id_b == b,
                    )
                )
            )
            if existing.scalar_one_or_none() is None:
                db.add(DuplicateIgnore(doc_id_a=a, doc_id_b=b))
                added += 1

    await db.commit()
    logger.info("Added %d ignore pair(s) for doc_ids=%s", added, body.doc_ids)
    return {"added": added}


@router.get("/ignored")
async def list_ignored(db: AsyncSession = Depends(get_db)):
    """Gibt alle ignorierten Paare zurück."""
    result = await db.execute(select(DuplicateIgnore).order_by(DuplicateIgnore.created_at.desc()))
    rows = result.scalars().all()
    return [
        {
            "id": row.id,
            "doc_id_a": row.doc_id_a,
            "doc_id_b": row.doc_id_b,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


@router.delete("/ignore/{doc_id_a}/{doc_id_b}")
async def remove_ignore(doc_id_a: int, doc_id_b: int, db: AsyncSession = Depends(get_db)):
    """Hebt die Ignorierung eines Paares auf."""
    a, b = sorted([doc_id_a, doc_id_b])
    result = await db.execute(
        delete(DuplicateIgnore).where(
            and_(
                DuplicateIgnore.doc_id_a == a,
                DuplicateIgnore.doc_id_b == b,
            )
        )
    )
    await db.commit()

    if getattr(result, "rowcount", 0) == 0:
        raise HTTPException(status_code=404, detail="Paar nicht gefunden")

    logger.info("Removed ignore pair (%d, %d)", a, b)
    return {"removed": True}
