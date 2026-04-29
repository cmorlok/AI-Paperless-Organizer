import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.database import get_db
from app.models.cloud_import import CloudSource, CloudImportLog
from app.services.cloud_import import CloudImportService, CloudSyncState, cloud_sync_loop, RcloneOAuthService
from app.container import container as di_container
from dishka.integrations.fastapi import inject
from dishka import FromDishka
from app.services.paperless import PaperlessClient
from app.services.config import ConfigService

router = APIRouter()
logger = logging.getLogger(__name__)

# Singleton rclone OAuth service (module-level, same lifetime as before)
_rclone_oauth = RcloneOAuthService()


# Persist enabled flag to AppSettings KV store (STATE-08)

async def _persist_cloud_sync_enabled(enabled: bool, config_svc: ConfigService) -> None:
    """Persist cloud_sync_enabled flag to AppSettings."""
    await config_svc.set(
        "cloud_sync_enabled",
        "true" if enabled else "false",
        "bool",
    )


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class CloudSourceCreate(BaseModel):
    name: str
    source_type: str = "webdav"  # webdav, rclone, local
    enabled: bool = True
    poll_interval_minutes: int = 5

    # WebDAV
    webdav_url: str = ""
    webdav_username: str = ""
    webdav_password: str = ""
    webdav_path: str = "/"

    # rclone
    rclone_remote: str = ""
    rclone_path: str = "/"
    rclone_config: str = ""

    # Local
    local_path: str = ""

    # Import settings
    filename_prefix: str = ""
    paperless_tag_ids: str = "[]"
    paperless_correspondent_id: Optional[int] = None
    paperless_document_type_id: Optional[int] = None
    after_import_action: str = "keep"  # keep, delete


class CloudSourceUpdate(CloudSourceCreate):
    pass


# ── Source CRUD ──────────────────────────────────────────────────────────────

@router.get("/sources")
async def list_sources(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(CloudSource).order_by(CloudSource.id))
    sources = result.scalars().all()
    return [_source_to_dict(s) for s in sources]


@router.post("/sources")
async def create_source(body: CloudSourceCreate, db: AsyncSession = Depends(get_db)):
    source = CloudSource(**body.model_dump())
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return _source_to_dict(source)


@router.put("/sources/{source_id}")
async def update_source(source_id: int, body: CloudSourceUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(CloudSource).where(CloudSource.id == source_id))
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Quelle nicht gefunden")

    for key, val in body.model_dump().items():
        setattr(source, key, val)
    await db.commit()
    await db.refresh(source)
    return _source_to_dict(source)


@router.delete("/sources/{source_id}")
async def delete_source(source_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(CloudSource).where(CloudSource.id == source_id))
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Quelle nicht gefunden")
    await db.delete(source)
    await db.commit()
    return {"ok": True}


# ── Connection test ──────────────────────────────────────────────────────────

@router.post("/sources/{source_id}/test")
@inject
async def test_source(source_id: int, db: AsyncSession = Depends(get_db), service: FromDishka[CloudImportService] = None):
    assert service is not None
    result = await db.execute(select(CloudSource).where(CloudSource.id == source_id))
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Quelle nicht gefunden")
    return await service.test_connection(source)


# ── Manual sync trigger ──────────────────────────────────────────────────────

@router.post("/sources/{source_id}/sync")
@inject
async def sync_source_now(source_id: int, db: AsyncSession = Depends(get_db), client: FromDishka[PaperlessClient] = None, service: FromDishka[CloudImportService] = None):
    assert client is not None
    assert service is not None
    result = await db.execute(select(CloudSource).where(CloudSource.id == source_id))
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Quelle nicht gefunden")

    try:
        stats = await service.sync_source(source, client, db)
        from datetime import datetime
        source.last_checked_at = datetime.utcnow()
        source.last_status = "idle"
        await db.commit()
        return {"ok": True, **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Folder browser ───────────────────────────────────────────────────────────

@router.get("/sources/{source_id}/folders")
@inject
async def browse_source_folders(source_id: int, path: str = "/", db: AsyncSession = Depends(get_db), service: FromDishka[CloudImportService] = None):
    assert service is not None
    """List folders on a source for folder picker UI."""
    result = await db.execute(select(CloudSource).where(CloudSource.id == source_id))
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Quelle nicht gefunden")
    try:
        folders = await getattr(service, "list_folders")(source, path)
        return {"path": path, "folders": folders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── File listing ─────────────────────────────────────────────────────────────

@router.get("/sources/{source_id}/files")
@inject
async def list_source_files(source_id: int, db: AsyncSession = Depends(get_db), service: FromDishka[CloudImportService] = None):
    assert service is not None
    result = await db.execute(select(CloudSource).where(CloudSource.id == source_id))
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Quelle nicht gefunden")
    try:
        files = await service.list_files(source)
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Import log ───────────────────────────────────────────────────────────────

@router.get("/log")
async def get_import_log(
    source_id: Optional[int] = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    query = select(CloudImportLog).order_by(desc(CloudImportLog.imported_at)).limit(limit)
    if source_id is not None:
        query = query.where(CloudImportLog.source_id == source_id)
    result = await db.execute(query)
    logs = result.scalars().all()
    return [
        {
            "id": log_entry.id,
            "source_id": log_entry.source_id,
            "source_name": log_entry.source_name,
            "file_name": log_entry.file_name,
            "file_path": log_entry.file_path,
            "paperless_doc_id": log_entry.paperless_doc_id,
            "import_status": log_entry.import_status,
            "error_message": log_entry.error_message,
            "imported_at": log_entry.imported_at.isoformat() if log_entry.imported_at else None,
        }
        for log_entry in logs
    ]


@router.delete("/log")
async def clear_import_log(source_id: Optional[int] = None, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import delete
    query = delete(CloudImportLog)
    if source_id is not None:
        query = query.where(CloudImportLog.source_id == source_id)
    await db.execute(query)
    await db.commit()
    return {"ok": True}


# ── Sync daemon control ──────────────────────────────────────────────────────

@router.get("/status")
@inject
async def get_sync_status(state: FromDishka[CloudSyncState] = None):
    assert state is not None
    return {
        "enabled": state.enabled,
        "running": state.running,
        "current_source_name": state.current_source_name,
        "current_file": state.current_file,
        "last_run": state.last_run,
        "files_imported_session": state.files_imported_session,
        "errors_session": state.errors_session,
    }


@router.post("/start")
@inject
async def start_sync_daemon(
    state: FromDishka[CloudSyncState] = None,
    config_svc: FromDishka[ConfigService] = None,
):
    assert state is not None
    assert config_svc is not None
    if state.enabled:
        return {"status": "already_running"}
    state.enabled = True
    state.files_imported_session = 0
    state.errors_session = 0
    state.task = asyncio.get_running_loop().create_task(cloud_sync_loop(di_container))
    logger.info("Cloud sync daemon gestartet")

    # Persist to AppSettings KV store (STATE-08)
    try:
        await _persist_cloud_sync_enabled(True, config_svc)
    except Exception as e:
        logger.warning(f"Could not persist cloud_sync_enabled to KV: {e}")

    return {"status": "started"}


@router.post("/stop")
@inject
async def stop_sync_daemon(
    state: FromDishka[CloudSyncState] = None,
    config_svc: FromDishka[ConfigService] = None,
):
    assert state is not None
    assert config_svc is not None
    state.enabled = False
    task = state.task
    if task and not task.done():
        task.cancel()
    state.running = False
    logger.info("Cloud sync daemon gestoppt")

    # Persist to AppSettings KV store (STATE-08)
    try:
        await _persist_cloud_sync_enabled(False, config_svc)
    except Exception as e:
        logger.warning(f"Could not persist cloud_sync_enabled to KV: {e}")

    return {"status": "stopped"}


# ── Paperless metadata for dropdowns ────────────────────────────────────────

@router.get("/paperless/tags")
@inject
async def get_paperless_tags(client: FromDishka[PaperlessClient] = None):
    assert client is not None
    try:
        tags = await client.get_tags(use_cache=False)
        return [{"id": t["id"], "name": t["name"]} for t in tags]
    except Exception:
        return []


@router.get("/paperless/correspondents")
@inject
async def get_paperless_correspondents(client: FromDishka[PaperlessClient] = None):
    assert client is not None
    try:
        corrs = await client.get_correspondents(use_cache=False)
        return [{"id": c["id"], "name": c["name"]} for c in corrs]
    except Exception:
        return []


@router.get("/paperless/document-types")
@inject
async def get_paperless_document_types(client: FromDishka[PaperlessClient] = None):
    assert client is not None
    try:
        types = await client.get_document_types(use_cache=False)
        return [{"id": t["id"], "name": t["name"]} for t in types]
    except Exception:
        return []


# ── rclone OAuth flow ────────────────────────────────────────────────────────

@router.post("/rclone/authorize")
async def start_rclone_authorize(provider: str = "gdrive"):
    """Start rclone OAuth flow with TCP proxy for Docker."""
    result = await _rclone_oauth.start_authorize(provider)
    if result.get("status") == "error" and "Unbekannter Provider" in result.get("error", ""):
        raise HTTPException(400, result["error"])
    return result


@router.get("/rclone/authorize/status")
async def get_rclone_authorize_status():
    """Poll for rclone OAuth status."""
    return _rclone_oauth.status


@router.post("/rclone/authorize/create-source")
async def create_source_from_rclone_auth(
    name: str = "Google Drive",
    remote_name: str = "gdrive",
    remote_path: str = "/",
    db: AsyncSession = Depends(get_db),
):
    """Create a CloudSource from a successful rclone authorization."""
    result = await _rclone_oauth.create_source_from_auth(name, remote_name, remote_path, db)
    if result is None:
        raise HTTPException(400, "Kein gültiger Token vorhanden. Bitte zuerst autorisieren.")
    return result


# ── Helpers ──────────────────────────────────────────────────────────────────

def _source_to_dict(s: CloudSource) -> dict:
    return {
        "id": s.id,
        "name": s.name,
        "source_type": s.source_type,
        "enabled": s.enabled,
        "poll_interval_minutes": s.poll_interval_minutes,
        "webdav_url": s.webdav_url,
        "webdav_username": s.webdav_username,
        "webdav_password": "***" if s.webdav_password else "",
        "webdav_path": s.webdav_path,
        "rclone_remote": s.rclone_remote,
        "rclone_path": s.rclone_path,
        "rclone_config": s.rclone_config,
        "local_path": s.local_path,
        "filename_prefix": s.filename_prefix,
        "paperless_tag_ids": s.paperless_tag_ids,
        "paperless_correspondent_id": s.paperless_correspondent_id,
        "paperless_document_type_id": s.paperless_document_type_id,
        "after_import_action": s.after_import_action,
        "last_checked_at": s.last_checked_at.isoformat() if s.last_checked_at else None,
        "last_status": s.last_status,
        "last_error": s.last_error,
        "files_imported": s.files_imported,
        "created_at": s.created_at.isoformat() if s.created_at else None,
    }
