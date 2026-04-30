"""Cloud Import Router — thin endpoints only."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.services.cloud_import import CloudImportService, RcloneOAuthService
from app.services.paperless import PaperlessClient
from app.services.config import ConfigService
from app.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from dishka.integrations.fastapi import inject
from dishka import FromDishka

router = APIRouter()
logger = logging.getLogger(__name__)

# Singleton rclone OAuth service
_rclone_oauth = RcloneOAuthService()


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class CloudSourceCreate(BaseModel):
    name: str
    source_type: str = "webdav"
    enabled: bool = True
    poll_interval_minutes: int = 5
    webdav_url: str = ""
    webdav_username: str = ""
    webdav_password: str = ""
    webdav_path: str = "/"
    rclone_remote: str = ""
    rclone_path: str = "/"
    rclone_config: str = ""
    local_path: str = ""
    filename_prefix: str = ""
    paperless_tag_ids: str = "[]"
    paperless_correspondent_id: Optional[int] = None
    paperless_document_type_id: Optional[int] = None
    after_import_action: str = "keep"


class CloudSourceUpdate(CloudSourceCreate):
    pass


# ── Source CRUD ──────────────────────────────────────────────────────────────

@router.get("/sources")
@inject
async def list_sources(service: FromDishka[CloudImportService] = None):
    assert service is not None
    return await service.list_sources()


@router.post("/sources")
@inject
async def create_source(body: CloudSourceCreate, service: FromDishka[CloudImportService] = None):
    assert service is not None
    return await service.create_source(body.model_dump())


@router.put("/sources/{source_id}")
@inject
async def update_source(source_id: int, body: CloudSourceUpdate, service: FromDishka[CloudImportService] = None):
    assert service is not None
    try:
        return await service.update_source(source_id, body.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/sources/{source_id}")
@inject
async def delete_source(source_id: int, service: FromDishka[CloudImportService] = None):
    assert service is not None
    try:
        await service.delete_source(source_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True}


# ── Connection test ──────────────────────────────────────────────────────────

@router.post("/sources/{source_id}/test")
@inject
async def test_source(source_id: int, service: FromDishka[CloudImportService] = None):
    assert service is not None
    source = await service.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Quelle nicht gefunden")
    return await service.test_connection(source)


# ── Manual sync trigger ──────────────────────────────────────────────────────

@router.post("/sources/{source_id}/sync")
@inject
async def sync_source_now(
    source_id: int,
    client: FromDishka[PaperlessClient] = None,
    service: FromDishka[CloudImportService] = None,
):
    assert client is not None
    assert service is not None
    try:
        return await service.sync_source_now(source_id, client)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Folder browser ───────────────────────────────────────────────────────────

@router.get("/sources/{source_id}/folders")
@inject
async def browse_source_folders(
    source_id: int, path: str = "/",
    service: FromDishka[CloudImportService] = None,
):
    assert service is not None
    source = await service.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Quelle nicht gefunden")
    try:
        folders = await service.list_folders(source, path)
        return {"path": path, "folders": folders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── File listing ─────────────────────────────────────────────────────────────

@router.get("/sources/{source_id}/files")
@inject
async def list_source_files(source_id: int, service: FromDishka[CloudImportService] = None):
    assert service is not None
    source = await service.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Quelle nicht gefunden")
    try:
        files = await service.list_files(source)
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Import log ───────────────────────────────────────────────────────────────

@router.get("/log")
@inject
async def get_import_log(
    source_id: Optional[int] = None,
    limit: int = 100,
    service: FromDishka[CloudImportService] = None,
):
    assert service is not None
    return await service.get_import_log(source_id, limit)


@router.delete("/log")
@inject
async def clear_import_log(
    source_id: Optional[int] = None,
    service: FromDishka[CloudImportService] = None,
):
    assert service is not None
    await service.clear_import_log(source_id)
    return {"ok": True}


# ── Sync daemon control ──────────────────────────────────────────────────────

@router.get("/status")
@inject
async def get_sync_status(service: FromDishka[CloudImportService] = None):
    assert service is not None
    return service.get_sync_status_dict()


@router.post("/start")
@inject
async def start_sync_daemon(
    config_svc: FromDishka[ConfigService] = None,
    service: FromDishka[CloudImportService] = None,
):
    assert config_svc is not None
    assert service is not None
    return await service.start_sync_daemon(config_svc)


@router.post("/stop")
@inject
async def stop_sync_daemon(
    config_svc: FromDishka[ConfigService] = None,
    service: FromDishka[CloudImportService] = None,
):
    assert config_svc is not None
    assert service is not None
    return await service.stop_sync_daemon(config_svc)


# ── Paperless metadata for dropdowns ────────────────────────────────────────

@router.get("/paperless/tags")
@inject
async def get_paperless_tags(client: FromDishka[PaperlessClient] = None, service: FromDishka[CloudImportService] = None):
    assert client is not None
    assert service is not None
    return await service.get_paperless_tags(client)


@router.get("/paperless/correspondents")
@inject
async def get_paperless_correspondents(client: FromDishka[PaperlessClient] = None, service: FromDishka[CloudImportService] = None):
    assert client is not None
    assert service is not None
    return await service.get_paperless_correspondents(client)


@router.get("/paperless/document-types")
@inject
async def get_paperless_document_types(client: FromDishka[PaperlessClient] = None, service: FromDishka[CloudImportService] = None):
    assert client is not None
    assert service is not None
    return await service.get_paperless_document_types(client)


# ── rclone OAuth flow ────────────────────────────────────────────────────────

@router.post("/rclone/authorize")
async def start_rclone_authorize(provider: str = "gdrive"):
    result = await _rclone_oauth.start_authorize(provider)
    if result.get("status") == "error" and "Unbekannter Provider" in result.get("error", ""):
        raise HTTPException(400, result["error"])
    return result


@router.get("/rclone/authorize/status")
async def get_rclone_authorize_status():
    return _rclone_oauth.status


@router.post("/rclone/authorize/create-source")
async def create_source_from_rclone_auth(
    name: str = "Google Drive",
    remote_name: str = "gdrive",
    remote_path: str = "/",
    db: AsyncSession = Depends(get_db),
):
    result = await _rclone_oauth.create_source_from_auth(name, remote_name, remote_path, db)
    if result is None:
        raise HTTPException(400, "Kein gültiger Token vorhanden. Bitte zuerst autorisieren.")
    return result
