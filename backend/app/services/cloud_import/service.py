from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional, Any

import httpx

from app.database import async_session

logger = logging.getLogger(__name__)


class CloudImportService:

    def __init__(self, session_factory: Optional[Any] = None, state: Optional[Any] = None):
        self.session_factory = session_factory or async_session
        self._state = state

    @property
    def _sync_state(self):
        if self._state is not None:
            return self._state
        from app.services.cloud_import.state import CloudSyncState
        return CloudSyncState()

    # ── WebDAV ──────────────────────────────────────────────────────────────

    async def list_files_webdav(self, source) -> List[Dict]:
        base_url = source.webdav_url.rstrip("/")
        path = source.webdav_path or "/"
        if not path.startswith("/"):
            path = "/" + path

        url = base_url + path
        auth = (source.webdav_username, source.webdav_password) if source.webdav_username else None

        body = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<propfind xmlns="DAV:"><prop>'
            "<resourcetype/><getcontenttype/><getcontentlength/><getlastmodified/>"
            "</prop></propfind>"
        )

        async with httpx.AsyncClient(timeout=30.0, verify=False, auth=auth) as client:
            response = await client.request(
                "PROPFIND",
                url,
                content=body.encode(),
                headers={"Depth": "1", "Content-Type": "application/xml"},
            )
            response.raise_for_status()

        return self._parse_propfind(response.text)

    def _parse_propfind(self, xml_text: str) -> List[Dict]:
        from app.services.cloud_import.state import _VALID_EXTENSIONS

        files = []
        try:
            root = ET.fromstring(xml_text)
            ns = {"d": "DAV:"}
            for resp in root.findall(".//d:response", ns):
                href_el = resp.find("d:href", ns)
                if href_el is None:
                    continue
                href = href_el.text or ""

                # Skip directories
                rt = resp.find(".//d:resourcetype", ns)
                if rt is not None and rt.find("d:collection", ns) is not None:
                    continue

                name = href.rstrip("/").split("/")[-1]
                if not name:
                    continue
                ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
                if ext not in _VALID_EXTENSIONS:
                    continue

                size_el = resp.find(".//d:getcontentlength", ns)
                mod_el = resp.find(".//d:getlastmodified", ns)
                files.append({
                    "path": href,
                    "name": name,
                    "size": int(size_el.text) if size_el is not None and size_el.text else 0,
                    "modified": mod_el.text if mod_el is not None else "",
                })
        except Exception as e:
            logger.error(f"WebDAV PROPFIND parse error: {e}")
        return files

    async def download_file_webdav(self, source, file_path: str) -> bytes:
        base_url = source.webdav_url.rstrip("/")
        if file_path.startswith("http"):
            url = file_path
        elif file_path.startswith("/"):
            url = base_url + file_path
        else:
            base_path = (source.webdav_path or "/").rstrip("/")
            url = base_url + base_path + "/" + file_path

        auth = (source.webdav_username, source.webdav_password) if source.webdav_username else None
        async with httpx.AsyncClient(timeout=120.0, verify=False, auth=auth) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.content

    async def delete_file_webdav(self, source, file_path: str):
        base_url = source.webdav_url.rstrip("/")
        if file_path.startswith("http"):
            url = file_path
        elif file_path.startswith("/"):
            url = base_url + file_path
        else:
            base_path = (source.webdav_path or "/").rstrip("/")
            url = base_url + base_path + "/" + file_path

        auth = (source.webdav_username, source.webdav_password) if source.webdav_username else None
        async with httpx.AsyncClient(timeout=30.0, verify=False, auth=auth) as client:
            r = await client.delete(url)
            r.raise_for_status()

    # ── rclone ──────────────────────────────────────────────────────────────

    def _rclone_conf_path(self, source) -> str:
        conf_dir = "/app/data/rclone"
        os.makedirs(conf_dir, exist_ok=True)
        path = f"{conf_dir}/source_{source.id}.conf"
        if source.rclone_config:
            with open(path, "w") as f:
                f.write(source.rclone_config)
        return path

    async def list_files_rclone(self, source) -> List[Dict]:
        from app.services.cloud_import.state import _VALID_EXTENSIONS

        conf = self._rclone_conf_path(source)
        remote = f"{source.rclone_remote}:{source.rclone_path or '/'}"
        proc = await asyncio.create_subprocess_exec(
            "rclone", "lsjson", remote, "--max-depth", "1", "--config", conf,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
        if proc.returncode != 0:
            raise RuntimeError(f"rclone lsjson failed: {stderr.decode()}")

        items = json.loads(stdout.decode())
        files = []
        for item in items:
            if item.get("IsDir"):
                continue
            name = item.get("Name", "")
            ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
            if ext not in _VALID_EXTENSIONS:
                continue
            files.append({
                "path": item.get("Path", name),
                "name": name,
                "size": item.get("Size", 0),
                "modified": item.get("ModTime", ""),
            })
        return files

    async def download_file_rclone(self, source, file_name: str) -> bytes:
        conf = self._rclone_conf_path(source)
        remote_base = (source.rclone_path or "/").rstrip("/")
        remote = f"{source.rclone_remote}:{remote_base}/{file_name}"
        ext = os.path.splitext(file_name)[1]

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "rclone", "copyto", remote, tmp_path, "--config", conf,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
            if proc.returncode != 0:
                raise RuntimeError(f"rclone copyto failed: {stderr.decode()}")
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def list_folders_rclone(self, source, path: str = "/") -> List[Dict]:
        """List folders on an rclone remote for folder browser."""
        conf = self._rclone_conf_path(source)
        remote = f"{source.rclone_remote}:{path}"
        proc = await asyncio.create_subprocess_exec(
            "rclone", "lsjson", remote, "--dirs-only", "--max-depth", "1", "--config", conf,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        if proc.returncode != 0:
            raise RuntimeError(f"rclone lsjson failed: {stderr.decode()}")
        items = json.loads(stdout.decode())
        folders = []
        for item in items:
            if item.get("IsDir"):
                name = item.get("Name", "")
                folder_path = path.rstrip("/") + "/" + name
                folders.append({"name": name, "path": folder_path})
        folders.sort(key=lambda x: x["name"].lower())
        return folders

    async def list_folders_webdav(self, source, path: str = "/") -> List[Dict]:
        """List folders on WebDAV for folder browser."""
        base_url = source.webdav_url.rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        url = base_url + path
        auth = (source.webdav_username, source.webdav_password) if source.webdav_username else None
        body = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<propfind xmlns="DAV:"><prop><resourcetype/></prop></propfind>'
        )
        async with httpx.AsyncClient(timeout=30.0, verify=False, auth=auth) as client:
            response = await client.request(
                "PROPFIND", url, content=body.encode(),
                headers={"Depth": "1", "Content-Type": "application/xml"},
            )
            response.raise_for_status()
        folders = []
        try:
            root = ET.fromstring(response.text)
            ns = {"d": "DAV:"}
            for resp in root.findall(".//d:response", ns):
                href_el = resp.find("d:href", ns)
                if href_el is None:
                    continue
                href = href_el.text or ""
                rt = resp.find(".//d:resourcetype", ns)
                if rt is None or rt.find("d:collection", ns) is None:
                    continue
                name = href.rstrip("/").split("/")[-1]
                if not name or href.rstrip("/") == path.rstrip("/"):
                    continue
                folders.append({"name": name, "path": href.rstrip("/")})
        except Exception as e:
            logger.error(f"WebDAV folder parse error: {e}")
        folders.sort(key=lambda x: x["name"].lower())
        return folders

    async def list_folders_local(self, source, path: str = "/") -> List[Dict]:
        """List folders in a local directory."""
        if not os.path.isdir(path):
            return []
        folders = []
        for entry in os.scandir(path):
            if entry.is_dir():
                folders.append({"name": entry.name, "path": entry.path})
        folders.sort(key=lambda x: x["name"].lower())
        return folders

    async def list_files(self, source) -> List[Dict]:
        """List files on any source type (type-based dispatch)."""
        if source.source_type == "webdav":
            return await self.list_files_webdav(source)
        elif source.source_type == "rclone":
            return await self.list_files_rclone(source)
        elif source.source_type == "local":
            return await self.list_files_local(source)
        return []

    async def list_folders(self, source, path: str = "/") -> List[Dict]:
        """List folders on any source type."""
        if source.source_type == "rclone":
            return await self.list_folders_rclone(source, path)
        elif source.source_type == "webdav":
            return await self.list_folders_webdav(source, path)
        elif source.source_type == "local":
            return await self.list_folders_local(source, path)
        return []

    async def delete_file_rclone(self, source, file_name: str):
        conf = self._rclone_conf_path(source)
        remote_base = (source.rclone_path or "/").rstrip("/")
        remote = f"{source.rclone_remote}:{remote_base}/{file_name}"
        proc = await asyncio.create_subprocess_exec(
            "rclone", "deletefile", remote, "--config", conf,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        if proc.returncode != 0:
            raise RuntimeError(f"rclone deletefile failed: {stderr.decode()}")

    # ── Local folder ────────────────────────────────────────────────────────

    async def list_files_local(self, source) -> List[Dict]:
        from app.services.cloud_import.state import _VALID_EXTENSIONS

        path = source.local_path
        if not path or not os.path.isdir(path):
            raise FileNotFoundError(f"Lokaler Pfad nicht gefunden: {path}")
        files = []
        for entry in os.scandir(path):
            if not entry.is_file():
                continue
            ext = entry.name.lower().rsplit(".", 1)[-1] if "." in entry.name else ""
            if ext not in _VALID_EXTENSIONS:
                continue
            stat = entry.stat()
            files.append({
                "path": entry.path,
                "name": entry.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        return files

    # ── Dedup ────────────────────────────────────────────────────────────────

    async def is_already_imported(self, db, source_id: int, file_path: str) -> bool:
        from sqlalchemy import select
        from app.models.cloud_import import CloudImportLog
        result = await db.execute(
            select(CloudImportLog).where(
                CloudImportLog.source_id == source_id,
                CloudImportLog.file_path == file_path,
                CloudImportLog.import_status == "success",
            )
        )
        return result.scalar_one_or_none() is not None

    # ── Sync one source ──────────────────────────────────────────────────────

    async def sync_source(self, source, pl_client, db) -> Dict:
        stats = {"imported": 0, "skipped": 0, "errors": 0}

        if source.source_type == "webdav":
            files = await self.list_files_webdav(source)
        elif source.source_type == "rclone":
            files = await self.list_files_rclone(source)
        elif source.source_type == "local":
            files = await self.list_files_local(source)
        else:
            raise ValueError(f"Unbekannter Quelltyp: {source.source_type}")

        tag_ids = []
        try:
            tag_ids = json.loads(source.paperless_tag_ids or "[]")
        except Exception:
            pass

        for file_info in files:
            if not self._sync_state.enabled:
                break

            file_path = file_info["path"]
            file_name = file_info["name"]
            self._sync_state.current_file = file_name

            if await self.is_already_imported(db, source.id, file_path):
                stats["skipped"] += 1
                continue

            # Download
            try:
                if source.source_type == "webdav":
                    file_bytes = await self.download_file_webdav(source, file_path)
                elif source.source_type == "rclone":
                    file_bytes = await self.download_file_rclone(source, file_name)
                else:
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
            except Exception as e:
                logger.error(f"Cloud import: Download fehlgeschlagen für {file_name}: {e}")
                await self._log(db, source, file_path, file_name, None, "error", str(e))
                stats["errors"] += 1
                continue

            # Upload to Paperless
            prefix = source.filename_prefix or ""
            paperless_name = prefix + file_name
            try:
                await pl_client.upload_document(
                    file_bytes=file_bytes,
                    filename=paperless_name,
                    correspondent_id=source.paperless_correspondent_id,
                    document_type_id=source.paperless_document_type_id,
                    tag_ids=tag_ids if tag_ids else None,
                )
                await self._log(db, source, file_path, file_name, None, "success", "")
                stats["imported"] += 1
                source.files_imported = (source.files_imported or 0) + 1
                self._sync_state.files_imported_session += 1

                # Post-import action
                if source.after_import_action == "delete":
                    try:
                        if source.source_type == "webdav":
                            await self.delete_file_webdav(source, file_path)
                        elif source.source_type == "rclone":
                            await self.delete_file_rclone(source, file_name)
                        elif source.source_type == "local":
                            os.unlink(file_path)
                    except Exception as e:
                        logger.warning(f"Cloud import: Quelldatei konnte nicht gelöscht werden {file_name}: {e}")

            except Exception as e:
                logger.error(f"Cloud import: Paperless-Upload fehlgeschlagen für {file_name}: {e}")
                await self._log(db, source, file_path, file_name, None, "error", str(e))
                stats["errors"] += 1
                self._sync_state.errors_session += 1

        return stats

    async def _log(self, db, source, file_path: str, file_name: str,
                   doc_id: Optional[int], status: str, error: str):
        from app.models.cloud_import import CloudImportLog
        entry = CloudImportLog(
            source_id=source.id,
            source_name=source.name,
            file_path=file_path,
            file_name=file_name,
            paperless_doc_id=doc_id,
            import_status=status,
            error_message=error,
        )
        db.add(entry)
        await db.commit()

    # ── Connection test ──────────────────────────────────────────────────────

    async def test_connection(self, source) -> Dict:
        try:
            if source.source_type == "webdav":
                files = await self.list_files_webdav(source)
            elif source.source_type == "rclone":
                files = await self.list_files_rclone(source)
            elif source.source_type == "local":
                files = await self.list_files_local(source)
            else:
                return {"ok": False, "message": "Unbekannter Quelltyp", "files": 0}
            return {"ok": True, "message": f"Verbindung OK – {len(files)} Dokument(e) gefunden", "files": len(files)}
        except Exception as e:
            return {"ok": False, "message": str(e), "files": 0}

    # ── Extracted router business logic ─────────────────────────────────────

    async def list_sources(self) -> list:
        """List all cloud sources."""
        from sqlalchemy import select
        from app.models.cloud_import import CloudSource
        async with self.session_factory() as db:
            result = await db.execute(select(CloudSource).order_by(CloudSource.id))
            sources = result.scalars().all()
            return [self._source_to_dict(s) for s in sources]

    async def create_source(self, data: dict) -> dict:
        """Create a new cloud source."""
        from app.models.cloud_import import CloudSource
        async with self.session_factory() as db:
            source = CloudSource(**data)
            db.add(source)
            await db.commit()
            await db.refresh(source)
            return self._source_to_dict(source)

    async def update_source(self, source_id: int, data: dict) -> dict:
        """Update a cloud source."""
        from sqlalchemy import select
        from app.models.cloud_import import CloudSource
        async with self.session_factory() as db:
            result = await db.execute(select(CloudSource).where(CloudSource.id == source_id))
            source = result.scalar_one_or_none()
            if not source:
                raise ValueError("Quelle nicht gefunden")
            for key, val in data.items():
                setattr(source, key, val)
            await db.commit()
            await db.refresh(source)
            return self._source_to_dict(source)

    async def delete_source(self, source_id: int) -> None:
        """Delete a cloud source."""
        from sqlalchemy import select
        from app.models.cloud_import import CloudSource
        async with self.session_factory() as db:
            result = await db.execute(select(CloudSource).where(CloudSource.id == source_id))
            source = result.scalar_one_or_none()
            if not source:
                raise ValueError("Quelle nicht gefunden")
            await db.delete(source)
            await db.commit()

    async def get_source(self, source_id: int):
        """Get a cloud source by ID."""
        from sqlalchemy import select
        from app.models.cloud_import import CloudSource
        async with self.session_factory() as db:
            result = await db.execute(select(CloudSource).where(CloudSource.id == source_id))
            return result.scalar_one_or_none()

    async def sync_source_now(self, source_id: int, client) -> dict:
        """Sync a source now and update timestamps."""
        from datetime import datetime
        from sqlalchemy import select
        source = await self.get_source(source_id)
        if not source:
            raise ValueError("Quelle nicht gefunden")
        async with self.session_factory() as db:
            result = await db.execute(
                select(source.__class__).where(source.__class__.id == source_id)
            )
            source = result.scalar_one_or_none()
            if source is None:
                return {"ok": False, "error": "Source not found"}
            stats = await self.sync_source(source, client, db)
            source.last_checked_at = datetime.utcnow()
            source.last_status = "idle"
            await db.commit()
            return {"ok": True, **stats}

    async def get_import_log(self, source_id: int = None, limit: int = 100) -> list:
        """Get import log entries."""
        from sqlalchemy import select, desc
        from app.models.cloud_import import CloudImportLog
        async with self.session_factory() as db:
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

    async def clear_import_log(self, source_id: int = None) -> None:
        """Clear import log entries."""
        from sqlalchemy import delete
        from app.models.cloud_import import CloudImportLog
        async with self.session_factory() as db:
            query = delete(CloudImportLog)
            if source_id is not None:
                query = query.where(CloudImportLog.source_id == source_id)
            await db.execute(query)
            await db.commit()

    async def persist_cloud_sync_enabled(self, enabled: bool, config_svc) -> None:
        """Persist cloud_sync_enabled flag to AppSettings."""
        await config_svc.set(
            "cloud_sync_enabled",
            "true" if enabled else "false",
            "bool",
        )

    async def start_sync_daemon(self, config_svc) -> dict:
        """Start the cloud sync daemon."""
        from app.services.cloud_import.sync_loop import cloud_sync_loop
        from app.container import container as di_container
        if self._sync_state.enabled:
            return {"status": "already_running"}
        self._sync_state.enabled = True
        self._sync_state.files_imported_session = 0
        self._sync_state.errors_session = 0
        self._sync_state.task = asyncio.get_running_loop().create_task(cloud_sync_loop(di_container))
        logger.info("Cloud sync daemon gestartet")
        try:
            await self.persist_cloud_sync_enabled(True, config_svc)
        except Exception as e:
            logger.warning(f"Could not persist cloud_sync_enabled to KV: {e}")
        return {"status": "started"}

    async def stop_sync_daemon(self, config_svc) -> dict:
        """Stop the cloud sync daemon."""
        self._sync_state.enabled = False
        task = self._sync_state.task
        if task and not task.done():
            task.cancel()
        self._sync_state.running = False
        logger.info("Cloud sync daemon gestoppt")
        try:
            await self.persist_cloud_sync_enabled(False, config_svc)
        except Exception as e:
            logger.warning(f"Could not persist cloud_sync_enabled to KV: {e}")
        return {"status": "stopped"}

    def get_sync_status_dict(self) -> dict:
        """Get sync daemon status."""
        return {
            "enabled": self._sync_state.enabled,
            "running": self._sync_state.running,
            "current_source_name": self._sync_state.current_source_name,
            "current_file": self._sync_state.current_file,
            "last_run": self._sync_state.last_run,
            "files_imported_session": self._sync_state.files_imported_session,
            "errors_session": self._sync_state.errors_session,
        }

    async def get_paperless_tags(self, client) -> list:
        """Get Paperless tags for dropdown."""
        try:
            tags = await client.get_tags(use_cache=False)
            return [{"id": t["id"], "name": t["name"]} for t in tags]
        except Exception:
            return []

    async def get_paperless_correspondents(self, client) -> list:
        """Get Paperless correspondents for dropdown."""
        try:
            corrs = await client.get_correspondents(use_cache=False)
            return [{"id": c["id"], "name": c["name"]} for c in corrs]
        except Exception:
            return []

    async def get_paperless_document_types(self, client) -> list:
        """Get Paperless document types for dropdown."""
        try:
            types = await client.get_document_types(use_cache=False)
            return [{"id": t["id"], "name": t["name"]} for t in types]
        except Exception:
            return []

    @staticmethod
    def _source_to_dict(s) -> dict:
        """Convert CloudSource model to dict."""
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
