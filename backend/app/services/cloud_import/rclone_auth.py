"""Rclone OAuth authorization flow — process management, TCP proxy, token extraction."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, Optional

from app.models.cloud_import import CloudSource

logger = logging.getLogger(__name__)

_RCLONE_PROVIDERS: Dict[str, Dict[str, str]] = {
    "gdrive": {"rclone_type": "drive", "label": "Google Drive"},
    "onedrive": {"rclone_type": "onedrive", "label": "OneDrive"},
    "dropbox": {"rclone_type": "dropbox", "label": "Dropbox"},
}


class RcloneOAuthService:
    """Manages the rclone OAuth authorization flow."""

    def __init__(self) -> None:
        self._state: Dict[str, Any] = {
            "process": None,
            "proxy_server": None,
            "provider": None,
            "auth_url": None,
            "token": None,
            "status": "idle",  # idle, waiting, success, error
            "error": None,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def providers(self) -> Dict[str, Dict[str, str]]:
        return _RCLONE_PROVIDERS

    @property
    def status(self) -> Dict[str, Any]:
        """Current auth state for polling endpoint."""
        return {
            "status": self._state["status"],
            "auth_url": self._state["auth_url"],
            "token": self._state["token"],
            "provider": self._state["provider"],
            "error": self._state["error"],
        }

    async def start_authorize(self, provider: str) -> Dict[str, Any]:
        """Start the rclone OAuth flow with TCP proxy for Docker.

        Returns a dict with ``status``, ``auth_url``, ``provider``, ``label``
        on success, or ``status=error`` with an ``error`` message.
        """
        if provider not in _RCLONE_PROVIDERS:
            return {"status": "error", "error": f"Unbekannter Provider: {provider}"}

        # Kill any previous rclone authorize + proxy
        await self._cleanup()

        # Also kill any orphan rclone on port 53682
        try:
            kill_proc = await asyncio.create_subprocess_exec(
                "fuser", "-k", "53682/tcp",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(kill_proc.wait(), timeout=3)
        except Exception:
            pass
        await asyncio.sleep(0.5)

        rclone_type = _RCLONE_PROVIDERS[provider]["rclone_type"]
        self._state.update({
            "provider": provider,
            "auth_url": None,
            "token": None,
            "status": "waiting",
            "error": None,
            "process": None,
            "proxy_server": None,
        })

        # 1) Start rclone authorize (binds to 127.0.0.1:53682)
        proc = await asyncio.create_subprocess_exec(
            "rclone", "authorize", rclone_type, "--auth-no-open-browser",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._state["process"] = proc

        # 2) Wait briefly for rclone to bind its port
        await asyncio.sleep(1)

        # 3) Start TCP proxy: 0.0.0.0:53683 → 127.0.0.1:53682
        #    Docker maps host:53682 → container:53683
        try:
            proxy = await asyncio.start_server(self._tcp_proxy_handler, "0.0.0.0", 53683)
            self._state["proxy_server"] = proxy
            logger.info("rclone OAuth TCP proxy started on 0.0.0.0:53683 → 127.0.0.1:53682")
        except Exception as e:
            logger.warning(f"rclone OAuth TCP proxy failed: {e}")

        # 4) Read output in background
        asyncio.get_running_loop().create_task(self._read_auth(proc))

        # 5) Wait for URL (up to 15s)
        for _ in range(30):
            await asyncio.sleep(0.5)
            if self._state["auth_url"]:
                return {
                    "status": "waiting",
                    "auth_url": self._state["auth_url"],
                    "provider": provider,
                    "label": _RCLONE_PROVIDERS[provider]["label"],
                }
            if self._state["status"] == "error":
                return {
                    "status": "error",
                    "error": self._state["error"],
                }

        return {"status": "waiting", "auth_url": None, "message": "Auth-URL wird geladen…"}

    async def create_source_from_auth(
        self,
        name: str,
        remote_name: str,
        remote_path: str,
        db: Any,
    ) -> Optional[Dict[str, Any]]:
        """Create a CloudSource from a successful rclone authorization.

        Returns the source dict on success, or ``None`` if no valid token.
        """
        if self._state["status"] != "success" or not self._state["token"]:
            return None

        provider = self._state["provider"] or "gdrive"
        rclone_type = _RCLONE_PROVIDERS.get(provider, {}).get("rclone_type", "drive")

        config_content = f"[{remote_name}]\ntype = {rclone_type}\ntoken = {self._state['token']}\n"

        source = CloudSource(
            name=name,
            source_type="rclone",
            rclone_remote=remote_name,
            rclone_path=remote_path,
            rclone_config=config_content,
            enabled=True,
        )
        db.add(source)
        await db.commit()
        await db.refresh(source)

        # Reset auth state
        self._state.update({"process": None, "status": "idle", "token": None, "auth_url": None})

        return {
            "id": source.id,
            "name": source.name,
            "source_type": source.source_type,
            "enabled": source.enabled,
            "poll_interval_minutes": source.poll_interval_minutes,
            "webdav_url": source.webdav_url,
            "webdav_username": source.webdav_username,
            "webdav_password": "***" if source.webdav_password else "",
            "webdav_path": source.webdav_path,
            "rclone_remote": source.rclone_remote,
            "rclone_path": source.rclone_path,
            "rclone_config": source.rclone_config,
            "local_path": source.local_path,
            "filename_prefix": source.filename_prefix,
            "paperless_tag_ids": source.paperless_tag_ids,
            "paperless_correspondent_id": source.paperless_correspondent_id,
            "paperless_document_type_id": source.paperless_document_type_id,
            "after_import_action": source.after_import_action,
            "last_checked_at": source.last_checked_at.isoformat() if source.last_checked_at else None,
            "last_status": source.last_status,
            "last_error": source.last_error,
            "files_imported": source.files_imported,
            "created_at": source.created_at.isoformat() if source.created_at else None,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _cleanup(self) -> None:
        """Kill running rclone authorize + proxy."""
        if self._state["process"] and self._state["process"].returncode is None:
            try:
                self._state["process"].kill()
                await asyncio.wait_for(self._state["process"].wait(), timeout=3)
            except Exception:
                pass
        if self._state["proxy_server"]:
            try:
                self._state["proxy_server"].close()
                await self._state["proxy_server"].wait_closed()
            except Exception:
                pass
        self._state["process"] = None
        self._state["proxy_server"] = None

    async def _tcp_proxy_handler(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Forward incoming TCP from 0.0.0.0:53683 → 127.0.0.1:53682 (rclone)."""
        try:
            target_reader, target_writer = await asyncio.open_connection("127.0.0.1", 53682)
        except Exception:
            writer.close()
            return

        async def forward(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            try:
                while True:
                    data = await src.read(8192)
                    if not data:
                        break
                    dst.write(data)
                    await dst.drain()
            except Exception:
                pass
            finally:
                try:
                    dst.close()
                except Exception:
                    pass

        await asyncio.gather(forward(reader, target_writer), forward(target_reader, writer))

    async def _read_auth(self, proc: asyncio.subprocess.Process) -> None:
        """Background: read rclone stdout/stderr, extract auth URL + token."""
        all_stdout: list[str] = []

        async def read_stream(stream: asyncio.StreamReader, name: str) -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                logger.info(f"rclone {name}: {text}")

                if name == "stdout":
                    all_stdout.append(text)

                # Extract auth URL — rclone prints "Please go to the following link: http://..."
                if "go to the following link" in text.lower() or ("http" in text and "/auth?" in text):
                    url_match = re.search(r'(https?://\S+)', text)
                    if url_match and not self._state["auth_url"]:
                        self._state["auth_url"] = url_match.group(1)
                        logger.info(f"rclone auth URL: {url_match.group(1)[:80]}…")

        assert proc.stdout is not None and proc.stderr is not None
        await asyncio.gather(
            read_stream(proc.stdout, "stdout"),
            read_stream(proc.stderr, "stderr"),
        )
        await proc.wait()

        # Close proxy
        if self._state["proxy_server"]:
            try:
                self._state["proxy_server"].close()
            except Exception:
                pass

        # Extract token from stdout lines
        token_json = None
        for line in all_stdout:
            line = line.strip()
            if line.startswith("{") and "access_token" in line:
                try:
                    json.loads(line)
                    token_json = line
                    break
                except Exception:
                    pass

        if token_json:
            self._state["token"] = token_json
            self._state["status"] = "success"
            logger.info("rclone auth: Token erfasst")
        elif proc.returncode == 0:
            self._state["status"] = "error"
            self._state["error"] = "Token konnte nicht gelesen werden. Bitte erneut versuchen."
        else:
            self._state["status"] = "error"
            self._state["error"] = f"rclone Fehler (exit {proc.returncode})"
