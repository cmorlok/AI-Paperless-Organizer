"""Business logic for debug and diagnostics endpoints."""

from __future__ import annotations

import socket
import subprocess
from typing import Any

import httpx

from app.services.classifier import AutoClassifyState
from app.services.cloud_import import CloudSyncState
from app.services.duplicate import DuplicateScanState
from app.services.ocr import OcrState


class DebugService:
    """Encapsulates complex business logic for debug/diagnostic endpoints."""

    # ------------------------------------------------------------------
    # Paperless API test
    # ------------------------------------------------------------------

    async def paperless_test(
        self,
        url: str,
        token: str | None = None,
        timeout: int = 10,
    ) -> dict[str, Any]:
        """Test Paperless-ngx API connection with URL variations and redirects.

        Returns a dict with success/failure info, tested URLs, and hints.
        """
        results: list[dict[str, Any]] = []
        url = url.rstrip("/")

        test_urls = [
            f"{url}/api/",
            f"{url}/api",
            url,
        ]

        # If HTTP, also try HTTPS
        if url.startswith("http://"):
            https_url = url.replace("http://", "https://")
            test_urls.extend([
                f"{https_url}/api/",
                https_url,
            ])

        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Token {token}"

        working_url: str | None = None
        final_status: int | None = None
        is_paperless = False
        redirect_target: str | None = None

        try:
            async with httpx.AsyncClient(timeout=timeout, verify=False, follow_redirects=True) as client:
                for test_url in test_urls:
                    try:
                        response = await client.get(test_url, headers=headers)
                        results.append({
                            "url": test_url,
                            "status": response.status_code,
                            "final_url": str(response.url),
                        })

                        if response.status_code == 200:
                            content = response.text.lower()
                            final_url_str = str(response.url)
                            is_paperless_api = (
                                "correspondents" in content
                                or "documents" in content
                                or "tags" in content
                                or "paperless" in content
                                or "/api/schema" in final_url_str
                                or "openapi" in content
                            )
                            if is_paperless_api:
                                working_url = test_url
                                final_status = response.status_code
                                is_paperless = True
                                if final_url_str != test_url:
                                    redirect_target = final_url_str
                                break
                    except Exception as e:
                        results.append({
                            "url": test_url,
                            "error": str(e),
                        })

            if working_url:
                return {
                    "success": True,
                    "url": url,
                    "working_url": working_url,
                    "redirect_target": redirect_target,
                    "status_code": final_status,
                    "is_paperless": is_paperless,
                    "tested_urls": results,
                    "message": f"Paperless API gefunden! Nutze: {redirect_target or working_url}",
                }

            # No working URL found — check for redirect
            redirect_info: str | None = None
            async with httpx.AsyncClient(timeout=timeout, verify=False, follow_redirects=False) as client:
                try:
                    resp = await client.get(f"{url}/api/", headers=headers)
                    if resp.status_code in [301, 302, 303, 307, 308]:
                        redirect_info = resp.headers.get("location")
                except Exception:
                    pass

            return {
                "success": False,
                "url": url,
                "tested_urls": results,
                "redirect_detected": redirect_info,
                "is_paperless": False,
                "message": (
                    f"Paperless API nicht gefunden. Redirect zu: {redirect_info}"
                    if redirect_info
                    else "Paperless API nicht gefunden"
                ),
                "hint": "Versuche HTTPS oder prüfe ob ein API-Token benötigt wird",
            }
        except Exception as e:
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "tested_urls": results,
                "message": f"Paperless nicht erreichbar: {e}",
            }

    # ------------------------------------------------------------------
    # Network info
    # ------------------------------------------------------------------

    def get_network_info(self) -> dict[str, Any]:
        """Gather container network information."""
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)

            interfaces: list[str] = []
            try:
                result = subprocess.run(
                    ["ip", "addr"], capture_output=True, text=True, timeout=5
                )
                interfaces = result.stdout.split("\n") if result.returncode == 0 else []
            except Exception:
                pass

            dns_servers: list[str] = []
            try:
                with open("/etc/resolv.conf", "r") as f:
                    for line in f:
                        if line.startswith("nameserver"):
                            dns_servers.append(line.split()[1])
            except Exception:
                pass

            return {
                "hostname": hostname,
                "local_ip": local_ip,
                "dns_servers": dns_servers,
                "interfaces": interfaces[:20] if interfaces else ["Nicht verfügbar"],
            }
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Common connectivity tests
    # ------------------------------------------------------------------

    async def run_common_tests(self) -> dict[str, Any]:
        """Run DNS and HTTPS connectivity tests against well-known hosts."""
        tests: list[dict[str, Any]] = []

        for host in ["google.com", "github.com"]:
            try:
                ips = socket.gethostbyname(host)
                tests.append({"test": f"DNS: {host}", "success": True, "result": ips})
            except Exception as e:
                tests.append({"test": f"DNS: {host}", "success": False, "result": str(e)})

        async with httpx.AsyncClient(timeout=5, verify=False) as client:
            for url in ["https://google.com", "https://api.openai.com"]:
                try:
                    r = await client.get(url)
                    tests.append({
                        "test": f"HTTPS: {url}",
                        "success": True,
                        "result": f"HTTP {r.status_code}",
                    })
                except Exception as e:
                    tests.append({
                        "test": f"HTTPS: {url}",
                        "success": False,
                        "result": str(e),
                    })

        return {"tests": tests}

    # ------------------------------------------------------------------
    # Service statuses
    # ------------------------------------------------------------------

    @staticmethod
    def _ocr_current_op(state: OcrState) -> str | None:
        """Get current OCR operation text."""
        if state.batch.running and state.batch.current_document:
            doc = state.batch.current_document
            title = doc.get("title", "?") if isinstance(doc, dict) else "?"
            return f"Batch: {title}"
        if state.processor.running:
            return f"Processor (Intervall: {state.processor.interval_minutes}min)"
        return None

    def get_service_statuses(
        self,
        ocr_state: OcrState,
        classify_state: AutoClassifyState,
        cloud_state: CloudSyncState,
        dup_state: DuplicateScanState,
    ) -> dict[str, Any]:
        """Build unified status dict for all background services."""
        return {
            "services": [
                {
                    "name": "ocr",
                    "label": "OCR",
                    "enabled": ocr_state.processor.enabled,
                    "running": ocr_state.processor.running or ocr_state.batch.running,
                    "current_op": self._ocr_current_op(ocr_state),
                    "detail": (
                        ocr_state.processor.model_dump()
                        if ocr_state.processor.running
                        else ocr_state.batch.model_dump()
                    ),
                },
                {
                    "name": "classifier",
                    "label": "Klassifizierung",
                    "enabled": classify_state.enabled,
                    "running": classify_state.enabled and classify_state.running,
                    "current_op": (
                        f"Dokument {classify_state.current_doc}"
                        if classify_state.current_doc
                        else None
                    ),
                    "detail": classify_state.model_dump(),
                },
                {
                    "name": "cloud_import",
                    "label": "Cloud-Import",
                    "enabled": cloud_state.enabled,
                    "running": cloud_state.running,
                    "current_op": cloud_state.current_file,
                    "detail": cloud_state.model_dump(),
                },
                {
                    "name": "duplicate",
                    "label": "Duplikat-Scan",
                    "enabled": False,  # duplicate scan is on-demand, not daemon
                    "running": dup_state.running,
                    "current_op": dup_state.phase if dup_state.running else None,
                    "detail": dup_state.model_dump(),
                },
            ]
        }
