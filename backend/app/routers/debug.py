"""Debug and diagnostics endpoints for network troubleshooting."""

import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import httpx
import socket
import asyncio
from urllib.parse import urlparse
from dishka.integrations.fastapi import inject
from dishka import FromDishka

logger = logging.getLogger(__name__)

from app.services.ocr import OcrState
from app.services.classifier import AutoClassifyState
from app.services.cloud_import import CloudSyncState
from app.services.duplicate import DuplicateScanState

router = APIRouter()


class PingRequest(BaseModel):
    host: str


class DnsRequest(BaseModel):
    hostname: str


class HttpTestRequest(BaseModel):
    url: str
    timeout: int = 10


@router.post("/dns-lookup")
async def dns_lookup(request: DnsRequest):
    """Perform DNS lookup for a hostname."""
    try:
        hostname = request.hostname
        # Remove protocol if present
        if "://" in hostname:
            hostname = urlparse(hostname).hostname or hostname
        
        # Get all IPs
        results = socket.getaddrinfo(hostname, None)
        ips = list(set([str(r[4][0]) for r in results]))
        
        return {
            "success": True,
            "hostname": hostname,
            "ips": ips,
            "message": f"DNS aufgelöst: {hostname} -> {', '.join(ips)}"
        }
    except socket.gaierror as e:
        logger.error("DNS lookup failed: %s", e)
        return {
            "success": False,
            "hostname": request.hostname,
            "error": "DNS-Auflösung fehlgeschlagen",
            "message": "DNS-Auflösung fehlgeschlagen"
        }
    except Exception as e:
        logger.error("DNS lookup error: %s", e)
        return {
            "success": False,
            "hostname": request.hostname,
            "error": "DNS-Auflösung fehlgeschlagen",
            "message": "DNS-Auflösung fehlgeschlagen"
        }


@router.post("/tcp-connect")
async def tcp_connect(request: PingRequest):
    """Test TCP connection to a host:port."""
    try:
        host = request.host
        port = 443  # Default HTTPS
        
        # Parse host and port
        if "://" in host:
            parsed = urlparse(host)
            host = parsed.hostname or host
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        elif ":" in host:
            parts = host.rsplit(":", 1)
            host = parts[0]
            port = int(parts[1])
        
        # Try to connect
        loop = asyncio.get_event_loop()
        start = loop.time()
        
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=5.0
        )
        
        elapsed = (loop.time() - start) * 1000
        writer.close()
        await writer.wait_closed()
        
        return {
            "success": True,
            "host": host,
            "port": port,
            "latency_ms": round(elapsed, 2),
            "message": f"TCP-Verbindung zu {host}:{port} erfolgreich ({elapsed:.0f}ms)"
        }
    except asyncio.TimeoutError:
        return {
            "success": False,
            "host": request.host,
            "error": "Timeout",
            "message": f"Timeout: Keine Verbindung zu {request.host} innerhalb von 5 Sekunden"
        }
    except Exception as e:
        logger.error("TCP connect failed: %s", e)
        return {
            "success": False,
            "host": request.host,
            "error": "Verbindung fehlgeschlagen",
            "message": "Verbindung fehlgeschlagen"
        }


@router.post("/http-test")
async def http_test(request: HttpTestRequest):
    """Test HTTP/HTTPS connection to a URL."""
    try:
        url = request.url
        if not url.startswith("http"):
            url = f"https://{url}"
        
        # First test without following redirects
        async with httpx.AsyncClient(timeout=request.timeout, verify=False, follow_redirects=False) as client:
            response = await client.get(url)
            
            redirect_info = None
            if response.status_code in [301, 302, 303, 307, 308]:
                redirect_location = response.headers.get("location", "")
                redirect_info = {
                    "redirects_to": redirect_location,
                    "hint": "Paperless leitet um! Versuche die Ziel-URL direkt."
                }
            
            return {
                "success": True,
                "url": url,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content_length": len(response.content),
                "redirect_info": redirect_info,
                "message": f"HTTP {response.status_code} - {len(response.content)} Bytes empfangen"
            }
    except httpx.ConnectError as e:
        logger.error("HTTP test failed: %s", e)
        return {
            "success": False,
            "url": request.url,
            "error": "ConnectError",
            "details": "Verbindungsfehler",
            "message": f"Verbindungsfehler: Kann {request.url} nicht erreichen"
        }
    except httpx.TimeoutException:
        return {
            "success": False,
            "url": request.url,
            "error": "Timeout",
            "message": f"Timeout nach {request.timeout} Sekunden"
        }
    except Exception as e:
        logger.error("HTTP test error: %s", e)
        return {
            "success": False,
            "url": request.url,
            "error": "Verbindungsfehler",
            "details": "Verbindungsfehler",
            "message": "Verbindungsfehler"
        }


class PaperlessTestRequest(BaseModel):
    url: str
    token: Optional[str] = None
    timeout: int = 10


@router.post("/paperless-test")
async def paperless_test(request: PaperlessTestRequest):
    """Test Paperless-ngx API connection."""
    from app.services.debug_service import DebugService
    svc = DebugService()
    return await svc.paperless_test(request.url, request.token, request.timeout)


@router.get("/network-info")
async def get_network_info():
    """Get container network information."""
    from app.services.debug_service import DebugService
    svc = DebugService()
    return svc.get_network_info()


@router.get("/common-tests")
async def run_common_tests():
    """Run common connectivity tests."""
    from app.services.debug_service import DebugService
    svc = DebugService()
    return await svc.run_common_tests()


# --- Service Statuses Endpoint ---


@router.get("/services")
@inject
async def get_service_statuses(
    ocr_state: FromDishka[OcrState] = None,
    classify_state: FromDishka[AutoClassifyState] = None,
    cloud_state: FromDishka[CloudSyncState] = None,
    dup_state: FromDishka[DuplicateScanState] = None,
):
    assert ocr_state is not None
    assert classify_state is not None
    assert cloud_state is not None
    assert dup_state is not None
    """Return unified status of all background services (read-only)."""
    from app.services.debug_service import DebugService
    svc = DebugService()
    return svc.get_service_statuses(ocr_state, classify_state, cloud_state, dup_state)

