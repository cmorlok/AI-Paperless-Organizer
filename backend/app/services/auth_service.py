"""Authentication service: Scrypt password hashing, session store, and middleware.

Per CONTEXT.md D-01 through D-14:
- D-02 cookie: HttpOnly=True, SameSite=Lax, Secure only in HTTPS (controlled via env)
- D-03 session cookie with no expiry (browser session only)
- D-05 Scrypt via cryptography package
- D-10 PUBLIC_PATHS: status, settings/app, login, setup; plus /api/health (RESEARCH Pitfall 8)

Architecture: Auth is enforced by default. The only way to disable it is the
DISABLE_LOGIN environment variable. SessionAuthMiddleware checks the env var
and the in-memory SESSIONS dict — no database query per request.
"""
from __future__ import annotations

import base64
import logging
import os
import secrets
from typing import Optional

import anyio
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

COOKIE_NAME = "paperless_ai_session"
SALT_SIZE = 16
SCRYPT_N = 2**14   # OWASP minimum (CONTEXT.md "Claude's Discretion"; RESEARCH.md A1)
SCRYPT_R = 8
SCRYPT_P = 1

# Module-level session store — lives here so middleware can read without DI context.
# Keyed by token (secrets.token_hex(32)). Value is literal "authenticated" sentinel.
SESSIONS: dict[str, str] = {}

# Public paths that bypass auth. Per CONTEXT.md D-10 plus /api/health (RESEARCH Pitfall 8).
PUBLIC_PATHS: set[tuple[str, str]] = {
    ("GET", "/api/auth/status"),
    ("GET", "/api/settings/app"),
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/setup"),
    ("GET", "/api/health"),
}


def _hash_password_sync(password: str) -> str:
    salt = os.urandom(SALT_SIZE)
    kdf = Scrypt(salt=salt, length=32, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    key = kdf.derive(password.encode("utf-8"))
    return base64.b64encode(salt + key).decode("ascii")


def _verify_password_sync(password: str, stored: str) -> bool:
    try:
        data = base64.b64decode(stored.encode("ascii"))
    except Exception:
        return False
    if len(data) < SALT_SIZE + 1:
        return False
    salt, key = data[:SALT_SIZE], data[SALT_SIZE:]
    kdf = Scrypt(salt=salt, length=32, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    try:
        kdf.verify(password.encode("utf-8"), key)
        return True
    except Exception:
        return False


async def hash_password(password: str) -> str:
    """Async Scrypt hash (offloads to worker thread — never blocks event loop)."""
    return await anyio.to_thread.run_sync(_hash_password_sync, password)


async def verify_password(password: str, stored: str) -> bool:
    """Async Scrypt verify (offloads to worker thread)."""
    if not stored:
        return False
    return await anyio.to_thread.run_sync(_verify_password_sync, password, stored)


def create_session() -> str:
    """Return a new session token (64 hex chars = 32 bytes, secrets.token_hex(32))."""
    token = secrets.token_hex(32)
    SESSIONS[token] = "authenticated"
    return token


def revoke_session(token: str) -> None:
    SESSIONS.pop(token, None)


def is_session_valid(token: Optional[str]) -> bool:
    return bool(token) and token in SESSIONS


def is_auth_disabled() -> bool:
    """Return True if the DISABLE_LOGIN environment variable is set to 'true'.

    When disabled, the middleware skips all authentication checks.
    """
    return os.getenv("DISABLE_LOGIN", "").lower() == "true"


def _cookie_secure_flag() -> bool:
    return os.getenv("COOKIE_SECURE", "").lower() == "true"


class SessionAuthMiddleware(BaseHTTPMiddleware):
    """Guard all /api/* paths except PUBLIC_PATHS.

    When DISABLE_LOGIN=true, all /api/* requests pass through immediately.
    Otherwise, a valid session cookie is required for all non-public /api/* paths.
    Performs ZERO database queries per request.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method

        # Non-API paths pass through (Vite static assets, frontend routes)
        if not path.startswith("/api/"):
            return await call_next(request)

        # Public routes
        if (method, path) in PUBLIC_PATHS:
            return await call_next(request)

        # Check if auth is globally disabled
        if is_auth_disabled():
            return await call_next(request)

        token = request.cookies.get(COOKIE_NAME)
        if not is_session_valid(token):
            return JSONResponse({"detail": "Nicht authentifiziert"}, status_code=401)

        return await call_next(request)
