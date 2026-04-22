"""SessionAuthMiddleware: guard all /api/* paths except PUBLIC_PATHS."""

from __future__ import annotations

import os
from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from app.services.auth.state import COOKIE_NAME, PUBLIC_PATHS


def is_auth_disabled() -> bool:
    return os.getenv("DISABLE_LOGIN", "").lower() == "true"


def is_session_valid(token: str | None) -> bool:
    from app.services.auth.state import SESSIONS
    return bool(token) and token in SESSIONS


class SessionAuthMiddleware:
    """Guard all /api/* paths except PUBLIC_PATHS.

    When DISABLE_LOGIN=true, all /api/* requests pass through immediately.
    Otherwise, a valid session cookie is required for all non-public /api/* paths.
    Performs ZERO database queries per request.

    Implemented as pure ASGI middleware (not BaseHTTPMiddleware) to avoid
    the Starlette double-body-read / exception-handling bug.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        path = request.url.path
        method = request.method

        # Non-API paths pass through (Vite static assets, frontend routes)
        if not path.startswith("/api/"):
            await self.app(scope, receive, send)
            return

        # Public routes
        if (method, path) in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        # Check if auth is globally disabled
        if is_auth_disabled():
            await self.app(scope, receive, send)
            return

        token = request.cookies.get(COOKIE_NAME)
        if not is_session_valid(token):
            response = JSONResponse({"detail": "Nicht authentifiziert"}, status_code=401)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)