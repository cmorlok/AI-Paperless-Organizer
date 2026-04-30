"""Auth service: session management and middleware."""

from app.services.auth.middleware import SessionAuthMiddleware
from app.services.auth.protocol import AuthService
from app.services.auth.service import (
    AuthServiceImpl,
    create_session,
    hash_password,
    is_auth_disabled,
    is_session_valid,
    revoke_session,
    verify_password,
)
from app.services.auth.state import COOKIE_NAME, PUBLIC_PATHS

__all__ = [
    "SessionAuthMiddleware",
    "AuthService",
    "AuthServiceImpl",
    "create_session",
    "hash_password",
    "is_auth_disabled",
    "is_session_valid",
    "revoke_session",
    "verify_password",
    "COOKIE_NAME",
    "PUBLIC_PATHS",
]
