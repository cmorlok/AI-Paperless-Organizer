"""Authentication service functions: password hashing, session management."""

from __future__ import annotations

import base64
import os
import secrets
from typing import Optional
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


def _hash_password_sync(password: str) -> str:
    salt = os.urandom(16)
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    key = kdf.derive(password.encode("utf-8"))
    return base64.b64encode(salt + key).decode("ascii")


def _verify_password_sync(password: str, stored: str) -> bool:
    try:
        data = base64.b64decode(stored.encode("ascii"))
    except Exception:
        return False
    if len(data) < 17:
        return False
    salt, key = data[:16], data[16:]
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    try:
        kdf.verify(password.encode("utf-8"), key)
        return True
    except Exception:
        return False


async def hash_password(password: str) -> str:
    """Async Scrypt hash (offloads to worker thread — never blocks event loop)."""
    from anyio.to_thread import run_sync
    return await run_sync(_hash_password_sync, password)


async def verify_password(password: str, stored: str) -> bool:
    """Async Scrypt verify (offloads to worker thread)."""
    if not stored:
        return False
    from anyio.to_thread import run_sync
    return await run_sync(_verify_password_sync, password, stored)


def create_session() -> str:
    """Return a new session token (64 hex chars = 32 bytes, secrets.token_hex(32))."""
    token = secrets.token_hex(32)
    from app.services.auth.state import SESSIONS
    SESSIONS[token] = "authenticated"
    return token


def revoke_session(token: str) -> None:
    from app.services.auth.state import SESSIONS
    SESSIONS.pop(token, None)


def is_session_valid(token: Optional[str]) -> bool:
    from app.services.auth.state import SESSIONS
    return bool(token) and token in SESSIONS


def is_auth_disabled() -> bool:
    """Return True if the DISABLE_LOGIN environment variable is set to 'true'."""
    return os.getenv("DISABLE_LOGIN", "").lower() == "true"


def _cookie_secure_flag() -> bool:
    return os.getenv("COOKIE_SECURE", "").lower() == "true"