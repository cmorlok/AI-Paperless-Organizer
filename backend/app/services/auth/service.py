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


class AuthServiceImpl:
    """Service for auth operations with DB access."""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    async def get_or_create_auth_config(self) -> dict:
        """Get or create auth config."""
        from sqlalchemy import select
        from app.models.auth_config import AuthConfig

        async with self._session_factory() as db:
            result = await db.execute(select(AuthConfig).where(AuthConfig.id == 1))
            config = result.scalar_one_or_none()
            if config is None:
                config = AuthConfig(id=1, password_hash="")
                db.add(config)
                await db.commit()
                await db.refresh(config)
            return {
                "id": config.id,
                "password_hash": config.password_hash,
            }

    async def set_password(self, password: str) -> None:
        """Set or update the admin password."""
        from sqlalchemy import select
        from app.models.auth_config import AuthConfig

        async with self._session_factory() as db:
            result = await db.execute(select(AuthConfig).where(AuthConfig.id == 1))
            config = result.scalar_one_or_none()
            if config is None:
                config = AuthConfig(id=1, password_hash="")
                db.add(config)
            config.password_hash = await hash_password(password)
            await db.commit()

    async def validate_password(self, password: str) -> bool:
        """Validate password against stored hash."""
        config = await self.get_or_create_auth_config()
        if not config.get("password_hash"):
            return False
        return await verify_password(password, config["password_hash"])

    async def change_password(self, current_password: str, new_password: str) -> bool:
        """Change password after verifying current password."""
        config = await self.get_or_create_auth_config()
        if not config.get("password_hash"):
            return False

        ok = await verify_password(current_password, config["password_hash"])
        if not ok:
            return False

        await self.set_password(new_password)
        return True

    def create_session(self) -> str:
        """Create a new session token."""
        return create_session()

    def revoke_session(self, token: str) -> None:
        """Revoke a session token."""
        revoke_session(token)

    def is_session_valid(self, token: Optional[str]) -> bool:
        """Check if a session token is valid."""
        return is_session_valid(token)

    def is_auth_disabled(self) -> bool:
        """Check if auth is disabled."""
        return is_auth_disabled()