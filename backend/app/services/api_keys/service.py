"""API keys service — manages API key CRUD and validation."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime
from typing import Callable, Optional

from sqlalchemy import select, delete as sa_delete


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class ApiKeysServiceImpl:
    """Service for managing API keys."""

    def __init__(self, session_factory: Callable):
        self._session_factory = session_factory

    async def list_keys(self) -> list:
        """List all API keys."""
        from app.models.rag import ApiKey

        async with self._session_factory() as db:
            result = await db.execute(select(ApiKey).order_by(ApiKey.created_at.desc()))
            keys = result.scalars().all()
            return [
                {
                    "id": k.id,
                    "name": k.name,
                    "key_prefix": k.key_prefix,
                    "is_active": k.is_active,
                    "created_at": k.created_at.isoformat() if k.created_at else None,
                    "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                }
                for k in keys
            ]

    async def create_key(self, name: str) -> dict:
        """Create a new API key."""
        from app.models.rag import ApiKey

        async with self._session_factory() as db:
            raw_key = f"po_{secrets.token_hex(24)}"
            key_hash = _hash_key(raw_key)
            key_prefix = raw_key[:10]

            api_key = ApiKey(
                name=name.strip(),
                key_hash=key_hash,
                key_prefix=key_prefix,
            )
            db.add(api_key)
            await db.commit()
            await db.refresh(api_key)

            return {
                "id": api_key.id,
                "name": api_key.name,
                "key": raw_key,
                "key_prefix": key_prefix,
                "message": "Speichere diesen Key sicher ab - er wird nicht erneut angezeigt!",
            }

    async def delete_key(self, key_id: int) -> None:
        """Delete an API key."""
        from app.models.rag import ApiKey

        async with self._session_factory() as db:
            result = await db.execute(select(ApiKey).where(ApiKey.id == key_id))
            key = result.scalar_one_or_none()
            if not key:
                raise ValueError("API-Key nicht gefunden")
            await db.execute(sa_delete(ApiKey).where(ApiKey.id == key_id))
            await db.commit()

    async def toggle_key(self, key_id: int) -> dict:
        """Toggle an API key's active status."""
        from app.models.rag import ApiKey

        async with self._session_factory() as db:
            result = await db.execute(select(ApiKey).where(ApiKey.id == key_id))
            key = result.scalar_one_or_none()
            if not key:
                raise ValueError("API-Key nicht gefunden")
            key.is_active = not key.is_active
            await db.commit()
            return {"id": key.id, "is_active": key.is_active}

    async def validate_key(self, token: str) -> Optional[dict]:
        """Validate an API key token and update last_used_at."""
        from app.models.rag import ApiKey

        async with self._session_factory() as db:
            key_hash = _hash_key(token)
            result = await db.execute(
                select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active)
            )
            api_key = result.scalar_one_or_none()
            if api_key:
                api_key.last_used_at = datetime.utcnow()
                await db.commit()
                return {
                    "id": api_key.id,
                    "name": api_key.name,
                    "key_prefix": api_key.key_prefix,
                    "is_active": api_key.is_active,
                }
            return None
