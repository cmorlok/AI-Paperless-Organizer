"""ConfigService implementation — reads/writes AppSettings KV store."""

from __future__ import annotations

from typing import Optional, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger
from app.models.settings_model import AppSettings

logger = get_logger(__name__)


class ConfigServiceImpl:
    """Implementation of ConfigService protocol using AppSettings KV store."""

    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._session_factory = session_factory

    async def get(self, key: str) -> Optional[str]:
        """Get a setting value by key. Returns None if not found."""
        async with self._session_factory() as db:
            result = await db.execute(
                select(AppSettings).where(AppSettings.key == key)
            )
            setting = result.scalar_one_or_none()
            return setting.value if setting else None

    async def set(self, key: str, value: str, value_type: str = "str") -> None:
        """Set a setting value. Creates new row if key doesn't exist, updates if it does."""
        async with self._session_factory() as db:
            result = await db.execute(
                select(AppSettings).where(AppSettings.key == key)
            )
            setting = result.scalar_one_or_none()
            if setting:
                setting.value = value
                setting.value_type = value_type
            else:
                setting = AppSettings(id=None, key=key, value=value, value_type=value_type)
                db.add(setting)
            await db.commit()
