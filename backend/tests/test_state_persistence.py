"""Tests for AppSettings KV persistence of enabled flags (STATE-08).

Verifies:
- _persist_auto_classify_enabled writes correct key/value to AppSettings
- _persist_ocr_watchdog_enabled writes correct key/value to AppSettings
- _persist_cloud_sync_enabled writes correct key/value to AppSettings
- set_setting creates a row when key does not exist
- set_setting updates an existing row when key already exists
- get_setting returns None when key is not found
- get_setting returns the persisted value when key exists
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select

from app.database import Base
from app.models.settings_model import AppSettings


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def test_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def test_session_factory(test_engine):
    return async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)


@pytest_asyncio.fixture
async def db_session(test_session_factory):
    async with test_session_factory() as session:
        yield session


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _read_key(session: AsyncSession, key: str) -> str | None:
    """Helper to read a key directly from AppSettings."""
    result = await session.execute(
        select(AppSettings).where(AppSettings.key == key)
    )
    row = result.scalar_one_or_none()
    return str(row.value) if row else None


# ── Tests for get_setting / set_setting ───────────────────────────────────────

class TestGetSetting:
    @pytest.mark.asyncio
    async def test_returns_none_when_key_not_found(self, db_session):
        from app.routers.settings import get_setting
        result = await get_setting("nonexistent_key", db_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_value_when_key_exists(self, db_session):
        from app.routers.settings import set_setting, get_setting
        await set_setting("test_key", "test_value", "str", db_session)
        result = await get_setting("test_key", db_session)
        assert result == "test_value"


class TestSetSetting:
    @pytest.mark.asyncio
    async def test_creates_row_when_key_not_exists(self, db_session):
        from app.routers.settings import set_setting
        await set_setting("new_key", "new_value", "str", db_session)
        stored = await _read_key(db_session, "new_key")
        assert stored == "new_value"

    @pytest.mark.asyncio
    async def test_updates_row_when_key_already_exists(self, db_session):
        from app.routers.settings import set_setting
        await set_setting("update_key", "initial", "str", db_session)
        await set_setting("update_key", "updated", "str", db_session)
        stored = await _read_key(db_session, "update_key")
        assert stored == "updated"

    @pytest.mark.asyncio
    async def test_creates_exactly_one_row_for_key(self, db_session):
        from app.routers.settings import set_setting
        await set_setting("unique_key", "v1", "str", db_session)
        await set_setting("unique_key", "v2", "str", db_session)
        result = await db_session.execute(
            select(AppSettings).where(AppSettings.key == "unique_key")
        )
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].value == "v2"


# ── Tests for _persist_auto_classify_enabled ──────────────────────────────────

class TestPersistAutoClassifyEnabled:
    @pytest.mark.asyncio
    async def test_persist_true_writes_true_string(self, db_session):
        from app.routers.classifier import _persist_auto_classify_enabled
        await _persist_auto_classify_enabled(True, db_session)
        stored = await _read_key(db_session, "auto_classify_enabled")
        assert stored == "true"

    @pytest.mark.asyncio
    async def test_persist_false_writes_false_string(self, db_session):
        from app.routers.classifier import _persist_auto_classify_enabled
        await _persist_auto_classify_enabled(False, db_session)
        stored = await _read_key(db_session, "auto_classify_enabled")
        assert stored == "false"

    @pytest.mark.asyncio
    async def test_persist_key_is_auto_classify_enabled(self, db_session):
        from app.routers.classifier import _persist_auto_classify_enabled
        await _persist_auto_classify_enabled(True, db_session)
        result = await db_session.execute(
            select(AppSettings).where(AppSettings.key == "auto_classify_enabled")
        )
        row = result.scalar_one_or_none()
        assert row is not None
        assert row.key == "auto_classify_enabled"


# ── Tests for _persist_ocr_watchdog_enabled ───────────────────────────────────

class TestPersistOcrWatchdogEnabled:
    @pytest.mark.asyncio
    async def test_persist_true_writes_true_string(self, db_session):
        from app.routers.ocr import _persist_ocr_watchdog_enabled
        await _persist_ocr_watchdog_enabled(True, db_session)
        stored = await _read_key(db_session, "ocr_watchdog_enabled")
        assert stored == "true"

    @pytest.mark.asyncio
    async def test_persist_false_writes_false_string(self, db_session):
        from app.routers.ocr import _persist_ocr_watchdog_enabled
        await _persist_ocr_watchdog_enabled(False, db_session)
        stored = await _read_key(db_session, "ocr_watchdog_enabled")
        assert stored == "false"

    @pytest.mark.asyncio
    async def test_persist_key_is_ocr_watchdog_enabled(self, db_session):
        from app.routers.ocr import _persist_ocr_watchdog_enabled
        await _persist_ocr_watchdog_enabled(True, db_session)
        result = await db_session.execute(
            select(AppSettings).where(AppSettings.key == "ocr_watchdog_enabled")
        )
        row = result.scalar_one_or_none()
        assert row is not None
        assert row.key == "ocr_watchdog_enabled"


# ── Tests for _persist_cloud_sync_enabled ─────────────────────────────────────

class TestPersistCloudSyncEnabled:
    @pytest.mark.asyncio
    async def test_persist_true_writes_true_string(self, db_session):
        from app.routers.cloud_import import _persist_cloud_sync_enabled
        await _persist_cloud_sync_enabled(True, db_session)
        stored = await _read_key(db_session, "cloud_sync_enabled")
        assert stored == "true"

    @pytest.mark.asyncio
    async def test_persist_false_writes_false_string(self, db_session):
        from app.routers.cloud_import import _persist_cloud_sync_enabled
        await _persist_cloud_sync_enabled(False, db_session)
        stored = await _read_key(db_session, "cloud_sync_enabled")
        assert stored == "false"

    @pytest.mark.asyncio
    async def test_persist_key_is_cloud_sync_enabled(self, db_session):
        from app.routers.cloud_import import _persist_cloud_sync_enabled
        await _persist_cloud_sync_enabled(True, db_session)
        result = await db_session.execute(
            select(AppSettings).where(AppSettings.key == "cloud_sync_enabled")
        )
        row = result.scalar_one_or_none()
        assert row is not None
        assert row.key == "cloud_sync_enabled"
