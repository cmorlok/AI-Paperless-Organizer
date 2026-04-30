"""Tests for AppSettings KV persistence of enabled flags (STATE-08).

Verifies:
- auto_classify_enabled persistence via ConfigService
- ocr_processor_enabled persistence via ConfigService
- cloud_sync_enabled persistence via ConfigService
- ConfigService.get returns None when key is not found
- ConfigService.set creates/updates rows correctly
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select

from app.database import Base
from app.models.settings_model import AppSettings
from app.services.config.service import ConfigServiceImpl


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
async def config_service(test_session_factory):
    return ConfigServiceImpl(session_factory=test_session_factory)


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


# ── Tests for ConfigService ──────────────────────────────────────────────────

class TestConfigService:
    @pytest.mark.asyncio
    async def test_returns_none_when_key_not_found(self, config_service):
        result = await config_service.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_value_when_key_exists(self, config_service):
        await config_service.set("test_key", "test_value", "str")
        result = await config_service.get("test_key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_creates_row_when_key_not_exists(self, config_service, test_session_factory):
        await config_service.set("new_key", "new_value", "str")
        async with test_session_factory() as session:
            stored = await _read_key(session, "new_key")
        assert stored == "new_value"

    @pytest.mark.asyncio
    async def test_updates_row_when_key_already_exists(self, config_service, test_session_factory):
        await config_service.set("update_key", "initial", "str")
        await config_service.set("update_key", "updated", "str")
        async with test_session_factory() as session:
            stored = await _read_key(session, "update_key")
        assert stored == "updated"

    @pytest.mark.asyncio
    async def test_creates_exactly_one_row_for_key(self, config_service, test_session_factory):
        await config_service.set("unique_key", "v1", "str")
        await config_service.set("unique_key", "v2", "str")
        async with test_session_factory() as session:
            result = await session.execute(
                select(AppSettings).where(AppSettings.key == "unique_key")
            )
            rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].value == "v2"


# ── Tests for auto_classify_enabled persistence via ConfigService ─────────────

class TestPersistAutoClassifyEnabled:
    @pytest.mark.asyncio
    async def test_persist_true_writes_true_string(self, config_service, test_session_factory):
        await config_service.set("auto_classify_enabled", "true", "bool")
        async with test_session_factory() as session:
            stored = await _read_key(session, "auto_classify_enabled")
        assert stored == "true"

    @pytest.mark.asyncio
    async def test_persist_false_writes_false_string(self, config_service, test_session_factory):
        await config_service.set("auto_classify_enabled", "false", "bool")
        async with test_session_factory() as session:
            stored = await _read_key(session, "auto_classify_enabled")
        assert stored == "false"

    @pytest.mark.asyncio
    async def test_persist_key_is_auto_classify_enabled(self, config_service, test_session_factory):
        await config_service.set("auto_classify_enabled", "true", "bool")
        async with test_session_factory() as session:
            result = await session.execute(
                select(AppSettings).where(AppSettings.key == "auto_classify_enabled")
            )
            row = result.scalar_one_or_none()
        assert row is not None
        assert row.key == "auto_classify_enabled"


# ── Tests for ocr_processor_enabled persistence via ConfigService ─────────────

class TestPersistOcrProcessorEnabled:
    @pytest.mark.asyncio
    async def test_persist_true_writes_true_string(self, config_service, test_session_factory):
        await config_service.set("ocr_processor_enabled", "true", "bool")
        async with test_session_factory() as session:
            stored = await _read_key(session, "ocr_processor_enabled")
        assert stored == "true"

    @pytest.mark.asyncio
    async def test_persist_false_writes_false_string(self, config_service, test_session_factory):
        await config_service.set("ocr_processor_enabled", "false", "bool")
        async with test_session_factory() as session:
            stored = await _read_key(session, "ocr_processor_enabled")
        assert stored == "false"

    @pytest.mark.asyncio
    async def test_persist_key_is_ocr_processor_enabled(self, config_service, test_session_factory):
        await config_service.set("ocr_processor_enabled", "true", "bool")
        async with test_session_factory() as session:
            result = await session.execute(
                select(AppSettings).where(AppSettings.key == "ocr_processor_enabled")
            )
            row = result.scalar_one_or_none()
        assert row is not None
        assert row.key == "ocr_processor_enabled"


# ── Tests for cloud_sync_enabled persistence via ConfigService ────────────────

class TestPersistCloudSyncEnabled:
    @pytest.mark.asyncio
    async def test_persist_true_writes_true_string(self, config_service, test_session_factory):
        await config_service.set("cloud_sync_enabled", "true", "bool")
        async with test_session_factory() as session:
            stored = await _read_key(session, "cloud_sync_enabled")
        assert stored == "true"

    @pytest.mark.asyncio
    async def test_persist_false_writes_false_string(self, config_service, test_session_factory):
        await config_service.set("cloud_sync_enabled", "false", "bool")
        async with test_session_factory() as session:
            stored = await _read_key(session, "cloud_sync_enabled")
        assert stored == "false"

    @pytest.mark.asyncio
    async def test_persist_key_is_cloud_sync_enabled(self, config_service, test_session_factory):
        await config_service.set("cloud_sync_enabled", "true", "bool")
        async with test_session_factory() as session:
            result = await session.execute(
                select(AppSettings).where(AppSettings.key == "cloud_sync_enabled")
            )
            row = result.scalar_one_or_none()
        assert row is not None
        assert row.key == "cloud_sync_enabled"
