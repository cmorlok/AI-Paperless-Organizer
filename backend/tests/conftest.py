import asyncio
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.database import Base


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
async def auth_app(test_session_factory, monkeypatch):
    # Override app.database.async_session BEFORE importing router
    import app.database as db_mod
    monkeypatch.setattr(db_mod, "async_session", test_session_factory)
    # Clear session store between tests
    from app.services.auth_service import SESSIONS
    SESSIONS.clear()

    from app.routers.auth import router as auth_router
    from app.services.auth_service import SessionAuthMiddleware

    app = FastAPI()
    app.add_middleware(SessionAuthMiddleware)
    app.include_router(auth_router, prefix="/api/auth")

    # Dummy public + protected routes for middleware tests
    @app.get("/api/settings/app")
    async def _dummy_settings():
        return {"password_set": False}

    @app.get("/api/test/protected")
    async def _protected():
        return {"ok": True}

    @app.get("/api/health")
    async def _health():
        return {"status": "healthy"}

    yield app


@pytest.fixture
def client(auth_app):
    return TestClient(auth_app)


@pytest_asyncio.fixture
async def seed_password(test_session_factory):
    """Async helper returning a coroutine-like that seeds an AuthConfig row with a Scrypt-hashed password."""
    from app.services.auth_service import hash_password
    from app.models.auth_config import AuthConfig
    from sqlalchemy import select

    async def _seed(plaintext: str):
        hashed = await hash_password(plaintext)
        async with test_session_factory() as session:
            result = await session.execute(select(AuthConfig).where(AuthConfig.id == 1))
            row = result.scalar_one_or_none()
            if row:
                row.password_hash = hashed
            else:
                row = AuthConfig(id=1, password_hash=hashed)
                session.add(row)
            await session.commit()
        return hashed

    return _seed
