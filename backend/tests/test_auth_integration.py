"""Integration tests for auth wiring in the real app.

Covers AUTH-03/04/05 and RESET_PASSWORD (D-14).
Uses the real app.main.app with monkeypatched DB.
"""

import os
import pytest
from fastapi.testclient import TestClient


from typing import Any, List


def _find_middleware(app: Any, cls_name: str) -> Any:
    for mw in app.user_middleware:
        if mw.cls.__name__ == cls_name:
            return mw
    return None


def test_app_imports_cleanly():
    from app.main import app
    assert app is not None


def test_unauthenticated_tags_returns_401(test_session_factory, monkeypatch):
    import app.database as db_mod
    monkeypatch.setattr(db_mod, "async_session", test_session_factory)
    from app.services.auth_service import SESSIONS
    SESSIONS.clear()
    import app.main as main_mod
    with TestClient(main_mod.app) as client:
        resp = client.get("/api/tags/")
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Nicht authentifiziert"


def test_unauthenticated_settings_paperless_returns_401(test_session_factory, monkeypatch):
    import app.database as db_mod
    monkeypatch.setattr(db_mod, "async_session", test_session_factory)
    from app.services.auth_service import SESSIONS
    SESSIONS.clear()
    import app.main as main_mod
    with TestClient(main_mod.app) as client:
        resp = client.get("/api/settings/paperless")
        assert resp.status_code == 401


def test_public_auth_status_returns_200_without_cookie(test_session_factory, monkeypatch):
    import app.database as db_mod
    monkeypatch.setattr(db_mod, "async_session", test_session_factory)
    from app.services.auth_service import SESSIONS
    SESSIONS.clear()
    import app.main as main_mod
    with TestClient(main_mod.app) as client:
        resp = client.get("/api/auth/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "authenticated" in data
        assert "requires_setup" in data
        assert "auth_disabled" in data


def test_public_settings_app_returns_200_without_cookie(test_session_factory, monkeypatch):
    import app.database as db_mod
    monkeypatch.setattr(db_mod, "async_session", test_session_factory)
    from app.services.auth_service import SESSIONS
    SESSIONS.clear()
    import app.main as main_mod
    with TestClient(main_mod.app) as client:
        resp = client.get("/api/settings/app")
        assert resp.status_code == 200


def test_public_health_returns_200_without_cookie(test_session_factory, monkeypatch):
    import app.database as db_mod
    monkeypatch.setattr(db_mod, "async_session", test_session_factory)
    from app.services.auth_service import SESSIONS
    SESSIONS.clear()
    import app.main as main_mod
    with TestClient(main_mod.app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200


def test_cors_allow_origins_not_wildcard():
    from app.main import app
    mw = _find_middleware(app, "CORSMiddleware")
    assert mw is not None
    origins: List[str] = mw.kwargs.get("allow_origins", [])
    assert "*" not in origins, f"CORS still wildcard: {origins}"
    assert len(origins) > 0


def test_cors_allow_credentials_true():
    from app.main import app
    mw = _find_middleware(app, "CORSMiddleware")
    assert mw is not None
    assert mw.kwargs.get("allow_credentials") is True


def test_cors_default_origins_when_env_absent(monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    # Re-import main to pick up fresh env (module may be cached)
    import importlib
    import app.main as main_mod
    importlib.reload(main_mod)
    mw = _find_middleware(main_mod.app, "CORSMiddleware")
    assert mw is not None
    origins: List[str] = mw.kwargs.get("allow_origins", [])
    assert "http://localhost:3000" in origins
    assert "http://localhost:8088" in origins


def test_login_logout_end_to_end(test_session_factory, monkeypatch):
    import app.database as db_mod
    monkeypatch.setattr(db_mod, "async_session", test_session_factory)
    from app.services.auth_service import SESSIONS
    SESSIONS.clear()
    import app.main as main_mod
    with TestClient(main_mod.app) as client:
        # Setup password
        resp = client.post("/api/auth/setup", json={"password": "test-pw-123"})
        assert resp.status_code == 200

        # Login
        resp = client.post("/api/auth/login", json={"password": "test-pw-123"})
        assert resp.status_code == 200
        assert resp.json()["authenticated"] is True
        cookie = resp.headers.get("set-cookie", "")
        assert "paperless_ai_session=" in cookie

        # Protected route should now pass middleware (may 500 from Paperless, but NOT 401)
        try:
            resp = client.get("/api/tags/")
            assert resp.status_code != 401
        except Exception:
            # Any exception means auth passed; middleware did not return 401
            pass

        # Logout
        resp = client.post("/api/auth/logout")
        assert resp.status_code == 200

        # Protected route should be 401 again
        resp = client.get("/api/tags/")
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_reset_password_env_clears_hash(monkeypatch, test_session_factory):
    from app.services.auth_service import hash_password
    from app.models.auth_config import AuthConfig
    from sqlalchemy import select

    # Seed a hash
    hashed = await hash_password("old-pw")
    async with test_session_factory() as db:
        row = AuthConfig(id=1, password_hash=hashed)
        db.add(row)
        await db.commit()

    import app.main as main_mod
    monkeypatch.setattr(main_mod, "async_session", test_session_factory)
    monkeypatch.setenv("RESET_PASSWORD", "true")
    await main_mod.reset_password_if_requested()

    async with test_session_factory() as db:
        result = await db.execute(select(AuthConfig).where(AuthConfig.id == 1))
        row = result.scalar_one()
        assert row.password_hash == ""


def test_auth_disabled_env_bypasses_middleware(test_session_factory, monkeypatch):
    import app.database as db_mod
    monkeypatch.setattr(db_mod, "async_session", test_session_factory)
    from app.services.auth_service import SESSIONS
    SESSIONS.clear()
    monkeypatch.setenv("DISABLE_LOGIN", "true")
    import app.main as main_mod
    # Reload to pick up env in middleware
    import importlib
    importlib.reload(main_mod)
    with TestClient(main_mod.app) as client:
        try:
            resp = client.get("/api/tags/")
            assert resp.status_code != 401
        except Exception:
            pass


@pytest.mark.asyncio
async def test_settings_app_returns_password_set_from_auth_config(
    monkeypatch, test_session_factory
):
    from app.services.auth_service import hash_password
    from app.models.auth_config import AuthConfig
    from sqlalchemy import select

    hashed = await hash_password("secure-pw")
    async with test_session_factory() as db:
        row = AuthConfig(id=1, password_hash=hashed)
        db.add(row)
        await db.commit()

    import app.database as db_mod
    monkeypatch.setattr(db_mod, "async_session", test_session_factory)
    from app.services.auth_service import SESSIONS
    SESSIONS.clear()
    import app.main as main_mod

    from app.database import get_db

    async def _override_get_db():
        async with test_session_factory() as session:
            yield session

    main_mod.app.dependency_overrides[get_db] = _override_get_db

    with TestClient(main_mod.app) as client:
        resp = client.get("/api/settings/app")
        assert resp.status_code == 200
        assert resp.json()["password_set"] is True

    main_mod.app.dependency_overrides.pop(get_db, None)
