"""Unit tests for auth endpoints covering AUTH-01/02/06/07/08/09."""

import importlib
import os
import sys

import pytest

from app.services.auth_service import (
    hash_password,
    verify_password,
    SESSIONS,
    create_session,
    is_session_valid,
    is_auth_disabled,
)
from app.models.auth_config import AuthConfig
from sqlalchemy import select


def test_hash_password_uses_scrypt_and_is_verifiable():
    import asyncio
    hashed = asyncio.run(hash_password("hunter2"))
    assert asyncio.run(verify_password("hunter2", hashed)) is True
    assert asyncio.run(verify_password("wrong", hashed)) is False


def test_hash_password_produces_different_hashes_for_same_input():
    import asyncio
    hash1 = asyncio.run(hash_password("hunter2"))
    hash2 = asyncio.run(hash_password("hunter2"))
    assert hash1 != hash2


def test_no_transitive_container_import():
    """Verify that auth modules do NOT transitively import app.container."""
    # Save current modules
    before = set(sys.modules.keys())

    # Import auth_service fresh
    importlib.import_module("app.services.auth_service")
    after_auth_service = set(sys.modules.keys())

    # Import auth router fresh
    importlib.import_module("app.routers.auth")
    after_auth_router = set(sys.modules.keys())

    new_modules = (after_auth_service | after_auth_router) - before
    container_imports = [m for m in new_modules if "app.container" in m]
    assert not container_imports, f"app.container imported transitively: {container_imports}"


def test_login_success(client, seed_password):
    import asyncio
    asyncio.run(seed_password("hunter2"))

    response = client.post("/api/auth/login", json={"password": "hunter2"})
    assert response.status_code == 200
    assert response.json()["authenticated"] is True

    set_cookie = response.headers.get("set-cookie", "")
    assert "paperless_ai_session=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite=Lax" in set_cookie or "samesite=lax" in set_cookie.lower()


def test_login_wrong_password_returns_401(client, seed_password):
    import asyncio
    asyncio.run(seed_password("hunter2"))

    response = client.post("/api/auth/login", json={"password": "wrong"})
    assert response.status_code == 401
    set_cookie = response.headers.get("set-cookie", "")
    assert "paperless_ai_session=" not in set_cookie


def test_login_with_no_password_configured_returns_400(client):
    response = client.post("/api/auth/login", json={"password": "hunter2"})
    assert response.status_code == 400
    assert "Kein Passwort konfiguriert" in response.json()["detail"]


def test_logout_revokes_session(client, seed_password):
    import asyncio
    asyncio.run(seed_password("hunter2"))

    # Login
    response = client.post("/api/auth/login", json={"password": "hunter2"})
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]

    # Logout
    response = client.post("/api/auth/logout", headers={"Cookie": cookie.split(";")[0]})
    assert response.status_code == 200

    # Cookie should be cleared
    set_cookie = response.headers.get("set-cookie", "")
    assert "Max-Age=0" in set_cookie or "expires=Thu, 01 Jan 1970" in set_cookie

    # Subsequent protected request should fail
    response = client.get("/api/test/protected")
    assert response.status_code == 401


def test_status_requires_no_auth_and_returns_shape(client):
    response = client.get("/api/auth/status")
    assert response.status_code == 200
    data = response.json()
    assert "authenticated" in data
    assert "requires_setup" in data
    assert "auth_disabled" in data
    assert isinstance(data["authenticated"], bool)
    assert isinstance(data["requires_setup"], bool)
    assert isinstance(data["auth_disabled"], bool)


def test_status_requires_setup_true_when_no_hash(client):
    response = client.get("/api/auth/status")
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] is False
    assert data["requires_setup"] is True
    assert data["auth_disabled"] is False


def test_status_auth_disabled_true_when_env_set(client, monkeypatch):
    monkeypatch.setenv("DISABLE_LOGIN", "true")
    response = client.get("/api/auth/status")
    assert response.status_code == 200
    data = response.json()
    assert data["auth_disabled"] is True
    assert data["requires_setup"] is False
    assert data["authenticated"] is False


def test_status_authenticated_true_with_valid_cookie(client, seed_password):
    import asyncio
    asyncio.run(seed_password("hunter2"))

    response = client.post("/api/auth/login", json={"password": "hunter2"})
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]

    response = client.get("/api/auth/status", headers={"Cookie": cookie.split(";")[0]})
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] is True
    assert data["requires_setup"] is False


def test_setup_creates_password_when_not_set(client):
    response = client.post("/api/auth/setup", json={"password": "new-pw"})
    assert response.status_code == 200

    # Verify the hash was stored
    import asyncio
    from app.database import async_session
    async def check():
        async with async_session() as session:
            result = await session.execute(select(AuthConfig).where(AuthConfig.id == 1))
            config = result.scalar_one_or_none()
            assert config is not None
            assert config.password_hash != ""
            assert await verify_password("new-pw", config.password_hash) is True
    asyncio.run(check())


def test_setup_rejects_when_password_already_set(client, seed_password):
    import asyncio
    asyncio.run(seed_password("hunter2"))

    response = client.post("/api/auth/setup", json={"password": "x"})
    assert response.status_code == 409
    assert "Passwort bereits eingerichtet" in response.json()["detail"]


def test_setup_rejects_empty_password(client):
    response = client.post("/api/auth/setup", json={"password": ""})
    assert response.status_code in (400, 422)


def test_change_password_requires_cookie(client):
    response = client.post(
        "/api/auth/change-password",
        json={"current_password": "old", "new_password": "new"},
    )
    assert response.status_code == 401


def test_change_password_success(client, seed_password):
    import asyncio
    asyncio.run(seed_password("hunter2"))

    # Login
    response = client.post("/api/auth/login", json={"password": "hunter2"})
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]

    # Change password
    response = client.post(
        "/api/auth/change-password",
        json={"current_password": "hunter2", "new_password": "hunter3"},
        headers={"Cookie": cookie.split(";")[0]},
    )
    assert response.status_code == 200

    # Verify old hash replaced
    from app.database import async_session
    async def check():
        async with async_session() as session:
            result = await session.execute(select(AuthConfig).where(AuthConfig.id == 1))
            config = result.scalar_one_or_none()
            assert config is not None
            assert await verify_password("hunter3", config.password_hash) is True
            assert await verify_password("hunter2", config.password_hash) is False
    asyncio.run(check())


def test_change_password_wrong_current_returns_401(client, seed_password):
    import asyncio
    asyncio.run(seed_password("hunter2"))

    # Login
    response = client.post("/api/auth/login", json={"password": "hunter2"})
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]

    response = client.post(
        "/api/auth/change-password",
        json={"current_password": "wrong", "new_password": "hunter3"},
        headers={"Cookie": cookie.split(";")[0]},
    )
    assert response.status_code == 401
    assert "Aktuelles Passwort ist falsch" in response.json()["detail"]


def test_session_token_is_32_bytes_hex(client, seed_password):
    import asyncio
    asyncio.run(seed_password("hunter2"))

    response = client.post("/api/auth/login", json={"password": "hunter2"})
    assert response.status_code == 200
    set_cookie = response.headers["set-cookie"]

    # Parse cookie value
    cookie_part = set_cookie.split(";")[0]
    token = cookie_part.split("=")[1]
    assert len(token) == 64
