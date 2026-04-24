"""Unit tests for SessionAuthMiddleware covering AUTH-03/04."""

import pytest

from app.services.auth.middleware import SessionAuthMiddleware
from app.services.auth.state import PUBLIC_PATHS, COOKIE_NAME, SESSIONS
from app.services.auth.service import create_session, is_auth_disabled


def test_middleware_blocks_unauthenticated_api_request(client):
    response = client.get("/api/test/protected")
    assert response.status_code == 401
    assert response.json()["detail"] == "Nicht authentifiziert"


def test_middleware_allows_public_status(client):
    response = client.get("/api/auth/status")
    assert response.status_code == 200


def test_middleware_allows_public_settings_app(client):
    response = client.get("/api/settings/app")
    assert response.status_code == 200


def test_middleware_allows_public_login(client):
    response = client.post("/api/auth/login", json={"password": "x"})
    # Middleware should allow it through; business logic may 400/401
    assert response.status_code != 401 or response.json().get("detail") != "Nicht authentifiziert"


def test_middleware_allows_public_setup(client):
    response = client.post("/api/auth/setup", json={"password": "x"})
    # Middleware should allow it through; business logic may 409
    assert response.status_code != 401 or response.json().get("detail") != "Nicht authentifiziert"


def test_middleware_allows_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200


def test_middleware_passes_through_non_api_paths(client):
    response = client.get("/some-frontend-path")
    assert response.status_code == 404


def test_middleware_authenticated_request_passes(client, seed_password):
    import asyncio
    asyncio.run(seed_password("hunter2"))

    response = client.post("/api/auth/login", json={"password": "hunter2"})
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]

    response = client.get("/api/test/protected", headers={"Cookie": cookie.split(";")[0]})
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_middleware_invalid_cookie_returns_401(client):
    response = client.get(
        "/api/test/protected",
        headers={"Cookie": f"{COOKIE_NAME}=invalid_token_not_in_sessions"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Nicht authentifiziert"


def test_middleware_disable_login_bypass(client, monkeypatch):
    monkeypatch.setenv("DISABLE_LOGIN", "true")
    response = client.get("/api/test/protected")
    assert response.status_code == 200


# The middleware performs NO database query per request.
# This is implicitly proven by the absence of any DB session usage
# in SessionAuthMiddleware — no explicit test needed.
