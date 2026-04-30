"""Auth router — implements AUTH-06/07/08/09 and change-password (D-08).

Session cookie contract (D-01/02/03):
- name: "paperless_ai_session"
- httponly=True, samesite="lax", secure=os.getenv("COOKIE_SECURE")=="true"
- no max_age / expires → browser session cookie
"""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from dishka.integrations.fastapi import inject
from dishka import FromDishka

from app.services.auth import AuthService, COOKIE_NAME

router = APIRouter()


class LoginSchema(BaseModel):
    password: str = Field(..., min_length=1)


class SetupSchema(BaseModel):
    password: str = Field(..., min_length=1)


class ChangePasswordSchema(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=1)


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=os.getenv("COOKIE_SECURE", "").lower() == "true",
        path="/",
    )


@router.get("/status")
@inject
async def auth_status(request: Request, service: FromDishka[AuthService] = None):
    """Public: returns {authenticated: bool, requires_setup: bool, auth_disabled: bool}."""
    assert service is not None
    config = await service.get_or_create_auth_config()
    token = request.cookies.get(COOKIE_NAME)
    authenticated = service.is_session_valid(token)
    auth_disabled = service.is_auth_disabled()
    requires_setup = not auth_disabled and not config.get("password_hash")
    return {
        "authenticated": authenticated,
        "requires_setup": requires_setup,
        "auth_disabled": auth_disabled,
    }


@router.post("/login")
@inject
async def login(data: LoginSchema, response: Response, service: FromDishka[AuthService] = None):
    assert service is not None
    config = await service.get_or_create_auth_config()
    if not config.get("password_hash"):
        raise HTTPException(
            status_code=400,
            detail="Kein Passwort konfiguriert. Führen Sie zuerst die Einrichtung aus.",
        )

    ok = await service.validate_password(data.password)
    if not ok:
        raise HTTPException(status_code=401, detail="Falsches Passwort")

    token = service.create_session()
    _set_session_cookie(response, token)
    return {"authenticated": True}


@router.post("/logout")
@inject
async def logout(request: Request, response: Response, service: FromDishka[AuthService] = None):
    assert service is not None
    token = request.cookies.get(COOKIE_NAME)
    if token:
        service.revoke_session(token)
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"success": True}


@router.post("/setup")
@inject
async def setup(data: SetupSchema, response: Response, service: FromDishka[AuthService] = None):
    """First-time setup: create admin password. Idempotent guard: rejects if password already set."""
    assert service is not None
    config = await service.get_or_create_auth_config()
    if config.get("password_hash"):
        raise HTTPException(status_code=409, detail="Passwort bereits eingerichtet.")

    await service.set_password(data.password)
    return {"success": True}


@router.post("/change-password")
@inject
async def change_password(
    data: ChangePasswordSchema,
    request: Request,
    service: FromDishka[AuthService] = None,
):
    """Change password: requires existing session + current password verification."""
    assert service is not None
    config = await service.get_or_create_auth_config()
    if not config.get("password_hash"):
        raise HTTPException(status_code=400, detail="Kein Passwort konfiguriert.")

    ok = await service.change_password(data.current_password, data.new_password)
    if not ok:
        raise HTTPException(status_code=401, detail="Aktuelles Passwort ist falsch.")

    return {"success": True}
