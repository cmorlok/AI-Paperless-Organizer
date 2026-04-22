"""Auth router — implements AUTH-06/07/08/09 and change-password (D-08).

Session cookie contract (D-01/02/03):
- name: "paperless_ai_session"
- httponly=True, samesite="lax", secure=os.getenv("COOKIE_SECURE")=="true"
- no max_age / expires → browser session cookie
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.auth_config import AuthConfig
from app.services.auth_service import (
    COOKIE_NAME,
    create_session,
    hash_password,
    is_auth_disabled,
    is_session_valid,
    revoke_session,
    verify_password,
)

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


async def _get_or_create_auth_config(db: AsyncSession) -> AuthConfig:
    result = await db.execute(select(AuthConfig).where(AuthConfig.id == 1))
    config = result.scalar_one_or_none()
    if config is None:
        config = AuthConfig(id=1, password_hash="")
        db.add(config)
        await db.commit()
        await db.refresh(config)
    return config


@router.get("/status")
async def auth_status(request: Request, db: AsyncSession = Depends(get_db)):
    """Public: returns {authenticated: bool, requires_setup: bool, auth_disabled: bool}."""
    config = await _get_or_create_auth_config(db)
    token = request.cookies.get(COOKIE_NAME)
    authenticated = is_session_valid(token)
    auth_disabled = is_auth_disabled()
    # When auth is disabled, setup is never required.
    requires_setup = not auth_disabled and not config.password_hash
    return {
        "authenticated": authenticated,
        "requires_setup": requires_setup,
        "auth_disabled": auth_disabled,
    }


@router.post("/login")
async def login(data: LoginSchema, response: Response, db: AsyncSession = Depends(get_db)):
    config = await _get_or_create_auth_config(db)
    if not config.password_hash:
        raise HTTPException(
            status_code=400,
            detail="Kein Passwort konfiguriert. Führen Sie zuerst die Einrichtung aus.",
        )

    ok = await verify_password(data.password, config.password_hash)
    if not ok:
        raise HTTPException(status_code=401, detail="Falsches Passwort")

    token = create_session()
    _set_session_cookie(response, token)
    return {"authenticated": True}


@router.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        revoke_session(token)
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"success": True}


@router.post("/setup")
async def setup(data: SetupSchema, response: Response, db: AsyncSession = Depends(get_db)):
    """First-time setup: create admin password. Idempotent guard: rejects if password already set."""
    config = await _get_or_create_auth_config(db)
    if config.password_hash:
        raise HTTPException(status_code=409, detail="Passwort bereits eingerichtet.")

    config.password_hash = await hash_password(data.password)
    await db.commit()
    return {"success": True}


@router.post("/change-password")
async def change_password(
    data: ChangePasswordSchema,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Change password: requires existing session + current password verification.

    Note: session validity is enforced by SessionAuthMiddleware; no redundant check here.
    """
    config = await _get_or_create_auth_config(db)
    if not config.password_hash:
        raise HTTPException(status_code=400, detail="Kein Passwort konfiguriert.")

    ok = await verify_password(data.current_password, config.password_hash)
    if not ok:
        raise HTTPException(status_code=401, detail="Aktuelles Passwort ist falsch.")

    config.password_hash = await hash_password(data.new_password)
    await db.commit()
    return {"success": True}
