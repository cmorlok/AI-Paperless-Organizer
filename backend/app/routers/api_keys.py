"""API Keys Router — thin endpoints only."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from dishka.integrations.fastapi import inject
from dishka import FromDishka

from app.services.api_keys import ApiKeysService

logger = logging.getLogger(__name__)
router = APIRouter()


class CreateKeyRequest(BaseModel):
    name: str


class ApiKeyResponse(BaseModel):
    id: int
    name: str
    key_prefix: str
    is_active: bool
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None


@router.post("/generate")
@inject
async def generate_api_key(
    request: CreateKeyRequest,
    service: FromDishka[ApiKeysService] = None,
):
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="Name darf nicht leer sein")
    assert service is not None
    return await service.create_key(request.name)


@router.get("/list")
@inject
async def list_api_keys(service: FromDishka[ApiKeysService] = None):
    assert service is not None
    return await service.list_keys()


@router.delete("/{key_id}")
@inject
async def delete_api_key(key_id: int, service: FromDishka[ApiKeysService] = None):
    assert service is not None
    try:
        await service.delete_key(key_id)
        return {"deleted": True}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{key_id}/toggle")
@inject
async def toggle_api_key(key_id: int, service: FromDishka[ApiKeysService] = None):
    assert service is not None
    try:
        return await service.toggle_key(key_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


def extract_api_token(request: Request) -> Optional[str]:
    """Extract API key token from Authorization header only.

    Query parameter api_key is no longer supported per D-21 — API keys
    in query params leak into server logs and browser history.
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    if auth_header.startswith("Api-Key "):
        return auth_header[8:]
    return None
