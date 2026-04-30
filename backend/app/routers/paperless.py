"""Paperless Router — thin endpoints only."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.paperless import PaperlessClient
from dishka.integrations.fastapi import inject
from dishka import FromDishka

router = APIRouter()


@router.get("/status")
@inject
async def get_paperless_status(client: FromDishka[PaperlessClient] = None):
    assert client is not None
    if not client.base_url:
        return {"connected": False, "error": "Keine URL konfiguriert"}
    try:
        is_connected = await client.test_connection()
        return {"connected": is_connected, "url": client.base_url}
    except Exception as e:
        return {"connected": False, "url": client.base_url, "error": str(e)}


@router.get("/correspondents")
@inject
async def get_correspondents(client: FromDishka[PaperlessClient] = None):
    assert client is not None
    return await client.get_correspondents()


@router.get("/tags")
@inject
async def get_tags(client: FromDishka[PaperlessClient] = None, db: AsyncSession = Depends(get_db)):
    assert client is not None
    return await client.get_tags_cached(db)


@router.get("/document-types")
@inject
async def get_document_types(client: FromDishka[PaperlessClient] = None):
    assert client is not None
    return await client.get_document_types()


@router.get("/documents")
@inject
async def get_documents(
    correspondent_id: int | None = None,
    tag_id: int | None = None,
    document_type_id: int | None = None,
    client: FromDishka[PaperlessClient] = None,
):
    assert client is not None
    return await client.get_documents(
        correspondent_id=correspondent_id,
        tag_id=tag_id,
        document_type_id=document_type_id,
    )


@router.post("/refresh-cache")
@inject
async def refresh_cache(client: FromDishka[PaperlessClient] = None, db: AsyncSession = Depends(get_db)):
    assert client is not None
    return await client.refresh_cache(db)


@router.get("/document-previews")
@inject
async def get_document_previews(
    correspondent_id: int | None = None,
    tag_id: int | None = None,
    document_type_id: int | None = None,
    limit: int = 5,
    client: FromDishka[PaperlessClient] = None,
):
    assert client is not None
    return await client.get_document_previews(
        correspondent_id=correspondent_id,
        tag_id=tag_id,
        document_type_id=document_type_id,
        limit=limit,
    )
