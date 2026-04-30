"""Router for managing ignored items in analyses."""

from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from dishka.integrations.fastapi import inject
from dishka import FromDishka

from app.services.ignored_items import IgnoredItemsService

router = APIRouter(tags=["Ignored Items"])


class IgnoredItemCreate(BaseModel):
    item_id: int
    item_name: str
    entity_type: str
    analysis_type: str
    reason: Optional[str] = ""


class IgnoredItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    item_id: int
    item_name: str
    entity_type: str
    analysis_type: str
    reason: str
    created_at: str


@router.get("")
@inject
async def get_ignored_items(
    entity_type: Optional[str] = None,
    analysis_type: Optional[str] = None,
    service: FromDishka[IgnoredItemsService] = None,
) -> List[IgnoredItemResponse]:
    """Get all ignored items, optionally filtered."""
    assert service is not None
    items = await service.list_ignored(entity_type, analysis_type)
    return [IgnoredItemResponse(**item) for item in items]


@router.post("")
@inject
async def add_ignored_item(
    data: IgnoredItemCreate,
    service: FromDishka[IgnoredItemsService] = None,
) -> IgnoredItemResponse:
    """Add an item to the ignore list."""
    assert service is not None
    try:
        item = await service.add_ignored(
            item_id=data.item_id,
            item_name=data.item_name,
            entity_type=data.entity_type,
            analysis_type=data.analysis_type,
            reason=data.reason or "",
        )
        return IgnoredItemResponse(**item)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{item_id}")
@inject
async def remove_ignored_item(
    item_id: int,
    service: FromDishka[IgnoredItemsService] = None,
):
    """Remove an item from the ignore list."""
    assert service is not None
    try:
        await service.remove_ignored(item_id)
        return {"status": "ok", "message": "Item von Ignorierliste entfernt"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/check/{entity_type}/{analysis_type}/{paperless_item_id}")
@inject
async def check_if_ignored(
    entity_type: str,
    analysis_type: str,
    paperless_item_id: int,
    service: FromDishka[IgnoredItemsService] = None,
) -> dict:
    """Check if a specific item is ignored."""
    assert service is not None
    return await service.check_if_ignored(entity_type, analysis_type, paperless_item_id)


@router.get("/ids/{entity_type}/{analysis_type}")
@inject
async def get_ignored_ids(
    entity_type: str,
    analysis_type: str,
    service: FromDishka[IgnoredItemsService] = None,
) -> List[int]:
    """Get list of ignored item IDs for filtering."""
    assert service is not None
    return await service.get_ignored_ids(entity_type, analysis_type)
