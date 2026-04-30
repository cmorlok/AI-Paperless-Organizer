"""Statistics Router — thin endpoints only."""

import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.statistics import StatisticsService
from app.services.paperless import PaperlessClient
from dishka.integrations.fastapi import inject
from dishka import FromDishka

logger = logging.getLogger(__name__)
router = APIRouter()


class RecordStatisticRequest(BaseModel):
    entity_type: str
    operation: str
    items_affected: int
    documents_affected: int = 0


@router.post("/record")
@inject
async def record_statistic(request: RecordStatisticRequest, stats_service: FromDishka[StatisticsService] = None):
    assert stats_service is not None
    await stats_service.record_operation(
        entity_type=request.entity_type,
        operation=request.operation,
        items_affected=request.items_affected,
        documents_affected=request.documents_affected,
    )
    return {"success": True}


@router.get("/summary")
@inject
async def get_statistics_summary(
    stats_service: FromDishka[StatisticsService] = None,
    paperless: FromDishka[PaperlessClient] = None,
    db: AsyncSession = Depends(get_db),
):
    assert stats_service is not None
    assert paperless is not None
    return await stats_service.get_statistics_summary(db, paperless)


@router.get("/recent")
@inject
async def get_recent_operations(limit: int = 10, stats_service: FromDishka[StatisticsService] = None):
    assert stats_service is not None
    return await stats_service.get_recent_operations(limit)


@router.get("/trend")
@inject
async def get_daily_trend(days: int = 7, stats_service: FromDishka[StatisticsService] = None):
    assert stats_service is not None
    return await stats_service.get_daily_trend(days)
