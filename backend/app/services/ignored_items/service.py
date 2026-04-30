"""Ignored items service — manages items excluded from analyses."""

from __future__ import annotations

from typing import Callable, List, Optional

from sqlalchemy import select, and_, delete as sa_delete


class IgnoredItemsServiceImpl:
    """Service for managing ignored items in analyses."""

    def __init__(self, session_factory: Callable):
        self._session_factory = session_factory

    async def list_ignored(
        self,
        entity_type: Optional[str] = None,
        analysis_type: Optional[str] = None,
    ) -> list:
        """Get all ignored items, optionally filtered."""
        from app.models.settings_model import IgnoredItem

        async with self._session_factory() as db:
            query = select(IgnoredItem)
            if entity_type:
                query = query.where(IgnoredItem.entity_type == entity_type)
            if analysis_type:
                query = query.where(IgnoredItem.analysis_type == analysis_type)
            query = query.order_by(IgnoredItem.created_at.desc())
            result = await db.execute(query)
            items = result.scalars().all()
            return [
                {
                    "id": item.id,
                    "item_id": item.item_id,
                    "item_name": item.item_name,
                    "entity_type": item.entity_type,
                    "analysis_type": item.analysis_type,
                    "reason": item.reason or "",
                    "created_at": item.created_at.isoformat() if item.created_at else "",
                }
                for item in items
            ]

    async def add_ignored(
        self,
        item_id: int,
        item_name: str,
        entity_type: str,
        analysis_type: str,
        reason: str = "",
    ) -> dict:
        """Add an item to the ignore list."""
        from app.models.settings_model import IgnoredItem

        async with self._session_factory() as db:
            existing = await db.execute(
                select(IgnoredItem).where(
                    and_(
                        IgnoredItem.item_id == item_id,
                        IgnoredItem.entity_type == entity_type,
                        IgnoredItem.analysis_type == analysis_type,
                    )
                )
            )
            if existing.scalar_one_or_none():
                raise ValueError("Item ist bereits auf der Ignorierliste")

            item = IgnoredItem(
                item_id=item_id,
                item_name=item_name,
                entity_type=entity_type,
                analysis_type=analysis_type,
                reason=reason or "",
            )
            db.add(item)
            await db.commit()
            await db.refresh(item)
            return {
                "id": item.id,
                "item_id": item.item_id,
                "item_name": item.item_name,
                "entity_type": item.entity_type,
                "analysis_type": item.analysis_type,
                "reason": item.reason or "",
                "created_at": item.created_at.isoformat() if item.created_at else "",
            }

    async def remove_ignored(self, item_id: int) -> None:
        """Remove an item from the ignore list."""
        from app.models.settings_model import IgnoredItem

        async with self._session_factory() as db:
            result = await db.execute(
                select(IgnoredItem).where(IgnoredItem.id == item_id)
            )
            item = result.scalar_one_or_none()
            if not item:
                raise ValueError("Item nicht gefunden")
            await db.execute(sa_delete(IgnoredItem).where(IgnoredItem.id == item_id))
            await db.commit()

    async def check_if_ignored(
        self,
        entity_type: str,
        analysis_type: str,
        paperless_item_id: int,
    ) -> dict:
        """Check if a specific item is ignored."""
        from app.models.settings_model import IgnoredItem

        async with self._session_factory() as db:
            result = await db.execute(
                select(IgnoredItem).where(
                    and_(
                        IgnoredItem.item_id == paperless_item_id,
                        IgnoredItem.entity_type == entity_type,
                        IgnoredItem.analysis_type == analysis_type,
                    )
                )
            )
            item = result.scalar_one_or_none()
            return {"ignored": item is not None, "item": item}

    async def get_ignored_ids(
        self,
        entity_type: str,
        analysis_type: str,
    ) -> List[int]:
        """Get list of ignored item IDs for filtering."""
        from app.models.settings_model import IgnoredItem

        async with self._session_factory() as db:
            result = await db.execute(
                select(IgnoredItem.item_id).where(
                    and_(
                        IgnoredItem.entity_type == entity_type,
                        IgnoredItem.analysis_type == analysis_type,
                    )
                )
            )
            return [row[0] for row in result.fetchall()]
