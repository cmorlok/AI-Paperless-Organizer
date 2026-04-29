"""Merge service for consolidating entities in Paperless."""

from __future__ import annotations

from typing import List, Dict, Optional, Any
from sqlalchemy import select
from app.core.logging import get_logger
from app.models import MergeHistory, MergeHistoryItem, CleanupStatistics
from app.services.paperless import PaperlessClient
from app.services.cache import get_cache

logger = get_logger(__name__)


class MergeService:
    """Service for merging similar entities in Paperless."""

    def __init__(self, paperless_client: Optional[PaperlessClient] = None, session_factory: Optional[Any] = None):
        self.paperless = paperless_client
        self.session_factory = session_factory

    async def merge_correspondents(
        self,
        target_id: int,
        target_name: str,
        source_ids: List[int]
    ) -> Dict:
        assert self.paperless is not None
        """Merge multiple correspondents into one target."""
        if not source_ids:
            return {"success": False, "error": "No source IDs provided"}

        # Remove target_id from source_ids if present
        source_ids = [sid for sid in source_ids if sid != target_id]

        if not source_ids:
            return {"success": False, "error": "No sources to merge"}

        documents_affected = 0
        merge_items = []

        # Get documents for each source and update them
        for source_id in source_ids:
            # Get correspondent info
            correspondents = await self.paperless.get_correspondents()
            source_info = next((c for c in correspondents if c["id"] == source_id), None)
            source_name = source_info["name"] if source_info else f"ID:{source_id}"

            # Get documents with this correspondent
            documents = await self.paperless.get_documents_by_correspondent(source_id)
            doc_ids = [d["id"] for d in documents]

            if doc_ids:
                # Update documents to use target correspondent
                for doc_id in doc_ids:
                    try:
                        await self.paperless.update_document(doc_id, {"correspondent": target_id})
                        documents_affected += 1
                    except Exception as e:
                        logger.error("Error updating document", extra={"document_id": doc_id, "error": str(e)})

            # Delete the source correspondent
            try:
                await self.paperless.delete_correspondent(source_id)
            except Exception as e:
                logger.error("Error deleting correspondent", extra={"correspondent_id": source_id, "error": str(e)})

            merge_items.append({
                "source_id": source_id,
                "source_name": source_name,
                "document_ids": doc_ids
            })

        if self.session_factory is None:
            return {"success": False, "error": "No session_factory configured"}

        async with self.session_factory() as db:
            # Record merge history
            history = MergeHistory(
                entity_type="correspondents",
                target_id=target_id,
                target_name=target_name,
                merged_count=len(source_ids),
                documents_affected=documents_affected,
                status="completed"
            )
            db.add(history)
            await db.flush()

            for item in merge_items:
                history_item = MergeHistoryItem(
                    merge_history_id=history.id,
                    source_id=item["source_id"],
                    source_name=item["source_name"],
                    document_ids=item["document_ids"]
                )
                db.add(history_item)

            # Record statistics
            stat = CleanupStatistics(
                entity_type="correspondents",
                operation="merged",
                items_affected=len(source_ids),
                documents_affected=documents_affected,
                details={"target_name": target_name, "merged_names": [i["source_name"] for i in merge_items]}
            )
            db.add(stat)

            await db.commit()

        # Clear all caches to ensure fresh data
        cache = get_cache()
        await cache.invalidate_paperless()

        return {
            "success": True,
            "merged_count": len(source_ids),
            "documents_affected": documents_affected,
            "history_id": history.id
        }
    
    async def merge_tags(
        self,
        target_id: int,
        target_name: str,
        source_ids: List[int]
    ) -> Dict:
        assert self.paperless is not None
        """Merge multiple tags into one target."""
        if not source_ids:
            return {"success": False, "error": "No source IDs provided"}

        source_ids = [sid for sid in source_ids if sid != target_id]

        if not source_ids:
            return {"success": False, "error": "No sources to merge"}

        documents_affected = 0
        merge_items = []

        for source_id in source_ids:
            # Get tag info
            tags = await self.paperless.get_tags()
            source_info = next((t for t in tags if t["id"] == source_id), None)
            source_name = source_info["name"] if source_info else f"ID:{source_id}"

            # Get documents with this tag
            documents = await self.paperless.get_documents_by_tag(source_id)
            doc_ids = [d["id"] for d in documents]

            if doc_ids:
                # Update each document: add target tag, remove source tag
                for doc in documents:
                    doc_id = doc["id"]
                    current_tags = doc.get("tags", [])

                    # Add target tag if not present, remove source tag
                    new_tags = [t for t in current_tags if t != source_id]
                    if target_id not in new_tags:
                        new_tags.append(target_id)

                    try:
                        await self.paperless.update_document(doc_id, {"tags": new_tags})
                        documents_affected += 1
                    except Exception as e:
                        logger.error("Error updating document", extra={"document_id": doc_id, "error": str(e)})

            # Delete the source tag
            try:
                await self.paperless.delete_tag(source_id)
            except Exception as e:
                logger.error("Error deleting tag", extra={"tag_id": source_id, "error": str(e)})

            merge_items.append({
                "source_id": source_id,
                "source_name": source_name,
                "document_ids": doc_ids
            })

        if self.session_factory is None:
            return {"success": False, "error": "No session_factory configured"}

        async with self.session_factory() as db:
            # Record merge history
            history = MergeHistory(
                entity_type="tags",
                target_id=target_id,
                target_name=target_name,
                merged_count=len(source_ids),
                documents_affected=documents_affected,
                status="completed"
            )
            db.add(history)
            await db.flush()

            for item in merge_items:
                history_item = MergeHistoryItem(
                    merge_history_id=history.id,
                    source_id=item["source_id"],
                    source_name=item["source_name"],
                    document_ids=item["document_ids"]
                )
                db.add(history_item)

            # Record statistics
            stat = CleanupStatistics(
                entity_type="tags",
                operation="merged",
                items_affected=len(source_ids),
                documents_affected=documents_affected,
                details={"target_name": target_name, "merged_names": [i["source_name"] for i in merge_items]}
            )
            db.add(stat)

            await db.commit()

        # Clear all caches to ensure fresh data
        cache = get_cache()
        await cache.invalidate_paperless()

        return {
            "success": True,
            "merged_count": len(source_ids),
            "documents_affected": documents_affected,
            "history_id": history.id
        }
    
    async def merge_document_types(
        self,
        target_id: int,
        target_name: str,
        source_ids: List[int]
    ) -> Dict:
        assert self.paperless is not None
        """Merge multiple document types into one target."""
        if not source_ids:
            return {"success": False, "error": "No source IDs provided"}

        source_ids = [sid for sid in source_ids if sid != target_id]

        if not source_ids:
            return {"success": False, "error": "No sources to merge"}

        documents_affected = 0
        merge_items = []

        for source_id in source_ids:
            # Get document type info
            doc_types = await self.paperless.get_document_types()
            source_info = next((dt for dt in doc_types if dt["id"] == source_id), None)
            source_name = source_info["name"] if source_info else f"ID:{source_id}"

            # Get documents with this document type
            documents = await self.paperless.get_documents_by_document_type(source_id)
            doc_ids = [d["id"] for d in documents]

            if doc_ids:
                # Update documents to use target document type
                for doc_id in doc_ids:
                    try:
                        await self.paperless.update_document(doc_id, {"document_type": target_id})
                        documents_affected += 1
                    except Exception as e:
                        logger.error("Error updating document", extra={"document_id": doc_id, "error": str(e)})

            # Delete the source document type
            try:
                await self.paperless.delete_document_type(source_id)
            except Exception as e:
                logger.error("Error deleting document type", extra={"doc_type_id": source_id, "error": str(e)})

            merge_items.append({
                "source_id": source_id,
                "source_name": source_name,
                "document_ids": doc_ids
            })

        if self.session_factory is None:
            return {"success": False, "error": "No session_factory configured"}

        async with self.session_factory() as db:
            # Record merge history
            history = MergeHistory(
                entity_type="document_types",
                target_id=target_id,
                target_name=target_name,
                merged_count=len(source_ids),
                documents_affected=documents_affected,
                status="completed"
            )
            db.add(history)
            await db.flush()

            for item in merge_items:
                history_item = MergeHistoryItem(
                    merge_history_id=history.id,
                    source_id=item["source_id"],
                    source_name=item["source_name"],
                    document_ids=item["document_ids"]
                )
                db.add(history_item)

            # Record statistics
            stat = CleanupStatistics(
                entity_type="document_types",
                operation="merged",
                items_affected=len(source_ids),
                documents_affected=documents_affected,
                details={"target_name": target_name, "merged_names": [i["source_name"] for i in merge_items]}
            )
            db.add(stat)

            await db.commit()

        # Clear all caches to ensure fresh data
        cache = get_cache()
        await cache.invalidate_paperless()

        return {
            "success": True,
            "merged_count": len(source_ids),
            "documents_affected": documents_affected,
            "history_id": history.id
        }
    
    async def get_history(self, entity_type: str | None = None) -> List[Dict]:
        """Get merge history, optionally filtered by entity type."""
        if self.session_factory is None:
            return []

        async with self.session_factory() as db:
            query = select(MergeHistory).order_by(MergeHistory.created_at.desc())

            if entity_type:
                query = query.where(MergeHistory.entity_type == entity_type)

            result = await db.execute(query)
            histories = result.scalars().all()

            return [
                {
                    "id": h.id,
                    "entity_type": h.entity_type,
                    "target_id": h.target_id,
                    "target_name": h.target_name,
                    "merged_count": h.merged_count,
                    "documents_affected": h.documents_affected,
                    "status": h.status,
                    "created_at": h.created_at.isoformat() if h.created_at else None
                }
                for h in histories
            ]