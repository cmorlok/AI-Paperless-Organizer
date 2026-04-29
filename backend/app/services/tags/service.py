"""Tags business-logic service — estimation, deletion, analysis persistence."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from sqlalchemy import select, delete
from app.models import SavedAnalysis, PaperlessCache
from app.services.paperless import PaperlessClient
from app.services.llm import LLMService
from app.services.config import ConfigService
from app.services.statistics import StatisticsService
from app.models.settings_model import LLM_KEY_CLASSIFIER_PROVIDER, LLM_KEY_CLASSIFIER_MODEL


class TagsServiceImpl:
    """Business-logic operations for tags — called by the thin router."""

    def __init__(
        self,
        paperless_client: PaperlessClient,
        llm_service: LLMService,
        config_service: ConfigService,
        statistics_service: StatisticsService,
        session_factory: Any,
    ):
        self._client = paperless_client
        self._llm = llm_service
        self._config = config_service
        self._stats = statistics_service
        self._session_factory = session_factory

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    async def estimate_tags(
        self,
        analysis_type: str = "nonsense",
    ) -> Dict:
        """Estimate tokens needed for a specific analysis type."""
        tags = await self._client.get_tags_with_counts()
        tags_count = len(tags)
        avg_tag_length = sum(len(t.get("name", "")) for t in tags) / max(tags_count, 1)

        prompt_sizes = {
            "nonsense": 800,
            "correspondent": 1200,
            "doctype": 1000,
            "similar": 600,
        }
        base_prompt = prompt_sizes.get(analysis_type, 800)

        if analysis_type == "correspondent":
            correspondents = await self._client.get_correspondents()
            corr_count = len(correspondents)
            avg_corr_length = sum(len(c.get("name", "")) for c in correspondents) / max(corr_count, 1)
            estimated_chars = base_prompt + tags_count * (avg_tag_length + 10) + corr_count * (avg_corr_length + 5)
            items_info = f"{tags_count} Tags + {corr_count} Korrespondenten"
        elif analysis_type == "doctype":
            doc_types = await self._client.get_document_types()
            dt_count = len(doc_types)
            avg_dt_length = sum(len(d.get("name", "")) for d in doc_types) / max(dt_count, 1)
            estimated_chars = base_prompt + tags_count * (avg_tag_length + 10) + dt_count * (avg_dt_length + 5)
            items_info = f"{tags_count} Tags + {dt_count} Dokumenttypen"
        else:
            estimated_chars = base_prompt + tags_count * (avg_tag_length + 10)
            items_info = f"{tags_count} Tags"

        estimated_tokens = estimated_chars // 4

        provider = await self._config.get(LLM_KEY_CLASSIFIER_PROVIDER) or ""
        model = await self._config.get(LLM_KEY_CLASSIFIER_MODEL) or ""
        token_limit = await self._llm.get_token_limit(provider, model)
        is_cloud = not self._llm.is_local_provider(provider)
        safe_limit = int(token_limit * 0.8)
        needs_batching = estimated_tokens > safe_limit
        recommended_batches = max(1, (estimated_tokens + safe_limit - 1) // safe_limit) if needs_batching else 1

        return {
            "analysis_type": analysis_type,
            "items_info": items_info,
            "estimated_tokens": estimated_tokens,
            "token_limit": token_limit,
            "is_cloud": is_cloud,
            "recommended_batches": recommended_batches,
            "warning": f"~{estimated_tokens:,} Tokens > {safe_limit:,} Limit. Wird in {recommended_batches} Batches aufgeteilt." if needs_batching else None,
        }

    # ------------------------------------------------------------------
    # Empty-tag deletion (parallel with stats recording)
    # ------------------------------------------------------------------

    async def delete_empty_tags(self) -> Dict:
        """Delete all tags with 0 documents — parallel with batch of 10."""
        tags = await self._client.get_tags_with_counts()
        empty = [t for t in tags if t.get("document_count", 0) == 0]

        if not empty:
            return {"deleted": 0, "total": 0, "errors": None}

        errors: List[str] = []
        deleted = 0
        batch_size = 10

        async def _delete_one(tag: dict):
            try:
                await self._client.delete_tag(tag["id"])
                return True, None
            except Exception as e:
                return False, f"{tag['name']}: {e}"

        for i in range(0, len(empty), batch_size):
            batch = empty[i : i + batch_size]
            results = await asyncio.gather(*[_delete_one(t) for t in batch])
            for success, error in results:
                if success:
                    deleted += 1
                elif error:
                    errors.append(error)

        if deleted > 0:
            await self._stats.record_operation(
                entity_type="tags",
                operation="deleted",
                items_affected=deleted,
                documents_affected=0,
                items_before=len(tags),
                items_after=len(tags) - deleted,
            )

        return {"deleted": deleted, "total": len(empty), "errors": errors or None}

    # ------------------------------------------------------------------
    # Bulk delete (parallel + cache update)
    # ------------------------------------------------------------------

    async def bulk_delete_tags(self, tag_ids: List[int]) -> Dict:
        """Bulk-delete tags in parallel and keep the DB cache in sync."""
        result = await self._client.delete_tags_bulk(tag_ids)

        deleted_set = set(result.get("deleted", []))
        if deleted_set:
            async with self._session_factory() as db:
                db_result = await db.execute(
                    select(PaperlessCache).where(PaperlessCache.cache_key == "tags")
                )
                db_entry = db_result.scalar_one_or_none()
                if db_entry and db_entry.data:
                    db_entry.data = [t for t in db_entry.data if t.get("id") not in deleted_set]
                    db_entry.count = len(db_entry.data)
                    await db.commit()

        return result

    # ------------------------------------------------------------------
    # Remove tags from ALL saved analyses
    # ------------------------------------------------------------------

    async def remove_tags_from_saved_analyses(self, tag_ids: List[int]) -> Dict:
        """Remove deleted tag IDs from every saved analysis so they don't reappear."""
        removed_ids = set(tag_ids)
        updated: Dict[str, Dict] = {}

        analysis_types = {
            "tags_nonsense": {"id_field": "id", "key": "nonsense"},
            "tags_correspondents": {"id_field": "tag_id", "key": "correspondent"},
            "tags_doctypes": {"id_field": "tag_id", "key": "doctype"},
        }

        async with self._session_factory() as db:
            for entity_type, config in analysis_types.items():
                result = await db.execute(
                    select(SavedAnalysis)
                    .where(SavedAnalysis.entity_type == entity_type)
                    .order_by(SavedAnalysis.created_at.desc())
                    .limit(1)
                )
                saved = result.scalar_one_or_none()
                if not saved or not saved.groups:
                    continue

                id_field = config["id_field"]
                original_count = len(saved.groups)
                filtered = [item for item in saved.groups if item.get(id_field) not in removed_ids]

                if len(filtered) < original_count:
                    saved.groups = filtered
                    saved.groups_count = len(filtered)
                    updated[config["key"]] = {"before": original_count, "after": len(filtered)}

            await db.commit()

        return {"success": True, "updated": updated}

    # ------------------------------------------------------------------
    # Analysis persistence — four specialised methods
    # ------------------------------------------------------------------

    async def save_similarity_analysis(self, result: Dict) -> None:
        """Persist a similarity analysis result (tags ↔ tags)."""
        groups = result.get("groups", [])
        stats = result.get("stats", {})
        await self._persist("tags", "similarity", groups, stats, stats.get("items_count", 0))

    async def save_nonsense_analysis(self, result: Dict) -> None:
        """Persist a nonsense-tag analysis result."""
        nonsense_tags = result.get("nonsense_tags", [])
        stats = result.get("stats", {})
        items_count = stats.get("analyzed_count", len(nonsense_tags))
        await self._persist("tags_nonsense", "nonsense", nonsense_tags, stats, items_count)

    async def save_correspondent_matches(self, result: Dict) -> None:
        """Persist a correspondent-matches analysis result."""
        correspondent_tags = result.get("correspondent_tags", [])
        stats = result.get("stats", {})
        items_count = stats.get("tags_count", len(correspondent_tags))
        await self._persist("tags_correspondents", "correspondent_matches", correspondent_tags, stats, items_count)

    async def save_doctype_matches(self, result: Dict) -> None:
        """Persist a doctype-matches analysis result."""
        doctype_tags = result.get("doctype_tags", [])
        stats = result.get("stats", {})
        items_count = stats.get("tags_count", len(doctype_tags))
        await self._persist("tags_doctypes", "doctype_matches", doctype_tags, stats, items_count)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _persist(
        self,
        entity_type: str,
        analysis_type: str,
        groups: list,
        stats: dict,
        items_count: int,
    ) -> None:
        """Delete old analysis of this type and insert a fresh one."""
        async with self._session_factory() as db:
            await db.execute(delete(SavedAnalysis).where(SavedAnalysis.entity_type == entity_type))
            db.add(
                SavedAnalysis(
                    entity_type=entity_type,
                    analysis_type=analysis_type,
                    groups=groups,
                    stats=stats,
                    items_count=items_count,
                    groups_count=len(groups),
                    processed_groups=[],
                )
            )
            await db.commit()
