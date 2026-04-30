"""Shared helpers for entity analysis services (correspondents, document_types, tags)."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import select, delete
from app.models import SavedAnalysis


async def persist_analysis(
    session_factory: Any,
    entity_type: str,
    analysis_type: str,
    groups: list,
    stats: dict,
    items_count: int,
) -> None:
    """Delete old analysis of this type and insert a fresh one."""
    async with session_factory() as db:
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


async def get_saved_analysis(session_factory: Any, entity_type: str) -> dict:
    """Check if there's a saved analysis for the given entity type."""
    async with session_factory() as db:
        result = await db.execute(
            select(SavedAnalysis)
            .where(SavedAnalysis.entity_type == entity_type)
            .order_by(SavedAnalysis.created_at.desc())
            .limit(1)
        )
        saved = result.scalar_one_or_none()
        if saved:
            return {
                "exists": True,
                "id": saved.id,
                "created_at": saved.created_at.isoformat() if saved.created_at else None,
                "items_count": saved.items_count,
                "groups_count": saved.groups_count,
                "processed_groups": saved.processed_groups or [],
            }
        return {"exists": False}


async def load_saved_analysis(session_factory: Any, entity_type: str) -> Optional[dict]:
    """Load the saved analysis results."""
    async with session_factory() as db:
        result = await db.execute(
            select(SavedAnalysis)
            .where(SavedAnalysis.entity_type == entity_type)
            .order_by(SavedAnalysis.created_at.desc())
            .limit(1)
        )
        saved = result.scalar_one_or_none()
        if not saved:
            return None
        return {
            "groups": saved.groups,
            "stats": saved.stats,
            "created_at": saved.created_at.isoformat() if saved.created_at else None,
            "processed_groups": saved.processed_groups or [],
        }


async def delete_saved_analysis(session_factory: Any, entity_type: str) -> None:
    """Delete saved analysis for the given entity type."""
    async with session_factory() as db:
        await db.execute(delete(SavedAnalysis).where(SavedAnalysis.entity_type == entity_type))
        await db.commit()


async def mark_group_processed(session_factory: Any, entity_type: str, group_index: int) -> None:
    """Mark a group as processed (merged or dismissed)."""
    async with session_factory() as db:
        result = await db.execute(
            select(SavedAnalysis)
            .where(SavedAnalysis.entity_type == entity_type)
            .order_by(SavedAnalysis.created_at.desc())
            .limit(1)
        )
        saved = result.scalar_one_or_none()
        if saved:
            processed = saved.processed_groups or []
            if group_index not in processed:
                processed.append(group_index)
                saved.processed_groups = processed
                await db.commit()


async def estimate_entity_tokens(
    items: List[Dict],
    llm_service: Any,
    config_service: Any,
    label: str,
    base_prompt_chars: int = 500,
) -> Dict:
    """Estimate tokens needed for analyzing a list of entities."""
    from app.models.settings_model import LLM_KEY_CLASSIFIER_PROVIDER, LLM_KEY_CLASSIFIER_MODEL

    items_count = len(items)
    avg_name_length = sum(len(i.get("name", "")) for i in items) / max(items_count, 1)
    estimated_input = base_prompt_chars + int(items_count * (avg_name_length + 10))
    estimated_tokens = estimated_input // 4

    provider = await config_service.get(LLM_KEY_CLASSIFIER_PROVIDER) or ""
    model = await config_service.get(LLM_KEY_CLASSIFIER_MODEL) or ""
    token_limit = await llm_service.get_token_limit(provider, model)
    is_cloud = not llm_service.is_local_provider(provider)
    safe_limit = int(token_limit * 0.8)
    needs_batching = estimated_tokens > safe_limit
    recommended_batches = max(1, (estimated_tokens + safe_limit - 1) // safe_limit) if needs_batching else 1

    return {
        "items_info": f"{items_count} {label}",
        "estimated_tokens": estimated_tokens,
        "token_limit": token_limit,
        "is_cloud": is_cloud,
        "recommended_batches": recommended_batches,
        "warning": f"~{estimated_tokens:,} Tokens > {safe_limit:,} Limit. Wird in {recommended_batches} Batches aufgeteilt." if needs_batching else None,
    }


async def delete_empty_entities(
    items: List[Dict],
    delete_fn: Callable,
    entity_type: str,
    stats_service: Any,
) -> Dict:
    """Delete all entities with 0 documents — parallel with batch of 10."""
    empty = [i for i in items if i.get("document_count", 0) == 0]

    if not empty:
        return {"deleted": 0, "total": 0, "errors": None}

    errors: List[str] = []
    deleted = 0
    batch_size = 10

    async def _delete_one(item: dict):
        try:
            await delete_fn(item["id"])
            return True, None
        except Exception as e:
            return False, f"{item['name']}: {e}"

    for i in range(0, len(empty), batch_size):
        batch = empty[i : i + batch_size]
        results = await asyncio.gather(*[_delete_one(t) for t in batch])
        for success, error in results:
            if success:
                deleted += 1
            elif error:
                errors.append(error)

    if deleted > 0:
        await stats_service.record_operation(
            entity_type=entity_type,
            operation="deleted",
            items_affected=deleted,
            documents_affected=0,
            items_before=len(items),
            items_after=len(items) - deleted,
        )

    return {"deleted": deleted, "total": len(empty), "errors": errors or None}
