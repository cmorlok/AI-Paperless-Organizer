"""Cloud sync polling loop for Paperless documents."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from dishka import AsyncContainer

from app.services.paperless import PaperlessClient
# Import CloudImportService from protocol (not service/impl) — fixed bug
from app.services.cloud_import import CloudImportService
# CloudSyncState is internal — keep deep import
from app.services.cloud_import.state import CloudSyncState

logger = logging.getLogger(__name__)


async def cloud_sync_loop(container: AsyncContainer) -> None:
    """Polling loop: checks all enabled sources on their configured interval."""
    while True:
        async with container() as ctx:
            state: CloudSyncState = await ctx.get(CloudSyncState)
            if not state.enabled:
                await _wait_for_enabled(container)
                continue

            await _run_sync_cycle(state, container)


async def _wait_for_enabled(container: AsyncContainer) -> None:
    while True:
        async with container() as ctx:
            state: CloudSyncState = await ctx.get(CloudSyncState)
            if state.enabled:
                return
        await asyncio.sleep(1)


async def _run_sync_cycle(state: CloudSyncState, container: AsyncContainer) -> None:
    from app.models.cloud_import import CloudSource
    from sqlalchemy import select as sa_select

    state.last_run = datetime.utcnow().isoformat()

    try:
        async with container() as ctx:
            client: PaperlessClient = await ctx.get(PaperlessClient)
            service: CloudImportService = await ctx.get(CloudImportService)

            async with getattr(service, "session_factory")() as db:
                src_q = await db.execute(sa_select(CloudSource).where(CloudSource.enabled))
                sources = src_q.scalars().all()

                now = datetime.utcnow()
                for source in sources:
                    if not state.enabled:
                        break

                    if source.last_checked_at:
                        elapsed = (now - source.last_checked_at).total_seconds() / 60
                        if elapsed < (source.poll_interval_minutes or 5):
                            continue

                    state.running = True
                    state.current_source_id = source.id
                    state.current_source_name = source.name
                    source.last_status = "syncing"
                    source.last_error = ""
                    await db.commit()

                    try:
                        stats = await service.sync_source(source, client, db)
                        source.last_status = "idle"
                        source.last_checked_at = datetime.utcnow()
                        logger.info(
                            f"Cloud sync '{source.name}': "
                            f"{stats['imported']} importiert, {stats['skipped']} übersprungen, {stats['errors']} Fehler"
                        )
                    except Exception as e:
                        source.last_status = "error"
                        source.last_error = str(e)
                        source.last_checked_at = datetime.utcnow()
                        logger.error(f"Cloud sync '{source.name}' fehlgschlagen: {e}")

                    state.running = False
                    state.current_source_id = None
                    state.current_source_name = None
                    state.current_file = None
                    await db.commit()

    except Exception as e:
        logger.error(f"Cloud sync loop error: {e}")
        state.running = False

    await asyncio.sleep(60)
    state.running = False
    logger.info("Cloud sync loop stopped")