from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from dishka import AsyncContainer

logger = logging.getLogger(__name__)


async def cloud_sync_loop(container: AsyncContainer):
    """Polling loop: checks all enabled sources on their configured interval."""
    from app.models.cloud_import import CloudSource
    from app.models.settings_model import PaperlessSettings
    from app.services.cloud_import.service import CloudImportService
    from app.services.cloud_import.state import _cloud_sync_state
    from sqlalchemy import select

    logger.info("Cloud sync loop started")

    async with container() as ctx:
        client = await ctx.get(PaperlessClient)
        service = await ctx.get(CloudImportService)

        while _cloud_sync_state["enabled"]:
            _cloud_sync_state["last_run"] = datetime.utcnow().isoformat()

            try:
                async with service.session_factory() as db:
                    src_q = await db.execute(select(CloudSource).where(CloudSource.enabled == True))
                    sources = src_q.scalars().all()

                    now = datetime.utcnow()
                    for source in sources:
                        if not _cloud_sync_state["enabled"]:
                            break

                        # Respect per-source poll interval
                        if source.last_checked_at:
                            elapsed = (now - source.last_checked_at).total_seconds() / 60
                            if elapsed < (source.poll_interval_minutes or 5):
                                continue

                        _cloud_sync_state["running"] = True
                        _cloud_sync_state["current_source_id"] = source.id
                        _cloud_sync_state["current_source_name"] = source.name
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

                        _cloud_sync_state["running"] = False
                        _cloud_sync_state["current_source_id"] = None
                        _cloud_sync_state["current_source_name"] = None
                        _cloud_sync_state["current_file"] = None
                        await db.commit()

            except Exception as e:
                logger.error(f"Cloud sync loop error: {e}")
                _cloud_sync_state["running"] = False

            # Main loop sleeps 60s, per-source interval is checked above
            await asyncio.sleep(60)

    _cloud_sync_state["running"] = False
    logger.info("Cloud sync loop stopped")
