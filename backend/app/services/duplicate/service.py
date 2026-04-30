"""DuplicateService orchestrator — delegates to operation modules."""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from app.database import async_session
from app.services.llm import LLMService

logger = logging.getLogger(__name__)


class DuplicateService:
    def __init__(self, session_factory, paperless_client, state=None, llm_service: Optional[LLMService] = None):
        self.session_factory = session_factory or async_session
        self.paperless_client = paperless_client
        self.state = state
        self.llm_service = llm_service

    async def scan_all(self, modes: List[str], similarity_threshold: float = 0.92):
        assert self.llm_service is not None
        assert self.state is not None
        from app.services.duplicate.state import DuplicateScanState
        from app.services.duplicate import exact, similar, invoices

        scan_state = self.state or DuplicateScanState()

        if scan_state.running:
            logger.warning("Duplicate scan already running, ignoring request")
            return

        scan_state.start_scan()

        try:
            pl_client = self.paperless_client
            for attempt in range(3):
                try:
                    await pl_client.test_connection()
                    break
                except Exception as e:
                    logger.warning(f"Paperless connection attempt {attempt+1}/3 failed: {e}")
                    if attempt < 2:
                        await asyncio.sleep(5)
            else:
                raise ConnectionError("Paperless-ngx nicht erreichbar nach 3 Versuchen")

            if "exact" in modes and not scan_state.is_cancelled():
                scan_state.update_progress("exact", 0, 0)
                scan_state.results["exact"] = await exact.scan_exact(
                    pl_client, self.session_factory, scan_state
                )

            if "similar" in modes and not scan_state.is_cancelled():
                scan_state.update_progress("similar", 0, 0)
                scan_state.results["similar"] = await similar.scan_similar(
                    pl_client, self.session_factory, scan_state, similarity_threshold
                )

            if "invoices" in modes and not scan_state.is_cancelled():
                scan_state.update_progress("invoices", 0, 0)
                scan_state.results["invoices"] = await invoices.scan_invoices(
                    pl_client, self.session_factory, scan_state, self.llm_service
                )

            scan_state.phase = "cancelled" if scan_state.is_cancelled() else "done"
            logger.info("Duplicate scan completed successfully")

        except Exception as e:
            logger.exception("Duplicate scan failed")
            scan_state.error = str(e)
        finally:
            scan_state.running = False

    # ── Extracted router business logic ─────────────────────────────────────

    async def ignore_group(self, doc_ids: list) -> dict:
        """Mark a group of documents as 'not a duplicate'."""
        from sqlalchemy import select, and_
        from app.models.duplicates import DuplicateIgnore

        if len(doc_ids) < 2:
            raise ValueError("Mindestens 2 Dokument-IDs erforderlich")

        added = 0
        async with self.session_factory() as db:
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    a, b = sorted([doc_ids[i], doc_ids[j]])
                    existing = await db.execute(
                        select(DuplicateIgnore).where(
                            and_(
                                DuplicateIgnore.doc_id_a == a,
                                DuplicateIgnore.doc_id_b == b,
                            )
                        )
                    )
                    if existing.scalar_one_or_none() is None:
                        db.add(DuplicateIgnore(doc_id_a=a, doc_id_b=b))
                        added += 1
            await db.commit()

        logger.info("Added %d ignore pair(s) for doc_ids=%s", added, doc_ids)
        return {"added": added}

    async def list_ignored(self) -> list:
        """Get all ignored pairs."""
        from sqlalchemy import select
        from app.models.duplicates import DuplicateIgnore

        async with self.session_factory() as db:
            result = await db.execute(select(DuplicateIgnore).order_by(DuplicateIgnore.created_at.desc()))
            rows = result.scalars().all()
            return [
                {
                    "id": row.id,
                    "doc_id_a": row.doc_id_a,
                    "doc_id_b": row.doc_id_b,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

    async def remove_ignore(self, doc_id_a: int, doc_id_b: int) -> None:
        """Remove an ignore pair."""
        from sqlalchemy import delete, and_
        from app.models.duplicates import DuplicateIgnore

        a, b = sorted([doc_id_a, doc_id_b])
        async with self.session_factory() as db:
            result = await db.execute(
                delete(DuplicateIgnore).where(
                    and_(
                        DuplicateIgnore.doc_id_a == a,
                        DuplicateIgnore.doc_id_b == b,
                    )
                )
            )
            await db.commit()
            if result.rowcount == 0:  # type: ignore[union-attr]
                raise ValueError("Paar nicht gefunden")

        logger.info("Removed ignore pair (%d, %d)", a, b)