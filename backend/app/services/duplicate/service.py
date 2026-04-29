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