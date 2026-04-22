"""DuplicateService orchestrator — delegates to operation modules."""

import asyncio
import logging
from typing import List

from app.database import async_session

logger = logging.getLogger(__name__)


class DuplicateService:
    def __init__(self, session_factory, paperless_client):
        self.session_factory = session_factory or async_session
        self.paperless_client = paperless_client

    async def scan_all(self, modes: List[str], similarity_threshold: float = 0.92):
        """Run selected scan modes as a background task.

        Args:
            modes: list that can include 'exact', 'similar', 'invoices'
            similarity_threshold: cosine similarity threshold for similar scan (default 0.92)
        """
        from app.services.duplicate.state import get_scan_state, _is_cancelled
        from app.services.duplicate import exact, similar, invoices

        scan_state = get_scan_state()

        if scan_state["running"]:
            logger.warning("Duplicate scan already running, ignoring request")
            return

        scan_state["running"] = True
        scan_state["phase"] = ""
        scan_state["progress"] = 0
        scan_state["total"] = 0
        scan_state["results"] = {"exact": [], "similar": [], "invoices": []}
        scan_state["error"] = None
        scan_state["cancel_requested"] = False

        try:
            # Use injected PaperlessClient (with retry)
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

            if "exact" in modes and not _is_cancelled():
                scan_state["phase"] = "exact"
                scan_state["progress"] = 0
                scan_state["results"]["exact"] = await exact.scan_exact(
                    pl_client, self.session_factory, scan_state
                )

            if "similar" in modes and not _is_cancelled():
                scan_state["phase"] = "similar"
                scan_state["progress"] = 0
                scan_state["results"]["similar"] = await similar.scan_similar(
                    pl_client, self.session_factory, scan_state, similarity_threshold
                )

            if "invoices" in modes and not _is_cancelled():
                scan_state["phase"] = "invoices"
                scan_state["progress"] = 0
                scan_state["results"]["invoices"] = await invoices.scan_invoices(
                    pl_client, self.session_factory, scan_state
                )

            scan_state["phase"] = "cancelled" if _is_cancelled() else "done"
            logger.info("Duplicate scan completed successfully")

        except Exception as e:
            logger.exception("Duplicate scan failed")
            scan_state["error"] = str(e)
        finally:
            scan_state["running"] = False