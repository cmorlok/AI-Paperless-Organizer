"""Auto-classify background loop for Paperless documents."""

from __future__ import annotations

from dishka import AsyncContainer
from sqlalchemy import select

from app.core.logging import get_logger
from app.models.classifier import ClassificationHistory
# Internal imports from same package — keep deep to avoid circular imports
from app.services.classifier import DocumentClassifierService
from app.services.classifier.state import AutoClassifyState
from app.services.paperless import PaperlessClient

logger = get_logger(__name__)


async def auto_classify_loop(container: AsyncContainer) -> None:
    """Background loop that classifies unprocessed documents."""
    while True:
        async with container() as ctx:
            state: AutoClassifyState = await ctx.get(AutoClassifyState)
            if not state.enabled:
                await _wait_for_enabled(container)
                continue

            await _run_classification_cycle(state, container)


async def _wait_for_enabled(container: AsyncContainer) -> None:
    import asyncio
    while True:
        async with container() as ctx:
            state: AutoClassifyState = await ctx.get(AutoClassifyState)
            if state.enabled:
                return
        await asyncio.sleep(1)


async def _run_classification_cycle(state: AutoClassifyState, container: AsyncContainer) -> None:
    import asyncio
    import time

    state.last_run = time.strftime("%Y-%m-%dT%H:%M:%S")
    state.running = False
    interval = 5  # default minutes between classification cycles

    async with container() as ctx:
        client: PaperlessClient = await ctx.get(PaperlessClient)
        service: DocumentClassifierService = await ctx.get(DocumentClassifierService)

        try:
            config = await service.get_config()
            mode = getattr(config, "auto_classify_mode", "review") or "review"
            interval = getattr(config, "auto_classify_interval", 5) or 5

            classified_ids: set = set()
            session_factory = getattr(service, "session_factory", None)
            if session_factory is not None:
                async with session_factory() as db_sess:
                    applied_q = await db_sess.execute(
                        select(ClassificationHistory.document_id).where(
                            ClassificationHistory.status.in_(["applied", "review", "pending"])
                        ).distinct()
                    )
                    classified_ids = {r[0] for r in applied_q.all()}

            found_any = False
            page = 1
            interval = 5  # default
            while state.enabled:
                result = await getattr(client, "_request")(
                    "GET", "/documents/",
                    params={"page_size": 50, "page": page, "ordering": "id"},
                )
                if not result:
                    break

                docs = result.get("results", [])
                if not docs:
                    break

                for doc in docs:
                    if not state.enabled:
                        break
                    doc_id = doc.get("id")
                    if doc_id in classified_ids:
                        continue

                    skip_tags = getattr(config, "auto_classify_skip_tag_ids", None) or []
                    if skip_tags:
                        doc_tags = doc.get("tags", [])
                        if any(t in skip_tags for t in doc_tags):
                            classified_ids.add(doc_id)
                            continue

                    found_any = True
                    state.running = True
                    state.current_doc = doc_id

                    try:
                        res = await service.classify_document_auto(doc_id, mode)
                        action = res.get("action", "")
                        if action == "applied":
                            state.processed += 1
                        elif action == "review":
                            state.reviewed += 1
                        elif action == "error":
                            state.errors += 1
                        else:
                            state.processed += 1
                        classified_ids.add(doc_id)
                        logger.info(f"Auto-classify doc {doc_id}: {action}")
                    except Exception as e:
                        state.errors += 1
                        classified_ids.add(doc_id)
                        logger.error(f"Auto-classify doc {doc_id} failed: {e}")
                    finally:
                        state.running = False
                        state.current_doc = None

                    await asyncio.sleep(2)

                if not result.get("next"):
                    break
                page += 1

            if not found_any:
                logger.info(f"Auto-classify: keine neuen Dokumente, warte {interval} min")

        except Exception as e:
            logger.error(f"Auto-classify loop error: {e}")

        state.running = False
        state.current_doc = None

        if state.enabled:
            await asyncio.sleep(interval * 60)