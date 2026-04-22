"""Auto-classify background loop for Paperless documents."""

# TODO (Phase 05): Move _auto_classify_state to services/classifier/state.py
from typing import Dict, Any
from app.database import async_session
from app.models.classifier import ClassificationHistory
from app.routers.classifier import _auto_classify_state

_auto_classify_state: Dict[str, Any] = {
    "enabled": False,
    "running": False,
    "task": None,
    "processed": 0,
    "errors": 0,
    "reviewed": 0,
    "current_doc": None,
    "last_run": None,
}


async def auto_classify_loop(container) -> None:
    """Background loop that classifies unprocessed documents."""
    import asyncio
    import time
    from dishka import AsyncContainer
    from app.services.classifier.protocol import DocumentClassifierService
    from app.services.llm.lock import acquire as ollama_acquire, release as ollama_release, is_locked as ollama_is_locked, current_holder as ollama_holder
    from app.services.paperless import PaperlessClient
    from sqlalchemy import select

    async with container() as ctx:
        client = await ctx.get(PaperlessClient)
        service = await ctx.get(DocumentClassifierService)

        while _auto_classify_state["enabled"]:
            _auto_classify_state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            _auto_classify_state["running"] = False

            try:
                config = await service.get_config()
                uses_ollama = config.active_provider == "ollama"
                mode = getattr(config, "auto_classify_mode", "review") or "review"
                interval = getattr(config, "auto_classify_interval", 5) or 5

                # Find all applied document IDs using service's session_factory
                classified_ids: set = set()
                if service.session_factory is not None:
                    async with service.session_factory() as db_sess:
                        applied_q = await db_sess.execute(
                            select(ClassificationHistory.document_id).where(
                                ClassificationHistory.status.in_(["applied", "review", "pending"])
                            ).distinct()
                        )
                        classified_ids = {r[0] for r in applied_q.all()}

                # Fetch documents in batches to find unclassified ones
                found_any = False
                page = 1
                while _auto_classify_state["enabled"]:
                    result = await client._request(
                        "GET", "/documents/",
                        params={"page_size": 50, "page": page, "ordering": "id"},
                    )
                    if not result:
                        break

                    docs = result.get("results", [])
                    if not docs:
                        break

                    for doc in docs:
                        if not _auto_classify_state["enabled"]:
                            break
                        doc_id = doc.get("id")
                        if doc_id in classified_ids:
                            continue

                        # Skip documents with tags in auto_classify_skip_tag_ids
                        skip_tags = getattr(config, "auto_classify_skip_tag_ids", None) or []
                        if skip_tags:
                            doc_tags = doc.get("tags", [])
                            if any(t in skip_tags for t in doc_tags):
                                classified_ids.add(doc_id)  # don't retry
                                continue

                        # Per-document Ollama lock: acquire before, release after
                        if uses_ollama:
                            if ollama_is_locked():
                                holder = ollama_holder()
                                logger.info(f"Auto-classify doc {doc_id}: Ollama belegt durch {holder}, warte...")
                                _auto_classify_state["current_doc"] = None
                                while ollama_is_locked() and _auto_classify_state["enabled"]:
                                    await asyncio.sleep(5)
                                if not _auto_classify_state["enabled"]:
                                    break
                            got_lock = await ollama_acquire("classifier", timeout=300)
                            if not got_lock:
                                logger.warning(f"Auto-classify doc {doc_id}: Lock-Timeout, ueberspringe")
                                await asyncio.sleep(10)
                                continue

                        found_any = True
                        _auto_classify_state["running"] = True
                        _auto_classify_state["current_doc"] = doc_id

                        try:
                            res = await service.classify_document_auto(doc_id, mode)
                            action = res.get("action", "")
                            if action == "applied":
                                _auto_classify_state["processed"] += 1
                            elif action == "review":
                                _auto_classify_state["reviewed"] += 1
                            elif action == "error":
                                _auto_classify_state["errors"] += 1
                            else:
                                _auto_classify_state["processed"] += 1
                            classified_ids.add(doc_id)
                            logger.info(f"Auto-classify doc {doc_id}: {action}")
                        except Exception as e:
                            _auto_classify_state["errors"] += 1
                            classified_ids.add(doc_id)
                            logger.error(f"Auto-classify doc {doc_id} failed: {e}")
                        finally:
                            if uses_ollama:
                                ollama_release("classifier")
                            _auto_classify_state["running"] = False
                            _auto_classify_state["current_doc"] = None

                        await asyncio.sleep(2)

                    if not result.get("next"):
                        break
                    page += 1

                if not found_any:
                    logger.info(f"Auto-classify: keine neuen Dokumente, warte {interval} min")

            except Exception as e:
                logger.error(f"Auto-classify loop error: {e}")

            _auto_classify_state["running"] = False
            _auto_classify_state["current_doc"] = None

            if _auto_classify_state["enabled"]:
                await asyncio.sleep(interval * 60)


# Import logger at module level
from app.core.logging import get_logger
logger = get_logger(__name__)
