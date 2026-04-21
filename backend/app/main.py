import json
import logging
import sys
import time
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dishka.integrations.fastapi import setup_dishka

from app.core.logging import init_logging, ensure_logging, get_logger
from app.database import run_migrations

# Initialize logging system FIRST, before any other modules that might use get_logger()
init_logging()
logger = get_logger("app.main")

from app.routers import paperless, correspondents, tags, document_types, settings, llm, debug, statistics, ignored_items, ocr, cleanup, classifier, rag, api_keys, cloud_import, duplicates
from app.routers.ocr import ocr_settings
from app.services.ocr_service import watchdog_state
from app.services.paperless_client import PaperlessClient
from app.services.protocols import OcrService, RAGService
from app.container import container as di_container


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    from app.database import async_session
    from app.models.settings_model import PaperlessSettings
    from sqlalchemy import select as sa_select

    # Run database migrations (Alembic upgrade to head)
    # NOTE: alembic/env.py calls fileConfig(alembic.ini) which resets the root logger
    # to level=WARN with a plain handler — ensure_logging() must run AFTER this.
    await asyncio.get_running_loop().run_in_executor(None, run_migrations)

    # Re-apply our logging config after Alembic's fileConfig reset the root logger.
    ensure_logging()
    logger.info("Logging active — worker process ready")

    # Auto-start watchdog if it was enabled before shutdown
    if ocr_settings.get("watchdog_enabled"):
        try:
            async with di_container() as ctx:
                client = await ctx.get(PaperlessClient)
                service = await ctx.get(OcrService)
                watchdog_state["enabled"] = True
                watchdog_state["interval_minutes"] = ocr_settings.get("watchdog_interval", 5)
                loop = asyncio.get_running_loop()
                watchdog_state["task"] = loop.create_task(service.watchdog_loop(client))
                logging.getLogger(__name__).info(
                    f"Watchdog auto-started (interval: {watchdog_state['interval_minutes']} min)"
                )
        except Exception as e:
            logging.getLogger(__name__).error(f"Watchdog auto-start failed: {e}")

    # Auto-start classifier background job if enabled
    try:
        from app.models.classifier import ClassifierConfig
        async with async_session() as db_sess:
            q = await db_sess.execute(sa_select(ClassifierConfig).where(ClassifierConfig.id == 1))
            cls_config = q.scalars().first()
            if cls_config and getattr(cls_config, "auto_classify_enabled", False):
                from app.routers.classifier import _auto_classify_state, _auto_classify_loop
                _auto_classify_state["enabled"] = True
                asyncio.get_running_loop().create_task(_auto_classify_loop(di_container))
                logging.getLogger(__name__).info("Auto-classify auto-started")
    except Exception as e:
        logging.getLogger(__name__).error(f"Auto-classify auto-start failed: {e}")

    # Reset stale RAG indexing status + auto-resume incomplete indexing
    try:
        from app.models.rag import RagIndexingState, RagConfig as RagConfigModel
        async with async_session() as db_sess:
            result = await db_sess.execute(sa_select(RagIndexingState).where(RagIndexingState.id == 1))
            rag_state = result.scalar_one_or_none()
            cfg_result = await db_sess.execute(sa_select(RagConfigModel).where(RagConfigModel.id == 1))
            rag_cfg = cfg_result.scalar_one_or_none()

            # Migrate away from mistral-nemo:12b — reset to documented default qwen3.5:4b
            if rag_cfg and getattr(rag_cfg, "chat_model", "") == "mistral-nemo:12b":
                rag_cfg.chat_model = "qwen3.5:4b"
                await db_sess.commit()
                logging.getLogger(__name__).info("RAG: migrated chat_model from mistral-nemo:12b to qwen3.5:4b")

            # Auto-enable RAG for existing users who already have indexed data
            if (
                rag_cfg
                and not getattr(rag_cfg, "rag_enabled", False)
                and rag_state
                and rag_state.indexed_documents > 0
            ):
                rag_cfg.rag_enabled = True
                await db_sess.commit()
                logging.getLogger(__name__).info("RAG: auto-enabled for existing user with indexed data")

            rag_active = rag_cfg and getattr(rag_cfg, "rag_enabled", False)

            if rag_state and rag_state.status == "indexing":
                rag_state.status = "idle"
                await db_sess.commit()
                logging.getLogger(__name__).info("Reset stale RAG indexing status to 'idle'")

            # Auto-resume if indexing was incomplete and RAG is enabled
            if (
                rag_active
                and rag_state
                and rag_state.status in ("idle", "error", "indexing")
                and rag_state.total_documents > 0
                and rag_state.indexed_documents < rag_state.total_documents
            ):
                async with di_container() as ctx:
                    rag_service = await ctx.get(RAGService)
                    asyncio.get_running_loop().create_task(rag_service.indexer.start_indexing(force=False))
                    logging.getLogger(__name__).info(
                        f"RAG: auto-resuming indexing ({rag_state.indexed_documents}/{rag_state.total_documents} already done)"
                    )
    except Exception as e:
        logging.getLogger(__name__).error(f"RAG status reset failed: {e}")

    # Auto-start cloud sync daemon if any sources are enabled
    try:
        from app.models.cloud_import import CloudSource
        from app.services.cloud_import_service import _cloud_sync_state, cloud_sync_loop
        async with async_session() as db_sess:
            src_q = await db_sess.execute(
                sa_select(CloudSource).where(CloudSource.enabled == True)
            )
            has_sources = src_q.scalars().first() is not None
        if has_sources:
            _cloud_sync_state["enabled"] = True
            asyncio.get_running_loop().create_task(cloud_sync_loop(di_container))
            logging.getLogger(__name__).info("Cloud sync daemon auto-started")
    except Exception as e:
        logging.getLogger(__name__).error(f"Cloud sync auto-start failed: {e}")

    yield
    # Shutdown: stop auto-classify + watchdog + cloud sync gracefully
    try:
        from app.routers.classifier import _auto_classify_state
        _auto_classify_state["enabled"] = False
        task = _auto_classify_state.get("task")
        if task and not task.done():
            task.cancel()
    except Exception:
        pass

    try:
        from app.services.cloud_import_service import _cloud_sync_state as _css
        _css["enabled"] = False
        task = _css.get("task")
        if task and not task.done():
            task.cancel()
    except Exception:
        pass

    if watchdog_state.get("enabled"):
        watchdog_state["enabled"] = False
        task = watchdog_state.get("task")
        if task and not task.done():
            task.cancel()


app = FastAPI(
    title="AI Paperless Organizer",
    description="Intelligente Bereinigung von Korrespondenten, Tags und Dokumententypen in Paperless-ngx",
    version="1.0.0",
    lifespan=lifespan
)

setup_dishka(di_container, app)


def _log(level: str, msg: str, *args):
    """Log via structured logger."""
    formatted = msg % args if args else msg
    log_level = getattr(logging, level.upper())
    logger.log(log_level, formatted)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs every HTTP request at INFO level and request/response bodies at DEBUG level."""
    method = request.method
    path = request.url.path
    query = request.url.query
    start = time.monotonic()

    # Read and log request body at DEBUG level
    req_body = ""
    if method in ("POST", "PUT", "PATCH"):
        try:
            body_bytes = await request.body()
            if body_bytes:
                req_body = body_bytes.decode(errors="replace")
                if len(req_body) > 2000:
                    req_body = req_body[:2000] + "...(truncated)"
                _log("DEBUG", "REQUEST BODY %s %s%s: %s", method, path,
                      f"?{query}" if query else "", req_body)
                async def receive():
                    return {"type": "http.request", "body": body_bytes, "more_body": False}
                request._receive = receive
        except Exception:
            pass

    if not req_body:
        _log("INFO", "REQUEST %s %s%s", method, path, f"?{query}" if query else "")

    try:
        response = await call_next(request)
    except Exception as exc:
        elapsed = time.monotonic() - start
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        _log("ERROR", "EXCEPTION %s %s after %.2fs: %s\n%s", method, path, elapsed, exc, tb)
        raise

    elapsed = time.monotonic() - start
    _log("INFO", "RESPONSE %s %s -> %s (%.2fs)", method, path, response.status_code, elapsed)

    if response.status_code >= 400:
        try:
            body_parts = []
            async for chunk in response.body_iterator:
                body_parts.append(chunk)
            resp_body = b"".join(body_parts).decode(errors="replace")[:2000]
            _log("ERROR", "ERROR RESPONSE %s %s -> %s | %s", method, path, response.status_code, resp_body)
            from starlette.responses import StreamingResponse
            async def iter_body():
                for part in body_parts:
                    yield part
            response = StreamingResponse(
                content=iter_body(),
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )
        except Exception:
            pass

    return response


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(
        "HTTP error %s on %s %s: %s",
        exc.status_code,
        request.method,
        request.url.path,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(
        "Unhandled exception on %s %s:\n%s",
        request.method,
        request.url.path,
        tb,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# Include routers
app.include_router(paperless.router, prefix="/api/paperless", tags=["Paperless"])
app.include_router(correspondents.router, prefix="/api/correspondents", tags=["Correspondents"])
app.include_router(tags.router, prefix="/api/tags", tags=["Tags"])
app.include_router(document_types.router, prefix="/api/document-types", tags=["Document Types"])
app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])
app.include_router(llm.router, prefix="/api/llm", tags=["LLM"])
app.include_router(debug.router, prefix="/api/debug", tags=["Debug"])
app.include_router(statistics.router, prefix="/api/statistics", tags=["Statistics"])
app.include_router(ignored_items.router, prefix="/api/ignored-items", tags=["Ignored Items"])
app.include_router(ocr.router, prefix="/api/ocr", tags=["OCR"])
app.include_router(cleanup.router, prefix="/api/cleanup", tags=["Cleanup"])
app.include_router(classifier.router, prefix="/api/classifier", tags=["Classifier"])
app.include_router(rag.router, prefix="/api/rag", tags=["RAG"])
app.include_router(api_keys.router, prefix="/api/api-keys", tags=["API Keys"])
app.include_router(cloud_import.router, prefix="/api/cloud-import", tags=["Cloud Import"])
app.include_router(duplicates.router, prefix="/api/cleanup/duplicates", tags=["Duplicates"])


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "AI Paperless Organizer"}

