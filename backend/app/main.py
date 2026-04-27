import logging
import os
import time
import traceback
from typing import Optional

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

from app.routers import paperless, correspondents, tags, document_types, settings, llm, debug, statistics, ignored_items, ocr, cleanup, classifier, rag, api_keys, cloud_import, duplicates, auth  # noqa: E402
from app.routers.ocr import load_ocr_settings  # noqa: E402
from app.services.ocr import OcrState, OcrService  # noqa: E402
from app.services.classifier import AutoClassifyState, auto_classify_loop  # noqa: E402
from app.services.paperless import PaperlessClient  # noqa: E402
from app.services.rag import RAGService  # noqa: E402
from app.services.auth import SessionAuthMiddleware  # noqa: E402
from app.database import async_session  # noqa: E402
from app.container import container as di_container  # noqa: E402

async def reset_password_if_requested() -> None:
    """Per CONTEXT.md D-14: RESET_PASSWORD=true clears AuthConfig password_hash."""
    import os
    if os.getenv("RESET_PASSWORD", "").lower() != "true":
        return
    from sqlalchemy import select as sa_select
    from app.models.auth_config import AuthConfig
    async with async_session() as db:
        result = await db.execute(sa_select(AuthConfig).where(AuthConfig.id == 1))
        row = result.scalar_one_or_none()
        if row is not None:
            row.password_hash = ""
            await db.commit()
            logger.warning("RESET_PASSWORD=true: Password cleared. Remove env var and restart for normal operation.")
        else:
            logger.warning("RESET_PASSWORD=true: No AuthConfig row to clear.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    from sqlalchemy import select as sa_select

    # Run database migrations (Alembic upgrade to head)
    # NOTE: alembic/env.py calls fileConfig(alembic.ini) which resets the root logger
    # to level=WARN with a plain handler — ensure_logging() must run AFTER this.
    await asyncio.get_running_loop().run_in_executor(None, run_migrations)

    # Re-apply our logging config after Alembic's fileConfig reset the root logger.
    ensure_logging()
    logger.info("Logging active — worker process ready")
    await reset_password_if_requested()

    # Get a database session for reading persisted settings
    db_sess = async_session()

    # Helper to read from KV store
    async def _read_kv_setting(key: str) -> Optional[bool]:
        try:
            from app.routers.settings import get_setting
            val = await get_setting(key, db_sess)
            return val == "true" if val is not None else None
        except Exception:
            return None

    # Auto-start processor if it was enabled before shutdown
    kv_processor_enabled = await _read_kv_setting("ocr_processor_enabled")
    startup_settings = load_ocr_settings()
    start_processor = kv_processor_enabled if kv_processor_enabled is not None else startup_settings.get("processor_enabled")

    if start_processor:
        try:
            async with di_container() as ctx:
                client = await ctx.get(PaperlessClient)
                service = await ctx.get(OcrService)
                ocr_state = await ctx.get(OcrState)
                ocr_state.reset()
                ocr_state.processor.enabled = True
                ocr_state.processor.interval_minutes = startup_settings.get("processor_interval", 5)
                loop = asyncio.get_running_loop()
                ocr_state.processor.task = loop.create_task(service.processor_loop(client))
                logging.getLogger(__name__).info(
                    f"Processor auto-started from KV store (interval: {ocr_state.processor.interval_minutes} min)"
                )
        except Exception as e:
            logging.getLogger(__name__).error(f"Processor auto-start failed: {e}")

    # Auto-start classifier background job if enabled (STATE-08: check KV store first)
    kv_classify_enabled = await _read_kv_setting("auto_classify_enabled")

    if kv_classify_enabled:
        try:
            async with di_container() as ctx:
                ac_state: AutoClassifyState = await ctx.get(AutoClassifyState)
                ac_state.reset()
                ac_state.enabled = True
            asyncio.get_running_loop().create_task(auto_classify_loop(di_container))
            logging.getLogger(__name__).info("Auto-classify auto-started from KV store")
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

    # Auto-start cloud sync daemon if enabled in KV store (STATE-08)
    kv_cloud_sync_enabled = await _read_kv_setting("cloud_sync_enabled")

    # Fall back to checking if any sources are enabled (legacy behavior)
    if kv_cloud_sync_enabled is None:
        try:
            from app.models.cloud_import import CloudSource
            src_q = await db_sess.execute(
                sa_select(CloudSource).where(CloudSource.enabled is True)
            )
            kv_cloud_sync_enabled = src_q.scalars().first() is not None
        except Exception:
            pass

    if kv_cloud_sync_enabled:
        try:
            from app.services.cloud_import import CloudSyncState, cloud_sync_loop
            async with di_container() as ctx:
                css: CloudSyncState = await ctx.get(CloudSyncState)
                css.reset()
                css.enabled = True
            asyncio.get_running_loop().create_task(cloud_sync_loop(di_container))
            logging.getLogger(__name__).info("Cloud sync daemon auto-started from KV store")
        except Exception as e:
            logging.getLogger(__name__).error(f"Cloud sync auto-start failed: {e}")

    # Close the session used for reading settings
    await db_sess.close()

    yield
    # Shutdown: stop auto-classify + watchdog + cloud sync gracefully

    # Auto-classify shutdown - resolve from container
    try:
        async with di_container() as ctx:
            ac_state: AutoClassifyState = await ctx.get(AutoClassifyState)
            ac_state.enabled = False
            task = ac_state._task
            if task and not task.done():
                task.cancel()
    except Exception:
        pass

    # Cloud sync shutdown - resolve from container
    try:
        async with di_container() as ctx:
            css: CloudSyncState = await ctx.get(CloudSyncState)
            css.enabled = False
            task = css.task
            if task and not task.done():
                task.cancel()
    except Exception:
        pass

    # OCR shutdown - resolve from container
    try:
        async with di_container() as ctx:
            ocr_state = await ctx.get(OcrState)
            if ocr_state.processor.enabled:
                ocr_state.processor.enabled = False
                task = ocr_state.processor.task
                if task and not task.done():
                    task.cancel()
    except Exception:
        pass


app = FastAPI(
    title="AI Paperless Organizer",
    description="Intelligente Bereinigung von Korrespondenten, Tags und Dokumententypen in Paperless-ngx",
    version="1.0.0",
    lifespan=lifespan
)

setup_dishka(di_container, app)
app.container = di_container


def _log(level: str, msg: str, *args):
    """Log via structured logger."""
    formatted = msg % args if args else msg
    log_level = getattr(logging, level.upper())
    logger.log(log_level, formatted)


class LoggingMiddleware:
    """Pure ASGI middleware that logs every request/response without BaseHTTPMiddleware bugs."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        path = scope.get("path", "")
        query = scope.get("query_string", b"").decode("utf-8")
        start = time.monotonic()

        # Buffer request body for logging and replay
        body_bytes = b""
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.request":
                body_bytes += message.get("body", b"")
                more_body = message.get("more_body", False)
            elif message["type"] == "http.disconnect":
                await self.app(scope, receive, send)
                return

        req_body = ""
        if method in ("POST", "PUT", "PATCH") and body_bytes:
            req_body = body_bytes.decode(errors="replace")
            if len(req_body) > 2000:
                req_body = req_body[:2000] + "...(truncated)"
            _log("DEBUG", "REQUEST BODY %s %s%s: %s", method, path,
                  f"?{query}" if query else "", req_body)

        if not req_body:
            _log("INFO", "REQUEST %s %s%s", method, path, f"?{query}" if query else "")

        # Synthetic receive that replays the buffered body
        request_messages = [{"type": "http.request", "body": body_bytes, "more_body": False}]
        message_index = 0

        async def receive_replay():
            nonlocal message_index
            if message_index < len(request_messages):
                msg = request_messages[message_index]
                message_index += 1
                return msg
            # After replay, forward disconnects only
            msg = await receive()
            return msg

        # Buffer response messages so we can inspect status/body
        response_messages = []
        status_code = 200

        async def send_capture(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
            response_messages.append(message)

        try:
            await self.app(scope, receive_replay, send_capture)
        except Exception as exc:
            elapsed = time.monotonic() - start
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            _log("ERROR", "EXCEPTION %s %s after %.2fs: %s\n%s", method, path, elapsed, exc, tb)
            raise

        elapsed = time.monotonic() - start
        _log("INFO", "RESPONSE %s %s -> %s (%.2fs)", method, path, status_code, elapsed)

        if status_code >= 400:
            try:
                resp_body = b""
                for msg in response_messages:
                    if msg["type"] == "http.response.body":
                        resp_body += msg.get("body", b"")
                resp_text = resp_body.decode(errors="replace")[:2000]
                _log("ERROR", "ERROR RESPONSE %s %s -> %s | %s", method, path, status_code, resp_text)
            except Exception:
                pass

        # Replay captured response messages to the real send
        for msg in response_messages:
            await send(msg)


# CORS configuration
_allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8088")
_allowed_origins = [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionAuthMiddleware)
app.add_middleware(LoggingMiddleware)


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
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
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

