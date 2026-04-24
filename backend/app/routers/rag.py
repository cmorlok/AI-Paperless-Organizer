import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.requests import Request
from dishka.integrations.fastapi import inject
from dishka import FromDishka

from app.services.rag.protocol import RAGService

logger = logging.getLogger(__name__)
router = APIRouter()


async def _check_api_auth(request: Request) -> bool:
    auth = request.headers.get("Authorization", "")
    api_key_param = request.query_params.get("api_key", "")
    if not auth and not api_key_param:
        return True
    from app.database import async_session
    from app.routers.api_keys import validate_api_key
    async with async_session() as db:
        key = await validate_api_key(request, db)
        return key is not None


# --- Request / Response Models ---

class SearchFilters(BaseModel):
    tags: Optional[List[int]] = None
    correspondent_id: Optional[int] = None
    document_type_id: Optional[int] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    filters: Optional[SearchFilters] = None


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=50)
    filters: Optional[SearchFilters] = None


class IndexRequest(BaseModel):
    force: bool = False


class ConfigUpdate(BaseModel):
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    bm25_weight: Optional[float] = None
    semantic_weight: Optional[float] = None
    max_sources: Optional[int] = None
    max_context_tokens: Optional[int] = None
    chat_model_provider: Optional[str] = None
    chat_model: Optional[str] = None
    chat_system_prompt: Optional[str] = None
    auto_index_enabled: Optional[bool] = None
    auto_index_interval: Optional[int] = None
    query_rewrite_enabled: Optional[bool] = None
    contextual_retrieval_enabled: Optional[bool] = None
    rag_enabled: Optional[bool] = None


# --- Chat Endpoints ---

@router.post("/chat")
@inject
async def chat(body: ChatRequest, request: Request, service: FromDishka[RAGService] = None):
    if not await _check_api_auth(request):
        raise HTTPException(status_code=401, detail="Ungültiger API-Key")
    filters = body.filters.model_dump(exclude_none=True) if body.filters else None

    async def event_stream():
        try:
            async for chunk in service.chat_stream(
                question=body.question,
                session_id=body.session_id,
                filters=filters,
            ):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            import json
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- Search Endpoint ---

@router.post("/search")
@inject
async def search(body: SearchRequest, request: Request, service: FromDishka[RAGService] = None):
    if not await _check_api_auth(request):
        raise HTTPException(status_code=401, detail="Ungültiger API-Key")
    filters = body.filters.model_dump(exclude_none=True) if body.filters else None

    results = await service.search(
        query=body.query,
        limit=body.limit,
        filters=filters,
    )
    return {
        "query": body.query,
        "results": [r.to_dict() for r in results],
        "total": len(results),
    }


# --- Indexing Endpoints ---

@router.post("/index/start")
@inject
async def start_indexing(request: IndexRequest, service: FromDishka[RAGService] = None):
    if service.indexer.is_indexing:
        raise HTTPException(status_code=409, detail="Indexierung läuft bereits")

    await service.indexer.start_indexing(force=request.force)
    return {"status": "started", "force": request.force}


@router.get("/index/status")
@inject
async def indexing_status(service: FromDishka[RAGService] = None):
    return await service.indexer.get_status()


# --- Session Endpoints ---

@router.get("/sessions")
@inject
async def list_sessions(service: FromDishka[RAGService] = None):
    return await service.get_sessions()


@router.get("/sessions/{session_id}")
@inject
async def get_session(session_id: str, service: FromDishka[RAGService] = None):
    session = await service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    return session


@router.delete("/sessions/{session_id}")
@inject
async def delete_session(session_id: str, service: FromDishka[RAGService] = None):
    deleted = await service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    return {"deleted": True}


# --- Config Endpoints ---

@router.get("/config")
@inject
async def get_config(service: FromDishka[RAGService] = None):
    return await service.get_config_dict()


@router.put("/config")
@inject
async def update_config(request: ConfigUpdate, service: FromDishka[RAGService] = None):
    updates = request.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="Keine Änderungen angegeben")
    return await service.update_config(updates)


# --- Health Check ---

@router.get("/health")
@inject
async def rag_health(service: FromDishka[RAGService] = None):
    config = await service.get_config_dict()
    index_status = await service.indexer.get_status()
    return {
        "embedding": await _probe_embedding(config),
        "index": index_status,
    }


async def _probe_embedding(config: dict) -> dict:
    from app.services.llm import llm_embedding
    provider = config["embedding_provider"]
    model = config["embedding_model"]
    try:
        await llm_embedding(
            model=model,
            provider=provider,
            input=["test"],
        )
        return {"healthy": True, "provider": provider, "model": model}
    except Exception as e:
        return {"healthy": False, "provider": provider, "model": model, "error": str(e)}
