from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional
from app.services.llm import LLMService
from app.models.settings_model import LLM_KEY_CLASSIFIER_PROVIDER, LLM_KEY_CLASSIFIER_MODEL
from app.services.config import ConfigService
from dishka.integrations.fastapi import inject
from dishka import FromDishka

router = APIRouter(tags=["llm"])


class TestPromptRequest(BaseModel):
    """Request to test a prompt."""
    prompt: str
    provider_name: Optional[str] = None


@router.post("/test")
@inject
async def test_llm_connection(
    provider: Optional[str] = Query(None, description="Provider name to test (e.g. 'ollama')"),
    model: Optional[str] = Query(None, description="Model name to test (e.g. 'qwen2.5vl:7b')"),
    llm_service: FromDishka[LLMService] = None,
    config_svc: FromDishka[ConfigService] = None,
):
    """Test LLM provider connection. If provider/model given, tests that specific combo; otherwise tests active classifier provider."""
    try:
        test_provider = provider
        test_model = model
        if not test_provider or not test_model:
            test_provider = test_provider or (await config_svc.get(LLM_KEY_CLASSIFIER_PROVIDER))
            test_model = test_model or (await config_svc.get(LLM_KEY_CLASSIFIER_MODEL))
        result = await llm_service.test_connection(provider=test_provider, model=test_model)
        return {"success": True, "provider": result["provider"], "model": result["model"]}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/test-prompt")
@inject
async def test_prompt(
    request: TestPromptRequest,
    llm_service: FromDishka[LLMService] = None,
    config_svc: FromDishka[ConfigService] = None,
):
    """Test a prompt with the active LLM provider."""
    try:
        provider = (await config_svc.get(LLM_KEY_CLASSIFIER_PROVIDER)) or None
        model = (await config_svc.get(LLM_KEY_CLASSIFIER_MODEL)) or None
        result = await llm_service.complete(
            provider=provider,
            model=model,
            messages=[{"role": "user", "content": request.prompt}],
        )
        return {"success": True, "response": (result.content or "").strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/status")
@inject
async def get_llm_status(
    llm_service: FromDishka[LLMService] = None,
) -> dict[str, dict]:
    """Get current LLM lock status for all local providers.

    Returns lock status for each provider that has an active lock,
    e.g. {"ollama": {"locked": true}}."""

    return llm_service.get_lock_status()

