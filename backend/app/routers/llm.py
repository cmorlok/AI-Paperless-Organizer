from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.protocols import LLMService as LLMProviderService
from dishka.integrations.fastapi import inject
from dishka import FromDishka

router = APIRouter()


class TestPromptRequest(BaseModel):
    """Request to test a prompt."""
    prompt: str
    provider_name: Optional[str] = None


@router.post("/test")
@inject
async def test_llm_connection(
    llm_service: FromDishka[LLMProviderService] = None
):
    """Test the active LLM provider connection."""
    try:
        result = await llm_service.test_connection()
        return {"success": True, "provider": result["provider"], "model": result["model"]}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/test-prompt")
@inject
async def test_prompt(
    request: TestPromptRequest,
    llm_service: FromDishka[LLMProviderService] = None
):
    """Test a prompt with the active LLM provider."""
    try:
        response = await llm_service.complete(request.prompt)
        return {"success": True, "response": response}
    except Exception as e:
        return {"success": False, "error": str(e)}

