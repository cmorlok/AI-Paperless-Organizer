"""Settings Router — thin endpoints only: validate input, call service, return response."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.llm import LLMService
from app.services import settings_service as svc
from dishka.integrations.fastapi import inject
from dishka import FromDishka

router = APIRouter()


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class SettingUpdateSchema(BaseModel):
    value: str
    value_type: Optional[str] = "str"


class PaperlessSettingsSchema(BaseModel):
    url: str
    api_token: str


class LLMProviderSchema(BaseModel):
    name: str
    display_name: str
    api_key: Optional[str] = ""
    api_base_url: Optional[str] = ""


class LLMProviderPatchSchema(BaseModel):
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None


class LLMProviderCreateSchema(BaseModel):
    name: str
    display_name: Optional[str] = None
    api_key: Optional[str] = ""
    api_base_url: Optional[str] = ""


class CustomPromptSchema(BaseModel):
    entity_type: str
    prompt_template: str
    is_active: bool = True


class IgnoredTagSchema(BaseModel):
    pattern: str
    reason: Optional[str] = ""
    is_regex: bool = False


class AppSettingsSchema(BaseModel):
    show_debug_menu: Optional[bool] = None
    sidebar_compact: Optional[bool] = None
    classifier_provider: Optional[str] = None
    classifier_model: Optional[str] = None
    ocr_model: Optional[str] = None


# ── Paperless Settings ───────────────────────────────────────────────────────

@router.get("/paperless")
async def get_paperless_settings(db: AsyncSession = Depends(get_db)):
    return await svc.get_paperless_settings(db)


@router.post("/paperless")
async def save_paperless_settings(data: PaperlessSettingsSchema, db: AsyncSession = Depends(get_db)):
    try:
        return await svc.save_paperless_settings(data.url, data.api_token, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── LLM Providers ────────────────────────────────────────────────────────────

@router.get("/llm-providers/db")
async def get_llm_providers_from_db(db: AsyncSession = Depends(get_db)):
    return await svc.get_llm_providers_from_db(db)


@router.get("/llm-providers")
@inject
async def get_llm_providers_from_litellm(llm_service: FromDishka[LLMService] = None):
    assert llm_service is not None
    return llm_service.list_providers()


@router.get("/llm-providers/models")
@inject
async def get_llm_provider_models(provider: str, llm_service: FromDishka[LLMService] = None):
    assert llm_service is not None
    try:
        models = await llm_service.list_models(provider)
        return {"provider": provider, "models": models}
    except Exception as e:
        return {"provider": provider, "models": [], "error": str(e)}


@router.put("/llm-providers/db/{provider_id}")
async def update_llm_provider(provider_id: int, data: LLMProviderSchema, db: AsyncSession = Depends(get_db)):
    try:
        return await svc.update_llm_provider(
            provider_id, data.name, data.display_name, data.api_key, data.api_base_url, db,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/llm-providers/db/{provider_id}")
async def patch_llm_provider(provider_id: int, data: LLMProviderPatchSchema, db: AsyncSession = Depends(get_db)):
    try:
        return await svc.patch_llm_provider(provider_id, data.api_key, data.api_base_url, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/llm-providers/db")
async def create_llm_provider(data: LLMProviderCreateSchema, db: AsyncSession = Depends(get_db)):
    try:
        return await svc.create_llm_provider(data.name, data.display_name, data.api_key, data.api_base_url, db)
    except ValueError as e:
        if str(e) == "already_exists":
            raise HTTPException(status_code=409, detail="Provider already exists")
        raise


# ── Custom Prompts ───────────────────────────────────────────────────────────

@router.get("/prompts")
async def get_prompts(db: AsyncSession = Depends(get_db)):
    return await svc.get_prompts(db)


@router.put("/prompts/{prompt_id}")
async def update_prompt(prompt_id: int, data: CustomPromptSchema, db: AsyncSession = Depends(get_db)):
    try:
        return await svc.update_prompt(prompt_id, data.prompt_template, data.is_active, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/prompts/reset/{entity_type}")
async def reset_prompt(entity_type: str, db: AsyncSession = Depends(get_db)):
    try:
        return await svc.reset_prompt(entity_type, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Ignored Tags ─────────────────────────────────────────────────────────────

@router.get("/ignored-tags")
async def get_ignored_tags(db: AsyncSession = Depends(get_db)):
    return await svc.get_ignored_tags(db)


@router.post("/ignored-tags")
async def add_ignored_tag(data: IgnoredTagSchema, db: AsyncSession = Depends(get_db)):
    try:
        return await svc.add_ignored_tag(data.pattern, data.reason or "", data.is_regex, db)
    except ValueError:
        raise HTTPException(status_code=400, detail="Pattern already exists")


@router.delete("/ignored-tags/{tag_id}")
async def delete_ignored_tag(tag_id: int, db: AsyncSession = Depends(get_db)):
    try:
        await svc.delete_ignored_tag(tag_id, db)
    except ValueError:
        raise HTTPException(status_code=404, detail="Pattern not found")
    return {"success": True}


# ── App Settings ─────────────────────────────────────────────────────────────

@router.get("/app")
async def get_app_settings(db: AsyncSession = Depends(get_db)):
    return await svc.get_app_settings(db)


@router.put("/app")
async def update_app_settings(data: AppSettingsSchema, db: AsyncSession = Depends(get_db)):
    await svc.update_app_settings(
        show_debug_menu=data.show_debug_menu,
        sidebar_compact=data.sidebar_compact,
        classifier_provider=data.classifier_provider,
        classifier_model=data.classifier_model,
        ocr_model=data.ocr_model,
        db=db,
    )
    return {"success": True}


# ── Key-Value Settings ───────────────────────────────────────────────────────

@router.get("/settings/{key}")
async def get_setting_endpoint(key: str, db: AsyncSession = Depends(get_db)):
    value = await svc.get_setting(key, db)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
    return {"key": key, "value": value}


@router.put("/settings/{key}")
async def set_setting_endpoint(key: str, data: SettingUpdateSchema, db: AsyncSession = Depends(get_db)):
    await svc.set_setting(key, data.value, data.value_type or "str", db)
    return {"success": True, "key": key, "value": data.value}


@router.post("/settings/seed-llm-keys")
async def seed_llm_keys(db: AsyncSession = Depends(get_db)):
    return await svc.seed_llm_keys(db)
