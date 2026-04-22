from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models import PaperlessSettings, LLMProvider, CustomPrompt, IgnoredTag, AppSettings
from app.models.auth_config import AuthConfig
from app.models.settings_model import (
    LLM_KEY_CLASSIFIER_PROVIDER,
    LLM_KEY_CLASSIFIER_MODEL,
    LLM_KEY_OCR_PROVIDER,
    LLM_KEY_OCR_MODEL,
)
from app.services.llm.service import list_llm_models, list_llm_providers, PROVIDER_DISPLAY_NAMES
from app.prompts.default_prompts import DEFAULT_PROMPTS

router = APIRouter()


# ── Key-Value Setting Helpers (LLM-08) ─────────────────────────────────────────

async def get_setting(key: str, db: AsyncSession) -> Optional[str]:
    """Get a setting value by key. Returns None if not found."""
    result = await db.execute(
        select(AppSettings).where(AppSettings.key == key)
    )
    setting = result.scalar_one_or_none()
    return setting.value if setting else None


async def set_setting(key: str, value: str, value_type: str = "str", db: AsyncSession = None):
    """Set a setting value. Creates new row if key doesn't exist, updates if it does."""
    if db is None:
        async for session in get_db():
            await set_setting(key, value, value_type, session)
            break   # exits cleanly; generator finally-block runs
        return
    result = await db.execute(
        select(AppSettings).where(AppSettings.key == key)
    )
    setting = result.scalar_one_or_none()
    if setting:
        setting.value = value
        setting.value_type = value_type
    else:
        setting = AppSettings(id=None, key=key, value=value, value_type=value_type)
        db.add(setting)
    await db.commit()


class SettingUpdateSchema(BaseModel):
    value: str
    value_type: Optional[str] = "str"


# Pydantic models for requests/responses
class PaperlessSettingsSchema(BaseModel):
    url: str
    api_token: str


class LLMProviderSchema(BaseModel):
    """Full LLMProvider update schema (used by PUT, now without model fields per D-05)."""
    name: str
    display_name: str
    api_key: Optional[str] = ""
    api_base_url: Optional[str] = ""


class LLMProviderPatchSchema(BaseModel):
    """Patch schema for updating connection fields only (per D-05)."""
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None


class LLMProviderCreateSchema(BaseModel):
    """Schema for creating a new LLM provider record."""
    name: str
    display_name: Optional[str] = None
    api_key: Optional[str] = ""
    api_base_url: Optional[str] = ""


class CustomPromptSchema(BaseModel):
    entity_type: str
    prompt_template: str
    is_active: bool = True


# Paperless Settings
@router.get("/paperless")
async def get_paperless_settings(db: AsyncSession = Depends(get_db)):
    """Get Paperless connection settings."""
    result = await db.execute(select(PaperlessSettings).where(PaperlessSettings.id == 1))
    settings = result.scalar_one_or_none()
    
    if not settings:
        return {"url": "", "api_token": "", "is_configured": False}
    
    return {
        "url": settings.url,
        "api_token": "***" if settings.api_token else "",
        "is_configured": settings.is_configured
    }


@router.post("/paperless")
async def save_paperless_settings(
    data: PaperlessSettingsSchema,
    db: AsyncSession = Depends(get_db)
):
    """Save Paperless connection settings."""
    result = await db.execute(select(PaperlessSettings).where(PaperlessSettings.id == 1))
    settings = result.scalar_one_or_none()
    
    if settings:
        settings.url = data.url
        # Only update token if it's not the masked value "***"
        if data.api_token and data.api_token != "***":
            settings.api_token = data.api_token
        # Check if configured (use existing token if masked)
        actual_token = settings.api_token if data.api_token == "***" else data.api_token
        settings.is_configured = bool(data.url and actual_token)
    else:
        # New settings - token must be provided
        if data.api_token == "***":
            raise HTTPException(status_code=400, detail="API Token muss angegeben werden")
        settings = PaperlessSettings(
            id=1,
            url=data.url,
            api_token=data.api_token,
            is_configured=bool(data.url and data.api_token)
        )
        db.add(settings)
    
    await db.commit()
    return {"success": True, "is_configured": settings.is_configured}


# LLM Providers - DB-based (for internal/admin use)
@router.get("/llm-providers/db")
async def get_llm_providers_from_db(db: AsyncSession = Depends(get_db)):
    """Get all LLM provider configurations from database.
    
    Returns only providers that have been explicitly configured via POST.
    The SettingsPanel uses the LiteLLM-based /llm-providers endpoint for the provider dropdown.
    """
    result = await db.execute(select(LLMProvider).order_by(LLMProvider.name))
    providers = result.scalars().all()
    
    return [
        {
            "id": p.id,
            "name": p.name,
            "display_name": p.display_name,
            "api_key": "***" if p.api_key else "",
            "api_base_url": p.api_base_url or "",
        }
        for p in providers
    ]


# LLM Providers - LiteLLM-based (for SettingsPanel UI)
@router.get("/llm-providers")
async def get_llm_providers_from_litellm():
    """Get all LiteLLM-supported providers from litellm.provider_list."""
    return list_llm_providers()


@router.get("/llm-providers/models")
async def get_llm_provider_models(provider: str, db: AsyncSession = Depends(get_db)):
    """Get available models for a specific LiteLLM provider.
    
    For local providers (ollama, lm_studio, vllm): queries the provider's
    API directly using configured base_url from the database.
    
    For other providers: falls back to litellm.model_list filtered by 
    provider prefix (e.g., "openai/").
    """
    try:
        models = await list_llm_models(provider, db)
        return {"provider": provider, "models": models}
    except Exception as e:
        return {"provider": provider, "models": [], "error": str(e)}


def _validate_base_url(url: str) -> str:
    """Validate URL scheme to prevent SSRF attacks (WR-04)."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme for provider: {parsed.scheme!r}")
    return url.rstrip("/")


def _format_model_display_name(model_name: str) -> str:
    """Format model name for display in dropdown."""
    # Common model name cleanups
    name = model_name.replace("-", " ").replace("_", " ")
    
    # Capitalize words
    name = " ".join(word.capitalize() for word in name.split())
    
    # Common replacements
    replacements = {
        "Gpt": "GPT",
        "Claude": "Claude",
        "Llama": "Llama",
        "Mistral": "Mistral",
        "Qwen": "Qwen",
        "Gemma": "Gemma",
        "Deepseek": "DeepSeek",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    return name


@router.put("/llm-providers/db/{provider_id}")
async def update_llm_provider(
    provider_id: int,
    data: LLMProviderSchema,
    db: AsyncSession = Depends(get_db)
):
    """Update an LLM provider configuration (connection fields only per D-05)."""
    result = await db.execute(select(LLMProvider).where(LLMProvider.id == provider_id))
    provider = result.scalar_one_or_none()
    
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    # Keep old key if *** or empty string is sent (don't accidentally clear the key)
    if data.api_key and data.api_key != "***":
        provider.api_key = data.api_key
    # else: keep existing provider.api_key
    if data.api_base_url is not None:
        provider.api_base_url = data.api_base_url
    
    await db.commit()
    return {"success": True}


@router.patch("/llm-providers/db/{provider_id}")
async def patch_llm_provider(
    provider_id: int,
    data: LLMProviderPatchSchema,
    db: AsyncSession = Depends(get_db)
):
    """Update only connection fields (api_key, api_base_url) of an LLM provider.

    Does NOT update model fields — those are set via AppSettings key-value per LLM-08.
    """
    result = await db.execute(select(LLMProvider).where(LLMProvider.id == provider_id))
    provider = result.scalar_one_or_none()
    
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    # Only update fields that are provided
    if data.api_key is not None:
        # Don't overwrite with empty string if *** is sent (masked = don't change)
        if data.api_key and data.api_key != "***":
            provider.api_key = data.api_key
    
    if data.api_base_url is not None:
        provider.api_base_url = data.api_base_url
    
    await db.commit()
    return {"success": True}


@router.post("/llm-providers/db")
async def create_llm_provider(
    data: LLMProviderCreateSchema,
    db: AsyncSession = Depends(get_db)
):
    """Create a new LLM provider configuration.
    
    Used when user selects a provider from LiteLLM list that doesn't exist in DB yet.
    """
    # Check if provider with this name already exists
    result = await db.execute(select(LLMProvider).where(LLMProvider.name == data.name))
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(status_code=409, detail="Provider already exists")
    
    # Use provided display_name or derive from PROVIDER_DISPLAY_NAMES
    display_name = data.display_name or PROVIDER_DISPLAY_NAMES.get(data.name, data.name.title())
    
    provider = LLMProvider(
        name=data.name,
        display_name=display_name,
        api_key=data.api_key or "",
        api_base_url=data.api_base_url or "",
    )
    
    db.add(provider)
    await db.commit()
    await db.refresh(provider)
    
    return {"id": provider.id, "name": provider.name, "display_name": provider.display_name}


# Custom Prompts - Display names for UI
PROMPT_DISPLAY_NAMES = {
    "correspondents": "Korrespondenten gruppieren",
    "tags": "Tags gruppieren",
    "document_types": "Dokumententypen gruppieren",
    "tags_nonsense": "Sinnlose Tags erkennen",
    "tags_are_correspondents": "Tags als Korrespondenten erkennen",
    "tags_are_document_types": "Tags als Dokumententypen erkennen",
}

@router.get("/prompts")
async def get_prompts(db: AsyncSession = Depends(get_db)):
    """Get all custom prompts."""
    result = await db.execute(select(CustomPrompt).order_by(CustomPrompt.entity_type))
    prompts = result.scalars().all()
    
    # Get existing entity types
    existing_types = {p.entity_type for p in prompts}
    
    # Add missing prompts from DEFAULT_PROMPTS
    for entity_type, template in DEFAULT_PROMPTS.items():
        if entity_type not in existing_types:
            prompt = CustomPrompt(
                entity_type=entity_type,
                prompt_template=template,
                is_active=True
            )
            db.add(prompt)
    
    # Commit if we added any
    if len(existing_types) < len(DEFAULT_PROMPTS):
        await db.commit()
        result = await db.execute(select(CustomPrompt).order_by(CustomPrompt.entity_type))
        prompts = result.scalars().all()
    
    return [
        {
            "id": p.id,
            "entity_type": p.entity_type,
            "display_name": PROMPT_DISPLAY_NAMES.get(p.entity_type, p.entity_type),
            "prompt_template": p.prompt_template,
            "is_active": p.is_active
        }
        for p in prompts
    ]


@router.put("/prompts/{prompt_id}")
async def update_prompt(
    prompt_id: int,
    data: CustomPromptSchema,
    db: AsyncSession = Depends(get_db)
):
    """Update a custom prompt."""
    result = await db.execute(select(CustomPrompt).where(CustomPrompt.id == prompt_id))
    prompt = result.scalar_one_or_none()
    
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    prompt.prompt_template = data.prompt_template
    prompt.is_active = data.is_active
    
    await db.commit()
    return {"success": True}


@router.post("/prompts/reset/{entity_type}")
async def reset_prompt(
    entity_type: str,
    db: AsyncSession = Depends(get_db)
):
    """Reset a prompt to its default."""
    if entity_type not in DEFAULT_PROMPTS:
        raise HTTPException(status_code=400, detail="Invalid entity type")
    
    result = await db.execute(
        select(CustomPrompt).where(CustomPrompt.entity_type == entity_type)
    )
    prompt = result.scalar_one_or_none()
    
    if prompt:
        prompt.prompt_template = DEFAULT_PROMPTS[entity_type]
        await db.commit()
    
    return {"success": True, "prompt_template": DEFAULT_PROMPTS[entity_type]}


# Ignored Tags
class IgnoredTagSchema(BaseModel):
    pattern: str
    reason: Optional[str] = ""
    is_regex: bool = False


# Default ignored patterns
DEFAULT_IGNORED_TAGS = [
    {"pattern": "INBOX", "reason": "System-Tag für Paperless Eingang"},
    {"pattern": "ai-done", "reason": "Paperless-AI Verarbeitungsmarker"},
    {"pattern": "paperless-ai", "reason": "Paperless-AI System-Tag"},
    {"pattern": "TODO", "reason": "Aufgaben-Marker"},
    {"pattern": "*@*", "reason": "E-Mail-Adressen (Muster)"},
]


@router.get("/ignored-tags")
async def get_ignored_tags(db: AsyncSession = Depends(get_db)):
    """Get all ignored tag patterns."""
    result = await db.execute(select(IgnoredTag).order_by(IgnoredTag.pattern))
    tags = result.scalars().all()
    
    # If empty, create defaults
    if not tags:
        for item in DEFAULT_IGNORED_TAGS:
            tag = IgnoredTag(**item)
            db.add(tag)
        await db.commit()
        
        result = await db.execute(select(IgnoredTag).order_by(IgnoredTag.pattern))
        tags = result.scalars().all()
    
    return [
        {
            "id": t.id,
            "pattern": t.pattern,
            "reason": t.reason,
            "is_regex": t.is_regex
        }
        for t in tags
    ]


@router.post("/ignored-tags")
async def add_ignored_tag(
    data: IgnoredTagSchema,
    db: AsyncSession = Depends(get_db)
):
    """Add a new ignored tag pattern."""
    # Check if already exists
    result = await db.execute(
        select(IgnoredTag).where(IgnoredTag.pattern == data.pattern)
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(status_code=400, detail="Pattern already exists")
    
    tag = IgnoredTag(
        pattern=data.pattern,
        reason=data.reason,
        is_regex=data.is_regex
    )
    db.add(tag)
    await db.commit()
    await db.refresh(tag)
    
    return {
        "id": tag.id,
        "pattern": tag.pattern,
        "reason": tag.reason,
        "is_regex": tag.is_regex
    }


@router.delete("/ignored-tags/{tag_id}")
async def delete_ignored_tag(
    tag_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete an ignored tag pattern."""
    result = await db.execute(select(IgnoredTag).where(IgnoredTag.id == tag_id))
    tag = result.scalar_one_or_none()
    
    if not tag:
        raise HTTPException(status_code=404, detail="Pattern not found")
    
    await db.delete(tag)
    await db.commit()
    
    return {"success": True}


# App Settings (Debug Toggle, etc.)
class AppSettingsSchema(BaseModel):
    show_debug_menu: Optional[bool] = None
    sidebar_compact: Optional[bool] = None
    classifier_provider: Optional[str] = None
    classifier_model: Optional[str] = None  # Stored in key-value store (LLM-09)
    ocr_model: Optional[str] = None  # Stored in key-value store (LLM-09)


@router.get("/app")
async def get_app_settings(db: AsyncSession = Depends(get_db)):
    """Get application settings."""
    result = await db.execute(select(AppSettings).where(AppSettings.id == 1))
    settings = result.scalar_one_or_none()
    
    if not settings:
        # Create default settings
        settings = AppSettings(id=1)
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
    
    # Check key-value store for classifier_provider first (LLM-08)
    kv_classifier_provider = await get_setting(LLM_KEY_CLASSIFIER_PROVIDER, db)
    classifier_provider = kv_classifier_provider or ""
    
    # Get key-value settings for model fields (LLM-09)
    kv_classifier_model = await get_setting(LLM_KEY_CLASSIFIER_MODEL, db)
    kv_ocr_model = await get_setting(LLM_KEY_OCR_MODEL, db)

    # AuthConfig is the source of truth for password_set (D-07 clean break)
    auth_result = await db.execute(select(AuthConfig).where(AuthConfig.id == 1))
    auth_config = auth_result.scalar_one_or_none()
    password_set = bool(auth_config and auth_config.password_hash)
    
    return {
        "password_set": password_set,
        "show_debug_menu": settings.show_debug_menu,
        "sidebar_compact": settings.sidebar_compact,
        "classifier_provider": classifier_provider,
        "classifier_model": kv_classifier_model or "",
        "ocr_model": kv_ocr_model or "",
    }


@router.put("/app")
async def update_app_settings(
    data: AppSettingsSchema,
    db: AsyncSession = Depends(get_db)
):
    """Update application settings."""
    result = await db.execute(select(AppSettings).where(AppSettings.id == 1))
    settings = result.scalar_one_or_none()
    
    if not settings:
        settings = AppSettings(id=1)
        db.add(settings)
    
    if data.show_debug_menu is not None:
        settings.show_debug_menu = data.show_debug_menu
    
    if data.sidebar_compact is not None:
        settings.sidebar_compact = data.sidebar_compact

    if data.classifier_provider is not None:
        settings.classifier_provider = data.classifier_provider
        # Also update the key-value store (LLM-08)
        await set_setting(LLM_KEY_CLASSIFIER_PROVIDER, data.classifier_provider, "str", db)
    
    if data.classifier_model is not None:
        # Store in key-value store (LLM-09)
        await set_setting(LLM_KEY_CLASSIFIER_MODEL, data.classifier_model, "str", db)
    
    if data.ocr_model is not None:
        # Store in key-value store (LLM-09)
        await set_setting(LLM_KEY_OCR_MODEL, data.ocr_model, "str", db)
    
    await db.commit()
    
    return {"success": True}


# ── Key-Value Settings Endpoints (LLM-08) ──────────────────────────────────────

@router.get("/settings/{key}")
async def get_setting_endpoint(key: str, db: AsyncSession = Depends(get_db)):
    """Get a setting value by key."""
    value = await get_setting(key, db)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
    return {"key": key, "value": value}


@router.put("/settings/{key}")
async def set_setting_endpoint(
    key: str,
    data: SettingUpdateSchema,
    db: AsyncSession = Depends(get_db)
):
    """Set a setting value."""
    await set_setting(key, data.value, data.value_type or "str", db)
    return {"success": True, "key": key, "value": data.value}


@router.post("/settings/seed-llm-keys")
async def seed_llm_keys(db: AsyncSession = Depends(get_db)):
    """Seed LLM key-value settings from existing LLMProvider records. Run once during migration."""
    result = await db.execute(select(LLMProvider).limit(1))
    provider = result.scalar_one_or_none()
    if provider and provider.name:
        await set_setting(LLM_KEY_CLASSIFIER_PROVIDER, provider.name, "str", db)
    return {"success": True}

