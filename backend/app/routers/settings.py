from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models import PaperlessSettings, LLMProvider, CustomPrompt, IgnoredTag, AppSettings
from app.models.settings_model import (
    LLM_KEY_CLASSIFIER_PROVIDER,
    LLM_KEY_CLASSIFIER_MODEL,
    LLM_KEY_OCR_PROVIDER,
    LLM_KEY_OCR_MODEL,
)
import hashlib
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
            return
    result = await db.execute(
        select(AppSettings).where(AppSettings.key == key)
    )
    setting = result.scalar_one_or_none()
    if setting:
        setting.value = value
        setting.value_type = value_type
    else:
        setting = AppSettings(key=key, value=value, value_type=value_type)
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
    is_active: bool = False


class LLMProviderPatchSchema(BaseModel):
    """Patch schema for updating connection fields only (per D-05)."""
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None


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
    
    This is the DB-based endpoint kept for internal/admin use.
    The SettingsPanel uses the LiteLLM-based /llm-providers endpoint instead.
    """
    result = await db.execute(select(LLMProvider).order_by(LLMProvider.name))
    providers = result.scalars().all()
    
    ALL_DEFAULTS = [
        {"name": "openai", "display_name": "OpenAI"},
        {"name": "anthropic", "display_name": "Anthropic Claude"},
        {"name": "azure", "display_name": "Azure OpenAI"},
        {"name": "ollama", "display_name": "Ollama (Lokal)", "api_base_url": "http://localhost:11434"},
        {"name": "mistral", "display_name": "Mistral AI"},
        {"name": "openrouter", "display_name": "OpenRouter"},
    ]

    if not providers:
        for p in ALL_DEFAULTS:
            db.add(LLMProvider(**p))
        await db.commit()
        result = await db.execute(select(LLMProvider).order_by(LLMProvider.name))
        providers = result.scalars().all()
    else:
        existing_names = {p.name for p in providers}
        added = False
        for p in ALL_DEFAULTS:
            if p["name"] not in existing_names:
                db.add(LLMProvider(**p))
                added = True
        if added:
            await db.commit()
            result = await db.execute(select(LLMProvider).order_by(LLMProvider.name))
            providers = result.scalars().all()
    
    return [
        {
            "id": p.id,
            "name": p.name,
            "display_name": p.display_name,
            "api_key": "***" if p.api_key else "",
            "api_base_url": p.api_base_url or "",
            "is_active": p.is_active,
            "is_configured": p.is_configured,
        }
        for p in providers
    ]


# Provider display name mapping (used by both LiteLLM provider list and DB-based list)
PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic Claude",
    "azure": "Azure OpenAI",
    "ollama": "Ollama (Lokal)",
    "mistral": "Mistral AI",
    "openrouter": "OpenRouter",
    "google": "Google AI (Gemini)",
    "deepseek": "DeepSeek",
    "cohere": "Cohere",
    "groq": "Groq",
    "fireworks": "Fireworks AI",
    "anyscale": "Anyscale",
    "togetherai": "TogetherAI",
    "replicate": "Replicate",
    "cloudflare": "Cloudflare Workers AI",
    "aws": "AWS Bedrock",
    "vertex_ai": "Google Vertex AI",
    "sagemaker": "AWS SageMaker",
    "gemini": "Google Gemini",
    "xai": "xAI",
    "perplexity": "Perplexity",
    "meta": "Meta AI",
    "qwen": "Qwen (Alibaba)",
    "samba": "SambaNova",
    "ai21": "AI21 Labs",
    "bedrock": "AWS Bedrock",
    "volcengine": "Volcengine",
    "LINGYUN": "Lingyun",
    "sageng": "SAGEN",
    "MISTRAL": "Mistral AI",
    "openllm": "OpenLLM",
    "lmstudio": "LM Studio",
    "ollama": "Ollama (Lokal)",
    "localai": "LocalAI",
    "vllm": "vLLM",
    "tensorzero": "TensorZero",
}


# LLM Providers - LiteLLM-based (for SettingsPanel UI)
@router.get("/llm-providers")
async def get_llm_providers_from_litellm(db: AsyncSession = Depends(get_db)):
    """Get all LiteLLM-supported providers dynamically from litellm.provider_list enum.
    
    Uses litellm.provider_list which provides all 132+ supported provider names
    as an enum. Falls back to hardcoded list if LiteLLM fails.
    """
    try:
        import litellm
        
        # Use litellm.provider_list — enum of all 132+ supported providers
        providers = []
        for provider_enum in litellm.provider_list:
            provider_name = provider_enum.value
            providers.append({
                "name": provider_name,
                "display_name": PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name.title()),
            })
        
        # Sort by display_name
        providers.sort(key=lambda x: x["display_name"])
        return providers
    except Exception as e:
        # Fallback to hardcoded list
        return [
            {"name": "openai", "display_name": "OpenAI"},
            {"name": "anthropic", "display_name": "Anthropic Claude"},
            {"name": "azure", "display_name": "Azure OpenAI"},
            {"name": "ollama", "display_name": "Ollama (Lokal)"},
            {"name": "mistral", "display_name": "Mistral AI"},
            {"name": "openrouter", "display_name": "OpenRouter"},
        ]


@router.get("/llm-providers/models")
async def get_llm_provider_models(provider: str, db: AsyncSession = Depends(get_db)):
    """Get available models for a specific LiteLLM provider.
    
    For Ollama: fetches live models from /api/tags endpoint using provider's
    configured api_base_url from the database.
    
    For other providers: falls back to litellm.model_list filtered by 
    provider prefix (e.g., "openai/").
    """
    try:
        # Special handling for Ollama — fetch live models from /api/tags
        if provider == "ollama":
            # Look up Ollama provider config from DB
            result = await db.execute(
                select(LLMProvider).where(LLMProvider.name == "ollama")
            )
            db_provider = result.scalar_one_or_none()
            api_base = (db_provider.api_base_url or "http://localhost:11434") if db_provider else "http://localhost:11434"
            
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{api_base}/api/tags")
                    response.raise_for_status()
                    models_data = response.json().get("models", [])
                    return {
                        "provider": provider,
                        "models": [
                            {
                                "id": m["name"],
                                "name": m["name"],
                                "display_name": _format_model_display_name(m["name"]),
                            }
                            for m in models_data
                        ]
                    }
            except Exception:
                # Fall through to litellm.model_list fallback below
                pass
        
        # Fall back to litellm.model_list for all providers
        import litellm
        all_models = litellm.model_list
        
        # Filter models for the specified provider
        # Provider format in model list: "provider/model-name" (e.g., "openai/gpt-4o")
        prefix = f"{provider}/"
        models = []
        seen = set()
        
        for model in all_models:
            if model.startswith(prefix):
                # Extract model name without provider prefix
                model_name = model[len(prefix):]
                if model_name not in seen:
                    seen.add(model_name)
                    models.append({
                        "id": model_name,
                        "name": model_name,
                        "display_name": _format_model_display_name(model_name),
                    })
        
        return {"provider": provider, "models": sorted(models, key=lambda x: x["name"])}
        
    except Exception as e:
        return {"provider": provider, "models": [], "error": str(e)}


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
    
    # If setting this provider as active, deactivate others
    if data.is_active:
        await db.execute(
            LLMProvider.__table__.update().values(is_active=False)
        )
    
    # Keep old key if *** or empty string is sent (don't accidentally clear the key)
    if data.api_key and data.api_key != "***":
        provider.api_key = data.api_key
    # else: keep existing provider.api_key
    if data.api_base_url is not None:
        provider.api_base_url = data.api_base_url
    provider.is_active = data.is_active
    # Update is_configured based on provider type (per D-05)
    provider.update_configured()
    
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
    
    # Update is_configured based on provider type
    provider.update_configured()
    
    await db.commit()
    return {"success": True}


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


# App Settings (Password, Debug Toggle, etc.)
class AppSettingsSchema(BaseModel):
    password_enabled: Optional[bool] = None
    password: Optional[str] = None  # Plain password, will be hashed
    show_debug_menu: Optional[bool] = None
    sidebar_compact: Optional[bool] = None
    classifier_provider: Optional[str] = None
    classifier_model: Optional[str] = None  # Stored in key-value store (LLM-09)
    ocr_model: Optional[str] = None  # Stored in key-value store (LLM-09)


class PasswordVerifySchema(BaseModel):
    password: str


def hash_password(password: str) -> str:
    """Simple password hashing."""
    return hashlib.sha256(password.encode()).hexdigest()


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
    classifier_provider = kv_classifier_provider or getattr(settings, "classifier_provider", "ollama") or "ollama"
    
    # Get key-value settings for model fields (LLM-09)
    kv_classifier_model = await get_setting(LLM_KEY_CLASSIFIER_MODEL, db)
    kv_ocr_model = await get_setting(LLM_KEY_OCR_MODEL, db)
    
    return {
        "password_enabled": settings.password_enabled,
        "password_set": bool(settings.password_hash),
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
    
    if data.password_enabled is not None:
        settings.password_enabled = data.password_enabled
    
    if data.password is not None and data.password:
        settings.password_hash = hash_password(data.password)
    
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


@router.post("/app/verify-password")
async def verify_password(
    data: PasswordVerifySchema,
    db: AsyncSession = Depends(get_db)
):
    """Verify the UI password."""
    result = await db.execute(select(AppSettings).where(AppSettings.id == 1))
    settings = result.scalar_one_or_none()
    
    if not settings or not settings.password_enabled:
        return {"valid": True, "password_required": False}
    
    if not settings.password_hash:
        return {"valid": True, "password_required": False}
    
    is_valid = settings.password_hash == hash_password(data.password)
    return {"valid": is_valid, "password_required": True}


@router.delete("/app/password")
async def remove_password(db: AsyncSession = Depends(get_db)):
    """Remove the UI password."""
    result = await db.execute(select(AppSettings).where(AppSettings.id == 1))
    settings = result.scalar_one_or_none()
    
    if settings:
        settings.password_enabled = False
        settings.password_hash = ""
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
    result = await db.execute(select(LLMProvider).where(LLMProvider.is_active == True))
    provider = result.scalar_one_or_none()
    if provider:
        if provider.name:
            await set_setting(LLM_KEY_CLASSIFIER_PROVIDER, provider.name, "str", db)
        if hasattr(provider, "classifier_model") and provider.classifier_model:
            await set_setting(LLM_KEY_CLASSIFIER_MODEL, provider.classifier_model, "str", db)
    return {"success": True}

