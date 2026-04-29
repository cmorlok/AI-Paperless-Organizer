"""Settings service — business logic extracted from routers/settings.py.

Houses key-value helpers, app settings aggregation/update, and prompt loading
with default insertion.  The router stays thin: validate → call service → return.
"""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import AppSettings, CustomPrompt
from app.models.auth_config import AuthConfig
from app.models.settings_model import (
    LLM_KEY_CLASSIFIER_MODEL,
    LLM_KEY_CLASSIFIER_PROVIDER,
    LLM_KEY_OCR_MODEL,
)
from app.prompts.default_prompts import DEFAULT_PROMPTS


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


# ── Prompt display names ────────────────────────────────────────────────────────

PROMPT_DISPLAY_NAMES = {
    "correspondents": "Korrespondenten gruppieren",
    "tags": "Tags gruppieren",
    "document_types": "Dokumententypen gruppieren",
    "tags_nonsense": "Sinnlose Tags erkennen",
    "tags_are_correspondents": "Tags als Korrespondenten erkennen",
    "tags_are_document_types": "Tags als Dokumententypen erkennen",
}


# ── App Settings ────────────────────────────────────────────────────────────────

async def get_app_settings(db: AsyncSession) -> dict:
    """Aggregate application settings from AppSettings row + key-value store + AuthConfig."""
    result = await db.execute(select(AppSettings).where(AppSettings.id == 1))
    settings = result.scalar_one_or_none()

    if not settings:
        settings = AppSettings(id=1)
        db.add(settings)
        await db.commit()
        await db.refresh(settings)

    # Key-value store overrides (LLM-08/09)
    kv_classifier_provider = await get_setting(LLM_KEY_CLASSIFIER_PROVIDER, db)
    classifier_provider = kv_classifier_provider or ""

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


async def update_app_settings(
    *,
    show_debug_menu: Optional[bool] = None,
    sidebar_compact: Optional[bool] = None,
    classifier_provider: Optional[str] = None,
    classifier_model: Optional[str] = None,
    ocr_model: Optional[str] = None,
    db: AsyncSession,
) -> None:
    """Update application settings — merges AppSettings row with KV store."""
    result = await db.execute(select(AppSettings).where(AppSettings.id == 1))
    settings = result.scalar_one_or_none()

    if not settings:
        settings = AppSettings(id=1)
        db.add(settings)

    if show_debug_menu is not None:
        settings.show_debug_menu = show_debug_menu

    if sidebar_compact is not None:
        settings.sidebar_compact = sidebar_compact

    if classifier_provider is not None:
        settings.classifier_provider = classifier_provider
        await set_setting(LLM_KEY_CLASSIFIER_PROVIDER, classifier_provider, "str", db)

    if classifier_model is not None:
        await set_setting(LLM_KEY_CLASSIFIER_MODEL, classifier_model, "str", db)

    if ocr_model is not None:
        await set_setting(LLM_KEY_OCR_MODEL, ocr_model, "str", db)

    await db.commit()


# ── Prompts ─────────────────────────────────────────────────────────────────────

async def get_prompts(db: AsyncSession) -> list[dict]:
    """Load all prompts, inserting missing defaults from DEFAULT_PROMPTS."""
    result = await db.execute(select(CustomPrompt).order_by(CustomPrompt.entity_type))
    prompts = result.scalars().all()

    existing_types = {p.entity_type for p in prompts}

    # Seed missing defaults
    for entity_type, template in DEFAULT_PROMPTS.items():
        if entity_type not in existing_types:
            prompt = CustomPrompt(
                entity_type=entity_type,
                prompt_template=template,
                is_active=True,
            )
            db.add(prompt)

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
            "is_active": p.is_active,
        }
        for p in prompts
    ]
