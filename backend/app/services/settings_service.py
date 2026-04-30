"""Settings service — business logic extracted from routers/settings.py.

Houses key-value helpers, app settings aggregation/update, and prompt loading
with default insertion.  The router stays thin: validate → call service → return.
"""

from typing import Optional

from sqlalchemy import select, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

import ipaddress
import socket
from urllib.parse import urlparse

from app.database import get_db
from app.models import AppSettings, CustomPrompt, PaperlessSettings, LLMProvider, IgnoredTag
from app.models.auth_config import AuthConfig
from app.models.settings_model import (
    LLM_KEY_CLASSIFIER_MODEL,
    LLM_KEY_CLASSIFIER_PROVIDER,
    LLM_KEY_OCR_MODEL,
)


# ── Key-Value Setting Helpers (LLM-08) ─────────────────────────────────────────

async def get_setting(key: str, db: AsyncSession) -> Optional[str]:
    """Get a setting value by key. Returns None if not found."""
    result = await db.execute(
        select(AppSettings).where(AppSettings.key == key)
    )
    setting = result.scalar_one_or_none()
    return setting.value if setting else None


async def set_setting(key: str, value: str, value_type: str = "str", db: AsyncSession | None = None):
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
    """Load all prompts from database. Services register their prompts on startup."""
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


# ── Paperless Settings ────────────────────────────────────────────────────────

async def get_paperless_settings(db: AsyncSession) -> dict:
    """Get Paperless connection settings."""
    result = await db.execute(select(PaperlessSettings).where(PaperlessSettings.id == 1))
    settings = result.scalar_one_or_none()
    if not settings:
        return {"url": "", "api_token": "", "is_configured": False}
    return {
        "url": settings.url,
        "api_token": "***" if settings.api_token else "",
        "is_configured": settings.is_configured,
    }


def _validate_base_url(url: str) -> None:
    """Validate URL against SSRF: reject private/link-local IP addresses.

    Note: DNS rebinding is not addressed — validation resolves the hostname at
    check time, but a malicious DNS server could return a public IP then a
    private IP at request time. Acceptable for a sidecar service over an
    internal network.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Ungültige URL: kein Hostname gefunden")
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL muss http oder https verwenden")
    # Resolve hostname to IP addresses
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise ValueError("Hostname konnte nicht aufgelöst werden")
    for family, _, _, _, sockaddr in addr_info:
        addr = sockaddr[0]
        ip = ipaddress.ip_address(addr)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError("URL resolves to a private/internal IP address")
        # Also block IPv6 private/loopback/link-local
        if ip.is_private:
            raise ValueError("URL resolves to a private/internal IP address")


async def save_paperless_settings(url: str, api_token: str, db: AsyncSession) -> dict:
    """Save Paperless connection settings."""
    _validate_base_url(url)
    result = await db.execute(select(PaperlessSettings).where(PaperlessSettings.id == 1))
    settings = result.scalar_one_or_none()
    if settings:
        settings.url = url
        if api_token and api_token != "***":
            settings.api_token = api_token
        actual_token = settings.api_token if api_token == "***" else api_token
        settings.is_configured = bool(url and actual_token)
    else:
        if api_token == "***":
            raise ValueError("API Token muss angegeben werden")
        settings = PaperlessSettings(
            id=1, url=url, api_token=api_token,
            is_configured=bool(url and api_token),
        )
        db.add(settings)
    await db.commit()
    return {"success": True, "is_configured": settings.is_configured}


# ── LLM Providers (DB-based) ─────────────────────────────────────────────────

async def get_llm_providers_from_db(db: AsyncSession) -> list[dict]:
    """Get all LLM provider configurations from database."""
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


async def update_llm_provider(
    provider_id: int, name: str, display_name: str,
    api_key: Optional[str], api_base_url: Optional[str],
    db: AsyncSession,
) -> dict:
    """Update an LLM provider configuration (connection fields only)."""
    result = await db.execute(select(LLMProvider).where(LLMProvider.id == provider_id))
    provider = result.scalar_one_or_none()
    if not provider:
        raise ValueError("Provider not found")
    if api_key and api_key != "***":
        provider.api_key = api_key
    if api_base_url is not None:
        provider.api_base_url = api_base_url
    await db.commit()
    return {"success": True}


async def patch_llm_provider(
    provider_id: int,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    db: AsyncSession = None,
) -> dict:
    """Update only connection fields (api_key, api_base_url) of an LLM provider."""
    result = await db.execute(select(LLMProvider).where(LLMProvider.id == provider_id))
    provider = result.scalar_one_or_none()
    if not provider:
        raise ValueError("Provider not found")
    if api_key is not None and api_key and api_key != "***":
        provider.api_key = api_key
    if api_base_url is not None:
        provider.api_base_url = api_base_url
    await db.commit()
    return {"success": True}


async def create_llm_provider(
    name: str, display_name: Optional[str], api_key: Optional[str],
    api_base_url: Optional[str], db: AsyncSession,
) -> dict:
    """Create a new LLM provider configuration."""
    result = await db.execute(select(LLMProvider).where(LLMProvider.name == name))
    existing = result.scalar_one_or_none()
    if existing:
        raise ValueError("already_exists")
    from app.services.llm import PROVIDER_DISPLAY_NAMES
    resolved_display_name = display_name or PROVIDER_DISPLAY_NAMES.get(name, name.title())
    provider = LLMProvider(
        name=name, display_name=resolved_display_name,
        api_key=api_key or "", api_base_url=api_base_url or "",
    )
    db.add(provider)
    await db.commit()
    await db.refresh(provider)
    return {"id": provider.id, "name": provider.name, "display_name": provider.display_name}


# ── Prompts (update/reset) ───────────────────────────────────────────────────

async def update_prompt(prompt_id: int, prompt_template: str, is_active: bool, db: AsyncSession) -> dict:
    """Update a custom prompt."""
    result = await db.execute(select(CustomPrompt).where(CustomPrompt.id == prompt_id))
    prompt = result.scalar_one_or_none()
    if not prompt:
        raise ValueError("Prompt not found")
    prompt.prompt_template = prompt_template
    prompt.is_active = is_active
    prompt.modified = True
    await db.commit()
    return {"success": True}


async def reset_prompt(entity_type: str, db: AsyncSession) -> dict:
    """Reset a prompt to its default by setting modified=False.

    On next registration (service startup), the prompt will be updated from
    the service's current source value.
    """
    result = await db.execute(select(CustomPrompt).where(CustomPrompt.entity_type == entity_type))
    prompt = result.scalar_one_or_none()
    if prompt:
        prompt.modified = False
        await db.commit()
    return {"success": True}


async def get_prompt(key: str, db: AsyncSession) -> Optional[str]:
    """Get a prompt template by entity type. Returns None if not found or not active."""
    result = await db.execute(
        select(CustomPrompt).where(
            CustomPrompt.entity_type == key,
            CustomPrompt.is_active
        )
    )
    prompt = result.scalar_one_or_none()
    return prompt.prompt_template if prompt else None


async def register_prompt(key: str, prompt: str, db: AsyncSession) -> None:
    """Register a prompt: insert if not exists, update if exists and not modified by user."""
    result = await db.execute(select(CustomPrompt).where(CustomPrompt.entity_type == key))
    existing = result.scalar_one_or_none()

    if existing is None:
        new_prompt = CustomPrompt(
            entity_type=key,
            prompt_template=prompt,
            is_active=True,
            modified=False,
        )
        db.add(new_prompt)
        await db.commit()
    elif not existing.modified:
        existing.prompt_template = prompt
        await db.commit()


# ── Ignored Tags ─────────────────────────────────────────────────────────────

DEFAULT_IGNORED_TAGS = [
    {"pattern": "INBOX", "reason": "System-Tag für Paperless Eingang"},
    {"pattern": "ai-done", "reason": "Paperless-AI Verarbeitungsmarker"},
    {"pattern": "paperless-ai", "reason": "Paperless-AI System-Tag"},
    {"pattern": "TODO", "reason": "Aufgaben-Marker"},
    {"pattern": "*@*", "reason": "E-Mail-Adressen (Muster)"},
]


async def get_ignored_tags(db: AsyncSession) -> list[dict]:
    """Get all ignored tag patterns, creating defaults if empty."""
    result = await db.execute(select(IgnoredTag).order_by(IgnoredTag.pattern))
    tags = result.scalars().all()
    if not tags:
        for item in DEFAULT_IGNORED_TAGS:
            tag = IgnoredTag(**item)
            db.add(tag)
        await db.commit()
        result = await db.execute(select(IgnoredTag).order_by(IgnoredTag.pattern))
        tags = result.scalars().all()
    return [
        {"id": t.id, "pattern": t.pattern, "reason": t.reason, "is_regex": t.is_regex}
        for t in tags
    ]


async def add_ignored_tag(pattern: str, reason: str, is_regex: bool, db: AsyncSession) -> dict:
    """Add a new ignored tag pattern."""
    result = await db.execute(select(IgnoredTag).where(IgnoredTag.pattern == pattern))
    existing = result.scalar_one_or_none()
    if existing:
        raise ValueError("already_exists")
    tag = IgnoredTag(pattern=pattern, reason=reason, is_regex=is_regex)
    db.add(tag)
    await db.commit()
    await db.refresh(tag)
    return {"id": tag.id, "pattern": tag.pattern, "reason": tag.reason, "is_regex": tag.is_regex}


async def delete_ignored_tag(tag_id: int, db: AsyncSession) -> None:
    """Delete an ignored tag pattern."""
    result = await db.execute(select(IgnoredTag).where(IgnoredTag.id == tag_id))
    tag = result.scalar_one_or_none()
    if not tag:
        raise ValueError("Pattern not found")
    await db.execute(sa_delete(IgnoredTag).where(IgnoredTag.id == tag_id))
    await db.commit()


# ── Key-Value Settings (seed) ────────────────────────────────────────────────

async def seed_llm_keys(db: AsyncSession) -> dict:
    """Seed LLM key-value settings from existing LLMProvider records."""
    result = await db.execute(select(LLMProvider).limit(1))
    provider = result.scalar_one_or_none()
    if provider and provider.name:
        await set_setting(LLM_KEY_CLASSIFIER_PROVIDER, provider.name, "str", db)
    return {"success": True}
