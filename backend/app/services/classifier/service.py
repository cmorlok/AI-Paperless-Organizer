"""Main classifier service that orchestrates document classification."""

from __future__ import annotations

import asyncio
import json as _json
import logging
import re
from collections import Counter
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from sqlalchemy import select, func as sa_func, case as sa_case

from app.models.classifier import (
    ClassifierConfig, StoragePathProfile, CustomFieldMapping, ClassificationHistory,
)
from app.models import LLMProvider
from app.models.settings_model import (
    LLM_KEY_CLASSIFIER_PROVIDER,
    LLM_KEY_CLASSIFIER_MODEL,
)
from app.services.settings_service import get_setting
from app.services.paperless import PaperlessClient
from app.services.classifier.base_provider import (
    BaseClassifierProvider, ClassificationResult, DocumentContext,
)
from app.services.classifier.litellm_provider import (
    LitellmToolCallingProvider,
    LitellmOllamaProvider,
)
from app.services.classifier.tool_executor import ToolExecutor
from app.services.classifier.state import AutoClassifyState
from app.services.llm import LLMService

logger = logging.getLogger(__name__)

# ── Legal form stripping ───────────────────────────────────────────────────────
# Ordered longest-first so "GmbH & Co. KG" is matched before "GmbH" or "KG"
_LEGAL_FORMS = [
    r"GmbH\s*&\s*Co\.?\s*KGaA",
    r"GmbH\s*&\s*Co\.?\s*KG",
    r"GmbH\s*&\s*Co\.?",
    r"AG\s*&\s*Co\.?\s*KG",
    r"UG\s*\(haftungsbeschr[äa]nkt\)",
    r"GmbH",
    r"AG",
    r"KGaA",
    r"KG",
    r"OHG",
    r"GbR",
    r"e\.?\s*V\.?",
    r"e\.?\s*G\.?",
    r"e\.?\s*K\.?",
    r"SE",
    r"UG",
    r"mbH",
    r"Ltd\.?",
    r"Inc\.?",
    r"Corp\.?",
    r"P\.?L\.?C\.?",
    r"SARL",
    r"S\.?\s*A\.?",
    r"N\.?\s*V\.?",
    r"B\.?\s*V\.?",
    r"i\.?\s*Gr\.?",      # in Gründung
    r"i\.?\s*L\.?",       # in Liquidation
]
_LEGAL_SUFFIX_RE = re.compile(
    r"[\s,]+(" + "|".join(_LEGAL_FORMS) + r")\s*$",
    re.IGNORECASE,
)


def _strip_legal_forms(name: str) -> str:
    """Remove German/international legal form suffixes from a company name."""
    if not name:
        return name
    # Apply up to 3 times to strip chained suffixes like "GmbH & Co. KG"
    for _ in range(3):
        stripped = _LEGAL_SUFFIX_RE.sub("", name).strip(" ,.")
        if stripped == name:
            break
        name = stripped
    return name.strip()


# Reference indicators that legitimise a YYYY-NNNNN number in a title.
_TITLE_REF_INDICATORS = re.compile(
    r"\b(?:Nr|Re|Ref|Rechnung|Auftrag|Aktenzeichen|Vertrags?|AZ|Az)\s*[-.:]\s*$",
    re.IGNORECASE,
)
# Matches "YYYY-NNNNN" style numbers (year-personalnumber) in titles.
_TITLE_YEAR_ID_RE = re.compile(r"\b((?:19|20)\d{2})-(\d{4,6})\b")


def _clean_title(title: str, created_date: str | None = None) -> str:
    """Minimal safety net: remove obvious personal-number patterns from titles.

    Only removes "YYYY-NNNNN" patterns that are NOT preceded by a reference
    keyword (Nr., Re-, Rechnung, ...). Everything else is left to the LLM.
    """
    if not title:
        return title

    def _replace_fake_ref(m: re.Match) -> str:
        before = title[: m.start()].rstrip()
        if _TITLE_REF_INDICATORS.search(before):
            return m.group(0)  # keep — it's a real reference number
        return ""  # remove the entire YYYY-NNNNN pattern

    cleaned = _TITLE_YEAR_ID_RE.sub(_replace_fake_ref, title)
    cleaned = re.sub(r"  +", " ", cleaned).strip(" -,.")
    return cleaned


# ── Custom field type prompts & validation ─────────────────────────────────────
FIELD_TYPE_PROMPTS = {
    "rechnungsnummer": "Extrahiere die Rechnungsnummer/Belegnummer. Suche nach 'Rechnungsnr', 'RE-', 'Invoice', 'Beleg-Nr' o.ae.",
    "betrag": "Extrahiere den Gesamtbetrag (brutto inkl. MwSt) als Zahl. Punkt als Dezimaltrenner, kein Waehrungszeichen, kein Tausendertrennzeichen. Beispiel: 149.99 statt 149,99 EUR. Bei mehreren Betraegen den Gesamtbetrag (Summe/Total) nehmen.",
    "gesamtbetrag": "Extrahiere den Gesamtbetrag (brutto inkl. MwSt) als Zahl. Punkt als Dezimaltrenner, kein Waehrungszeichen, kein Tausendertrennzeichen. Beispiel: 149.99 statt 149,99 EUR. Bei mehreren Betraegen den Gesamtbetrag (Summe/Total) nehmen.",
    "iban": "Extrahiere die IBAN/Kontonummer des ABSENDERS/EMPFAENGERS (nicht die eigene!). Format: ohne Leerzeichen. Bei aelteren Dokumenten ggf. Kontonummer+BLZ.",
    "kontonummer": "Extrahiere die IBAN/Kontonummer des ABSENDERS/EMPFAENGERS (nicht die eigene!). Format: ohne Leerzeichen. Bei aelteren Dokumenten ggf. Kontonummer+BLZ.",
    "kundennummer": "Extrahiere die Kundennummer/Vertragsnummer. Suche nach 'Kundennr', 'Kd-Nr', 'Vertragsnr' o.ae.",
    "steuernummer": "Extrahiere die Steuernummer oder USt-IdNr. Format: DE + 9 Ziffern (USt-ID) oder XX/XXX/XXXXX.",
    "faelligkeitsdatum": "Extrahiere das Faelligkeitsdatum/Zahlungsziel. Format: YYYY-MM-DD. Suche nach 'zahlbar bis', 'faellig am'.",
    "lieferscheinnummer": "Extrahiere die Lieferscheinnummer. Suche nach 'Lieferschein-Nr', 'LS-Nr', 'Delivery Note' o.ae.",
    "bestellnummer": "Extrahiere die Bestellnummer. Suche nach 'Bestell-Nr', 'Order', 'Auftragsnr' o.ae.",
}

FIELD_TYPE_VALIDATION = {
    "iban": r"^[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}$",
    "betrag": r"^\d+(\.\d{1,2})?$",
    "gesamtbetrag": r"^\d+(\.\d{1,2})?$",
    "faelligkeitsdatum": r"^\d{4}-\d{2}-\d{2}$",
}


class DocumentClassifierService:
    """Orchestrates document classification using the configured provider."""

    def __init__(
        self,
        paperless: Optional[PaperlessClient] = None,
        session_factory: Optional[Any] = None,
        state: Optional[AutoClassifyState] = None,
        llm_service: Optional[LLMService] = None,
    ):
        self.paperless = paperless
        self.session_factory = session_factory
        self.state = state
        self.llm_service = llm_service

    async def get_config(self) -> ClassifierConfig:
        if self.session_factory is None:
            raise RuntimeError("No session_factory configured")
        async with self.session_factory() as db:
            result = await db.execute(
                select(ClassifierConfig).where(ClassifierConfig.id == 1)
            )
            config = result.scalar_one_or_none()
            if not config:
                config = ClassifierConfig(id=1)
                db.add(config)
                await db.commit()
                await db.refresh(config)
            return config

    async def save_config(self, data: Dict[str, Any]) -> ClassifierConfig:
        if self.session_factory is None:
            raise RuntimeError("No session_factory configured")
        async with self.session_factory() as db:
            result = await db.execute(
                select(ClassifierConfig).where(ClassifierConfig.id == 1)
            )
            config = result.scalar_one_or_none()
            if not config:
                config = ClassifierConfig(id=1)
                db.add(config)
            for key, value in data.items():
                if hasattr(config, key) and key not in ("id", "created_at", "updated_at"):
                    setattr(config, key, value)
            await db.commit()
            await db.refresh(config)
            return config

    async def get_config_response(self, active_provider: str, active_model: str) -> Dict[str, Any]:
        """Build full config response dict for the API."""
        cfg = await self.get_config()
        return {
            "active_provider": active_provider,
            "active_model": active_model,
            "enable_title": cfg.enable_title,
            "enable_tags": cfg.enable_tags,
            "enable_correspondent": cfg.enable_correspondent,
            "enable_document_type": cfg.enable_document_type,
            "enable_storage_path": cfg.enable_storage_path,
            "enable_created_date": cfg.enable_created_date,
            "enable_custom_fields": cfg.enable_custom_fields,
            "tag_behavior": cfg.tag_behavior,
            "tags_min": cfg.tags_min or 1,
            "tags_max": cfg.tags_max or 5,
            "tags_keep_existing": cfg.tags_keep_existing if cfg.tags_keep_existing is not None else True,
            "tags_ignore": cfg.tags_ignore or [],
            "tags_protected": cfg.tags_protected or [],
            "dates_ignore": cfg.dates_ignore or [],
            "storage_path_behavior": cfg.storage_path_behavior or "always",
            "storage_path_override_names": cfg.storage_path_override_names or ["Zuweisen"],
            "correspondent_behavior": cfg.correspondent_behavior,
            "review_mode": cfg.review_mode,
            "batch_size": cfg.batch_size,
            "prompt_title": cfg.prompt_title or "",
            "prompt_tags": cfg.prompt_tags or "",
            "prompt_correspondent": cfg.prompt_correspondent or "",
            "prompt_document_type": cfg.prompt_document_type or "",
            "prompt_date": cfg.prompt_date or "",
            "system_prompt": cfg.system_prompt,
            "excluded_tag_ids": cfg.excluded_tag_ids or [],
            "excluded_correspondent_ids": cfg.excluded_correspondent_ids or [],
            "excluded_document_type_ids": cfg.excluded_document_type_ids or [],
            "correspondent_trim_prompt": bool(getattr(cfg, "correspondent_trim_prompt", False)),
            "correspondent_strip_legal": bool(getattr(cfg, "correspondent_strip_legal", False)),
            "correspondent_ignore": getattr(cfg, "correspondent_ignore", None) or [],
            "auto_classify_enabled": bool(getattr(cfg, "auto_classify_enabled", False)),
            "auto_classify_interval": getattr(cfg, "auto_classify_interval", 5) or 5,
            "auto_classify_mode": getattr(cfg, "auto_classify_mode", "review") or "review",
        }

    async def get_storage_profiles(self) -> List[StoragePathProfile]:
        assert self.paperless is not None
        if self.session_factory is None:
            return []
        async with self.session_factory() as db:
            result = await db.execute(
                select(StoragePathProfile).order_by(StoragePathProfile.person_name)
            )
            return list(result.scalars().all())

    async def save_storage_profile(self, data: Dict[str, Any]) -> StoragePathProfile:
        assert self.paperless is not None
        if self.session_factory is None:
            raise RuntimeError("No session_factory configured")
        async with self.session_factory() as db:
            path_id = data.get("paperless_path_id")
            result = await db.execute(
                select(StoragePathProfile).where(
                    StoragePathProfile.paperless_path_id == path_id
                )
            )
            profile = result.scalar_one_or_none()
            if not profile:
                profile = StoragePathProfile(paperless_path_id=path_id)
                db.add(profile)

            for key, value in data.items():
                if hasattr(profile, key) and key not in ("id", "created_at", "updated_at"):
                    setattr(profile, key, value)

            await db.commit()
            await db.refresh(profile)
            return profile

    async def get_custom_field_mappings(self) -> List[CustomFieldMapping]:
        assert self.paperless is not None
        assert self.llm_service is not None
        if self.session_factory is None:
            return []
        async with self.session_factory() as db:
            result = await db.execute(
                select(CustomFieldMapping).order_by(CustomFieldMapping.paperless_field_name)
            )
            return list(result.scalars().all())

    async def save_custom_field_mapping(self, data: Dict[str, Any]) -> CustomFieldMapping:
        assert self.paperless is not None
        assert self.llm_service is not None
        if self.session_factory is None:
            raise RuntimeError("No session_factory configured")
        async with self.session_factory() as db:
            field_id = data.get("paperless_field_id")
            result = await db.execute(
                select(CustomFieldMapping).where(
                    CustomFieldMapping.paperless_field_id == field_id
                )
            )
            mapping = result.scalar_one_or_none()
            if not mapping:
                mapping = CustomFieldMapping(paperless_field_id=field_id)
                db.add(mapping)

            for key, value in data.items():
                if hasattr(mapping, key) and key not in ("id", "created_at", "updated_at"):
                    setattr(mapping, key, value)

            await db.commit()
            await db.refresh(mapping)
            return mapping

    def _build_tool_executor(
        self, config: ClassifierConfig,
        storage_profiles: list, field_mappings: list,
    ) -> ToolExecutor:
        assert self.paperless is not None
        return ToolExecutor(
            paperless=self.paperless,
            storage_profiles=storage_profiles,
            custom_field_mappings=field_mappings,
            excluded_tag_ids=config.excluded_tag_ids or [],
            excluded_correspondent_ids=config.excluded_correspondent_ids or [],
            excluded_document_type_ids=config.excluded_document_type_ids or [],
            tags_ignore=config.tags_ignore or [],
        )

    async def _get_llm_provider(self, provider_name: str) -> 'LLMProvider':
        assert self.paperless is not None
        assert self.llm_service is not None
        """Get a configured LLMProvider from the central table."""
        if self.session_factory is None:
            raise RuntimeError("No session_factory configured")
        async with self.session_factory() as db:
            result = await db.execute(
                select(LLMProvider).where(LLMProvider.name == provider_name)
            )
            provider = result.scalar_one_or_none()
            if not provider:
                raise ValueError(f"Provider '{provider_name}' not found in LLM settings.")
            return provider

    async def _get_classifier_provider_name(self) -> str:
        assert self.paperless is not None
        assert self.llm_service is not None
        """Get the classifier provider name from AppSettings key-value store (LLM-08)."""
        if self.session_factory is None:
            raise ValueError("classifier_provider is not configured")
        # Try key-value store first
        from app.services.settings_service import get_setting
        async with self.session_factory() as db:
            kv_provider = await get_setting(LLM_KEY_CLASSIFIER_PROVIDER, db)
            if kv_provider:
                return kv_provider
            # Fall back to scalar column for backward compatibility
            from app.models import AppSettings
            result = await db.execute(select(AppSettings).where(AppSettings.id == 1))
            app_settings = result.scalar_one_or_none()
            if app_settings and getattr(app_settings, "classifier_provider", None):
                return app_settings.classifier_provider
            raise ValueError("classifier_provider is not configured")

    async def _build_provider(self, config: ClassifierConfig) -> BaseClassifierProvider:
        assert self.paperless is not None
        assert self.llm_service is not None
        """Build the appropriate provider based on central LLM settings."""
        storage_profiles = await self.get_storage_profiles()
        field_mappings = await self.get_custom_field_mappings()
        tool_executor = self._build_tool_executor(config, storage_profiles, field_mappings)

        provider_name = await self._get_classifier_provider_name()
        return await self._create_provider_instance(provider_name, tool_executor)

    async def _create_provider_instance(
        self, provider_name: str, tool_executor: ToolExecutor,
        model_override: Optional[str] = None,
    ) -> BaseClassifierProvider:
        assert self.llm_service is not None
        assert self.paperless is not None
        """Create a provider instance from the central LLMProvider table."""
        if self.session_factory is not None:
            async with self.session_factory() as db:
                model = model_override or await get_setting(LLM_KEY_CLASSIFIER_MODEL, db) or ""
        else:
            model = model_override or ""

        if provider_name == "ollama":
            return LitellmOllamaProvider(model=model, provider=provider_name, tool_executor=tool_executor, llm_service=self.llm_service)

        from app.services.llm import PROVIDER_DISPLAY_NAMES
        label = PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name.replace("_", " ").title())
        return LitellmToolCallingProvider(model=model, provider=provider_name, tool_executor=tool_executor, provider_label=label, llm_service=self.llm_service)

    async def _get_active_classifier_provider_name(self) -> str:
        assert self.paperless is not None
        """Alias for backward compat."""
        return await self._get_classifier_provider_name()

    def _build_config_dict(self, config: ClassifierConfig) -> Dict[str, Any]:
        return {
            "enable_title": config.enable_title,
            "enable_tags": config.enable_tags,
            "enable_correspondent": config.enable_correspondent,
            "enable_document_type": config.enable_document_type,
            "enable_storage_path": config.enable_storage_path,
            "enable_created_date": config.enable_created_date,
            "enable_custom_fields": config.enable_custom_fields,
            "tag_behavior": config.tag_behavior,
            "tags_min": config.tags_min or 1,
            "tags_max": config.tags_max or 5,
            "correspondent_behavior": config.correspondent_behavior,
            "prompt_title": config.prompt_title or "",
            "prompt_tags": config.prompt_tags or "",
            "prompt_correspondent": config.prompt_correspondent or "",
            "prompt_document_type": config.prompt_document_type or "",
            "prompt_date": config.prompt_date or "",
            "system_prompt": config.system_prompt,
            "tags_ignore": config.tags_ignore or [],
            "storage_path_behavior": getattr(config, "storage_path_behavior", "always") or "always",
            "storage_path_override_names": getattr(config, "storage_path_override_names", ["Zuweisen"]) or ["Zuweisen"],
            "correspondent_trim_prompt": bool(getattr(config, "correspondent_trim_prompt", False)),
            "correspondent_strip_legal": bool(getattr(config, "correspondent_strip_legal", False)),
        }

    async def _build_document_context(self, document_id: int) -> tuple:
        assert self.paperless is not None
        """Build DocumentContext + raw doc_data from Paperless. Returns (context, doc_data) or raises."""
        doc_data = await self.paperless.get_document(document_id)
        if not doc_data:
            return None, None

        all_tags = await self.paperless.get_tags(use_cache=True)
        tag_map = {t["id"]: t["name"] for t in all_tags}
        current_tag_names = [tag_map.get(tid, str(tid)) for tid in doc_data.get("tags", [])]

        all_correspondents = await self.paperless.get_correspondents(use_cache=True)
        corr_map = {c["id"]: c["name"] for c in all_correspondents}
        current_corr = corr_map.get(doc_data.get("correspondent"), None)

        all_types = await self.paperless.get_document_types(use_cache=True)
        type_map = {dt["id"]: dt["name"] for dt in all_types}
        current_type = type_map.get(doc_data.get("document_type"), None)

        all_paths = await self.paperless.get_storage_paths(use_cache=True)
        path_map = {p["id"]: p["name"] for p in all_paths}
        current_path_id = doc_data.get("storage_path")
        current_path_name = path_map.get(current_path_id) if current_path_id else None

        document = DocumentContext(
            document_id=document_id,
            current_title=doc_data.get("title", ""),
            content=doc_data.get("content", ""),
            current_tags=current_tag_names,
            current_correspondent=current_corr,
            current_document_type=current_type,
            current_storage_path=current_path_name,
            created_date=doc_data.get("created"),
        )
        return document, doc_data

    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize various date formats to YYYY-MM-DD for comparison."""
        if not date_str:
            return None
        date_str = date_str.strip()
        # Already ISO: 1987-06-17
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return date_str
        # German DD.MM.YYYY
        m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", date_str)
        if m:
            return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
        # German DD.MM.YY
        m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{2})$", date_str)
        if m:
            year = int(m.group(3))
            year = 2000 + year if year < 50 else 1900 + year
            return f"{year}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
        return None

    def _normalize_custom_fields(self, custom_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize custom field values to consistent formats regardless of LLM output."""
        if not custom_fields:
            return custom_fields

        normalized = {}
        for key, value in custom_fields.items():
            if value is None:
                normalized[key] = None
                continue

            val = str(value).strip()

            if not val or val.lower() in ("null", "none", "n/a", "-", "--", "nicht gefunden", "nicht vorhanden"):
                normalized[key] = None
                continue

            key_lower = key.lower()

            if any(k in key_lower for k in ("iban", "kontonummer", "konto")):
                cleaned = re.sub(r'[\s\-\.]+', '', val)
                if re.match(r'^[A-Z]{2}\d', cleaned, re.IGNORECASE):
                    val = cleaned.upper()
                elif re.sub(r'\D', '', cleaned):
                    val = cleaned
                else:
                    normalized[key] = None
                    continue

            elif any(k in key_lower for k in ("betrag", "summe", "gesamt", "preis", "kosten")):
                val = re.sub(r'[€$\s]', '', val)
                val = val.replace('\u00a0', '')
                if re.match(r'^\d{1,3}(\.\d{3})+(,\d{1,2})?$', val):
                    val = val.replace('.', '').replace(',', '.')
                elif re.match(r'^\d{1,3}(\.\d{3})+$', val):
                    val = val.replace('.', '')
                elif ',' in val and '.' not in val:
                    val = val.replace(',', '.')
                try:
                    normalized[key] = round(float(val), 2)
                    continue
                except ValueError:
                    pass

            normalized[key] = val

        return normalized

    def _build_protected_matchers(self, config) -> list:
        """Build regex matchers from tags_protected patterns."""
        matchers = []
        for pat in (config.tags_protected or []):
            if "*" in pat:
                regex_pat = re.escape(pat).replace(r"\*", ".*")
                matchers.append(("regex", re.compile(f"^{regex_pat}$", re.IGNORECASE)))
            else:
                matchers.append(("exact", pat.lower()))
        return matchers

    def _is_tag_protected(self, tag_name: str, matchers: list) -> bool:
        """Check if a tag matches any protected pattern."""
        for kind, matcher in matchers:
            if kind == "exact" and tag_name.lower() == matcher:
                return True
            elif kind == "regex" and matcher.match(tag_name):
                return True
        return False

    def _deduplicate_tags(self, tags: List[str]) -> List[str]:
        """Remove redundant tags where one is a substring of another."""
        if len(tags) <= 1:
            return tags
        result = []
        sorted_tags = sorted(tags, key=len)
        for i, tag in enumerate(sorted_tags):
            is_redundant = False
            for j, other in enumerate(sorted_tags):
                if i == j:
                    continue
                if len(tag) < len(other) and tag.lower() in other.lower():
                    is_redundant = True
                    break
                if tag.lower() == other.lower() and i < j:
                    is_redundant = True
                    break
            if not is_redundant:
                result.append(tag)
        return result

    def _build_context_words(self, result: ClassificationResult, doc_content: str = "") -> set:
        """Build a set of context words from all available document info."""
        words = set()
        for source in [result.title, result.correspondent, result.summary]:
            if source:
                words.update(w.lower() for w in re.split(r'[\s,.\-/]+', source) if len(w) > 3)
        if result.document_type:
            words.add(result.document_type.lower())
        if doc_content:
            snippet = doc_content[:3000].lower()
            words.update(w for w in re.split(r'[\s,.\-/]+', snippet) if len(w) > 4)
        return words

    def _matches_ignore_patterns(self, tag: str, config: 'ClassifierConfig') -> bool:
        """Check if a tag matches any configured ignore pattern (exact or wildcard)."""
        tag_lower = tag.lower().strip()
        for pat in (config.tags_ignore or []):
            if "*" in pat:
                regex_pat = re.escape(pat).replace(r"\*", ".*")
                if re.match(f"^{regex_pat}$", tag, re.IGNORECASE):
                    return True
            elif pat.lower() == tag_lower:
                return True
        return False

    def _verify_result_coherence(
        self, result: ClassificationResult, config: ClassifierConfig,
        doc_content: str = "",
    ):
        """Verify result coherence. Removes system tags, enforces tag limits,
        and uses relevance scoring to decide which tags to keep."""
        tags_max = config.tags_max or 5

        issues = []
        if config.enable_document_type and not result.document_type:
            issues.append("document_type is empty")
        if config.enable_correspondent and not result.correspondent:
            issues.append("correspondent is empty")
        if config.enable_title and not result.title:
            issues.append("title is empty")
        if issues:
            logger.warning(f"Verification: missing fields: {', '.join(issues)}")

        if result.tags:
            before_count = len(result.tags)
            result.tags = [t for t in result.tags if not self._matches_ignore_patterns(t, config)]
            removed_ignore = before_count - len(result.tags)
            if removed_ignore:
                logger.info(f"Verification: removed {removed_ignore} tags via ignore patterns")

            result.tags = self._deduplicate_tags(result.tags)
            tags_min = config.tags_min or 1
            context_words = self._build_context_words(result, doc_content)

            def _tag_score(tag: str) -> int:
                tag_lower = tag.lower()
                tag_words = set(re.split(r'[\s\-/]+', tag_lower))
                score = 0
                for tw in tag_words:
                    if len(tw) < 3:
                        continue
                    for cw in context_words:
                        if tw in cw or cw in tw:
                            score += 2
                            break
                if result.title and tag_lower in result.title.lower():
                    score += 3
                if result.summary and tag_lower in result.summary.lower():
                    score += 3
                if doc_content and tag_lower in doc_content[:4000].lower():
                    score += 5
                return score

            # Remove tags with zero relevance to document content (if enough tags remain)
            if doc_content or result.title or result.summary:
                scored = [(t, _tag_score(t)) for t in result.tags]
                relevant = [(t, s) for t, s in scored if s > 0]
                irrelevant = [t for t, s in scored if s == 0]
                if irrelevant and len(relevant) >= tags_min:
                    logger.info(f"Verification: removed {len(irrelevant)} irrelevant tags (score=0): {irrelevant}")
                    result.tags = [t for t, _ in relevant]

            # Score and trim if still over limit
            if len(result.tags) > tags_max:
                scored_tags = [(t, _tag_score(t)) for t in result.tags]
                scored_tags.sort(key=lambda x: x[1], reverse=True)
                removed = [t for t, _ in scored_tags[tags_max:]]
                logger.info(f"Trimming {len(result.tags)} tags to max {tags_max}, removed: {removed}")
                result.tags = [t for t, _ in scored_tags[:tags_max]]

        logger.info(f"Verification complete: tags={result.tags}, doc_type={result.document_type}, "
                     f"corr={result.correspondent}, sp={result.storage_path_id}")

    async def _post_process(
        self, result: ClassificationResult, config: ClassifierConfig,
        doc_content: str = "",
    ):
        assert self.paperless is not None
        """Post-process: normalize fields, filter tags, verify coherence."""
        logger.info(f"Post-process start: tags_from_model={result.tags}")
        result.custom_fields = self._normalize_custom_fields(result.custom_fields)

        # Remove obvious personal-number patterns (YYYY-NNNNN) from title
        if result.title:
            cleaned = _clean_title(result.title)
            if cleaned != result.title:
                logger.info(f"Title cleaned: '{result.title}' → '{cleaned}'")
                result.title = cleaned

        if result.storage_path_id:
            all_paths = await self.paperless.get_storage_paths(use_cache=True)
            for p in all_paths:
                if p.get("id") == result.storage_path_id:
                    result.storage_path_name = p.get("name", "")
                    break

        # --- Storage path behavior: revert to existing if behavior says so ---
        sp_behavior = getattr(config, "storage_path_behavior", "always") or "always"
        if sp_behavior != "always" and result.existing_storage_path_id:
            existing_name = (result.existing_storage_path_name or "").strip().lower()
            keep_existing = False
            if sp_behavior == "keep_if_set":
                keep_existing = True
            elif sp_behavior == "keep_except_list":
                override_names = [n.lower() for n in (getattr(config, "storage_path_override_names", None) or ["Zuweisen"])]
                keep_existing = existing_name not in override_names
            if keep_existing:
                ai_suggestion = result.storage_path_name or f"ID {result.storage_path_id}"
                ai_reason = result.storage_path_reason or ""
                logger.info(
                    f"Post-process: reverting storage path to existing '{result.existing_storage_path_name}' "
                    f"(id={result.existing_storage_path_id}) due to behavior='{sp_behavior}'"
                )
                result.storage_path_id = result.existing_storage_path_id
                result.storage_path_name = result.existing_storage_path_name
                result.storage_path_reason = (
                    f"Bestehender Pfad beibehalten (Regel: {sp_behavior}). "
                    f"KI-Vorschlag war: {ai_suggestion}"
                    + (f" – {ai_reason}" if ai_reason else "")
                )

        if result.correspondent:
            # Check against ignore list first
            corr_ignore = getattr(config, "correspondent_ignore", None) or []
            corr_ignore_lower = [n.lower().strip() for n in corr_ignore if n.strip()]
            corr_name_lower = result.correspondent.lower()
            ignored_by = next(
                (ign for ign in corr_ignore_lower
                 if ign in corr_name_lower or corr_name_lower in ign),
                None,
            )
            if ignored_by:
                logger.info(f"Correspondent ignored: '{result.correspondent}' (matched ignore entry '{ignored_by}')")
                result.correspondent = None
                result.correspondent_is_new = False

        if result.correspondent:
            # Strip legal forms if enabled (post-processing, independent of prompt option)
            if getattr(config, "correspondent_strip_legal", False):
                original = result.correspondent
                result.correspondent = _strip_legal_forms(result.correspondent)
                if result.correspondent != original:
                    logger.info(f"Correspondent legal-strip: '{original}' → '{result.correspondent}'")

            all_correspondents = await self.paperless.get_correspondents(use_cache=False)
            existing_corr_names = {c["name"].lower() for c in all_correspondents}
            result.correspondent_is_new = result.correspondent.lower() not in existing_corr_names

        if result.tags:
            all_tags = await self.paperless.get_tags(use_cache=True)
            existing_tag_names = {t["name"].lower() for t in all_tags}

            all_doc_types = await self.paperless.get_document_types(use_cache=True)
            all_doc_type_names = {dt["name"].lower() for dt in all_doc_types}

            all_correspondents = await self.paperless.get_correspondents(use_cache=True)
            all_corr_names = {c["name"].lower() for c in all_correspondents}

            # Build ignore matchers: exact strings + wildcard patterns
            ignore_exact = set()
            ignore_patterns = []
            for t in (config.tags_ignore or []):
                if "*" in t:
                    regex_pat = re.escape(t).replace(r"\*", ".*")
                    ignore_patterns.append(re.compile(f"^{regex_pat}$", re.IGNORECASE))
                else:
                    ignore_exact.add(t.lower())

            def _is_ignored(tag_name: str) -> bool:
                if tag_name.lower() in ignore_exact:
                    return True
                return any(p.match(tag_name) for p in ignore_patterns)

            filtered_tags = []
            for tag in result.tags:
                tag_lower = tag.lower()
                if tag_lower in all_doc_type_names:
                    logger.info(f"Tag '{tag}' removed: matches a document type name")
                    continue
                if tag_lower in all_corr_names:
                    logger.info(f"Tag '{tag}' removed: matches a correspondent name")
                    continue
                if _is_ignored(tag):
                    logger.info(f"Tag '{tag}' removed: matches ignore pattern")
                    continue
                filtered_tags.append(tag)

            result.tags = filtered_tags
            result.tags_new = [t for t in result.tags if t.lower() not in existing_tag_names]
            logger.info(f"Post-process filtered tags: {result.tags} (new: {result.tags_new})")

        # --- Filter ignored dates ---
        if result.created_date and config.dates_ignore:
            normalized_result_date = self._normalize_date(result.created_date)
            for ignored in config.dates_ignore:
                if normalized_result_date and normalized_result_date == self._normalize_date(ignored):
                    logger.info(f"Date '{result.created_date}' matches ignore list ('{ignored}') -- cleared")
                    result.created_date = None
                    break

        self._verify_result_coherence(result, config, doc_content)

    async def classify_document(self, document_id: int) -> ClassificationResult:
        assert self.paperless is not None
        """Classify a single document and return proposals."""
        # Fresh Paperless data for every classification — new tags/correspondents
        # created by a previous apply must be visible immediately.
        from app.services.cache import get_cache
        await get_cache().clear("paperless:")

        config = await self.get_config()
        provider = await self._build_provider(config)

        document, doc_data = await self._build_document_context(document_id)
        if not document:
            return ClassificationResult(error=f"Document {document_id} not found")

        config_dict = self._build_config_dict(config)
        result = await provider.classify(document, config_dict)

        # Set existing metadata BEFORE _post_process so behavior logic can use it
        result.existing_tags = document.current_tags
        result.existing_correspondent = document.current_correspondent
        result.existing_document_type = document.current_document_type
        result.existing_storage_path_name = document.current_storage_path

        # Resolve existing storage path ID
        existing_sp_id = None
        if document.current_storage_path:
            all_paths = await self.paperless.get_storage_paths(use_cache=True)
            for p in all_paths:
                if p["name"] == document.current_storage_path:
                    existing_sp_id = p["id"]
                    break
        result.existing_storage_path_id = existing_sp_id

        # --- Fallback: if LLM returned nothing, keep the existing value ---
        if not result.title and document.current_title:
            result.title = document.current_title
            logger.info(f"Title fallback: kept existing '{document.current_title}'")
        if not result.correspondent and document.current_correspondent:
            result.correspondent = document.current_correspondent
            logger.info(f"Correspondent fallback: kept existing '{document.current_correspondent}'")
        if not result.document_type and document.current_document_type:
            result.document_type = document.current_document_type
            logger.info(f"DocType fallback: kept existing '{document.current_document_type}'")
        if result.storage_path_id is None and existing_sp_id:
            result.storage_path_id = existing_sp_id
            result.storage_path_name = document.current_storage_path
            result.storage_path_reason = "Vorhandener Speicherpfad beibehalten"
            logger.info(f"StoragePath fallback: kept existing id={existing_sp_id} '{document.current_storage_path}'")

        await self._post_process(result, config, document.content)

        classifier_provider_name = await self._get_classifier_provider_name()

        if self.session_factory is not None:
            async with self.session_factory() as db:
                # Remove old "pending" entries for this document before inserting the new one.
                # This prevents stale results from appearing in history / being loaded again.
                from sqlalchemy import delete as sa_delete
                await db.execute(
                    sa_delete(ClassificationHistory)
                    .where(ClassificationHistory.document_id == document_id)
                    .where(ClassificationHistory.status == "pending")
                )

                try:
                    history_model = await get_setting(LLM_KEY_CLASSIFIER_MODEL, db) or "unknown"
                except Exception:
                    history_model = "unknown"

                history = ClassificationHistory(
                    document_id=document_id,
                    document_title=doc_data.get("title", ""),
                    provider=classifier_provider_name,
                    model=history_model,
                    result_json=asdict(result),
                    tokens_input=result.tokens_input,
                    tokens_output=result.tokens_output,
                    cost_usd=result.cost_usd,
                    duration_seconds=result.duration_seconds,
                    tool_calls_count=result.tool_calls_count,
                    status="error" if result.error else "pending",
                    error_message=result.error or "",
                )
                db.add(history)
                await db.commit()

        return result

    async def _build_provider_by_name(
        self, provider_name: str, config: ClassifierConfig,
        model_override: Optional[str] = None,
    ) -> BaseClassifierProvider:
        assert self.paperless is not None
        """Build a specific provider with optional model override (for benchmarks)."""
        storage_profiles = await self.get_storage_profiles()
        field_mappings = await self.get_custom_field_mappings()
        tool_executor = self._build_tool_executor(config, storage_profiles, field_mappings)

        return await self._create_provider_instance(provider_name, tool_executor, model_override)

    async def benchmark_document(
        self, document_id: int,
        slots: List[tuple],
    ) -> Dict[str, Any]:
        assert self.paperless is not None
        """Run classification with N provider/model combos, strictly sequential."""

        config = await self.get_config()
        document, doc_data = await self._build_document_context(document_id)
        if not document:
            return {"error": f"Document {document_id} not found"}

        config_dict = self._build_config_dict(config)

        # Resolve existing storage path ID once (shared across benchmark slots)
        bench_existing_sp_id = None
        if document.current_storage_path:
            all_paths = await self.paperless.get_storage_paths(use_cache=True)
            for p in all_paths:
                if p["name"] == document.current_storage_path:
                    bench_existing_sp_id = p["id"]
                    break

        async def run_single(name: str, model: Optional[str]) -> Dict[str, Any]:
            assert self.paperless is not None
            # Resolve actual model for display: use provided model, or fetch from KV store
            if model:
                actual_model = model
            elif name == "openai":
                actual_model = config.openai_model
            else:
                # Fetch from KV store (LLM_KEY_CLASSIFIER_MODEL) for non-openai providers
                actual_model = None
                if self.session_factory is not None:
                    async with self.session_factory() as db:
                        actual_model = await get_setting(LLM_KEY_CLASSIFIER_MODEL, db)
                if not actual_model:
                    actual_model = f"{name}:default"
            try:
                provider = await self._build_provider_by_name(name, config, model)
                result = await provider.classify(document, config_dict)
                # Set existing metadata before _post_process so behavior logic works
                result.existing_tags = document.current_tags
                result.existing_correspondent = document.current_correspondent
                result.existing_document_type = document.current_document_type
                result.existing_storage_path_name = document.current_storage_path
                result.existing_storage_path_id = bench_existing_sp_id
                await self._post_process(result, config, document.content)
                return {
                    "provider": name,
                    "model": actual_model,
                    "result": asdict(result),
                }
            except Exception as e:
                logger.error(f"Benchmark {name}/{actual_model} failed: {e}", exc_info=True)
                return {
                    "provider": name,
                    "model": actual_model,
                    "result": asdict(ClassificationResult(error=str(e))),
                }

        # All slots run strictly sequential to avoid GPU contention
        all_results = []
        for name, model in slots:
            r = await run_single(name, model)
            all_results.append(r)

        return {
            "document_id": document_id,
            "document_title": doc_data.get("title", ""),
            "results": all_results,
        }

    async def apply_classification(
        self, document_id: int, classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        assert self.paperless is not None
        """Apply a (potentially edited) classification to a document in Paperless."""
        config = await self.get_config()
        update_data = {}

        if classification.get("title"):
            update_data["title"] = classification["title"]

        created = classification.get("created_date")
        if created and created != "null" and re.match(r"\d{4}-\d{2}-\d{2}", str(created)):
            update_data["created"] = created

        # Resolve tags to IDs
        if classification.get("tags"):
            all_tags = await self.paperless.get_tags(use_cache=True)
            tag_name_to_id = {t["name"].lower(): t["id"] for t in all_tags}
            tag_id_to_name = {t["id"]: t["name"] for t in all_tags}
            tag_ids = []
            for tag_name in classification["tags"]:
                tid = tag_name_to_id.get(tag_name.lower())
                if tid:
                    tag_ids.append(tid)
                else:
                    new_tag = await self.paperless.get_or_create_tag(tag_name)
                    if new_tag:
                        tag_ids.append(new_tag["id"])

            doc = await self.paperless.get_document(document_id)
            existing_tag_ids = doc.get("tags", []) if doc else []

            if config.tags_keep_existing:
                for etid in existing_tag_ids:
                    if etid not in tag_ids:
                        tag_ids.append(etid)
            else:
                # Replacing mode: keep protected tags from existing document
                protected_patterns = self._build_protected_matchers(config)
                if protected_patterns:
                    for etid in existing_tag_ids:
                        tag_name = tag_id_to_name.get(etid, "")
                        if etid not in tag_ids and self._is_tag_protected(tag_name, protected_patterns):
                            tag_ids.append(etid)
                            logger.info(f"Apply: keeping protected tag '{tag_name}' (id={etid})")

            if tag_ids:
                update_data["tags"] = tag_ids

        # Classification Tag: if enabled, ensure the configured tag is on every classified document
        if getattr(config, "classification_tag_enabled", False):
            tag_name = (getattr(config, "classification_tag_name", None) or "KI-klassifiziert").strip()
            if tag_name:
                cls_tag = await self.paperless.get_or_create_tag(tag_name)
                if cls_tag:
                    current_tag_ids = list(update_data.get("tags") or [])
                    if cls_tag["id"] not in current_tag_ids:
                        if not current_tag_ids:
                            # tags not yet fetched — load existing tags from document
                            doc = await self.paperless.get_document(document_id)
                            current_tag_ids = doc.get("tags", []) if doc else []
                        current_tag_ids.append(cls_tag["id"])
                        update_data["tags"] = current_tag_ids
                        logger.info(f"Apply: added classification tag '{tag_name}' (id={cls_tag['id']}) to doc {document_id}")

        # Resolve correspondent
        if classification.get("correspondent"):
            corr = await self.paperless.get_or_create_correspondent(
                classification["correspondent"]
            )
            if corr:
                update_data["correspondent"] = corr["id"]

        # Resolve document type
        if classification.get("document_type"):
            all_types = await self.paperless.get_document_types(use_cache=True)
            for dt in all_types:
                if dt["name"].lower() == classification["document_type"].lower():
                    update_data["document_type"] = dt["id"]
                    break

        # Storage path -- respect configured behavior
        if classification.get("storage_path_id"):
            sp_behavior = getattr(config, "storage_path_behavior", "always") or "always"
            sp_override_names = getattr(config, "storage_path_override_names", ["Zuweisen"]) or ["Zuweisen"]
            existing_sp_id = classification.get("existing_storage_path_id")
            existing_sp_name = (classification.get("existing_storage_path_name") or "").strip()

            # Fallback: fetch live from Paperless if not provided (extra safety)
            if not existing_sp_id and sp_behavior != "always":
                live_doc = await self.paperless.get_document(document_id)
                if live_doc and live_doc.get("storage_path"):
                    existing_sp_id = live_doc["storage_path"]
                    all_paths = await self.paperless.get_storage_paths(use_cache=True)
                    sp_map = {p["id"]: p["name"] for p in all_paths}
                    existing_sp_name = sp_map.get(existing_sp_id, "")
                    logger.info(f"Storage path fallback from Paperless: '{existing_sp_name}' (id={existing_sp_id})")

            apply_sp = True
            if sp_behavior == "keep_if_set":
                # Never change if document already has a path
                apply_sp = not existing_sp_id
            elif sp_behavior == "keep_except_list":
                # Keep existing UNLESS the current path name is in the override list
                if existing_sp_id:
                    override_names_lower = [n.lower() for n in sp_override_names]
                    apply_sp = existing_sp_name.lower() in override_names_lower
                # If no path set yet, always assign
            # "always" => apply_sp stays True

            if apply_sp:
                update_data["storage_path"] = classification["storage_path_id"]
            else:
                logger.info(
                    f"Storage path skipped (behavior={sp_behavior}): "
                    f"existing='{existing_sp_name}' (id={existing_sp_id})"
                )

        # Custom fields
        if classification.get("custom_fields"):
            field_mappings = await self.get_custom_field_mappings()
            field_name_to_id = {m.paperless_field_name: m.paperless_field_id for m in field_mappings}
            custom_field_updates = []
            for field_name, value in classification["custom_fields"].items():
                fid = field_name_to_id.get(field_name)
                if fid and value is not None:
                    custom_field_updates.append({"field": fid, "value": value})
            if custom_field_updates:
                update_data["custom_fields"] = custom_field_updates

        if not update_data:
            return {"applied": False, "reason": "No changes to apply"}

        logger.info(f"Applying to doc {document_id}: {update_data}")
        result = await self.paperless.update_document(document_id, update_data)

        # Mark latest pending/review history entry for this document as applied
        if self.session_factory is not None:
            try:
                from sqlalchemy import select, desc
                from app.models.classifier import ClassificationHistory
                async with self.session_factory() as db:
                    hist_q = await db.execute(
                        select(ClassificationHistory)
                        .where(ClassificationHistory.document_id == document_id)
                        .where(ClassificationHistory.status.in_(["pending", "review"]))
                        .order_by(desc(ClassificationHistory.id))
                        .limit(1)
                    )
                    hist = hist_q.scalars().first()
                    if hist:
                        hist.status = "applied"
                        await db.commit()
                        logger.info(f"History entry {hist.id} marked as applied for doc {document_id} (was: {hist.status})")
            except Exception as e:
                logger.warning(f"Could not update history status: {e}")

        # Always refresh cache after apply -- new tags/correspondents must be
        # visible immediately for the next classification call.
        from app.services.cache import get_cache
        cache = get_cache()
        await cache.clear("paperless:")
        logger.info("Cache cleared after apply -- next classification gets fresh Paperless data")

        return {"applied": True, "updated_fields": list(update_data.keys()), "result": result}

    @staticmethod
    def _needs_review(result: ClassificationResult) -> str:
        """Check if a classification result needs manual review. Returns reason or empty string.

        Only triggers for real problems — not for normal new correspondents/tags,
        since those are clearly visible in the history view.
        """
        reasons = []

        if result.error:
            reasons.append("Fehler bei Klassifizierung")
        if not result.title:
            reasons.append("Kein Titel erkannt")
        if not result.correspondent and not result.document_type:
            reasons.append("Weder Korrespondent noch Dokumenttyp erkannt")
        if not result.tags:
            reasons.append("Keine Tags erkannt")

        return "; ".join(reasons)

    async def classify_document_auto(self, document_id: int, mode: str = "review") -> Dict[str, Any]:
        assert self.paperless is not None
        """Classify a document in auto-mode.

        New tags suggested by the AI are NOT created automatically.
        Instead they are saved as 'tag_ideas' on the history entry for
        manual review. The document gets classified with existing tags only.
        """
        result = await self.classify_document(document_id)
        review_reason = self._needs_review(result)
        config = await self.get_config()

        if result.error:
            return {"document_id": document_id, "action": "error", "reason": result.error}

        # Sync tags_new with actual result.tags (verification may have removed some)
        if result.tags_new and result.tags:
            final_tag_set = {t.lower() for t in result.tags}
            result.tags_new = [t for t in result.tags_new if t.lower() in final_tag_set]

        # Extract new tag ideas before applying
        tag_ideas = list(result.tags_new) if result.tags_new else []
        has_tag_ideas = len(tag_ideas) > 0

        # Build a version of the result that only uses existing tags
        apply_data = asdict(result)
        if has_tag_ideas:
            existing_tags = [t for t in (result.tags or []) if t not in tag_ideas]
            apply_data["tags"] = existing_tags
            apply_data["tags_new"] = []
            logger.info(
                f"Auto-classify doc {document_id}: {len(tag_ideas)} tag idea(s) saved: {tag_ideas}"
            )

        if mode == "auto_apply" and not review_reason:
            await self.apply_classification(document_id, apply_data)
            # Save tag ideas on the history entry
            if has_tag_ideas:
                await self._save_tag_ideas(document_id, tag_ideas)
                if getattr(config, "tag_ideas_tag_enabled", False):
                    await self._add_status_tag(document_id, getattr(config, "tag_ideas_tag_name", None) or "KI-tag-ideen")
            return {
                "document_id": document_id,
                "action": "applied",
                "tag_ideas": tag_ideas,
            }

        # Mark as "review" in history if needed
        if review_reason:
            try:
                if self.session_factory is not None:
                    async with self.session_factory() as db:
                        hist_q = await db.execute(
                            select(ClassificationHistory)
                            .where(ClassificationHistory.document_id == document_id)
                            .where(ClassificationHistory.status == "pending")
                            .order_by(ClassificationHistory.id.desc())
                            .limit(1)
                        )
                        hist = hist_q.scalars().first()
                        if hist:
                            hist.status = "review"
                            hist.error_message = review_reason
                            if has_tag_ideas:
                                hist.tag_ideas = tag_ideas
                            await db.commit()
            except Exception as e:
                logger.warning(f"Could not mark as review: {e}")
            if getattr(config, "review_tag_enabled", False):
                await self._add_status_tag(document_id, getattr(config, "review_tag_name", None) or "KI-prüfen")
            if has_tag_ideas and getattr(config, "tag_ideas_tag_enabled", False):
                await self._add_status_tag(document_id, getattr(config, "tag_ideas_tag_name", None) or "KI-tag-ideen")
        else:
            # No review needed — auto-apply with existing tags only
            await self.apply_classification(document_id, apply_data)
            if has_tag_ideas:
                await self._save_tag_ideas(document_id, tag_ideas)
                if getattr(config, "tag_ideas_tag_enabled", False):
                    await self._add_status_tag(document_id, getattr(config, "tag_ideas_tag_name", None) or "KI-tag-ideen")
            return {
                "document_id": document_id,
                "action": "applied",
                "tag_ideas": tag_ideas,
            }

        return {
            "document_id": document_id,
            "action": "review" if review_reason else "pending",
            "reason": review_reason,
            "tag_ideas": tag_ideas,
        }

    async def _save_tag_ideas(self, document_id: int, tag_ideas: List[str]):
        assert self.paperless is not None
        """Save tag ideas on the latest history entry for a document."""
        try:
            if self.session_factory is not None:
                async with self.session_factory() as db:
                    hist_q = await db.execute(
                        select(ClassificationHistory)
                        .where(ClassificationHistory.document_id == document_id)
                        .order_by(ClassificationHistory.id.desc())
                        .limit(1)
                    )
                    hist = hist_q.scalars().first()
                    if hist:
                        hist.tag_ideas = tag_ideas
                        await db.commit()
        except Exception as e:
            logger.warning(f"Could not save tag ideas for doc {document_id}: {e}")

    async def _add_status_tag(self, document_id: int, tag_name: str):
        assert self.paperless is not None
        """Append a single tag to a document in Paperless (creates the tag if missing)."""
        try:
            tag_name = tag_name.strip()
            if not tag_name:
                return
            tag = await self.paperless.get_or_create_tag(tag_name)
            if not tag:
                return
            doc = await self.paperless.get_document(document_id)
            if not doc:
                return
            existing = list(doc.get("tags", []))
            if tag["id"] not in existing:
                existing.append(tag["id"])
                await self.paperless.update_document(document_id, {"tags": existing})
                logger.info(f"Status tag '{tag_name}' (id={tag['id']}) added to doc {document_id}")
        except Exception as e:
            logger.warning(f"Could not add status tag '{tag_name}' to doc {document_id}: {e}")

    # ── Extracted router business logic ────────────────────────────────────────

    async def get_stats(self, client: PaperlessClient) -> Dict[str, Any]:
        """Get classification statistics: how many done, open, costs, etc."""
        if self.session_factory is None:
            raise RuntimeError("No session_factory configured")

        try:
            total_docs = await client.get_document_count()
        except Exception:
            total_docs = 0

        async with self.session_factory() as db:
            history_q = await db.execute(
                select(
                    sa_func.count(ClassificationHistory.id).label("total"),
                    sa_func.sum(sa_case((ClassificationHistory.status == "applied", 1), else_=0)).label("applied"),
                    sa_func.sum(sa_case((ClassificationHistory.status == "error", 1), else_=0)).label("errors"),
                    sa_func.sum(ClassificationHistory.tokens_input).label("total_tokens_in"),
                    sa_func.sum(ClassificationHistory.tokens_output).label("total_tokens_out"),
                    sa_func.sum(ClassificationHistory.cost_usd).label("total_cost"),
                    sa_func.avg(ClassificationHistory.duration_seconds).label("avg_duration"),
                )
            )
            row = history_q.first()

            unique_q = await db.execute(
                select(sa_func.count(sa_func.distinct(ClassificationHistory.document_id)))
            )
            unique_classified = unique_q.scalar() or 0

            applied_unique_q = await db.execute(
                select(sa_func.count(sa_func.distinct(ClassificationHistory.document_id))).where(
                    ClassificationHistory.status == "applied"
                )
            )
            applied_unique = applied_unique_q.scalar() or 0

            provider_q = await db.execute(
                select(
                    ClassificationHistory.provider,
                    ClassificationHistory.model,
                    sa_func.count(ClassificationHistory.id).label("count"),
                    sa_func.sum(ClassificationHistory.cost_usd).label("cost"),
                    sa_func.avg(ClassificationHistory.duration_seconds).label("avg_duration"),
                ).group_by(ClassificationHistory.provider, ClassificationHistory.model)
            )
            providers = [
                {
                    "provider": r.provider,
                    "model": r.model,
                    "count": r.count,
                    "cost": round(float(r.cost or 0), 6),
                    "avg_duration": round(float(r.avg_duration or 0), 1),
                }
                for r in provider_q.all()
            ]

            recent_q = await db.execute(
                select(ClassificationHistory)
                .order_by(ClassificationHistory.created_at.desc())
                .limit(10)
            )
            recent = [
                {
                    "document_id": h.document_id,
                    "document_title": h.document_title,
                    "provider": h.provider,
                    "model": h.model,
                    "status": h.status,
                    "cost_usd": h.cost_usd,
                    "duration_seconds": h.duration_seconds,
                    "created_at": str(h.created_at) if h.created_at else None,
                }
                for h in recent_q.scalars().all()
            ]

        return {
            "total_documents_paperless": total_docs,
            "unique_classified": unique_classified,
            "unique_applied": applied_unique,
            "remaining": max(0, total_docs - applied_unique),
            "total_runs": row.total or 0,
            "total_applied": int(row.applied or 0),
            "total_errors": int(row.errors or 0),
            "total_tokens_in": int(row.total_tokens_in or 0),
            "total_tokens_out": int(row.total_tokens_out or 0),
            "total_cost_usd": round(float(row.total_cost or 0), 6),
            "avg_duration_seconds": round(float(row.avg_duration or 0), 1),
            "by_provider": providers,
            "recent": recent,
        }

    async def get_next_unclassified(
        self, after_id: int, client: PaperlessClient,
    ) -> Dict[str, Any]:
        """Find the next document ID not yet applied/classified."""
        if self.session_factory is None:
            raise RuntimeError("No session_factory configured")

        async with self.session_factory() as db:
            applied_q = await db.execute(
                select(ClassificationHistory.document_id).where(
                    ClassificationHistory.status == "applied"
                )
            )
            applied_ids: set = {row[0] for row in applied_q.all()}

        BATCH = 50
        page = 1
        while True:
            params = {"page_size": BATCH, "page": page, "ordering": "id"}
            if after_id:
                params["id__gt"] = after_id

            result = await getattr(client, "_request")("GET", "/documents/", params=params)
            if not result:
                break

            docs = result.get("results", [])
            for doc in docs:
                doc_id = doc.get("id")
                if doc_id and doc_id not in applied_ids:
                    return {"found": True, "document_id": doc_id, "title": doc.get("title", "")}

            if not result.get("next"):
                break
            page += 1

        return {"found": False, "document_id": None, "title": ""}

    async def get_storage_path_profiles_merged(
        self, client: PaperlessClient,
    ) -> List[Dict[str, Any]]:
        """Get all storage paths merged with saved profiles."""
        all_paths = await client.get_storage_paths(use_cache=True)
        saved_profiles = await self.get_storage_profiles()
        saved_by_id = {p.paperless_path_id: p for p in saved_profiles}

        result = []
        for path in all_paths:
            path_id = path.get("id")
            profile = saved_by_id.get(int(path_id)) if path_id is not None else None
            result.append({
                "id": profile.id if profile else None,
                "paperless_path_id": path_id,
                "paperless_path_name": path.get("name", ""),
                "paperless_path_path": path.get("path", ""),
                "enabled": profile.enabled if profile else True,
                "person_name": profile.person_name if profile else "",
                "path_type": profile.path_type if profile else "private",
                "context_prompt": profile.context_prompt if profile else "",
            })
        return result

    async def get_custom_field_mappings_merged(
        self, client: PaperlessClient,
    ) -> List[Dict[str, Any]]:
        """Get all Paperless custom fields merged with saved mappings."""
        all_fields = await client.get_custom_fields(use_cache=True)
        saved_mappings = await self.get_custom_field_mappings()
        saved_by_id = {m.paperless_field_id: m for m in saved_mappings}

        result = []
        for field in all_fields:
            fid = field.get("id")
            mapping = saved_by_id.get(int(fid)) if fid is not None else None
            field_type = field.get("data_type", "string")
            auto_prompt = FIELD_TYPE_PROMPTS.get(field.get("name", "").lower(), "")

            result.append({
                "id": mapping.id if mapping else None,
                "paperless_field_id": fid,
                "paperless_field_name": field.get("name", ""),
                "paperless_field_type": field_type,
                "enabled": mapping.enabled if mapping else False,
                "extraction_prompt": mapping.extraction_prompt if mapping and mapping.extraction_prompt else auto_prompt,
                "example_values": mapping.example_values if mapping else "",
                "validation_regex": mapping.validation_regex if mapping else FIELD_TYPE_VALIDATION.get(field.get("name", "").lower(), ""),
                "ignore_values": mapping.ignore_values if mapping else "",
            })
        return result

    async def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get classification history including stored result_json for review."""
        if self.session_factory is None:
            return []
        async with self.session_factory() as db:
            result = await db.execute(
                select(ClassificationHistory)
                .order_by(ClassificationHistory.created_at.desc())
                .limit(limit)
            )
            entries = result.scalars().all()
            return [
                {
                    "id": e.id,
                    "document_id": e.document_id,
                    "document_title": e.document_title,
                    "provider": e.provider,
                    "model": e.model,
                    "tokens_input": e.tokens_input,
                    "tokens_output": e.tokens_output,
                    "cost_usd": e.cost_usd,
                    "duration_seconds": e.duration_seconds,
                    "tool_calls_count": e.tool_calls_count,
                    "status": e.status,
                    "error_message": e.error_message,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                    "result_json": e.result_json,
                }
                for e in entries
            ]

    async def get_tag_stats(self) -> Dict[str, Any]:
        """Aggregate tag usage statistics from all applied history entries."""
        if self.session_factory is None:
            return {"top_tags": [], "total_unique_tags": 0, "total_tag_assignments": 0, "total_new_tags_created": 0}
        async with self.session_factory() as db:
            q = await db.execute(
                select(ClassificationHistory).where(
                    ClassificationHistory.result_json.isnot(None)
                )
            )
            entries = q.scalars().all()

        tag_counts: dict = {}
        tag_new_counts: dict = {}
        tag_applied_counts: dict = {}

        for e in entries:
            rj = e.result_json
            if isinstance(rj, str):
                try:
                    rj = _json.loads(rj)
                except Exception:
                    continue
            if not isinstance(rj, dict):
                continue

            tags = rj.get("tags") or []
            tags_new = rj.get("tags_new") or []

            for tag in tags:
                if not tag:
                    continue
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                if e.status == "applied":
                    tag_applied_counts[tag] = tag_applied_counts.get(tag, 0) + 1

            for tag in tags_new:
                if tag:
                    tag_new_counts[tag] = tag_new_counts.get(tag, 0) + 1

        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return {
            "top_tags": [
                {
                    "name": name,
                    "count": count,
                    "applied_count": tag_applied_counts.get(name, 0),
                    "new_count": tag_new_counts.get(name, 0),
                }
                for name, count in sorted_tags[:40]
            ],
            "total_unique_tags": len(tag_counts),
            "total_tag_assignments": sum(tag_counts.values()),
            "total_new_tags_created": len(tag_new_counts),
        }

    async def get_review_queue(self) -> List[Dict[str, Any]]:
        """Get all classification entries that need manual review."""
        if self.session_factory is None:
            return []
        async with self.session_factory() as db:
            result = await db.execute(
                select(ClassificationHistory)
                .where(ClassificationHistory.status == "review")
                .order_by(ClassificationHistory.created_at.desc())
            )
            entries = result.scalars().all()
            return [
                {
                    "id": e.id,
                    "document_id": e.document_id,
                    "document_title": e.document_title,
                    "provider": e.provider,
                    "model": e.model,
                    "status": e.status,
                    "error_message": e.error_message,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                    "result_json": e.result_json,
                }
                for e in entries
            ]

    async def approve_review(
        self, entry_id: int, document_id: int, classification: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Approve a review entry: apply classification and mark as applied."""
        if self.session_factory is None:
            raise RuntimeError("No session_factory configured")

        result = await self.apply_classification(document_id, classification)

        async with self.session_factory() as db:
            q = await db.execute(
                select(ClassificationHistory).where(ClassificationHistory.id == entry_id)
            )
            entry = q.scalars().first()
            if not entry:
                raise ValueError(f"Entry {entry_id} not found")
            entry.status = "applied"
            entry.error_message = ""
            await db.commit()

        return result

    async def dismiss_review(self, entry_id: int) -> Dict[str, str]:
        """Dismiss a review entry without applying."""
        if self.session_factory is None:
            raise RuntimeError("No session_factory configured")
        async with self.session_factory() as db:
            q = await db.execute(
                select(ClassificationHistory).where(ClassificationHistory.id == entry_id)
            )
            entry = q.scalars().first()
            if not entry:
                raise ValueError(f"Entry {entry_id} not found")
            entry.status = "rejected"
            await db.commit()
        return {"status": "dismissed"}

    async def get_tag_ideas(self) -> List[Dict[str, Any]]:
        """Get all history entries that have pending tag ideas."""
        if self.session_factory is None:
            return []
        async with self.session_factory() as db:
            result = await db.execute(
                select(ClassificationHistory)
                .where(ClassificationHistory.tag_ideas.isnot(None))
                .order_by(ClassificationHistory.created_at.desc())
            )
            entries = result.scalars().all()

        items = []
        for e in entries:
            ideas = e.tag_ideas
            if not ideas or (isinstance(ideas, list) and len(ideas) == 0):
                continue
            if isinstance(ideas, str):
                try:
                    ideas = _json.loads(ideas)
                except Exception:
                    continue
            if not ideas:
                continue
            items.append({
                "id": e.id,
                "document_id": e.document_id,
                "document_title": e.document_title,
                "provider": e.provider,
                "model": e.model,
                "status": e.status,
                "tag_ideas": ideas,
                "result_json": e.result_json,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            })
        return items

    async def get_tag_ideas_stats(self) -> Dict[str, Any]:
        """Aggregate stats: which new tags are suggested most frequently."""
        if self.session_factory is None:
            return {"total_ideas": 0, "unique_tags": 0, "documents_with_ideas": 0, "top_tags": []}
        async with self.session_factory() as db:
            result = await db.execute(
                select(ClassificationHistory)
                .where(ClassificationHistory.tag_ideas.isnot(None))
            )
            entries = result.scalars().all()

        tag_counter: Counter = Counter()
        tag_docs: dict = {}

        for e in entries:
            ideas = e.tag_ideas
            if isinstance(ideas, str):
                try:
                    ideas = _json.loads(ideas)
                except Exception:
                    continue
            if not ideas or not isinstance(ideas, list):
                continue
            for tag_name in ideas:
                tag_counter[tag_name] += 1
                if tag_name not in tag_docs:
                    tag_docs[tag_name] = []
                tag_docs[tag_name].append(e.document_id)

        top_tags = [
            {"name": name, "count": count, "document_ids": tag_docs.get(name, [])}
            for name, count in tag_counter.most_common(50)
        ]
        return {
            "total_ideas": sum(tag_counter.values()),
            "unique_tags": len(tag_counter),
            "documents_with_ideas": len([e for e in entries if e.tag_ideas and (isinstance(e.tag_ideas, list) and len(e.tag_ideas) > 0)]),
            "top_tags": top_tags,
        }

    async def approve_tag_idea(
        self, entry_id: int, tag_name: str, client: PaperlessClient,
    ) -> Dict[str, Any]:
        """Approve a single tag idea: create tag in Paperless and add to document."""
        if self.session_factory is None:
            raise RuntimeError("No session_factory configured")

        async with self.session_factory() as db:
            q = await db.execute(
                select(ClassificationHistory).where(ClassificationHistory.id == entry_id)
            )
            entry = q.scalars().first()
            if not entry:
                raise ValueError(f"Entry {entry_id} not found")

            tag = await client.get_or_create_tag(tag_name)
            if not tag:
                raise ValueError(f"Could not create tag '{tag_name}'")

            doc = await client.get_document(entry.document_id)
            if doc:
                existing_tags = doc.get("tags", [])
                if tag["id"] not in existing_tags:
                    existing_tags.append(tag["id"])
                    await client.update_document(entry.document_id, {"tags": existing_tags})
                    logger.info(f"Tag idea approved: '{tag_name}' added to doc {entry.document_id}")

            ideas = entry.tag_ideas
            if isinstance(ideas, str):
                try:
                    ideas = _json.loads(ideas)
                except Exception:
                    ideas = []
            if isinstance(ideas, list):
                ideas = [t for t in ideas if t != tag_name]
            entry.tag_ideas = ideas
            await db.commit()

        from app.services.cache import get_cache
        await get_cache().clear("paperless:")

        return {"status": "approved", "tag_name": tag_name, "remaining_ideas": ideas}

    async def dismiss_tag_idea(
        self, entry_id: int, tag_name: str,
    ) -> Dict[str, Any]:
        """Dismiss a single tag idea (remove from suggestions without creating)."""
        if self.session_factory is None:
            raise RuntimeError("No session_factory configured")

        async with self.session_factory() as db:
            q = await db.execute(
                select(ClassificationHistory).where(ClassificationHistory.id == entry_id)
            )
            entry = q.scalars().first()
            if not entry:
                raise ValueError(f"Entry {entry_id} not found")

            ideas = entry.tag_ideas
            if isinstance(ideas, str):
                try:
                    ideas = _json.loads(ideas)
                except Exception:
                    ideas = []
            if isinstance(ideas, list):
                ideas = [t for t in ideas if t != tag_name]
            entry.tag_ideas = ideas
            await db.commit()

        return {"status": "dismissed", "tag_name": tag_name, "remaining_ideas": ideas}

    async def approve_all_tag_ideas(
        self, entry_id: int, client: PaperlessClient,
    ) -> Dict[str, Any]:
        """Approve ALL tag ideas for a single document."""
        if self.session_factory is None:
            raise RuntimeError("No session_factory configured")

        async with self.session_factory() as db:
            q = await db.execute(
                select(ClassificationHistory).where(ClassificationHistory.id == entry_id)
            )
            entry = q.scalars().first()
            if not entry:
                raise ValueError(f"Entry {entry_id} not found")

            ideas = entry.tag_ideas
            if isinstance(ideas, str):
                try:
                    ideas = _json.loads(ideas)
                except Exception:
                    ideas = []
            if not ideas:
                return {"status": "nothing_to_approve"}

            doc = await client.get_document(entry.document_id)
            existing_tags = doc.get("tags", []) if doc else []

            approved = []
            for tag_name in ideas:
                tag = await client.get_or_create_tag(tag_name)
                if tag and tag["id"] not in existing_tags:
                    existing_tags.append(tag["id"])
                    approved.append(tag_name)

            if existing_tags and doc:
                await client.update_document(entry.document_id, {"tags": existing_tags})

            entry.tag_ideas = []
            await db.commit()

        from app.services.cache import get_cache
        await get_cache().clear("paperless:")

        logger.info(f"All tag ideas approved for doc {entry.document_id}: {approved}")
        return {"status": "approved_all", "approved": approved}

    async def bulk_approve_tag_idea(
        self, tag_name: str, client: PaperlessClient,
    ) -> Dict[str, Any]:
        """Approve a specific tag across ALL documents that suggest it."""
        if self.session_factory is None:
            raise RuntimeError("No session_factory configured")

        async with self.session_factory() as db:
            q = await db.execute(
                select(ClassificationHistory).where(
                    ClassificationHistory.tag_ideas.isnot(None),
                    sa_func.length(ClassificationHistory.tag_ideas) > 2,
                )
            )
            entries = q.scalars().all()

            affected = 0
            tag_obj = await client.get_or_create_tag(tag_name)
            if not tag_obj:
                raise ValueError(f"Tag '{tag_name}' konnte nicht erstellt werden")

            for entry in entries:
                ideas = entry.tag_ideas
                if isinstance(ideas, str):
                    try:
                        ideas = _json.loads(ideas)
                    except Exception:
                        continue
                if tag_name not in ideas:
                    continue

                try:
                    doc = await client.get_document(entry.document_id)
                    if doc:
                        existing_tags = doc.get("tags", [])
                        if tag_obj["id"] not in existing_tags:
                            existing_tags.append(tag_obj["id"])
                            await client.update_document(entry.document_id, {"tags": existing_tags})
                except Exception as e:
                    logger.warning(f"Failed to add tag to doc {entry.document_id}: {e}")

                ideas = [t for t in ideas if t != tag_name]
                entry.tag_ideas = ideas
                affected += 1

            await db.commit()

        from app.services.cache import get_cache
        await get_cache().clear("paperless:")

        logger.info(f"Bulk approved tag '{tag_name}' for {affected} documents")
        return {"status": "bulk_approved", "tag_name": tag_name, "documents_affected": affected}

    async def bulk_dismiss_tag_idea(self, tag_name: str) -> Dict[str, Any]:
        """Dismiss a specific tag across ALL documents that suggest it."""
        if self.session_factory is None:
            raise RuntimeError("No session_factory configured")

        async with self.session_factory() as db:
            q = await db.execute(
                select(ClassificationHistory).where(ClassificationHistory.tag_ideas != "[]")
            )
            entries = q.scalars().all()

            affected = 0
            for entry in entries:
                ideas = entry.tag_ideas
                if isinstance(ideas, str):
                    try:
                        ideas = _json.loads(ideas)
                    except Exception:
                        continue
                if tag_name not in ideas:
                    continue

                ideas = [t for t in ideas if t != tag_name]
                entry.tag_ideas = ideas
                affected += 1

            await db.commit()

        logger.info(f"Bulk dismissed tag '{tag_name}' from {affected} documents")
        return {"status": "bulk_dismissed", "tag_name": tag_name, "documents_affected": affected}

    async def assign_existing_tag(
        self, entry_id: int, tag_name: str, client: PaperlessClient,
    ) -> Dict[str, Any]:
        """Assign an existing Paperless tag to a document from the tag-ideas view."""
        if self.session_factory is None:
            raise RuntimeError("No session_factory configured")

        async with self.session_factory() as db:
            q = await db.execute(
                select(ClassificationHistory).where(ClassificationHistory.id == entry_id)
            )
            entry = q.scalars().first()
            if not entry:
                raise ValueError(f"Entry {entry_id} not found")

            tag = await client.get_or_create_tag(tag_name)
            if not tag:
                raise ValueError(f"Tag '{tag_name}' nicht gefunden")

            doc = await client.get_document(entry.document_id)
            if not doc:
                raise ValueError(f"Dokument {entry.document_id} nicht gefunden")

            existing_tags = doc.get("tags", [])
            if tag["id"] not in existing_tags:
                existing_tags.append(tag["id"])
                await client.update_document(entry.document_id, {"tags": existing_tags})

        from app.services.cache import get_cache
        await get_cache().clear("paperless:")

        logger.info(f"Assigned existing tag '{tag_name}' to doc {entry.document_id}")
        return {"status": "assigned", "tag_name": tag_name, "document_id": entry.document_id}

    # ── Extracted router business logic (Plan 10-02) ───────────────────────

    async def refresh_paperless_cache(self, client: PaperlessClient) -> dict:
        """Clear cache and reload all Paperless metadata."""
        from app.services.cache import get_cache
        await get_cache().clear("paperless:")
        tags, correspondents, doc_types, paths = await asyncio.gather(
            client.get_tags(use_cache=False),
            client.get_correspondents(use_cache=False),
            client.get_document_types(use_cache=False),
            client.get_storage_paths(use_cache=False),
        )
        return {
            "refreshed": True,
            "tags": len(tags),
            "correspondents": len(correspondents),
            "document_types": len(doc_types),
            "storage_paths": len(paths),
        }

    async def start_auto_classify_task(self, config_svc: Any, di_container: Any) -> dict:
        """Start auto-classify background task."""
        if self.state and self.state.enabled:
            return {"status": "already_running"}
        if self.state:
            self.state.enabled, self.state.processed, self.state.errors, self.state.reviewed = True, 0, 0, 0
            from app.services.classifier.state import auto_classify_loop
            self.state._task = asyncio.create_task(auto_classify_loop(di_container))
        try:
            await config_svc.set("auto_classify_enabled", "true", "bool")
        except Exception:
            pass
        return {"status": "started"}

    async def stop_auto_classify_task(self, config_svc: Any) -> dict:
        """Stop auto-classify background task."""
        if self.state:
            self.state.enabled = False
            if self.state._task and not self.state._task.done():
                self.state._task.cancel()
            self.state.running, self.state.current_doc = False, None
        try:
            await config_svc.set("auto_classify_enabled", "false", "bool")
        except Exception:
            pass
        return {"status": "stopped"}

    def get_auto_classify_status_dict(self, llm_service: Any = None) -> dict:
        """Get auto-classify status including lock info."""
        if not self.state:
            return {"enabled": False, "running": False, "processed": 0, "errors": 0, "reviewed": 0, "current_doc": None, "last_run": None, "waiting_for": None}
        lock = llm_service.get_lock_status() if llm_service else {}
        waiting = next(
            (p for p, s in lock.items() if s["locked"] and p != "classifier"), None,
        ) if self.state.enabled else None
        return {
            "enabled": self.state.enabled,
            "running": self.state.running,
            "processed": self.state.processed,
            "errors": self.state.errors,
            "reviewed": self.state.reviewed,
            "current_doc": self.state.current_doc,
            "last_run": self.state.last_run,
            "waiting_for": waiting,
        }
