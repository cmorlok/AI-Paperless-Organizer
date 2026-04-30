"""Classifier provider using Ollama with sequential focused LLM calls."""

import json
import random
import re
import time
import logging
from typing import Any, Dict, List, Optional

from app.services.classifier.base_provider import (
    BaseClassifierProvider, ClassificationResult, DocumentContext,
)
from app.services.llm import LLMService
from app.services.classifier.tool_executor import ToolExecutor
from app.services.classifier.prompts import (
    SYSTEM_PROMPT_OLLAMA_ANALYZE,
    SYSTEM_PROMPT_OLLAMA_STORAGE_PATH, SYSTEM_PROMPT_OLLAMA_CUSTOM_FIELDS,
    SYSTEM_PROMPT_OLLAMA_VERIFY, RULES_TITLE, RULES_TAGS, RULES_CORRESPONDENT,
    RULES_DOCTYPE, RULES_DATE, get_correspondent_rules,
)
from app.services.classifier.llm_schemas import (
    _MAX_TOOL_ROUNDS, _MAX_CONTENT_CHARS, _LOCAL_LLM_CALL_TIMEOUT,
    _THINKING_MODEL_PREFIXES, _STRICT_SCHEMA_MODELS,
    _SCHEMA_ANALYZE, _SCHEMA_TAGS, _SCHEMA_DOCTYPE,
    _SCHEMA_STORAGE_PATH, _SCHEMA_VERIFY,
)

logger = logging.getLogger(__name__)


class OllamaLlmProvider(BaseClassifierProvider):
    """Classifies documents using Ollama with sequential focused calls.

    Since Ollama models don't support tool calling, we split the classification
    into focused sequential LLM calls:
      1. Analyze: title, correspondent, date, summary
      2. Document Type: simple pick from list
      3. Tags: pick from list with context
      4. Storage Path: assign based on profiles + all context
      5. Custom Fields: extract configured values

    Each call does ONE thing only -- small models work best with focused tasks.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        provider: str = "ollama",
        tool_executor: Optional[ToolExecutor] = None,
        llm_service: LLMService | None = None,
    ):
        self.model = model
        self.provider = provider
        self.tool_executor = tool_executor
        self.llm_service = llm_service
        self._is_thinking = any(
            k in self.model.lower() for k in _THINKING_MODEL_PREFIXES
        )
        self._use_strict_schemas = any(
            k in self.model.lower() for k in _STRICT_SCHEMA_MODELS
        )

    def get_name(self) -> str:
        return f"Ollama ({self.model})"

    def supports_tool_calling(self) -> bool:
        return False

    async def test_connection(self) -> Dict[str, Any]:
        assert self.llm_service is not None
        try:
            await self.llm_service.complete(
                provider=self.provider,
                model=self.model,
                messages=[{"role": "user", "content": "Ping"}],
                max_tokens=5,
            )
            return {"connected": True, "model": self.model}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    async def classify(
        self,
        document: DocumentContext,
        config: Dict[str, Any],
    ) -> ClassificationResult:
        start_time = time.time()
        total_calls = 0

        content = document.content
        original_content_len = len(content)
        if len(content) > _MAX_CONTENT_CHARS:
            content = content[:_MAX_CONTENT_CHARS] + "\n[... gekuerzt ...]"

        content_snippet = content

        self._total_input_tokens = 0
        self._total_output_tokens = 0

        result = ClassificationResult()
        result.debug_info["content_original_chars"] = original_content_len
        result.debug_info["content_sent_chars"] = len(content)
        result.debug_info["model"] = self.model
        result.debug_info["is_thinking"] = self._is_thinking

        candidate_tags: List[str] = []
        candidate_set_lower: Dict[str, str] = {}
        type_names: List[str] = []
        valid_path_ids: List[int] = []

        try:
            if config.get("enable_title", True):
                result.title = None
            if config.get("enable_correspondent", True):
                result.correspondent = None
            if config.get("enable_created_date", True):
                result.created_date = None
            if config.get("enable_document_type", True):
                result.document_type = None
            if config.get("enable_tags", True):
                result.tags = []
            if config.get("enable_storage_path", True):
                result.storage_path_id = None
                result.storage_path_reason = None
            result.custom_fields = {}

            analyze_prompt = SYSTEM_PROMPT_OLLAMA_ANALYZE
            if config.get("prompt_title") and config["prompt_title"].strip():
                analyze_prompt = analyze_prompt.replace(RULES_TITLE, config["prompt_title"])
            if config.get("prompt_correspondent") and config["prompt_correspondent"].strip():
                analyze_prompt = analyze_prompt.replace(RULES_CORRESPONDENT, config["prompt_correspondent"])
            elif config.get("correspondent_trim_prompt"):
                analyze_prompt = analyze_prompt.replace(
                    RULES_CORRESPONDENT, get_correspondent_rules(trim_prompt=True)
                )
            if config.get("prompt_date") and config["prompt_date"].strip():
                analyze_prompt = analyze_prompt.replace(RULES_DATE, config["prompt_date"])

            existing_hints = []
            if document.current_correspondent:
                existing_hints.append(f"Korrespondent: {document.current_correspondent}")
            if document.current_document_type:
                existing_hints.append(f"Dokumenttyp: {document.current_document_type}")
            if document.current_storage_path:
                existing_hints.append(f"Speicherpfad: {document.current_storage_path}")
            if document.current_tags:
                existing_hints.append(f"Tags: {', '.join(document.current_tags)}")

            existing_block = ""
            if existing_hints:
                existing_block = (
                    "\n\nBEREITS VORHANDENE METADATEN (behalte sie wenn passend, "
                    "verbessere sie nur wenn du Besseres im Inhalt findest):\n"
                    + "\n".join(f"- {h}" for h in existing_hints)
                )

            first_lines = [line.strip() for line in content.split("\n") if line.strip()][:3]
            header_hint = "\n".join(first_lines)

            analyze_user_msg = (
                f"Aktueller Titel: {document.current_title}{existing_block}"
                f"\n\n=== DOKUMENT-KOPF (WICHTIGSTE ZEILEN!) ===\n{header_hint}"
                f"\n\n--- VOLLSTAENDIGER DOKUMENTINHALT ---\n{content}"
            )

            analysis = ""
            analysis_data: Dict[str, Any] = {}
            _attempt = 0
            for _attempt in range(3):
                analysis = await self._call_ollama(
                    analyze_prompt,
                    analyze_user_msg,
                    max_tokens=500,
                    json_schema=_SCHEMA_ANALYZE,
                )
                total_calls += 1
                analysis_data = self._parse_json(analysis)
                if analysis_data.get("title") or analysis_data.get("summary"):
                    break
                if _attempt < 2:
                    logger.warning(
                        f"Analyze call attempt {_attempt + 1} returned empty result "
                        f"(raw={repr(analysis[:100])}), retrying…"
                    )

            result.debug_info["analyze_raw_response"] = analysis[:500] if analysis else ""
            result.debug_info["analyze_attempts"] = _attempt + 1

            if config.get("enable_title", True):
                result.title = analysis_data.get("title")
            if config.get("enable_correspondent", True):
                corr = analysis_data.get("correspondent")
                if corr and self._is_hallucinated_correspondent(corr):
                    logger.warning(f"Correspondent looks like hallucination, discarding: {corr[:80]}")
                    corr = None
                result.correspondent = corr
            if config.get("enable_created_date", True):
                result.created_date = analysis_data.get("created_date")

            summary = analysis_data.get("summary") or document.current_title or ""
            result.summary = summary

            if config.get("enable_document_type", True) and self.tool_executor:
                doc_types_data = await self.tool_executor.execute("get_document_types", {})
                doc_types = json.loads(doc_types_data)
                type_names = [dt["name"] for dt in doc_types] if doc_types else []

                if type_names:
                    dtype_prompt = (
                        f"Bestimme den Dokumenttyp anhand des folgenden Dokuments.\n\n"
                        f"EXTRAHIERTE METADATEN:\n"
                        f"- Titel: {result.title or 'unbekannt'}\n"
                        f"- Korrespondent: {result.correspondent or 'unbekannt'}\n"
                        f"- KI-Zusammenfassung: {summary}\n\n"
                        f"DOKUMENTINHALT (Anfang):\n{content_snippet}\n\n"
                        f"ENTSCHEIDUNGSREGELN:\n"
                        f"- Rechnungsnummer (RE-..., RG-..., INV-..., R-...) im Inhalt? -> 'Rechnung'\n"
                        f"- Netto/Brutto/MwSt-Angaben im Inhalt? -> 'Rechnung'\n"
                        f"- 'Zahlungseingang bestaetigt' + Rechnungsnummer? -> trotzdem 'Rechnung'\n"
                        f"- 'Bestaetigung' NUR fuer Auftrags-/Bestellbestaetigung OHNE Rechnungsnummer\n"
                        f"- Monatlicher Kontoauszug? -> 'Kontoauszug'\n"
                        f"- Vertrag/Kuendigungsschreiben? -> 'Vertrag'\n\n"
                        f"VERFUEGBARE TYPEN: {', '.join(type_names)}\n\n"
                        'Antworte als JSON: {"document_type": "Name"}'
                    )
                    dtype_schema = _SCHEMA_DOCTYPE
                    if self._use_strict_schemas:
                        dtype_schema = {
                            "type": "object",
                            "required": ["document_type"],
                            "properties": {
                                "document_type": {"type": ["string", "null"], "enum": type_names + [None]},
                            },
                            "additionalProperties": False,
                        }
                    dtype_response = await self._call_ollama(dtype_prompt, "", max_tokens=80, json_schema=dtype_schema)
                    total_calls += 1
                    dtype_data = self._parse_json(dtype_response)
                    if isinstance(dtype_data, dict):
                        dt = dtype_data.get("document_type") or dtype_data.get("type") or dtype_data.get("dokumenttyp")
                        if dt:
                            dt_str = str(dt).strip()
                            for tn in type_names:
                                if tn.lower() == dt_str.lower():
                                    result.document_type = tn
                                    break
                            if not result.document_type:
                                for tn in type_names:
                                    if dt_str.lower() in tn.lower() or tn.lower() in dt_str.lower():
                                        result.document_type = tn
                                        break
                    result.debug_info["doctype_raw_response"] = dtype_response
                    logger.info(f"DocType call: raw='{dtype_response[:100]}' -> '{result.document_type}'")

            if config.get("enable_tags", True) and self.tool_executor:
                all_tags_data = await self.tool_executor.execute("search_tags", {"query": ""})
                all_tags = json.loads(all_tags_data)

                if all_tags:
                    tags_min = config.get("tags_min", 1)
                    tags_max = config.get("tags_max", 5)

                    content_tags = list(all_tags)
                    ignore_patterns = []
                    for pat in (config.get("tags_ignore") or []):
                        if "*" in pat:
                            regex_pat = re.escape(pat).replace(r"\*", ".*")
                            ignore_patterns.append(re.compile(f"^{regex_pat}$", re.IGNORECASE))

                    if ignore_patterns:
                        before = len(content_tags)
                        content_tags = [
                            t for t in content_tags
                            if not any(p.match(t.get("name", "")) for p in ignore_patterns)
                        ]
                        logger.info(f"Wildcard ignore patterns removed {before - len(content_tags)} tags")

                    candidate_tags = [t.get("name", "") for t in content_tags]

                    logger.info(f"Tags: {len(all_tags)} total -> {len(candidate_tags)} sent to model "
                               f"(removed {len(all_tags) - len(candidate_tags)} system/ignored)")

                    result.debug_info["tags_total"] = len(all_tags)
                    result.debug_info["tags_after_blacklist"] = len(content_tags)
                    result.debug_info["tags_sent_to_model"] = candidate_tags
                    result.debug_info["summary_used"] = summary

                    tags_rule = config.get("prompt_tags") or RULES_TAGS
                    tag_prompt = (
                        f"DOKUMENT-KONTEXT:\n"
                        f"- Titel: {result.title or 'unbekannt'}\n"
                        f"- Typ: {result.document_type or 'unbekannt'}\n"
                        f"- Korrespondent: {result.correspondent or 'unbekannt'}\n"
                        f"- KI-Zusammenfassung: {summary}\n\n"
                        f"DOKUMENTINHALT (Anfang):\n{content_snippet}\n\n"
                        f"VERFUEGBARE TAGS:\n{', '.join(candidate_tags)}\n\n"
                        f"{tags_rule}\n"
                        f"Waehle {tags_min}-{tags_max} Tags. "
                        f'Antworte als JSON: {{"tags": ["Tag1", "Tag2"]}}'
                    )

                    result.debug_info["tag_prompt_length"] = len(tag_prompt)

                    candidate_set_lower = {t.lower(): t for t in candidate_tags}

                    tag_response = await self._call_ollama(tag_prompt, "", max_tokens=200, json_schema=_SCHEMA_TAGS)
                    total_calls += 1
                    result.debug_info["tag_raw_response"] = tag_response[:300]

                    tag_data = self._parse_json(tag_response)
                    raw_tags: List[str] = []
                    if isinstance(tag_data, dict):
                        raw_tags = tag_data.get("tags", [])
                        if not isinstance(raw_tags, list):
                            raw_tags = []
                    elif isinstance(tag_data, list):
                        raw_tags = tag_data

                    valid_tags = []
                    for t in raw_tags:
                        if not isinstance(t, str) or not t.strip():
                            continue
                        t = t.strip()
                        if t in candidate_tags:
                            valid_tags.append(t)
                        elif t.lower() in candidate_set_lower:
                            valid_tags.append(candidate_set_lower[t.lower()])
                        else:
                            logger.info(f"Tag '{t}' is NEW (not in existing tags)")
                            valid_tags.append(t)

                    result.tags = valid_tags
                    logger.info(f"Tags call result: {result.tags} (raw: {raw_tags})")

            if config.get("enable_storage_path", True) and self.tool_executor:
                paths_data = await self.tool_executor.execute("get_storage_paths", {})
                paths = json.loads(paths_data)

                if paths:
                    profiles_text = "\n".join(
                        f"- ID {p['id']}: {p['name']} (Person: {p.get('person_name', '-')}, "
                        f"Typ: {p.get('type', '-')})\n  Kontext: {p.get('context_prompt', 'Kein Kontext')}"
                        for p in paths
                    )
                    path_prompt = SYSTEM_PROMPT_OLLAMA_STORAGE_PATH.format(
                        path_profiles=profiles_text,
                        title=result.title or "unbekannt",
                        summary=summary,
                        content_snippet=content_snippet,
                        correspondent=result.correspondent or "unbekannt",
                        document_type=result.document_type or "unbekannt",
                        tags=", ".join(result.tags) if result.tags else "keine",
                    )

                    result.debug_info["storage_path_profiles"] = [
                        {"id": p["id"], "name": p["name"],
                         "person": p.get("person_name", ""),
                         "type": p.get("type", ""),
                         "context": p.get("context_prompt", "")}
                        for p in paths
                    ]
                    result.debug_info["storage_path_prompt_length"] = len(path_prompt)

                    valid_path_ids = [p["id"] for p in paths]

                    if self._use_strict_schemas:
                        path_schema = {
                            "type": "object",
                            "required": ["path_id", "reason"],
                            "properties": {
                                "path_id": {"enum": valid_path_ids + [None]},
                                "reason":  {"type": "string"},
                            },
                            "additionalProperties": False,
                        }
                    else:
                        path_schema = _SCHEMA_STORAGE_PATH

                    path_response = await self._call_ollama(path_prompt, "", max_tokens=200, json_schema=path_schema)
                    total_calls += 1
                    result.debug_info["storage_path_raw_response"] = path_response

                    path_data = self._parse_json(path_response)
                    if isinstance(path_data, dict):
                        raw_path_id = path_data.get("path_id")
                        if raw_path_id is not None and raw_path_id in valid_path_ids:
                            result.storage_path_id = raw_path_id
                        elif raw_path_id is not None:
                            logger.warning(f"path_id {raw_path_id} not in valid list {valid_path_ids} – discarded")
                        result.storage_path_reason = path_data.get("reason")

            if config.get("enable_custom_fields", False) and self.tool_executor:
                fields_data = await self.tool_executor.execute("get_custom_field_definitions", {})
                fields = json.loads(fields_data)

                if fields:
                    fields_text = "\n".join(
                        f"- {f['field_name']} (Typ: {f['field_type']}): {f['extraction_prompt']}"
                        + (f"\n  Beispiele: {f['example_values']}" if f.get("example_values") else "")
                        for f in fields
                    )
                    cf_prompt = SYSTEM_PROMPT_OLLAMA_CUSTOM_FIELDS.format(
                        field_definitions=fields_text,
                    )
                    cf_response = await self._call_ollama(
                        cf_prompt,
                        f"--- DOKUMENTINHALT ---\n{content[:4000]}",
                        max_tokens=400,
                    )
                    total_calls += 1
                    cf_data = self._parse_json(cf_response)
                    if isinstance(cf_data, dict):
                        result.custom_fields = cf_data

            has_gaps = (
                (config.get("enable_storage_path") and not result.storage_path_id) or
                (config.get("enable_tags") and len(result.tags) == 0) or
                (config.get("enable_document_type") and not result.document_type) or
                (config.get("enable_correspondent") and not result.correspondent)
            )

            if has_gaps:
                logger.info(f"Verification needed: sp={result.storage_path_id}, "
                            f"tags={len(result.tags)}, dt={result.document_type}, "
                            f"corr={result.correspondent}")

                paths_text = "Keine verfuegbar"
                if self.tool_executor:
                    try:
                        sp_data = await self.tool_executor.execute("get_storage_paths", {})
                        sp_list = json.loads(sp_data)
                        if sp_list:
                            paths_text = "\n".join(
                                f"- ID {p['id']}: {p['name']} ({p.get('type', '-')}) "
                                f"Kontext: {p.get('context_prompt', '-')}"
                                for p in sp_list
                            )
                    except Exception:
                        pass

                verify_prompt = SYSTEM_PROMPT_OLLAMA_VERIFY.format(
                    summary=summary,
                    title=result.title or "fehlt",
                    correspondent=result.correspondent or "fehlt",
                    document_type=result.document_type or "fehlt",
                    tags=", ".join(result.tags) if result.tags else "keine",
                    storage_path_id=result.storage_path_id or "null",
                    storage_path_reason=result.storage_path_reason or "fehlt",
                    created_date=result.created_date or "fehlt",
                    storage_paths=paths_text,
                )

                if self._use_strict_schemas:
                    verify_schema: Dict[str, Any] = {
                        "type": "object",
                        "properties": {
                            "storage_path_id": {"enum": valid_path_ids + [None]},
                            "storage_path_reason": {"type": "string"},
                            "tags": {"type": "array", "items": {
                                "type": "string", "enum": candidate_tags or [""],
                            }},
                            "document_type": {"type": ["string", "null"],
                                              "enum": (type_names + [None]) if type_names else [None]},
                            "correspondent": {"type": ["string", "null"]},
                        },
                    }
                else:
                    verify_schema = _SCHEMA_VERIFY

                verify_response = await self._call_ollama(verify_prompt, "", max_tokens=300, json_schema=verify_schema)
                total_calls += 1
                verify_data = self._parse_json(verify_response)

                if isinstance(verify_data, dict) and verify_data:
                    logger.info(f"Verification corrections: {verify_data}")
                    result.debug_info["verification_corrections"] = verify_data

                    if "storage_path_id" in verify_data and verify_data["storage_path_id"] is not None:
                        sp_id = verify_data["storage_path_id"]
                        if not valid_path_ids and self.tool_executor:
                            try:
                                sp_data_v = await self.tool_executor.execute("get_storage_paths", {})
                                valid_path_ids = [p["id"] for p in json.loads(sp_data_v)]
                            except Exception:
                                pass
                        if sp_id in valid_path_ids:
                            result.storage_path_id = sp_id
                            raw_reason = verify_data.get("storage_path_reason", "")
                            if raw_reason and not self._contains_non_latin(raw_reason):
                                result.storage_path_reason = raw_reason
                        else:
                            logger.warning(f"Verification path_id {sp_id} invalid – ignored")

                    if "tags" in verify_data and isinstance(verify_data["tags"], list):
                        result.tags = [
                            candidate_set_lower[t.lower()] if t.lower() in candidate_set_lower else t
                            for t in verify_data["tags"]
                            if isinstance(t, str) and (t in candidate_tags or t.lower() in candidate_set_lower)
                        ]

                    if "document_type" in verify_data and verify_data["document_type"]:
                        dt_v = verify_data["document_type"]
                        if not type_names or any(
                            dt_v.lower() == tn.lower() for tn in type_names
                        ):
                            result.document_type = next(
                                (tn for tn in type_names if tn.lower() == dt_v.lower()), dt_v
                            ) if type_names else dt_v
                        else:
                            logger.warning(f"Verification document_type '{dt_v}' not in valid list – ignored")

                    if "correspondent" in verify_data and verify_data["correspondent"]:
                        corr_v = verify_data["correspondent"]
                        if not self._is_hallucinated_correspondent(corr_v):
                            result.correspondent = corr_v
                        else:
                            logger.warning(f"Verification correspondent looks like hallucination – ignored: {corr_v[:80]}")
                else:
                    logger.info("Verification: no corrections needed or empty response")
                    result.debug_info["verification_corrections"] = None

        except Exception as e:
            logger.error(f"Ollama classification failed: {e}", exc_info=True)
            result.error = str(e)

        await self._unload_model()

        result.duration_seconds = time.time() - start_time
        result.tool_calls_count = total_calls
        result.tokens_input = self._total_input_tokens
        result.tokens_output = self._total_output_tokens
        result.cost_usd = 0.0
        result.debug_info["total_tokens"] = self._total_input_tokens + self._total_output_tokens
        logger.info(f"Ollama total: {self._total_input_tokens}+{self._total_output_tokens} tokens, "
                     f"{total_calls} calls, {result.duration_seconds:.1f}s")
        return result

    async def _call_ollama(
        self, system_prompt: str, user_message: str,
        max_tokens: int = 500, keep_alive: str = "5m",
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        assert self.llm_service is not None
        if self._is_thinking:
            return await self._call_ollama_generate(
                system_prompt, user_message, max_tokens, keep_alive, json_schema
            )
        return await self._call_ollama_chat(
            system_prompt, user_message, max_tokens, keep_alive, json_schema
        )

    async def _call_ollama_chat(
        self, system_prompt: str, user_message: str,
        max_tokens: int, keep_alive: str,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        assert self.llm_service is not None
        messages = [{"role": "system", "content": system_prompt}]
        if user_message:
            messages.append({"role": "user", "content": user_message})

        try:
            result = await self.llm_service.complete(
                provider=self.provider,
                model=self.model,
                messages=messages,
                num_predict=max_tokens,
                keep_alive=keep_alive,
                seed=random.randint(1, 2**31 - 1),
                json_schema=json_schema,
                json_output=True,
                timeout=_LOCAL_LLM_CALL_TIMEOUT,
            )
            content = result.content or ""
            self._total_input_tokens  += result.input_tokens
            self._total_output_tokens += result.output_tokens
            logger.info(f"Ollama chat: {len(content)} chars: {content[:200]}")
            return content
        except Exception as e:
            if json_schema is not None and ("format" in str(e).lower() or "schema" in str(e).lower()):
                logger.warning(f"Ollama schema enforcement rejected, retrying without schema: {e}")
                result = await self.llm_service.complete(
                    provider=self.provider,
                    model=self.model,
                    messages=messages,
                    num_predict=max_tokens,
                    keep_alive=keep_alive,
                    seed=random.randint(1, 2**31 - 1),
                    json_output=True,
                    timeout=_LOCAL_LLM_CALL_TIMEOUT,
                )
                content = result.content or ""
                self._total_input_tokens  += result.input_tokens
                self._total_output_tokens += result.output_tokens
                return content
            raise

    async def _call_ollama_generate(
        self, system_prompt: str, user_message: str,
        max_tokens: int, keep_alive: str,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        assert self.llm_service is not None
        prompt_parts = [
            "Du bist ein JSON-Extraktor. Antworte AUSSCHLIESSLICH mit validem JSON.",
            "KEIN Denkprozess, KEINE Erklaerung, KEIN Markdown -- NUR das JSON-Objekt.",
            "",
            "AUFGABE:",
            system_prompt,
        ]
        if user_message:
            prompt_parts.extend(["", "INPUT:", user_message])
        prompt_parts.extend(["", "JSON-ANTWORT:"])
        raw_prompt = "\n".join(prompt_parts)

        messages = [{"role": "user", "content": raw_prompt}]

        try:
            result = await self.llm_service.complete(
                provider=self.provider,
                model=self.model,
                messages=messages,
                num_predict=max_tokens,
                keep_alive=keep_alive,
                think=False,
                json_schema=json_schema,
                json_output=True,
                timeout=_LOCAL_LLM_CALL_TIMEOUT,
            )
            content = result.content or ""
            self._total_input_tokens  += result.input_tokens
            self._total_output_tokens += result.output_tokens
            return content

        except Exception as e:
            if json_schema is not None and ("format" in str(e).lower() or "schema" in str(e).lower()):
                logger.warning(f"Ollama generate schema rejected, retrying without schema: {e}")
                result = await self.llm_service.complete(
                    provider=self.provider,
                    model=self.model,
                    messages=messages,
                    num_predict=max_tokens,
                    keep_alive=keep_alive,
                    think=False,
                    json_output=True,
                    timeout=_LOCAL_LLM_CALL_TIMEOUT,
                )
                content = result.content or ""
                self._total_input_tokens  += result.input_tokens
                self._total_output_tokens += result.output_tokens
                return content
            raise

    async def _unload_model(self):
        assert self.llm_service is not None
        try:
            await self.llm_service.unload_local_model(self.provider, self.model)
            logger.info(f"Model '{self.model}' unloaded from GPU")
        except Exception as e:
            logger.warning(f"Could not unload model: {e}")

    def _parse_json(self, text: str) -> Any:
        if not text or not text.strip():
            logger.warning("Empty text passed to _parse_json")
            return {}

        text = text.strip()

        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        text = self._strip_thinking_text(text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for match in re.finditer(r'\{', text):
            candidate = text[match.start():]
            depth = 0
            end = -1
            for i, ch in enumerate(candidate):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end > 0:
                try:
                    return json.loads(candidate[:end + 1])
                except json.JSONDecodeError:
                    continue

        arr_match = re.search(r'\[.*\]', text, re.DOTALL)
        if arr_match:
            try:
                return json.loads(arr_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse JSON from Ollama response: {text[:300]}")
        return {}

    @staticmethod
    def _contains_non_latin(text: str) -> bool:
        for ch in text:
            cp = ord(ch)
            if (
                0x4E00 <= cp <= 0x9FFF
                or 0x3400 <= cp <= 0x4DBF
                or 0x20000 <= cp <= 0x2A6DF
                or 0x3040 <= cp <= 0x30FF
                or 0xAC00 <= cp <= 0xD7AF
                or 0x0600 <= cp <= 0x06FF
                or 0x0400 <= cp <= 0x04FF
            ):
                return True
        return False

    def _is_hallucinated_correspondent(self, value: str) -> bool:
        if not value:
            return False
        v = value.lower().strip()
        if v.startswith(("http://", "https://", "www.")):
            return True
        if len(value) > 120:
            return True
        if self._contains_non_latin(value):
            return True
        _HALLUCINATION_PATTERNS = (
            "placeholder", "field in the json", "ckan", "api reference",
            "openapi", "swagger", "graphql", "rest api", "json api",
            "example.com", "lorem ipsum", "insert here", "your name",
        )
        return any(p in v for p in _HALLUCINATION_PATTERNS)

    def _strip_thinking_text(self, text: str) -> str:
        if not text:
            return text

        first_brace = text.find('{')
        first_bracket = text.find('[')

        candidates = [i for i in (first_brace, first_bracket) if i >= 0]
        if not candidates:
            return text

        json_start = min(candidates)

        if json_start > 0:
            prefix = text[:json_start].strip()
            if prefix and not prefix.startswith(('{', '[')):
                logger.info(f"Stripped {json_start} chars of thinking preamble")
                text = text[json_start:]

        return text.strip()