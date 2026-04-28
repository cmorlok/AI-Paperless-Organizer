"""Classifier providers for LiteLLM — replaces openai_provider.py and ollama_provider.py."""

import json
import random
import re
import time
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from app.services.classifier.base_provider import (
    BaseClassifierProvider, ClassificationResult, DocumentContext,
)

if TYPE_CHECKING:
    from app.services.llm.service import LitellmService
from app.services.classifier.tool_definitions import (
    CLASSIFIER_TOOLS,
)
from app.services.classifier.tool_executor import ToolExecutor
from app.services.classifier.prompts import (
    SYSTEM_PROMPT_OPENAI, SYSTEM_PROMPT_OLLAMA_ANALYZE,
    SYSTEM_PROMPT_OLLAMA_STORAGE_PATH, SYSTEM_PROMPT_OLLAMA_CUSTOM_FIELDS,
    SYSTEM_PROMPT_OLLAMA_VERIFY, RULES_TITLE, RULES_TAGS, RULES_CORRESPONDENT,
    RULES_DOCTYPE, RULES_DATE, get_correspondent_rules,
)

logger = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 10
MAX_CONTENT_CHARS = 10000
LOCAL_LLM_CALL_TIMEOUT = 180.0
THINKING_MODEL_PREFIXES = ("qwen3", "deepseek-r1", "qwq")
_STRICT_SCHEMA_MODELS = ("mistral",)

# --- JSON Schemas for structured Ollama output ---
# Forces grammar-based constrained generation – prevents models like mistral-nemo
# from returning arbitrary JSON structures that don't match the expected schema.
_SCHEMA_ANALYZE = {
    "type": "object",
    "required": ["title", "correspondent", "created_date", "summary", "language"],
    "properties": {
        "title":         {"type": ["string", "null"]},
        "correspondent": {"type": ["string", "null"]},
        "created_date":  {"type": ["string", "null"]},
        "summary":       {"type": "string"},
        "language":      {"type": "string"},
    },
    "additionalProperties": False,
}

_SCHEMA_TAGS = {
    "type": "object",
    "required": ["tags"],
    "properties": {
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

_SCHEMA_DOCTYPE = {
    "type": "object",
    "required": ["document_type"],
    "properties": {
        "document_type": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

_SCHEMA_STORAGE_PATH = {
    "type": "object",
    "required": ["path_id", "reason"],
    "properties": {
        "path_id": {"type": ["integer", "null"]},
        "reason":  {"type": "string"},
    },
    "additionalProperties": False,
}

_SCHEMA_VERIFY = {
    "type": "object",
    "properties": {
        "storage_path_id":     {"type": ["integer", "null"]},
        "storage_path_reason": {"type": "string"},
        "tags":                {"type": "array", "items": {"type": "string"}},
        "document_type":       {"type": ["string", "null"]},
        "correspondent":       {"type": ["string", "null"]},
    },
}


class LitellmToolCallingProvider(BaseClassifierProvider):
    """Classifies documents using LiteLLM tool calling (replaces OpenAIToolCallingProvider).

    Supports OpenAI, Mistral, OpenRouter, Anthropic, and any other LiteLLM-compatible
    provider that supports tool calling.

    Flow:
    1. Send document content + tool definitions to the LLM via litellm.acompletion
    2. LLM may request tool calls (search_tags, search_correspondents, ...)
    3. We execute those calls locally against our Paperless cache
    4. Send results back to the LLM
    5. Repeat until the LLM returns the final classification
    """

    def __init__(
        self,
        model: str,
        provider: str,
        tool_executor: Optional[ToolExecutor] = None,
        provider_label: str = "",
        llm_service: "LitellmService | None" = None,
    ):
        self.model = model
        self.provider = provider
        self.tool_executor = tool_executor
        self._provider_label = provider_label or provider
        self.llm_service = llm_service

    def get_name(self) -> str:
        return f"{self._provider_label} ({self.model})"

    def supports_tool_calling(self) -> bool:
        return True

    async def test_connection(self) -> Dict[str, Any]:
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
        total_input_tokens = 0
        total_output_tokens = 0
        total_tool_calls = 0

        enabled_fields = self._get_enabled_fields(config)

        system_prompt = config.get("system_prompt") or SYSTEM_PROMPT_OPENAI
        system_prompt += f"\n\nAktivierte Felder: {', '.join(enabled_fields)}"
        tags_min = config.get("tags_min", 1)
        tags_max = config.get("tags_max", 5)
        system_prompt += f"\nTag-Anzahl: Mindestens {tags_min}, maximal {tags_max} Tags."
        if "custom_fields" in enabled_fields:
            system_prompt += "\nDu MUSST get_custom_field_definitions aufrufen und die Felder extrahieren!"
        else:
            system_prompt += "\nCustom Fields sind deaktiviert, ignoriere get_custom_field_definitions."
        if "storage_path" in enabled_fields:
            system_prompt += "\nDu MUSST get_storage_paths aufrufen und einen Pfad zuordnen! storage_path_id und storage_path_reason MUESSEN im Ergebnis stehen!"
        else:
            system_prompt += "\nSpeicherpfad ist deaktiviert, ignoriere get_storage_paths."

        # Replace default rules with user-configured prompts when set.
        # If trim_prompt is enabled and no manual prompt_correspondent is set,
        # swap in the short-name variant automatically.
        trim_prompt = config.get("correspondent_trim_prompt", False)
        effective_correspondent_rule = config.get("prompt_correspondent") or (
            get_correspondent_rules(trim_prompt) if trim_prompt else None
        )
        replacements = {
            RULES_TITLE: config.get("prompt_title"),
            RULES_TAGS: config.get("prompt_tags"),
            RULES_CORRESPONDENT: effective_correspondent_rule,
            RULES_DOCTYPE: config.get("prompt_document_type"),
            RULES_DATE: config.get("prompt_date"),
        }
        for default_rule, user_rule in replacements.items():
            if user_rule and user_rule.strip():
                system_prompt = system_prompt.replace(default_rule, user_rule)

        user_content = self._build_user_message(document)
        logger.info(f"LiteLLM tool-calling user message length: {len(user_content)} chars")

        active_tools = self._filter_tools(config)
        logger.info(f"LiteLLM active tools: {[t['function']['name'] for t in active_tools]}")

        messages: List[Dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            for _round in range(MAX_TOOL_ROUNDS):
                result = await self.llm_service.complete(
                    provider=self.provider,
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    tools=active_tools if active_tools else None,
                )
                total_input_tokens += result.input_tokens
                total_output_tokens += result.output_tokens

                if result.finish_reason == "tool_calls" and result.tool_calls:
                    messages.append(result.assistant_message)
                    for tool_call in result.tool_calls:
                        total_tool_calls += 1
                        fn_name = tool_call.name
                        fn_args = json.loads(tool_call.arguments)

                        logger.info(f"Tool call: {fn_name}({fn_args})")

                        result_str = await self.tool_executor.execute(fn_name, fn_args)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str,
                        })
                    continue

                # No more tool calls -- parse final response
                raw = result.content or "{}"
                raw = raw.strip()
                if "```" in raw:
                    import re as _re
                    m = _re.search(r'```(?:json)?\s*\n(.*?)```', raw, _re.DOTALL)
                    if m:
                        raw = m.group(1).strip()
                if not raw.startswith("{"):
                    brace = raw.find("{")
                    if brace >= 0:
                        raw = raw[brace:]

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response: {raw[:500]}")
                    return ClassificationResult(
                        error=f"Invalid JSON from LLM: {raw[:200]}",
                        tokens_input=total_input_tokens,
                        tokens_output=total_output_tokens,
                        duration_seconds=time.time() - start_time,
                    )

                logger.info(f"LiteLLM tool-calling result keys: {list(data.keys())}")
                logger.info(f"LiteLLM raw tags: {data.get('tags', [])}")
                if "storage_path_id" not in data:
                    logger.warning(f"LiteLLM did NOT return storage_path_id! "
                                   f"Tool calls made: {total_tool_calls}")

                result = self._parse_result(
                    data, total_input_tokens, total_output_tokens,
                    total_tool_calls, start_time,
                )
                result.debug_info["content_sent_chars"] = len(user_content)
                result.debug_info["model"] = self.model
                result.debug_info["tools_called"] = total_tool_calls
                result.debug_info["raw_tags_from_llm"] = data.get("tags", [])
                return result

            return ClassificationResult(
                error=f"Max tool call rounds ({MAX_TOOL_ROUNDS}) reached",
                tokens_input=total_input_tokens,
                tokens_output=total_output_tokens,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"LiteLLM tool-calling classification failed: {e}", exc_info=True)
            return ClassificationResult(
                error=str(e),
                tokens_input=total_input_tokens,
                tokens_output=total_output_tokens,
                duration_seconds=time.time() - start_time,
            )

    def _get_enabled_fields(self, config: Dict) -> List[str]:
        fields = []
        mapping = {
            "enable_title": "title",
            "enable_tags": "tags",
            "enable_correspondent": "correspondent",
            "enable_document_type": "document_type",
            "enable_storage_path": "storage_path",
            "enable_created_date": "created_date",
            "enable_custom_fields": "custom_fields",
        }
        for key, name in mapping.items():
            if config.get(key, False):
                fields.append(name)
        return fields

    def _build_user_message(self, doc: DocumentContext) -> str:
        parts = [f"Dokument-ID: {doc.document_id}"]
        if doc.current_title:
            parts.append(f"Aktueller Titel: {doc.current_title}")
        if doc.current_tags:
            parts.append(f"Aktuelle Tags: {', '.join(doc.current_tags)}")
        if doc.current_correspondent:
            parts.append(f"Aktueller Korrespondent: {doc.current_correspondent}")
        if doc.current_document_type:
            parts.append(f"Aktueller Dokumenttyp: {doc.current_document_type}")

        content = doc.content
        if len(content) > 15000:
            content = content[:15000] + "\n[... Inhalt gekuerzt ...]"

        parts.append(f"\n--- DOKUMENTINHALT ---\n{content}")
        return "\n".join(parts)

    def _filter_tools(self, config: Dict) -> List[Dict]:
        """Only include tools for enabled features."""
        tools = []
        for tool in CLASSIFIER_TOOLS:
            fn_name = tool["function"]["name"]
            if fn_name == "search_tags" and config.get("enable_tags"):
                tools.append(tool)
            elif fn_name == "search_correspondents" and config.get("enable_correspondent"):
                tools.append(tool)
            elif fn_name == "get_document_types" and config.get("enable_document_type"):
                tools.append(tool)
            elif fn_name == "get_storage_paths" and config.get("enable_storage_path"):
                tools.append(tool)
            elif fn_name == "get_custom_field_definitions" and config.get("enable_custom_fields"):
                tools.append(tool)
        return tools

    def _parse_result(
        self, data: Dict, input_tokens: int, output_tokens: int,
        tool_calls: int, start_time: float,
    ) -> ClassificationResult:
        model_info = {
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.50, 10.00),
            "mistral-small-latest": (0.10, 0.30),
            "mistral-medium-latest": (0.40, 1.20),
            "mistral-large-latest": (2.00, 6.00),
            "codestral-latest": (0.30, 0.90),
            "open-mistral-nemo": (0.15, 0.15),
            "ministral-8b-latest": (0.10, 0.10),
        }
        input_price, output_price = model_info.get(self.model, (0.15, 0.60))
        cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000

        return ClassificationResult(
            title=data.get("title"),
            tags=data.get("tags", []),
            correspondent=data.get("correspondent"),
            document_type=data.get("document_type"),
            storage_path_id=data.get("storage_path_id"),
            storage_path_reason=data.get("storage_path_reason"),
            created_date=data.get("created_date"),
            custom_fields=data.get("custom_fields", {}),
            tokens_input=input_tokens,
            tokens_output=output_tokens,
            cost_usd=cost,
            duration_seconds=time.time() - start_time,
            tool_calls_count=tool_calls,
        )


class LitellmOllamaProvider(BaseClassifierProvider):
    """Classifies documents using Ollama via LiteLLM with sequential focused calls.

    Replaces OllamaMultiCallProvider. Since Ollama models don't support tool calling,
    we split the classification into focused sequential LLM calls:
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
        llm_service: "LitellmService | None" = None,
    ):
        self.model = model
        self.provider = provider
        self.tool_executor = tool_executor
        self.llm_service = llm_service
        self._is_thinking = any(
            k in self.model.lower() for k in THINKING_MODEL_PREFIXES
        )
        # Use strict enum-constrained schemas for models known to hallucinate values
        self._use_strict_schemas = any(
            k in self.model.lower() for k in _STRICT_SCHEMA_MODELS
        )

    def get_name(self) -> str:
        return f"Ollama ({self.model})"

    def supports_tool_calling(self) -> bool:
        return False

    async def test_connection(self) -> Dict[str, Any]:
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
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS] + "\n[... gekuerzt ...]"

        # All follow-up calls get the same full content as Call 1
        # (local Ollama = no cost, longer is fine)
        content_snippet = content

        self._total_input_tokens = 0
        self._total_output_tokens = 0

        result = ClassificationResult()
        result.debug_info["content_original_chars"] = original_content_len
        result.debug_info["content_sent_chars"] = len(content)
        result.debug_info["model"] = self.model
        result.debug_info["is_thinking"] = self._is_thinking

        # Shared across calls – populated in their respective call blocks
        candidate_tags: List[str] = []
        candidate_set_lower: Dict[str, str] = {}
        type_names: List[str] = []
        valid_path_ids: List[int] = []

        try:
            # --- Call 1: Analyze (title, correspondent, date, summary) ---
            # Replace default rules with user-configured prompts when set
            analyze_prompt = SYSTEM_PROMPT_OLLAMA_ANALYZE
            if config.get("prompt_title") and config["prompt_title"].strip():
                analyze_prompt = analyze_prompt.replace(RULES_TITLE, config["prompt_title"])
            # Correspondent: user override takes priority; otherwise use trim variant if enabled
            if config.get("prompt_correspondent") and config["prompt_correspondent"].strip():
                analyze_prompt = analyze_prompt.replace(RULES_CORRESPONDENT, config["prompt_correspondent"])
            elif config.get("correspondent_trim_prompt"):
                analyze_prompt = analyze_prompt.replace(
                    RULES_CORRESPONDENT, get_correspondent_rules(trim_prompt=True)
                )
            if config.get("prompt_date") and config["prompt_date"].strip():
                analyze_prompt = analyze_prompt.replace(RULES_DATE, config["prompt_date"])

            # Build existing-metadata hint for the LLM
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

            # Extract first 3 non-empty lines as prominent header hint
            first_lines = [line.strip() for line in content.split("\n") if line.strip()][:3]
            header_hint = "\n".join(first_lines)

            analyze_user_msg = (
                f"Aktueller Titel: {document.current_title}{existing_block}"
                f"\n\n=== DOKUMENT-KOPF (WICHTIGSTE ZEILEN!) ===\n{header_hint}"
                f"\n\n--- VOLLSTAENDIGER DOKUMENTINHALT ---\n{content}"
            )

            # Retry analyze up to 3 times – thinking models sometimes return empty JSON
            analysis = ""
            analysis_data: Dict[str, Any] = {}
            for _attempt in range(3):
                analysis = await self._call_ollama(
                    analyze_prompt,
                    analyze_user_msg,
                    max_tokens=500,
                    json_schema=_SCHEMA_ANALYZE,
                )
                total_calls += 1
                analysis_data = self._parse_json(analysis)
                # Accept if we got at least a title or summary
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
                # Discard obvious hallucinations: URLs, placeholder strings, suspiciously long values
                if corr and self._is_hallucinated_correspondent(corr):
                    logger.warning(f"Correspondent looks like hallucination, discarding: {corr[:80]}")
                    corr = None
                result.correspondent = corr
            if config.get("enable_created_date", True):
                result.created_date = analysis_data.get("created_date")

            summary = analysis_data.get("summary") or document.current_title or ""
            result.summary = summary

            # --- Call 2: Document Type (simple, dedicated) ---
            if config.get("enable_document_type", True) and self.tool_executor:
                doc_types_data = await self.tool_executor.execute("get_document_types", {})
                doc_types = json.loads(doc_types_data)
                type_names = [dt["name"] for dt in doc_types] if doc_types else []  # shared scope

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
                    # Strict models: enum constrains output to only valid type names
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

            # --- Call 3: Tags (all tags after configured exclusions) ---
            if config.get("enable_tags", True) and self.tool_executor:
                all_tags_data = await self.tool_executor.execute("search_tags", {"query": ""})
                all_tags = json.loads(all_tags_data)

                if all_tags:
                    tags_min = config.get("tags_min", 1)
                    tags_max = config.get("tags_max", 5)

                    # ToolExecutor already filtered excluded_tag_ids + exact tags_ignore.
                    # Here we additionally apply wildcard patterns from tags_ignore.
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

                    # Use user-configured prompt if set, otherwise fall back to RULES_TAGS default
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

                    tags_schema = _SCHEMA_TAGS

                    tag_response = await self._call_ollama(tag_prompt, "", max_tokens=200, json_schema=tags_schema)
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

            # --- Call 4: Storage Path (with ALL context from previous calls) ---
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

                    valid_path_ids = [p["id"] for p in paths]  # shared scope

                    # Strict models: enum constrains path_id to only valid IDs
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
                        # Post-processing: validate that path_id is actually in the list
                        if raw_path_id is not None and raw_path_id in valid_path_ids:
                            result.storage_path_id = raw_path_id
                        elif raw_path_id is not None:
                            logger.warning(f"path_id {raw_path_id} not in valid list {valid_path_ids} – discarded")
                        result.storage_path_reason = path_data.get("reason")

            # --- Call 5: Custom Fields ---
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

            # --- Call 6: Self-Verification (LLM reviews its own result) ---
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

                # Build storage paths text for verification
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

                # For strict-schema models: add enum constraints to verification too
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

                    # storage_path_id: only accept if it's a valid ID from our list
                    if "storage_path_id" in verify_data and verify_data["storage_path_id"] is not None:
                        sp_id = verify_data["storage_path_id"]
                        # Use already-fetched valid_path_ids if available, else re-fetch
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
                        # Only accept if it's a known document type
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
        """Make a single Ollama call via LiteLLM. Uses generate path with raw prompt for
        thinking models (bypasses chat template that triggers thinking),
        /api/chat with format=json/schema for standard models.
        """
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
        """Standard models: chat endpoint with format=json or JSON schema via LiteLLM."""
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
                timeout=LOCAL_LLM_CALL_TIMEOUT,
            )
            content = result.content or ""
            self._total_input_tokens  += result.input_tokens
            self._total_output_tokens += result.output_tokens
            logger.info(f"Ollama chat via litellm: {len(content)} chars: {content[:200]}")
            return content
        except Exception as e:
            # REVIEW FEEDBACK HIGH: Schema enforcement not supported — retry without schema (HTTP 400/422 fallback)
            # LiteLLM raises BadRequestError (not httpx.HTTPStatusError) for Ollama HTTP 400/422.
            # The retry condition checks the exception message for "format" or "schema" keywords
            # since LiteLLM normalizes the error.
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
                    timeout=LOCAL_LLM_CALL_TIMEOUT,
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
        """Thinking models: raw prompt approach via LiteLLM chat endpoint.
        We construct a raw prompt that forces direct JSON output without
        triggering the model's thinking behavior. think=False suppresses
        thinking output via extra_body per D-01.
        """
        # Build a raw prompt that forces direct JSON output.
        # LiteLLM uses /api/chat; we inject the raw prompt as a user turn.
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
                timeout=LOCAL_LLM_CALL_TIMEOUT,
            )
            content = result.content or ""
            self._total_input_tokens  += result.input_tokens
            self._total_output_tokens += result.output_tokens
            return content

        except Exception as e:
            # Schema fallback for thinking models too
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
                    timeout=LOCAL_LLM_CALL_TIMEOUT,
                )
                content = result.content or ""
                self._total_input_tokens  += result.input_tokens
                self._total_output_tokens += result.output_tokens
                return content
            raise

    @staticmethod
    def _contains_non_latin(text: str) -> bool:
        """Return True if the text contains CJK or other non-Latin script characters."""
        for ch in text:
            cp = ord(ch)
            # CJK Unified Ideographs + Extensions, Katakana, Hiragana, Hangul, Arabic, etc.
            if (
                0x4E00 <= cp <= 0x9FFF   # CJK main block
                or 0x3400 <= cp <= 0x4DBF  # CJK Ext-A
                or 0x20000 <= cp <= 0x2A6DF  # CJK Ext-B
                or 0x3040 <= cp <= 0x30FF   # Hiragana / Katakana
                or 0xAC00 <= cp <= 0xD7AF   # Hangul
                or 0x0600 <= cp <= 0x06FF   # Arabic
                or 0x0400 <= cp <= 0x04FF   # Cyrillic
            ):
                return True
        return False

    def _is_hallucinated_correspondent(self, value: str) -> bool:
        """Detect correspondent values that are clearly hallucinations."""
        if not value:
            return False
        v = value.lower().strip()
        # URL patterns
        if v.startswith(("http://", "https://", "www.")):
            return True
        # Suspiciously long (> 120 chars)
        if len(value) > 120:
            return True
        # Non-Latin script (Chinese, Japanese, Korean, Arabic, Cyrillic …)
        if self._contains_non_latin(value):
            return True
        # Known hallucination fragments from LLM training data bleed-through
        _HALLUCINATION_PATTERNS = (
            "placeholder", "field in the json", "ckan", "api reference",
            "openapi", "swagger", "graphql", "rest api", "json api",
            "example.com", "lorem ipsum", "insert here", "your name",
        )
        return any(p in v for p in _HALLUCINATION_PATTERNS)

    def _strip_thinking_text(self, text: str) -> str:
        """Aggressively remove thinking preamble from response."""
        if not text:
            return text

        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

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

    async def _unload_model(self):
        """Unload model from GPU memory after classification."""
        try:
            await self.llm_service.unload_local_model(self.provider, self.model)
            logger.info(f"Model '{self.model}' unloaded from GPU")
        except Exception as e:
            logger.warning(f"Could not unload model: {e}")

    def _parse_json(self, text: str) -> Any:
        """Extract JSON from LLM response with aggressive fallbacks."""
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
