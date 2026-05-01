"""Classifier provider using LiteLLM tool calling (OpenAI, Mistral, OpenRouter, Anthropic, etc.)."""

import json
import time
import logging
from typing import Dict, Any, List, Optional

from app.services.classifier.base_provider import (
    BaseClassifierProvider, ClassificationResult, DocumentContext,
)
from app.services.llm import LLMService
from app.services.classifier.tool_definitions import (
    CLASSIFIER_TOOLS,
)
from app.services.classifier.tool_executor import ToolExecutor
from app.services.classifier.prompts import (
    SYSTEM_PROMPT_OPENAI, RULES_TITLE, RULES_TAGS, RULES_CORRESPONDENT,
    RULES_DOCTYPE, RULES_DATE, RULES_CUSTOM_FIELDS, get_correspondent_rules, PROMPTS,
)
from app.services.classifier.llm_schemas import _MAX_TOOL_ROUNDS

logger = logging.getLogger(__name__)


class ToolCallingLlmProvider(BaseClassifierProvider):
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
        llm_service: LLMService | None = None,
        session_factory=None,
    ):
        self.model = model
        self.provider = provider
        self.tool_executor = tool_executor
        self._provider_label = provider_label or provider
        self.llm_service = llm_service
        self.session_factory = session_factory
        self._prompts_registered = False

    async def _register_prompts(self, db: Any) -> None:
        """Register all classifier prompts with settings_service. Called once on first use."""
        from app.services.settings_service import register_prompt
        for key, prompt_template in PROMPTS.items():
            await register_prompt(key, prompt_template, db)
        self._prompts_registered = True

    async def _get_prompt(self, key: str, variables: Optional[Dict[str, Any]] = None) -> str:
        """Get a prompt template by key, optionally rendered with variables."""
        from app.services.settings_service import get_prompt
        if self.session_factory is None:
            template_str = PROMPTS.get(key, "")
            if variables:
                from jinja2 import Template
                return Template(template_str, autoescape=False).render(**variables)
            return template_str
        async with self.session_factory() as db:
            if not self._prompts_registered:
                await self._register_prompts(db)
            prompt_template = await get_prompt(key, db, variables)
            if prompt_template:
                return prompt_template
            template_str = PROMPTS.get(key, "")
            if variables:
                from jinja2 import Template
                return Template(template_str, autoescape=False).render(**variables)
            return template_str

    def get_name(self) -> str:
        return f"{self._provider_label} ({self.model})"

    def supports_tool_calling(self) -> bool:
        return True

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
        assert self.llm_service is not None
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        total_tool_calls = 0

        enabled_fields = self._get_enabled_fields(config)

        trim_prompt = config.get("correspondent_trim_prompt", False)

        rules_title = await self._get_prompt("classifier_rules_title")
        rules_tags = await self._get_prompt("classifier_rules_tags")
        rules_correspondent = await self._get_prompt(
            "classifier_rules_correspondent_short" if trim_prompt else "classifier_rules_correspondent"
        )
        rules_doctype = await self._get_prompt("classifier_rules_doctype")
        rules_date = await self._get_prompt("classifier_rules_date")
        rules_custom_fields = await self._get_prompt("classifier_rules_custom_fields")

        system_prompt = await self._get_prompt(
            "classifier_openai",
            variables={
                "RULES_TITLE": rules_title,
                "RULES_TAGS": rules_tags,
                "RULES_CORRESPONDENT": rules_correspondent,
                "RULES_DOCTYPE": rules_doctype,
                "RULES_DATE": rules_date,
                "RULES_CUSTOM_FIELDS": rules_custom_fields,
            },
        )

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

        user_content = self._build_user_message(document)
        logger.info(f"LiteLLM tool-calling user message length: {len(user_content)} chars")

        active_tools = self._filter_tools(config)
        logger.info(f"LiteLLM active tools: {[t['function']['name'] for t in active_tools]}")

        messages: List[Dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            for _round in range(_MAX_TOOL_ROUNDS):
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
                    if result.assistant_message is not None:
                        messages.append(result.assistant_message)
                    for tool_call in result.tool_calls:
                        total_tool_calls += 1
                        fn_name = tool_call.name
                        fn_args = json.loads(tool_call.arguments)

                        logger.info(f"Tool call: {fn_name}({fn_args})")

                        if self.tool_executor is not None:
                            result_str = await self.tool_executor.execute(fn_name, fn_args)
                        else:
                            result_str = ""

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str,
                        })
                    continue

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
                error=f"Max tool call rounds ({_MAX_TOOL_ROUNDS}) reached",
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