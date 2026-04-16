"""Unified LLM service — uses LiteLLM for all providers."""

import json
import re
import logging
from typing import Optional, Dict, Any, List

import litellm
from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import LLMProvider
from app.models.settings_model import LLM_KEY_CLASSIFIER_MODEL, LLM_KEY_CLASSIFIER_PROVIDER
from app.routers.settings import get_setting

logger = logging.getLogger(__name__)


class LitellmService:
    """Service for interacting with various LLM providers via LiteLLM."""

    # Static model info for UI display (pricing/context)
    # NOTE: For actual LLM calls, LiteLLM handles provider routing internally via model name prefix
    MODEL_INFO: Dict[str, Dict[str, Any]] = {
        # OpenAI
        "gpt-4o": {"context": 128000, "input_price": 2.50, "output_price": 10.00, "description": "Flagship model", "provider": "openai"},
        "gpt-4o-mini": {"context": 128000, "input_price": 0.15, "output_price": 0.60, "description": "Günstig & schnell", "provider": "openai"},
        "gpt-4-turbo": {"context": 128000, "input_price": 10.00, "output_price": 30.00, "description": "Vorgänger von GPT-4o", "provider": "openai"},
        "gpt-4": {"context": 8192, "input_price": 30.00, "output_price": 60.00, "description": "Original GPT-4", "provider": "openai"},
        "o1-preview": {"context": 128000, "input_price": 15.00, "output_price": 60.00, "description": "Reasoning-Modell", "provider": "openai"},
        "o1-mini": {"context": 128000, "input_price": 3.00, "output_price": 12.00, "description": "Schnelleres Reasoning", "provider": "openai"},
        # Anthropic
        "claude-3-5-sonnet-20241022": {"context": 200000, "input_price": 3.00, "output_price": 15.00, "description": "Neuestes Sonnet", "provider": "anthropic"},
        "claude-3-5-haiku-20241022": {"context": 200000, "input_price": 0.80, "output_price": 4.00, "description": "Schnell & günstig", "provider": "anthropic"},
        "claude-3-opus-20240229": {"context": 200000, "input_price": 15.00, "output_price": 75.00, "description": "Stärkstes Claude", "provider": "anthropic"},
        "claude-3-sonnet-20240229": {"context": 200000, "input_price": 3.00, "output_price": 15.00, "description": "Älteres Sonnet", "provider": "anthropic"},
        # Mistral
        "mistral-small-latest": {"context": 32000, "input_price": 0.10, "output_price": 0.30, "description": "Schnell & günstig", "provider": "mistral"},
        "mistral-large-latest": {"context": 128000, "input_price": 2.00, "output_price": 6.00, "description": "Stärkstes Mistral", "provider": "mistral"},
        # Ollama/Local
        "llama3.2": {"context": 128000, "input_price": 0, "output_price": 0, "description": "Lokal - kostenlos", "provider": "ollama"},
        "llama3.1": {"context": 128000, "input_price": 0, "output_price": 0, "description": "Lokal - kostenlos", "provider": "ollama"},
        "llama3": {"context": 8192, "input_price": 0, "output_price": 0, "description": "Lokal - kostenlos", "provider": "ollama"},
        "mistral": {"context": 32768, "input_price": 0, "output_price": 0, "description": "Lokal - kostenlos", "provider": "ollama"},
        "mixtral": {"context": 32768, "input_price": 0, "output_price": 0, "description": "Lokal - MoE Modell", "provider": "ollama"},
        "qwen2.5": {"context": 32768, "input_price": 0, "output_price": 0, "description": "Lokal - kostenlos", "provider": "ollama"},
        "gemma2": {"context": 8192, "input_price": 0, "output_price": 0, "description": "Lokal - kostenlos", "provider": "ollama"},
        # Gemini
        "gemini-2.0-flash": {"context": 1000000, "input_price": 0.00, "output_price": 0.00, "description": "Google Gemini", "provider": "gemini"},
    }

    @classmethod
    def get_available_models(cls, provider: str = None) -> list:
        """Get list of available models with their info."""
        models = []
        for model_id, info in cls.MODEL_INFO.items():
            if provider is None or info.get("provider") == provider:
                models.append({
                    "id": model_id,
                    "provider": info.get("provider"),
                    "context": info.get("context"),
                    "input_price": info.get("input_price"),
                    "output_price": info.get("output_price"),
                    "description": info.get("description")
                })
        return models

    @classmethod
    def get_model_info(cls, model_id: str) -> Optional[Dict[str, Any]]:
        """Get info dict for a specific model_id."""
        if model_id in cls.MODEL_INFO:
            info = cls.MODEL_INFO[model_id].copy()
            info["model"] = model_id
            return info
        return None

    def __init__(self, provider: Optional[LLMProvider] = None, model: Optional[str] = None):
        self.provider = provider
        self.model = model

    async def complete(self, prompt: str, model_override: Optional[str] = None) -> str:
        """Send a completion request to the LLM provider via LiteLLM.
        
        LiteLLM handles provider routing internally based on model name prefix
        (e.g., 'anthropic/claude-3-5-sonnet', 'ollama/llama3').
        """
        model = model_override or self.model
        if not model:
            raise ValueError("No model specified")

        # Build litellm completion kwargs
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }

        # Add provider credentials if available
        if self.provider:
            if self.provider.api_key:
                kwargs["api_key"] = self.provider.api_key
            if self.provider.api_base_url:
                kwargs["api_base"] = self.provider.api_base_url

        response = await litellm.acompletion(**kwargs)
        return (response.choices[0].message.content or "").strip()

    async def test_connection(self) -> Dict:
        """Test connection to the LLM provider."""
        if not self.provider:
            raise ValueError("No LLM provider configured")
        response = await self.complete("Antworte nur mit: OK")
        return {
            "provider": self.provider.name,
            "model": self.model,
            "response": response
        }

    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Estimate token count using LiteLLM's token counter.
        
        Args:
            text: Text to count tokens for
            model: Model to use for tokenization (uses self.model if not specified)
        """
        model_for_count = model or self.model or "gpt-4"
        try:
            return litellm.token_counter(model=model_for_count, text=text)
        except Exception:
            # Fallback to char-based estimation
            return len(text) // 4

    def get_token_limit(self, model: Optional[str] = None) -> int:
        """Get the token limit for a model based on MODEL_INFO."""
        model_name = model or self.model
        if model_name and model_name in self.MODEL_INFO:
            info = self.MODEL_INFO[model_name]
            context = info.get("context", 8000)
            return int(context * 0.95)  # Leave buffer for output
        return 8000  # Conservative default

    def get_instance_model_info(self) -> Optional[Dict[str, Any]]:
        """Get info about the current provider and model."""
        if not self.provider:
            return None

        return {
            "model": self.model or "Nicht konfiguriert",
            "provider_name": self.provider.name,
            "context": self.get_token_limit()
        }

    async def analyze_for_similarity(self, prompt_template: str, items: list) -> Dict:
        """Analyze items for similarity using the configured LLM."""
        if not items:
            return {"groups": [], "stats": {"items_count": 0, "estimated_tokens": 0}}

        # Format items as a list
        items_str = json.dumps([item["name"] for item in items], ensure_ascii=False, indent=2)

        # Fill in the prompt template
        prompt = prompt_template.replace("{items}", items_str)

        # Estimate tokens
        estimated_input_tokens = self.estimate_tokens(prompt)

        logger.info(f"[LLM] Analyzing {len(items)} items, estimated tokens: {estimated_input_tokens}")

        # Check if likely too large
        token_warning = None
        max_recommended = 8000
        if estimated_input_tokens > max_recommended:
            token_warning = f"Viele Items ({len(items)})! Geschätzte Tokens: ~{estimated_input_tokens}. Könnte das Limit überschreiten."

        # Get LLM response
        logger.info(f"[LLM] Sending request to LLM provider...")
        response = await self.complete(prompt)
        logger.info(f"[LLM] Got response, length: {len(response)} chars, first 200: {response[:200]}")

        # Estimate output tokens
        estimated_output_tokens = self.estimate_tokens(response)

        # Build stats
        stats = {
            "items_count": len(items),
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_total_tokens": estimated_input_tokens + estimated_output_tokens,
            "warning": token_warning
        }

        # Parse JSON from response
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group()

                # Fix common JSON errors
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)

                result = None
                parse_error = None

                # Attempt 1: Direct parse
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as e:
                    parse_error = e
                    logger.warning(f"[LLM] JSON parse attempt 1 failed: {e}")

                # Attempt 2: Find balanced JSON
                if result is None:
                    try:
                        depth = 0
                        last_valid = 0
                        in_string = False
                        escape_next = False

                        for i, c in enumerate(json_str):
                            if escape_next:
                                escape_next = False
                                continue
                            if c == '\\':
                                escape_next = True
                                continue
                            if c == '"' and not escape_next:
                                in_string = not in_string
                                continue
                            if in_string:
                                continue
                            if c == '{':
                                depth += 1
                            elif c == '}':
                                depth -= 1
                                if depth == 0:
                                    last_valid = i + 1
                                    break

                        if last_valid > 0:
                            json_str = json_str[:last_valid]
                            result = json.loads(json_str)
                            logger.info(f"[LLM] JSON parse attempt 2 succeeded (truncated to {last_valid} chars)")
                    except json.JSONDecodeError as e:
                        parse_error = e
                        logger.warning(f"[LLM] JSON parse attempt 2 failed: {e}")

                # Attempt 3: Try to extract just the groups array
                if result is None:
                    try:
                        groups_match = re.search(r'"groups"\s*:\s*\[([\s\S]*?)\](?=\s*[,}]|$)', json_str)
                        if groups_match:
                            groups_content = groups_match.group(1)
                            group_objects = []
                            depth = 0
                            start = -1
                            in_str = False
                            esc = False

                            for i, c in enumerate(groups_content):
                                if esc:
                                    esc = False
                                    continue
                                if c == '\\':
                                    esc = True
                                    continue
                                if c == '"':
                                    in_str = not in_str
                                    continue
                                if in_str:
                                    continue
                                if c == '{':
                                    if depth == 0:
                                        start = i
                                    depth += 1
                                elif c == '}':
                                    depth -= 1
                                    if depth == 0 and start >= 0:
                                        try:
                                            obj = json.loads(groups_content[start:i+1])
                                            group_objects.append(obj)
                                        except Exception:
                                            pass
                                        start = -1

                            if group_objects:
                                result = {"groups": group_objects}
                                logger.info(f"[LLM] JSON parse attempt 3 succeeded, extracted {len(group_objects)} groups")
                    except Exception as e:
                        logger.warning(f"[LLM] JSON parse attempt 3 failed: {e}")

                if result is None:
                    raise parse_error or json.JSONDecodeError("Could not parse JSON", json_str, 0)

                # Enrich groups with IDs from original items
                items_dict = {item["name"]: item for item in items}
                items_dict_lower = {item["name"].lower(): item for item in items}

                for group in result.get("groups", []):
                    enriched_members = []
                    for member_name in group.get("members", []):
                        matched = False

                        # 1. Exact match
                        if member_name in items_dict:
                            enriched_members.append(items_dict[member_name])
                            matched = True
                        # 2. Case-insensitive match
                        elif member_name.lower() in items_dict_lower:
                            enriched_members.append(items_dict_lower[member_name.lower()])
                            matched = True
                        else:
                            # 3. Fuzzy match
                            for name, item in items_dict.items():
                                if (member_name.lower() in name.lower() or
                                    name.lower() in member_name.lower() or
                                    member_name.lower().replace(" ", "") == name.lower().replace(" ", "")):
                                    enriched_members.append(item)
                                    matched = True
                                    break

                        if not matched:
                            logger.warning(f"[LLM] Member '{member_name}' not found in items!")

                    group["members"] = enriched_members
                    if len(enriched_members) < len(group.get("members", [])):
                        logger.warning(f"[LLM] Group '{group.get('suggested_name')}': Only {len(enriched_members)} of {len(group.get('members', []))} members matched")

                result["stats"] = stats
                return result
            else:
                return {"groups": [], "error": "Keine JSON-Antwort vom LLM erhalten", "stats": stats, "raw_response": response[:500]}
        except json.JSONDecodeError as e:
            error_context = response[max(0, e.pos-100):e.pos+100] if hasattr(e, 'pos') else response[:200]
            return {
                "groups": [],
                "error": f"JSON-Fehler: {str(e)}. Kontext: ...{error_context}...",
                "stats": stats
            }


# Module-level MODEL_INFO export for backward compatibility
MODEL_INFO = LitellmService.MODEL_INFO


async def get_llm_service(db: AsyncSession = Depends(get_db)) -> LitellmService:
    """Dependency to get LLM service with provider from AppSettings."""
    provider_name = await get_setting(LLM_KEY_CLASSIFIER_PROVIDER, db)
    provider = None
    if provider_name:
        result = await db.execute(
            select(LLMProvider).where(LLMProvider.name == provider_name)
        )
        provider = result.scalars().first()
    
    if not provider:
        result = await db.execute(select(LLMProvider).limit(1))
        provider = result.scalars().first()
    
    model = await get_setting(LLM_KEY_CLASSIFIER_MODEL, db)
    return LitellmService(provider, model=model)
