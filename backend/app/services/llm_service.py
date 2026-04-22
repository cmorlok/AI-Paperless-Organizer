"""Unified LLM service — uses LiteLLM for all providers."""

from __future__ import annotations

import json
import re
import time
import traceback
from typing import Optional, Dict, Any, List

import httpx
import litellm

from app.core.logging import get_logger

logger = get_logger("llm")

# =========================================================================
# LiteLLM callback-based request/response logging
# =========================================================================

_callbacks_registered = False


def _log_llm(msg: str, data: dict[str, Any]) -> None:
    logger.debug("%s: %s", msg, json.dumps(data, indent=2, default=str))


def _get_data_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract common data from litellm kwargs for logging."""
    provider = kwargs.get("metadata", {}).get("custom_llm_provider") if "metadata" in kwargs else None
    model = kwargs.get("model", "unknown")
    additional_args = kwargs.get("additional_args", {})
    request = additional_args.get("complete_input_dict", {})
    if request == {}:
        request = {
            "model": model,
            "messages": kwargs.get("messages", []),
            **kwargs.get("optional_params", {}),
        }
    return {"provider": provider, "model": model, "request": request}


def _llm_input_callback(kwargs: dict[str, Any]) -> None:
    api_base = kwargs.get("additional_args", {}).get("api_base")
    request_url = "unknown"
    if api_base:
        if isinstance(api_base, str):
            request_url = api_base
        elif isinstance(api_base, (list, tuple)):
            try:
                scheme = str(api_base[0]) if len(api_base) > 0 and api_base[0] else "http"
                host = str(api_base[2]) if len(api_base) > 2 else ""
                port = api_base[3] if len(api_base) > 3 and api_base[3] else None
                path = str(api_base[4]) if len(api_base) > 4 else ""
                url = f"{scheme}://{host}"
                if port:
                    url = f"{url}:{port}"
                if path:
                    url = f"{url}{path}"
                request_url = url
            except Exception:
                request_url = str(api_base)
    if request_url == "unknown":
        request_url = str(kwargs.get("base_url") or kwargs.get("api_base") or "unknown")

    llm_data = _get_data_from_kwargs(kwargs)
    llm_data.update({"event": "input", "request_url": request_url})
    _log_llm("LLM input", llm_data)


async def _llm_success_callback(kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
    duration = 0.0
    try:
        diff = end_time - start_time
        duration = diff.total_seconds() if hasattr(diff, "total_seconds") else float(diff)
    except Exception:
        pass

    response = response_obj.model_dump() if hasattr(response_obj, "model_dump") else str(response_obj)
    llm_data = _get_data_from_kwargs(kwargs)
    llm_data.update({"event": "success", "response": response, "duration": duration, "status": "success"})
    _log_llm("LLM success", llm_data)


async def _llm_failure_callback(kwargs: dict[str, Any], exception: Exception, start_time: Any, end_time: Any) -> None:
    duration = 0.0
    try:
        diff = end_time - start_time
        duration = diff.total_seconds() if hasattr(diff, "total_seconds") else float(diff)
    except Exception:
        pass

    error_msg = str(exception) if exception else ""
    if not error_msg or error_msg == "None":
        error_msg = str(kwargs.get("exception", kwargs.get("error", "")))

    llm_data = _get_data_from_kwargs(kwargs)
    llm_data.update({"event": "failure", "error": error_msg, "duration": duration, "status": "error"})
    _log_llm("LLM failure", llm_data)


def _register_litellm_callbacks() -> None:
    """Register input/success/failure callbacks with LiteLLM (idempotent)."""
    global _callbacks_registered
    if _callbacks_registered:
        return
    _callbacks_registered = True
    litellm.logging_callback_manager.add_litellm_input_callback(_llm_input_callback)
    litellm.logging_callback_manager.add_litellm_success_callback(_llm_success_callback)
    litellm.logging_callback_manager.add_litellm_failure_callback(_llm_failure_callback)


# Register callbacks on module load
_register_litellm_callbacks()


def extract_litellm_error(exc: Exception) -> str:
    """Extract full error details from a litellm exception (response body, headers, etc.)."""
    parts = [str(exc)]
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            body = resp.text if hasattr(resp, "text") else ""
            if body:
                parts.append(f"Response body: {body[:2000]}")
        except Exception:
            pass
        try:
            status = resp.status_code if hasattr(resp, "status_code") else ""
            parts.append(f"Response status: {status}")
        except Exception:
            pass
    for attr in ("llm_provider", "model", "status_code", "body", "litellm_debug_info"):
        val = getattr(exc, attr, None)
        if val is not None:
            parts.append(f"{attr}: {val}")
    return " | ".join(parts)


def log_llm_error(msg: str, exc: Exception):
    """Log error with full details + traceback via print() (bypasses uvicorn logging suppression)."""
    detail = extract_litellm_error(exc)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(f"ERROR app.services.llm_service: {msg}: {detail}\n{tb}", flush=True)
    logger.error("%s: %s", msg, detail, exc_info=True)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session
from app.models import LLMProvider
from app.models.settings_model import LLM_KEY_CLASSIFIER_MODEL, LLM_KEY_CLASSIFIER_PROVIDER

# Static extra headers injected per provider on every call.
_PROVIDER_EXTRA_HEADERS: Dict[str, Dict[str, str]] = {
    "openrouter": {"HTTP-Referer": "https://github.com/syberx/AI-Paperless-Organizer"},
}


async def _resolve_provider_credentials(provider: str) -> Dict[str, Any]:
    """Look up api_key, api_base, and extra_headers for a provider from llm_providers table."""
    async with async_session() as db:
        result = await db.execute(select(LLMProvider).where(LLMProvider.name == provider))
        llm = result.scalar_one_or_none()

    creds: Dict[str, Any] = {}
    if llm:
        if llm.api_key:
            creds["api_key"] = llm.api_key
        if llm.api_base_url:
            api_base = llm.api_base_url.rstrip("/")
            if provider in ("lm_studio", "vllm") and not api_base.endswith("/v1"):
                api_base = f"{api_base}/v1"
            creds["api_base"] = api_base
    if provider in _PROVIDER_EXTRA_HEADERS:
        creds["extra_headers"] = _PROVIDER_EXTRA_HEADERS[provider]
    return creds


def build_ollama_params(
    temperature: float = 0.0,
    top_p: float = 0.1,
    num_ctx: int = 16384,
    num_predict: Optional[int] = None,
    keep_alive: Optional[str] = None,
    think: Optional[bool] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    json_output: bool = False,
    seed: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the extra_body dict for Ollama calls via LiteLLM (private — use via llm_completion).

    Per D-01: model params go in extra_body['options'].
    'keep_alive' and 'think' are top-level keys.
    'format' carries a JSON schema or the string "json" for structured output.
    """
    options: Dict[str, Any] = {"temperature": temperature, "top_p": top_p, "num_ctx": num_ctx}
    if num_predict is not None:
        options["num_predict"] = num_predict
    if seed is not None:
        options["seed"] = seed
    if repeat_penalty is not None:
        options["repeat_penalty"] = repeat_penalty

    body: Dict[str, Any] = {"options": options}
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    if think is not None:
        body["think"] = think
    if json_schema is not None:
        body["format"] = json_schema
    elif json_output:
        body["format"] = "json"
    return body


async def llm_completion(
    model: str,
    messages: List[Dict[str, Any]],
    provider: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 0.1,
    stream: bool = False,
    keep_alive: Optional[str] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    json_output: bool = False,
    num_ctx: Optional[int] = None,
    num_predict: Optional[int] = None,
    seed: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
    think: Optional[bool] = None,
    **kwargs,
):
    """Wrapper around litellm.acompletion.

    For provider=\"ollama\" calls, applies sensible defaults internally:
      num_ctx=16384, json_output=True, keep_alive as specified (or None).
    Callers only need to override what differs from the default.
    """
    if provider and not model.startswith(f"{provider}/"):
        model = f"{provider}/{model}"

    litellm_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
        **kwargs,
    }

    if provider == "ollama" and "extra_body" not in kwargs:
        litellm_kwargs["extra_body"] = build_ollama_params(
            temperature=temperature,
            top_p=top_p,
            num_ctx=num_ctx or 16384,
            num_predict=num_predict,
            keep_alive=keep_alive,
            think=think,
            json_schema=json_schema,
            json_output=json_output,
            seed=seed,
            repeat_penalty=repeat_penalty,
        )

    if provider:
        for k, v in (await _resolve_provider_credentials(provider)).items():
            litellm_kwargs.setdefault(k, v)
    try:
        return await litellm.acompletion(**litellm_kwargs)
    except Exception as e:
        log_llm_error(f"llm_completion failed (model={model}, provider={provider})", e)
        raise


async def llm_embedding(
    model: str,
    input: List[str],
    provider: Optional[str] = None,
    **kwargs,
) -> List[List[float]]:
    """Wrapper around litellm.aembedding. Resolves credentials from llm_providers table."""
    if provider and not model.startswith(f"{provider}/"):
        model = f"{provider}/{model}"
    litellm_kwargs: Dict[str, Any] = {
        "model": model,
        "input": input,
        **kwargs,
    }
    if provider:
        for k, v in (await _resolve_provider_credentials(provider)).items():
            litellm_kwargs.setdefault(k, v)
    try:
        response = await litellm.aembedding(**litellm_kwargs)
        return [item.embedding for item in response.data]
    except Exception as e:
        log_llm_error(f"llm_embedding failed (model={model}, provider={provider})", e)
        raise


def _derive_openai_compatible_url(base_url: str, provider: str) -> str:
    """Derive the OpenAI-compatible /models endpoint URL for a provider.
    
    Ollama:     http://host:11434 → http://host:11434/api/tags
    LM Studio:  http://host:1234  → http://host:1234/v1/models
    vLLM:       http://host:8000  → http://host:8000/v1/models
    Other:      use as-is ( LiteLLM handles standard OpenAI-compatible endpoints)
    """
    base = base_url.rstrip("/")
    if provider == "ollama":
        return f"{base}/api/tags"
    elif provider in ("lm_studio", "vllm"):
        return f"{base}/v1/models"
    return f"{base}/v1/models"


async def list_llm_models(provider: str, db: Optional[AsyncSession] = None) -> List[Dict[str, str]]:
    """List available models for a provider.
    
    1. Look up provider config (base_url, api_key) from DB if db session provided.
    2. For local providers (ollama, lm_studio, vllm): query the /models endpoint
       using the derived OpenAI-compatible URL.
    3. Fall back to litellm.model_list if the API call fails or no db session.
    """
    db_provider = None
    if db:
        result = await db.execute(select(LLMProvider).where(LLMProvider.name == provider))
        db_provider = result.scalar_one_or_none()

    api_base = None
    api_key = None
    if db_provider:
        api_base = db_provider.api_base_url
        api_key = db_provider.api_key

    # Try live fetch from the provider's API if base_url is configured
    if api_base:
        try:
            model_url = _derive_openai_compatible_url(api_base, provider)
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(model_url, headers=headers if headers else None)
                response.raise_for_status()
                data = response.json()

            # Normalize different response formats
            if provider == "ollama":
                models_data = data.get("models", [])
            else:
                # OpenAI-compatible /v1/models format
                models_data = data.get("data", data.get("models", []))

            return [
                {
                    "id": m.get("name") or m.get("id"),
                    "name": m.get("name") or m.get("id"),
                    "display_name": _format_model_display_name(m.get("name") or m.get("id", "")),
                }
                for m in models_data
            ]
        except Exception as e:
            logger.info("Failed to fetch live models from %s for %s, falling back: %s", api_base, provider, e)

    # Fall back to LiteLLM registry
    return _list_models_from_litellm(provider)


def _list_models_from_litellm(provider: str) -> List[Dict[str, str]]:
    """Get models from LiteLLM registry using models_by_provider."""
    if provider == "ollama":
        return []  # Ollama not tracked in models_by_provider

    provider_models = litellm.models_by_provider.get(provider, set())
    return [
        {
            "id": model_name,
            "name": model_name,
            "display_name": _format_model_display_name(model_name),
        }
        for model_name in sorted(provider_models)
    ]


def _format_model_display_name(model_name: str) -> str:
    """Format model name for display in dropdown."""
    name = model_name.replace("-", " ").replace("_", " ")
    return " ".join(word.capitalize() for word in name.split()) if name else model_name


PROVIDER_DISPLAY_NAMES = {
    "a2a": "A2A",
    "a2a_agent": "A2A Agent",
    "ai21": "AI21 Labs",
    "bedrock": "AWS Bedrock",
    "sagemaker": "AWS SageMaker",
    "ai21_chat": "AI21 Chat",
    "aiml": "AIML",
    "aiohttp_openai": "AIOHTTP OpenAI",
    "amazon_nova": "Amazon Nova",
    "anthropic": "Anthropic Claude",
    "anthropic_text": "Anthropic Text",
    "apertis": "Apertis",
    "assemblyai": "AssemblyAI",
    "auto_router": "Auto Router",
    "aws_polly": "AWS Polly",
    "azure": "Azure OpenAI",
    "azure_ai": "Azure AI",
    "azure_text": "Azure Text",
    "baseten": "Baseten",
    "bedrock_mantle": "Bedrock Mantle",
    "black_forest_labs": "Black Forest Labs",
    "bytez": "Bytez",
    "cerebras": "Cerebras",
    "charity_engine": "Charity Engine",
    "chatgpt": "ChatGPT",
    "chutes": "Chutes",
    "clarifai": "Clarifai",
    "cloudflare": "Cloudflare Workers AI",
    "codestral": "Codestral",
    "cohere": "Cohere",
    "cohere_chat": "Cohere Chat",
    "cometapi": "CometAPI",
    "compactifai": "CompactifAI",
    "cursor": "Cursor",
    "custom": "Custom",
    "custom_openai": "Custom OpenAI",
    "dashscope": "DashScope",
    "databricks": "Databricks",
    "datarobot": "DataRobot",
    "deepseek": "DeepSeek",
    "deepgram": "Deepgram",
    "deepinfra": "DeepInfra",
    "docker_model_runner": "Docker Model Runner",
    "dotprompt": "Dotprompt",
    "elevenlabs": "ElevenLabs",
    "empower": "Empower",
    "fal_ai": "fal.ai",
    "featherless_ai": "Featherless AI",
    "fireworks_ai": "Fireworks AI",
    "friendliai": "FriendliAI",
    "galadriel": "Galadriel",
    "gigachat": "GigaChat",
    "github": "GitHub",
    "github_copilot": "GitHub Copilot",
    "gemini": "Google Gemini",
    "vertex_ai": "Google Vertex AI",
    "gradient_ai": "Gradient AI",
    "groq": "Groq",
    "helicone": "Helicone",
    "heroku": "Heroku",
    "hosted_vllm": "Hosted vLLM",
    "huggingface": "Hugging Face",
    "humanloop": "Humanloop",
    "hyperbolic": "Hyperbolic",
    "infinity": "Infinity",
    "jina_ai": "Jina AI",
    "lambda_ai": "Lambda AI",
    "langfuse": "Langfuse",
    "langgraph": "LangGraph",
    "lemonade": "Lemonade",
    "litellm_agent": "LiteLLM Agent",
    "litellm_proxy": "LiteLLM Proxy",
    "llamafile": "Llamafile",
    "lm_studio": "LM Studio",
    "manus": "Manus",
    "maritalk": "MariTalk",
    "meta_llama": "Meta Llama",
    "milvus": "Milvus",
    "minimax": "MiniMax",
    "mistral": "Mistral AI",
    "moonshot": "Moonshot",
    "morph": "Morph",
    "nano-gpt": "NanoGPT",
    "nebius": "Nebius AI",
    "nlp_cloud": "NLP Cloud",
    "novita": "Novita AI",
    "nscale": "Nscale",
    "nvidia_nim": "NVIDIA NIM",
    "oci": "Oracle Cloud Infrastructure",
    "ollama": "Ollama (Local)",
    "ollama_chat": "Ollama Chat",
    "oobabooga": "Oobabooga",
    "openai": "OpenAI",
    "openrouter": "OpenRouter",
    "openai_like": "OpenAI-Compatible API",
    "ovhcloud": "OVHcloud",
    "perplexity": "Perplexity",
    "petals": "Petals",
    "pg_vector": "pgvector",
    "poe": "Poe",
    "predibase": "Predibase",
    "publicai": "PublicAI",
    "ragflow": "RAGFlow",
    "recraft": "Recraft",
    "replicate": "Replicate",
    "runwayml": "RunwayML",
    "s3_vectors": "S3 Vectors",
    "sagemaker_chat": "SageMaker Chat",
    "sagemaker_nova": "SageMaker Nova",
    "sambanova": "SambaNova",
    "sap": "SAP",
    "snowflake": "Snowflake",
    "stability": "Stability AI",
    "synthetic": "Synthetic",
    "text-completion-codestral": "Text Completion (Codestral)",
    "text-completion-openai": "Text Completion (OpenAI)",
    "together_ai": "Together AI",
    "topaz": "Topaz",
    "triton": "Triton",
    "v0": "v0",
    "vercel_ai_gateway": "Vercel AI Gateway",
    "vertex_ai_beta": "Vertex AI (Beta)",
    "volcengine": "Volcengine",
    "voyage": "Voyage AI",
    "wandb": "Weights & Biases (W&B)",
    "watsonx": "watsonx",
    "watsonx_text": "watsonx Text",
    "xiaomi_mimo": "Xiaomi Mimo",
    "xinference": "Xinference",
    "zai": "Zai",
    "vllm": "vLLM",
    "xai": "xAI",
}


def list_llm_providers() -> List[Dict[str, str]]:
    """List all available LLM providers from LiteLLM registry."""
    try:
        providers = []
        for provider_enum in litellm.provider_list:
            provider_name = provider_enum.value
            providers.append({
                "name": provider_name,
                "display_name": PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name.title()),
            })
        providers.sort(key=lambda x: x["display_name"])
        return providers
    except Exception:
        return []


async def get_setting(key: str, db: AsyncSession) -> Optional[str]:
    """Get a setting value from the key-value store."""
    from app.routers.settings import get_setting as gs
    return await gs(key, db)


class LitellmService:
    """Service for interacting with various LLM providers via LiteLLM."""

    def __init__(self, provider: Optional[LLMProvider] = None, model: Optional[str] = None, session_factory: Optional[Any] = None):
        self.provider = provider
        self.model = model
        self.session_factory = session_factory
        self._config_loaded = False

    async def _ensure_config(self) -> None:
        """Lazy-load provider/model from DB on first use."""
        if self._config_loaded or self.provider is not None or self.session_factory is None:
            return
        from app.models.settings_model import LLM_KEY_CLASSIFIER_PROVIDER, LLM_KEY_CLASSIFIER_MODEL
        async with self.session_factory() as db:
            provider_name = await get_setting(LLM_KEY_CLASSIFIER_PROVIDER, db)
            if provider_name:
                result = await db.execute(
                    select(LLMProvider).where(LLMProvider.name == provider_name)
                )
                self.provider = result.scalars().first()
            if not self.provider:
                result = await db.execute(select(LLMProvider).limit(1))
                self.provider = result.scalars().first()
            self.model = await get_setting(LLM_KEY_CLASSIFIER_MODEL, db) or self.model
        self._config_loaded = True

    async def complete(self, prompt: str, model_override: Optional[str] = None) -> str:
        """Send a completion request to the LLM provider via LiteLLM.

        LiteLLM handles provider routing internally based on model name prefix
        (e.g., 'anthropic/claude-3-5-sonnet', 'ollama/llama3').
        """
        await self._ensure_config()
        model = model_override or self.model
        if not model:
            raise ValueError("No model specified")

        response = await llm_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            provider=self.provider.name if self.provider else None,
            temperature=0.0,
            top_p=0.1,
        )
        return (response.choices[0].message.content or "").strip()

    async def test_connection(self) -> Dict:
        """Test connection to the LLM provider."""
        await self._ensure_config()
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



