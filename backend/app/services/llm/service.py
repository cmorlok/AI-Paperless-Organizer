"""Unified LLM service — uses LiteLLM for all providers."""

from __future__ import annotations

import asyncio
import json
import re
import traceback
from collections import defaultdict
from typing import Optional, Dict, Any, List, AsyncGenerator

import httpx
import litellm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.database import async_session
from app.models import LLMProvider

# Local LLM providers that need GPU lock serialization
LOCAL_LLM_PROVIDERS = frozenset({
    "ollama", "ollama_chat", "lm_studio", "lm_studio_chat",
    "vllm", "llama.cpp", "llama-cpp", "local"
})

# Static extra headers injected per provider on every call.
_PROVIDER_EXTRA_HEADERS: Dict[str, Dict[str, str]] = {
    "openrouter": {"HTTP-Referer": "https://github.com/syberx/AI-Paperless-Organizer"},
}


class LLMLockTimeoutError(Exception):
    """Raised when a local LLM is busy and times out waiting for the lock."""
    pass


logger = get_logger("llm")

# =========================================================================
# LiteLLM callback-based request/response logging
# =========================================================================


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
    from app.services.llm.state import _callbacks_registered
    if _callbacks_registered:
        return
    # Mark as registered before setting the flag to avoid races
    import app.services.llm.state as llm_state
    llm_state._callbacks_registered = True
    litellm.logging_callback_manager.add_litellm_input_callback(_llm_input_callback)
    litellm.logging_callback_manager.add_litellm_success_callback(_llm_success_callback)
    litellm.logging_callback_manager.add_litellm_failure_callback(_llm_failure_callback)


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


def _format_model_display_name(model_name: str) -> str:
    """Format model name for display in dropdown."""
    name = model_name.replace("-", " ").replace("_", " ")
    return " ".join(word.capitalize() for word in name.split()) if name else model_name


PROVIDER_DISPLAY_NAMES: Dict[str, str] = {
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


async def get_setting(key: str, db: AsyncSession) -> Optional[str]:
    """Get a setting value from the key-value store."""
    from app.routers.settings import get_setting as gs
    return await gs(key, db)


class LitellmService:
    """Service for interacting with various LLM providers via LiteLLM."""

    def __init__(self, provider: Optional[LLMProvider] = None, model: Optional[str] = None, session_factory: Optional[Any] = None):
        # Register LiteLLM callbacks on construction (idempotent)
        _register_litellm_callbacks()
        self.provider = provider
        self.model = model
        self.session_factory = session_factory
        self._config_loaded = False
        # Per-provider lock dict for local LLM serialization
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _is_local_provider(self, provider: str) -> bool:
        """Check if provider is a local LLM (needs GPU lock serialization)."""
        if not provider:
            return False
        provider_lower = provider.lower().split("/")[0]
        return provider_lower in LOCAL_LLM_PROVIDERS

    @staticmethod
    def _extract_litellm_error(exc: Exception) -> str:
        """Extract full error details from a litellm exception."""
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

    @staticmethod
    def _log_llm_error(msg: str, exc: Exception) -> None:
        """Log error with full details + traceback via print() (bypasses uvicorn log suppression)."""
        detail = LitellmService._extract_litellm_error(exc)
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(f"ERROR app.services.llm.service: {msg}: {detail}\n{tb}", flush=True)
        logger.error("%s: %s", msg, detail, exc_info=True)

    async def _resolve_credentials(self, provider: str) -> dict:
        """Look up api_key, api_base, and extra_headers for a provider from llm_providers table."""
        async with async_session() as db:
            result = await db.execute(select(LLMProvider).where(LLMProvider.name == provider))
            llm = result.scalar_one_or_none()
        creds: dict = {}
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

    def _build_ollama_extra_body(
        self,
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
        """Build the extra_body dict for Ollama calls via LiteLLM.

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

    async def _prepare_kwargs(
        self,
        provider: str,
        model: str,
        messages: list,
        *,
        temperature: float = 0.0,
        top_p: float = 0.1,
        timeout: float = 30.0,
        tools: list | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        keep_alive: str | None = None,
        think: bool | None = None,
        seed: int | None = None,
        repeat_penalty: float | None = None,
        json_output: bool = False,
        json_schema: dict | None = None,
        stream: bool = False,
        **kwargs,
    ) -> dict:
        """Build complete litellm.acompletion kwargs dict."""
        model_name = model if "/" in model else f"{provider}/{model}"
        litellm_kwargs: dict = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "timeout": timeout,
            "stream": stream,
            **kwargs,
        }
        if tools:
            litellm_kwargs["tools"] = tools
        # Ollama-specific extra_body (only when provider is ollama variant)
        if provider in ("ollama", "ollama_chat") and "extra_body" not in kwargs:
            litellm_kwargs["extra_body"] = self._build_ollama_extra_body(
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
        # Merge provider credentials (api_key, api_base, extra_headers)
        for k, v in (await self._resolve_credentials(provider)).items():
            litellm_kwargs.setdefault(k, v)
        return litellm_kwargs

    async def _execute_completion(self, provider: str, litellm_kwargs: dict):
        """Acquire per-provider GPU lock for local providers, then call litellm.acompletion."""
        timeout = litellm_kwargs.get("timeout", 30.0)
        if not self._is_local_provider(provider):
            try:
                return await litellm.acompletion(**litellm_kwargs)
            except Exception as e:
                LitellmService._log_llm_error(
                    f"_execute_completion failed (provider={provider})", e)
                raise
        lock = self._locks[provider]
        try:
            async with asyncio.timeout(timeout):
                async with lock:
                    try:
                        return await litellm.acompletion(**litellm_kwargs)
                    except Exception as e:
                        LitellmService._log_llm_error(
                            f"_execute_completion failed (provider={provider})", e)
                        raise
        except asyncio.TimeoutError:
            raise LLMLockTimeoutError(
                f"Local LLM {provider} busy (held by another background job). "
                "Try again in a moment."
            )

    def _build_response(self, raw) -> "LLMResponse":
        """Convert raw litellm response into LLMResponse."""
        from app.services.llm.types import LLMResponse, ToolCall
        choice = raw.choices[0]
        usage = raw.usage
        input_tokens = (usage.prompt_tokens or 0) if usage else 0
        output_tokens = (usage.completion_tokens or 0) if usage else 0
        finish_reason = choice.finish_reason

        if finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_calls = [
                ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                for tc in choice.message.tool_calls
            ]
            raw_msg = choice.message.model_dump(exclude_none=True)
            clean_msg: dict = {"role": raw_msg["role"]}
            if raw_msg.get("content"):
                clean_msg["content"] = raw_msg["content"]
            if raw_msg.get("tool_calls"):
                clean_msg["tool_calls"] = raw_msg["tool_calls"]
            return LLMResponse(
                content=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                assistant_message=clean_msg,
            )
        return LLMResponse(
            content=choice.message.content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def complete_llm(
        self,
        provider: str,
        model: str,
        messages: list,
        *,
        temperature: float = 0.0,
        top_p: float = 0.1,
        timeout: float = 30.0,
        tools: list | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        keep_alive: str | None = None,
        think: bool | None = None,
        seed: int | None = None,
        repeat_penalty: float | None = None,
        json_output: bool = False,
        json_schema: dict | None = None,
        **kwargs,
    ) -> "LLMResponse":
        """Call LLM and return LLMResponse. GPU lock is acquired transparently for local providers."""
        litellm_kwargs = await self._prepare_kwargs(
            provider, model, messages,
            temperature=temperature, top_p=top_p, timeout=timeout, tools=tools,
            num_ctx=num_ctx, num_predict=num_predict, keep_alive=keep_alive,
            think=think, seed=seed, repeat_penalty=repeat_penalty,
            json_output=json_output, json_schema=json_schema,
            stream=False,
            **kwargs,
        )
        raw = await self._execute_completion(provider, litellm_kwargs)
        return self._build_response(raw)

    async def stream_llm(
        self,
        provider: str,
        model: str,
        messages: list,
        *,
        temperature: float = 0.0,
        top_p: float = 0.1,
        timeout: float = 30.0,
        num_ctx: int | None = None,
        keep_alive: str | None = None,
        think: bool | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response as str chunks. GPU lock acquired transparently for local providers."""
        litellm_kwargs = await self._prepare_kwargs(
            provider, model, messages,
            temperature=temperature, top_p=top_p, timeout=timeout,
            num_ctx=num_ctx, keep_alive=keep_alive, think=think,
            stream=True,
            **kwargs,
        )
        raw_stream = await self._execute_completion(provider, litellm_kwargs)

        async def _generate():
            async for chunk in raw_stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                content = getattr(delta, "content", None) or ""
                finish = getattr(chunk.choices[0], "finish_reason", None) if chunk.choices else None
                if content:
                    yield content
                if finish is not None and finish != "length":
                    break

        return _generate()

    async def embed(
        self,
        provider: str,
        model: str,
        input: list[str],
        *,
        timeout: float = 30.0,
        **kwargs,
    ) -> list[list[float]]:
        """Generate embeddings via litellm.aembedding. No GPU lock needed."""
        model_name = model if "/" in model else f"{provider}/{model}"
        litellm_kwargs: dict = {"model": model_name, "input": input, "timeout": timeout, **kwargs}
        for k, v in (await self._resolve_credentials(provider)).items():
            litellm_kwargs.setdefault(k, v)
        try:
            response = await litellm.aembedding(**litellm_kwargs)
            return [item.embedding for item in response.data]
        except Exception as e:
            LitellmService._log_llm_error(f"embed failed (provider={provider}, model={model})", e)
            raise

    def list_providers(self) -> list[dict]:
        """List all LiteLLM-supported providers."""
        try:
            providers = []
            for provider_enum in litellm.provider_list:
                provider_name = provider_enum.value
                providers.append({
                    "name": provider_name,
                    "display_name": PROVIDER_DISPLAY_NAMES.get(
                        provider_name, provider_name.title()
                    ),
                })
            providers.sort(key=lambda x: x["display_name"])
            return providers
        except Exception:
            return []

    async def list_models(self, provider: str) -> list[dict]:
        """List available models for a provider. Uses own session_factory for DB lookup."""
        async with async_session() as db:
            result = await db.execute(
                select(LLMProvider).where(LLMProvider.name == provider)
            )
            db_provider = result.scalar_one_or_none()

        api_base = db_provider.api_base_url if db_provider else None
        api_key = db_provider.api_key if db_provider else None

        if api_base:
            try:
                model_url = _derive_openai_compatible_url(api_base, provider)
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        model_url, headers=headers if headers else None
                    )
                    response.raise_for_status()
                    data = response.json()
                if provider == "ollama":
                    models_data = data.get("models", [])
                else:
                    models_data = data.get("data", data.get("models", []))
                return [
                    {
                        "id": m.get("name") or m.get("id"),
                        "name": m.get("name") or m.get("id"),
                        "display_name": _format_model_display_name(
                            m.get("name") or m.get("id", "")
                        ),
                    }
                    for m in models_data
                ]
            except Exception as e:
                logger.info("Failed to fetch live models for %s, falling back: %s", provider, e)

        if provider == "ollama":
            return []
        provider_models = litellm.models_by_provider.get(provider, set())
        return [
            {
                "id": name,
                "name": name,
                "display_name": _format_model_display_name(name),
            }
            for name in sorted(provider_models)
        ]

    # ── Existing public methods (updated) ───────────────────────────────────────

    async def complete(self, prompt: str, model_override=None) -> str:
        """Send a completion request to the LLM provider via LiteLLM."""
        await self._ensure_config()
        model = model_override or self.model
        if not model:
            raise ValueError("No model specified")
        provider_name = self.provider.name if self.provider else None
        if not provider_name:
            raise ValueError("No LLM provider configured")
        result = await self.complete_llm(
            provider=provider_name,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=0.1,
        )
        return (result.content or "").strip()

    async def test_connection(self, provider: str = None, model: str = None) -> dict:
        """Test connection to the LLM provider."""
        test_provider = provider or (self.provider.name if self.provider else None)
        test_model = model or self.model
        if not test_provider:
            raise ValueError("No LLM provider configured")
        result = await self.complete_llm(
            provider=test_provider,
            model=test_model,
            messages=[{"role": "user", "content": "Antworte nur mit: OK"}],
        )
        return {
            "provider": test_provider,
            "model": test_model,
            "response": (result.content or "").strip(),
        }

    def get_lock_status(self) -> dict[str, dict]:
        """Get current lock status for all local LLM providers."""
        status = {}
        for prov, lock in self._locks.items():
            status[prov] = {"locked": lock.locked()}
        return status

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

    async def check_provider_health(self, provider: str) -> bool:
        """Returns True if the provider at its configured URL is reachable."""
        creds = await self._resolve_credentials(provider)
        url = creds.get("api_base")
        if not url:
            return False
        endpoint = _derive_openai_compatible_url(url, provider)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(endpoint)
                return r.status_code == 200
        except Exception:
            return False

    async def unload_local_model(self, provider: str, model: str) -> bool:
        """Unload a local LLM model from GPU memory."""
        creds = await self._resolve_credentials(provider)
        url = (creds.get("api_base") or "").rstrip("/")
        api_key = creds.get("api_key")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if provider in ("ollama", "ollama_chat"):
                    await client.post(f"{url}/api/chat",
                        json={"model": model, "messages": [], "keep_alive": 0}, headers=headers)
                elif provider in ("lm_studio", "lm_studio_chat"):
                    await client.post(f"{url}/api/v1/models/unload",
                        json={"instance_id": model}, headers=headers)
                elif provider in ("llama.cpp", "llama-cpp", "llamafile"):
                    await client.post(f"{url}/models/unload",
                        json={"model": model}, headers=headers)
                else:
                    return False  # vllm and "local" have no standard unload API
            return True
        except Exception:
            return False

    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Estimate token count using LiteLLM's token counter."""
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
        logger.info("[LLM] Sending request to LLM provider...")
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
