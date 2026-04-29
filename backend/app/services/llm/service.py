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


class LitellmService:
    """Service for interacting with various LLM providers via LiteLLM."""

    _LOCAL_LLM_PROVIDERS = frozenset({
        "ollama", "ollama_chat", "lm_studio", "lm_studio_chat",
        "vllm", "llama.cpp", "llama-cpp", "local"
    })

    DEFAULT_CONTEXT_WINDOW: int = 32000

    _PROVIDER_EXTRA_HEADERS: Dict[str, Dict[str, str]] = {
        "openrouter": {"HTTP-Referer": "https://github.com/syberx/AI-Paperless-Organizer"},
    }

    def __init__(self):
        _register_litellm_callbacks()
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # ── Public API ─────────────────────────────────────────────────────────────

    async def complete(
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

    async def stream(
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
        """List available models for a provider. Queries the provider's API or falls back to LiteLLM registry."""
        async with async_session() as db:
            result = await db.execute(
                select(LLMProvider).where(LLMProvider.name == provider)
            )
            db_provider = result.scalar_one_or_none()

        api_base = db_provider.api_base_url if db_provider else None
        api_key = db_provider.api_key if db_provider else None

        if api_base:
            try:
                model_url = self._derive_openai_compatible_url(api_base, provider)
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
                        "display_name": self._format_model_display_name(
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
                "display_name": self._format_model_display_name(name),
            }
            for name in sorted(provider_models)
        ]

    async def test_connection(self, provider: str, model: str) -> dict:
        """Test connection to the LLM provider."""
        result = await self.complete(
            provider=provider,
            model=model,
            messages=[{"role": "user", "content": "Antworte nur mit: OK"}],
        )
        return {
            "provider": provider,
            "model": model,
            "response": (result.content or "").strip(),
        }

    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Estimate token count using LiteLLM's token counter."""
        model_for_count = model or "gpt-4"
        try:
            return litellm.token_counter(model=model_for_count, text=text)
        except Exception:
            return len(text) // 4

    def get_lock_status(self) -> dict[str, dict]:
        """Get current lock status for all local LLM providers."""
        status = {}
        for prov, lock in self._locks.items():
            status[prov] = {"locked": lock.locked()}
        return status

    def is_local_provider(self, provider: str) -> bool:
        if not provider:
            return False
        provider_lower = provider.lower().split("/")[0]
        return provider_lower in self._LOCAL_LLM_PROVIDERS

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _resolve_credentials(self, provider: str) -> dict:
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
        if provider in self._PROVIDER_EXTRA_HEADERS:
            creds["extra_headers"] = self._PROVIDER_EXTRA_HEADERS[provider]
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
        for k, v in (await self._resolve_credentials(provider)).items():
            litellm_kwargs.setdefault(k, v)
        return litellm_kwargs

    async def _execute_completion(self, provider: str, litellm_kwargs: dict):
        timeout = litellm_kwargs.get("timeout", 30.0)
        if not self.is_local_provider(provider):
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

    @staticmethod
    def _derive_openai_compatible_url(base_url: str, provider: str) -> str:
        base = base_url.rstrip("/")
        if provider == "ollama":
            return f"{base}/api/tags"
        elif provider in ("lm_studio", "vllm"):
            return f"{base}/v1/models"
        return f"{base}/v1/models"

    @staticmethod
    def _format_model_display_name(model_name: str) -> str:
        name = model_name.replace("-", " ").replace("_", " ")
        return " ".join(word.capitalize() for word in name.split()) if name else model_name

    @staticmethod
    def _extract_litellm_error(exc: Exception) -> str:
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
        detail = LitellmService._extract_litellm_error(exc)
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(f"ERROR app.services.llm.service: {msg}: {detail}\n{tb}", flush=True)
        logger.error("%s: %s", msg, detail, exc_info=True)

    async def check_provider_health(self, provider: str) -> bool:
        creds = await self._resolve_credentials(provider)
        url = creds.get("api_base")
        if not url:
            return False
        endpoint = self._derive_openai_compatible_url(url, provider)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(endpoint)
                return r.status_code == 200
        except Exception:
            return False

    async def unload_local_model(self, provider: str, model: str) -> bool:
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
                    return False
            return True
        except Exception:
            return False

    async def get_token_limit(self, provider: str, model: str) -> int:
        """Get the context window (max input tokens) for a specific model.
        
        Resolution order:
        1. LiteLLM model_info database (for known OpenAI/Anthropic/etc. models)
        2. Provider-specific API (Ollama /api/show, LM Studio model info)
        3. DEFAULT_CONTEXT_WINDOW (32000) as fallback
        """
        context_length: Optional[int] = None
        max_context_length: Optional[int] = None

        # 1. Try LiteLLM's model_info database first
        try:
            model_name = model if "/" in model else f"{provider}/{model}"
            model_info: Any = litellm.get_model_info(model=model_name)
            if model_info:
                max_tokens: Optional[int] = model_info.get("max_input_tokens")
                if max_tokens:
                    return max_tokens
        except Exception as e:
            logger.debug(f"LiteLLM model_info failed for {model}: {e}")

        # 2. Try provider-specific API (Ollama, LM Studio)
        try:
            context_length, max_context_length = await self._get_model_info(provider, model)
        except Exception as e:
            logger.debug(f"Provider-specific model_info failed for {model}: {e}")

        # 3. If still no result, use default
        if not context_length:
            return self.DEFAULT_CONTEXT_WINDOW

        return max_context_length or context_length


    async def _get_model_info(
        self,
        provider: str,
        model_name: str,
    ) -> tuple[Optional[int], Optional[int]]:
        """Fetch model info including context length from the provider API.
        
        Returns (context_length, max_context_length).
        """
        creds = await self._resolve_credentials(provider)
        base_url = creds.get("api_base")
        api_key = creds.get("api_key")
        
        if not base_url:
            return None, None

        base_url = base_url.rstrip("/")

        async with httpx.AsyncClient() as client:
            try:
                if provider in ("ollama", "ollama_chat"):
                    target_url = base_url[:-3] if base_url.endswith("/v1") else base_url
                    try:
                        resp = await client.post(
                            f"{target_url}/api/show",
                            json={"name": model_name},
                            timeout=10.0,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        model_info = data.get("model_info", {})
                        details = data.get("details", {})

                        ctx: Optional[int] = None
                        for key in model_info.keys():
                            if key.endswith(".context_length"):
                                ctx = model_info[key]
                                break
                        if not ctx:
                            ctx = details.get("context_length")

                        if ctx:
                            return ctx, ctx
                    except Exception:
                        pass

                elif provider in ("lm_studio", "lm_studio_chat"):
                    native_url = base_url.replace("/v1", "")
                    _, loaded_ctx, max_ctx = await self._get_lm_studio_model_info(
                        client, native_url, model_name, api_key
                    )
                    if loaded_ctx:
                        return loaded_ctx, max_ctx

                    loaded_ctx = await self._load_lm_studio_model(
                        client, native_url, model_name, api_key
                    )
                    if loaded_ctx:
                        return loaded_ctx, max_ctx

                    _, loaded_ctx, max_ctx = await self._get_lm_studio_model_info(
                        client, native_url, model_name, api_key
                    )
                    return loaded_ctx, max_ctx

                return None, None

            except Exception as e:
                logger.debug(f"Error fetching model info for {provider}: {e}")
                return None, None


    async def _get_lm_studio_model_info(
        self,
        client: httpx.AsyncClient,
        native_url: str,
        model_name: str,
        api_key: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[int], Optional[int]]:
        """Fetch model info including context length from LM Studio."""
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        max_ctx = None
        try:
            resp = await client.get(f"{native_url}/api/v1/models", headers=headers, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                for model in data.get("models", []):
                    if model.get("key") == model_name:
                        max_ctx = model.get("max_context_length")
                        for instance in model.get("loaded_instances", []):
                            loaded_ctx: Optional[int] = instance.get("config", {}).get(
                                "context_length"
                            )
                            if loaded_ctx:
                                return instance.get("id"), loaded_ctx, max_ctx

        except Exception:
            pass

        return None, None, max_ctx


    async def _load_lm_studio_model(
        self,
        client: httpx.AsyncClient,
        native_url: str,
        model_name: str,
        api_key: Optional[str] = None,
    ) -> Optional[int]:
        """Load a model in LM Studio and return its context length."""
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            resp = await client.post(
                f"{native_url}/api/v1/models/load",
                headers=headers,
                json={"model": model_name},
                timeout=60.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("context_length")
        except Exception:
            pass
        return None

