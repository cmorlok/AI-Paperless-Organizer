"""Protocol for LLM service (LiteLLM-based)."""
from __future__ import annotations
from typing import Protocol, runtime_checkable, Any, Optional, Dict, AsyncGenerator

from app.services.llm.types import LLMResponse


@runtime_checkable
class LLMService(Protocol):
    """Protocol for LLM service (LiteLLM-based)."""

    @property
    def provider(self) -> Any: ...

    @property
    def model(self) -> Optional[str]: ...

    async def complete(self, prompt: str, model_override: Optional[str] = None) -> str: ...

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
    ) -> LLMResponse: ...

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
    ) -> AsyncGenerator[str, None]: ...

    async def embed(
        self,
        provider: str,
        model: str,
        input: list[str],
        *,
        timeout: float = 30.0,
        **kwargs,
    ) -> list[list[float]]: ...

    async def test_connection(self, provider: str = None, model: str = None) -> Dict[str, Any]: ...

    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int: ...

    def get_lock_status(self) -> Dict[str, Dict[str, bool]]: ...

    def list_providers(self) -> list[dict]: ...

    async def list_models(self, provider: str) -> list[dict]: ...
