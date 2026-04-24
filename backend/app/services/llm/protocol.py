"""Protocol for LLM service (LiteLLM-based)."""

from typing import Protocol, runtime_checkable, Any, Optional, Dict


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
        model: str,
        messages: list,
        *,
        timeout: float = 30.0,
        stream: bool = False,
        temperature: float = 0.0,
        provider: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs
    ) -> Any: ...

    async def test_connection(self) -> Dict[str, Any]: ...

    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int: ...

    def get_lock_status(self) -> Dict[str, Dict[str, bool]]: ...