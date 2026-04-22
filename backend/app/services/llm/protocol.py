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

    async def test_connection(self) -> Dict[str, Any]: ...

    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int: ...