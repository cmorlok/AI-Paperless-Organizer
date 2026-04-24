"""LLM service: LiteLLM-based completions, embeddings."""

from app.services.llm.protocol import LLMService
from app.services.llm.service import (
    llm_completion,
    llm_embedding,
    list_llm_models,
    list_llm_providers,
    PROVIDER_DISPLAY_NAMES,
    LLMLockTimeoutError,
    LitellmService,
)

__all__ = [
    "LLMService",
    "llm_completion",
    "llm_embedding",
    "list_llm_models",
    "list_llm_providers",
    "PROVIDER_DISPLAY_NAMES",
    "LLMLockTimeoutError",
    "LitellmService",
]
