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
from app.services.llm.lock import (
    acquire,
    release,
    is_locked,
    current_holder,
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
    "acquire",
    "release",
    "is_locked",
    "current_holder",
]
