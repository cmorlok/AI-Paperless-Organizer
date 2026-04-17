import asyncio
import logging
import httpx
from typing import List

from app.services.llm_service import llm_embedding

logger = logging.getLogger(__name__)


EMBEDDING_DIMS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    # German-optimised models (recommended upgrade)
    "jina/jina-embeddings-v2-base-de": 768,   # ollama pull jina/jina-embeddings-v2-base-de
    "jina-embeddings-v2-base-de": 768,
    "bge-m3": 1024,                            # ollama pull bge-m3
}
DEFAULT_DIM = 768


class EmbeddingService:
    """Generates text embeddings via any LiteLLM-supported provider."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "mxbai-embed-large",
        api_base: str = "http://localhost:11434",
        api_key: str = "",
    ):
        self.provider = provider
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.dim = EMBEDDING_DIMS.get(model, DEFAULT_DIM)

    # Most Ollama embedding models (mxbai-embed-large, nomic-embed-text) cap at 512 tokens.
    # 512 tokens * ~1.5 chars/token for German OCR text ≈ 768 chars. Use 750 to be safe.
    _MAX_EMBED_CHARS = 750
    _EMBED_RETRY_SLEEPS = [5, 15, 30]  # seconds between retries on transient failures

    def _litellm_model(self) -> str:
        """Returns the LiteLLM-formatted model string for this provider."""
        if self.provider == "ollama" and not self.model.startswith("ollama/"):
            return f"ollama/{self.model}"
        return self.model

    async def generate(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        batch_size = 50
        model = self._litellm_model()
        max_attempts = len(self._EMBED_RETRY_SLEEPS) + 1

        for i in range(0, len(texts), batch_size):
            batch = [t[:self._MAX_EMBED_CHARS] if len(t) > self._MAX_EMBED_CHARS else t
                     for t in texts[i:i + batch_size]]

            for attempt in range(max_attempts):
                try:
                    embeddings = await llm_embedding(
                        model=model,
                        input=batch,
                        api_base=self.api_base or None,
                        api_key=self.api_key or None,
                    )
                    if len(embeddings) == len(batch):
                        all_embeddings.extend(embeddings)
                    else:
                        logger.error(f"Batch {i}: expected {len(batch)} embeddings, got {len(embeddings)}")
                        all_embeddings.extend(embeddings)
                        all_embeddings.extend([[0.0] * self.dim] * (len(batch) - len(embeddings)))
                    break
                except Exception as e:
                    if attempt < max_attempts - 1:
                        sleep = self._EMBED_RETRY_SLEEPS[attempt]
                        logger.warning(f"Embedding batch {i} attempt {attempt+1} failed (provider busy?), retry in {sleep}s: {e}")
                        await asyncio.sleep(sleep)
                    else:
                        logger.error(f"Embedding batch {i} failed after {max_attempts} attempts: {e}")
                        all_embeddings.extend([[0.0] * self.dim] * len(batch))

        return all_embeddings

    async def check_health(self) -> dict:
        if self.provider == "ollama":
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(f"{self.api_base}/api/tags")
                    resp.raise_for_status()
                    models = [m["name"] for m in resp.json().get("models", [])]
                    model_available = any(self.model in m for m in models)
                    return {"healthy": True, "model_available": model_available, "models": models}
            except Exception as e:
                return {"healthy": False, "error": str(e)}
        return {"healthy": True, "provider": self.provider}
