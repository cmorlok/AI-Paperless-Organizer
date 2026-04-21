import asyncio
import logging
from typing import List

from app.services.llm_service import llm_embedding

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates text embeddings via any LiteLLM-supported provider."""

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    # Most Ollama embedding models (mxbai-embed-large, nomic-embed-text) cap at 512 tokens.
    # 512 tokens * ~1.5 chars/token for German OCR text ≈ 768 chars. Use 750 to be safe.
    _MAX_EMBED_CHARS = 750
    _EMBED_RETRY_SLEEPS = [5, 15, 30]  # seconds between retries on transient failures

    async def generate(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        batch_size = 50
        max_attempts = len(self._EMBED_RETRY_SLEEPS) + 1

        for i in range(0, len(texts), batch_size):
            batch = [t[:self._MAX_EMBED_CHARS] if len(t) > self._MAX_EMBED_CHARS else t
                     for t in texts[i:i + batch_size]]

            for attempt in range(max_attempts):
                try:
                    embeddings = await llm_embedding(
                        model=self.model,
                        provider=self.provider,
                        input=batch,
                    )
                    all_embeddings.extend(embeddings)
                    break
                except Exception as e:
                    if attempt < max_attempts - 1:
                        sleep = self._EMBED_RETRY_SLEEPS[attempt]
                        logger.warning(f"Embedding batch {i} attempt {attempt+1} failed, retry in {sleep}s: {e}")
                        await asyncio.sleep(sleep)
                    else:
                        logger.error(f"Embedding batch {i} failed after {max_attempts} attempts: {e}")
                        raise

        return all_embeddings
