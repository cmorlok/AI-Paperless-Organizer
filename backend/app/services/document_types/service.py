"""Document types business-logic service — estimation, deletion, analysis persistence."""

from __future__ import annotations

from typing import Any, Dict

from app.services.paperless import PaperlessClient
from app.services.llm import LLMService
from app.services.config import ConfigService
from app.services.statistics import StatisticsService
from app.services.entity_analysis import (
    persist_analysis,
    estimate_entity_tokens,
    delete_empty_entities,
)


class DocumentTypesServiceImpl:
    """Business-logic operations for document types — called by the thin router."""

    def __init__(
        self,
        paperless_client: PaperlessClient,
        llm_service: LLMService,
        config_service: ConfigService,
        statistics_service: StatisticsService,
        session_factory: Any,
    ):
        self._client = paperless_client
        self._llm = llm_service
        self._config = config_service
        self._stats = statistics_service
        self._session_factory = session_factory

    async def estimate_document_types(self) -> Dict:
        """Estimate tokens needed for document type analysis."""
        doc_types = await self._client.get_document_types_with_counts()
        return await estimate_entity_tokens(
            items=doc_types,
            llm_service=self._llm,
            config_service=self._config,
            label="Dokumententypen",
        )

    async def save_similarity_analysis(self, result: Dict) -> None:
        """Persist a similarity analysis result for document types."""
        groups = result.get("groups", [])
        stats = result.get("stats", {})
        await persist_analysis(
            session_factory=self._session_factory,
            entity_type="document_types",
            analysis_type="similarity",
            groups=groups,
            stats=stats,
            items_count=stats.get("items_count", 0),
        )

    async def delete_empty_document_types(self) -> Dict:
        """Delete all document types with 0 documents — parallel with stats recording."""
        doc_types = await self._client.get_document_types_with_counts()
        return await delete_empty_entities(
            items=doc_types,
            delete_fn=self._client.delete_document_type,
            entity_type="document_types",
            stats_service=self._stats,
        )
