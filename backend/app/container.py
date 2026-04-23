from __future__ import annotations

from dishka import Provider, Scope, provide, make_async_container, AsyncContainer
from dishka.integrations.fastapi import setup_dishka
from typing import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database import async_session
from app.services.llm.protocol import LLMService
from app.services.paperless.protocol import PaperlessClient
from app.services.similarity.protocol import SimilarityService
from app.services.merge.protocol import MergeService
from app.services.statistics.protocol import StatisticsService
from app.services.ocr.protocol import OcrService
from app.services.ocr.state import OcrState
from app.services.rag.protocol import RAGService
from app.services.classifier.protocol import DocumentClassifierService
from app.services.classifier.state import AutoClassifyState
from app.services.duplicate.protocol import DuplicateService
from app.services.cloud_import.protocol import CloudImportService
# Implementation imports use aliases to avoid name collision with Protocols
from app.services.llm.service import LitellmService
from app.services.paperless.service import PaperlessClient as PaperlessClientImpl
from app.services.similarity.service import SimilarityService as SimilarityServiceImpl
from app.services.merge.service import MergeService as MergeServiceImpl
from app.services.statistics.service import StatisticsService as StatisticsServiceImpl
from app.services.ocr.service import OcrService as OcrServiceImpl
from app.services.rag.service import RAGService as RAGServiceImpl
from app.services.classifier.service import DocumentClassifierService as DocumentClassifierServiceImpl
from app.services.duplicate.service import DuplicateService as DuplicateServiceImpl
from app.services.cloud_import.service import CloudImportService as CloudImportServiceImpl


class AppProvider(Provider):
    # State classes — APP-scoped singletons (Phase 05)
    # Each service plan adds its own state provider here:
    #   Plan 02: OcrState provider
    #   Plan 03: AutoClassifyState provider
    #   Plan 04: CloudSyncState and DuplicateScanState providers

    @provide(scope=Scope.APP)
    def ocr_state(self) -> OcrState:
        return OcrState()

    @provide(scope=Scope.APP)
    def auto_classify_state(self) -> AutoClassifyState:
        return AutoClassifyState()

    @provide(scope=Scope.APP)
    def session_factory(self) -> async_sessionmaker:
        return async_session

    @provide(scope=Scope.REQUEST)
    async def db_session(self) -> AsyncIterator[AsyncSession]:
        async with async_session() as session:
            yield session

    @provide(scope=Scope.APP)
    def paperless_client(self, session_factory: async_sessionmaker) -> PaperlessClient:
        return PaperlessClientImpl(session_factory=session_factory)

    @provide(scope=Scope.APP)
    def llm_service(self, session_factory: async_sessionmaker) -> LLMService:
        return LitellmService(session_factory=session_factory)

    @provide(scope=Scope.APP)
    def similarity_service(
        self,
        paperless_client: PaperlessClient,
        llm_service: LLMService,
        session_factory: async_sessionmaker,
    ) -> SimilarityService:
        return SimilarityServiceImpl(
            paperless_client=paperless_client,
            llm_service=llm_service,
            session_factory=session_factory,
        )

    @provide(scope=Scope.APP)
    def merge_service(
        self,
        paperless_client: PaperlessClient,
        session_factory: async_sessionmaker,
    ) -> MergeService:
        return MergeServiceImpl(
            paperless_client=paperless_client,
            session_factory=session_factory,
        )

    @provide(scope=Scope.APP)
    def statistics_service(
        self,
        session_factory: async_sessionmaker,
    ) -> StatisticsService:
        return StatisticsServiceImpl(session_factory=session_factory)

    @provide(scope=Scope.APP)
    def ocr_service(
        self,
        session_factory: async_sessionmaker,
    ) -> OcrService:
        return OcrServiceImpl(session_factory=session_factory)

    @provide(scope=Scope.APP)
    def rag_service(
        self,
        session_factory: async_sessionmaker,
        paperless_client: PaperlessClient,
    ) -> RAGService:
        return RAGServiceImpl(
            session_factory=session_factory,
            paperless_client=paperless_client,
        )

    @provide(scope=Scope.APP)
    def classifier_service(
        self,
        paperless_client: PaperlessClient,
        session_factory: async_sessionmaker,
    ) -> DocumentClassifierService:
        return DocumentClassifierServiceImpl(
            paperless=paperless_client,
            session_factory=session_factory,
        )

    @provide(scope=Scope.APP)
    def duplicate_service(
        self,
        session_factory: async_sessionmaker,
        paperless_client: PaperlessClient,
    ) -> DuplicateService:
        return DuplicateServiceImpl(
            session_factory=session_factory,
            paperless_client=paperless_client,
        )

    @provide(scope=Scope.APP)
    def cloud_import_service(
        self,
        session_factory: async_sessionmaker,
    ) -> CloudImportService:
        return CloudImportServiceImpl(session_factory=session_factory)


# Module-level container singleton
container: AsyncContainer = make_async_container(AppProvider())
