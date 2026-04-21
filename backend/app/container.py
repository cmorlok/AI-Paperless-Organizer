from __future__ import annotations

from dishka import Provider, Scope, provide, make_async_container, AsyncContainer
from dishka.integrations.fastapi import setup_dishka
from typing import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database import async_session
from app.services.protocols import (
    LLMService,
    PaperlessClient,
    SimilarityService,
    MergeService,
    StatisticsService,
    OcrService,
    RAGService,
    DocumentClassifierService,
    DuplicateService,
    CloudImportService,
)
# Implementation imports use aliases to avoid name collision with Protocols
from app.services.llm_service import LitellmService
from app.services.paperless_client import PaperlessClient as PaperlessClientImpl
from app.services.similarity import SimilarityService as SimilarityServiceImpl
from app.services.merge import MergeService as MergeServiceImpl
from app.services.statistics import StatisticsService as StatisticsServiceImpl
from app.services.ocr_service import OcrService as OcrServiceImpl
from app.services.rag.service import RAGService as RAGServiceImpl
from app.services.classifier.service import DocumentClassifierService as DocumentClassifierServiceImpl
from app.services.duplicate_service import DuplicateService as DuplicateServiceImpl
from app.services.cloud_import_service import CloudImportService as CloudImportServiceImpl


class AppProvider(Provider):
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
