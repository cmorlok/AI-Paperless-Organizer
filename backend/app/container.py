from __future__ import annotations

from dishka import Provider, Scope, provide, make_async_container, AsyncContainer
from typing import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database import async_session
# Protocol/state imports — use package-level
from app.services.llm import LLMService
from app.services.paperless import PaperlessClient
from app.services.similarity import SimilarityService
from app.services.merge import MergeService
from app.services.statistics import StatisticsService
from app.services.ocr import OcrService, OcrState, OcrCompareState
from app.services.rag import RAGService
from app.services.classifier import DocumentClassifierService, AutoClassifyState
from app.services.duplicate import DuplicateService, DuplicateScanState
from app.services.cloud_import import CloudImportService, CloudSyncState
from app.services.config import ConfigService
from app.services.tags import TagsService
from app.services.correspondents import CorrespondentsService
from app.services.document_types import DocumentTypesService
from app.services.ignored_items import IgnoredItemsService
from app.services.api_keys import ApiKeysService
from app.services.auth import AuthService
# Implementation imports use aliases to avoid name collision with Protocols
from app.services.config.service import ConfigServiceImpl
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
from app.services.tags.service import TagsServiceImpl
from app.services.correspondents.service import CorrespondentsServiceImpl
from app.services.document_types.service import DocumentTypesServiceImpl
from app.services.ignored_items.service import IgnoredItemsServiceImpl
from app.services.api_keys.service import ApiKeysServiceImpl
from app.services.auth.service import AuthServiceImpl


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
    def ocr_compare_state(self) -> OcrCompareState:
        return OcrCompareState()

    @provide(scope=Scope.APP)
    def auto_classify_state(self) -> AutoClassifyState:
        return AutoClassifyState()

    @provide(scope=Scope.APP)
    def cloud_sync_state(self) -> CloudSyncState:
        return CloudSyncState()

    @provide(scope=Scope.APP)
    def duplicate_scan_state(self) -> DuplicateScanState:
        return DuplicateScanState()

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
    def llm_service(self) -> LLMService:
        return LitellmService()

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
        state: OcrState,
        llm_service: LLMService,
        session_factory: async_sessionmaker,
    ) -> OcrService:
        return OcrServiceImpl(
            state=state,
            llm_service=llm_service,
            session_factory=session_factory,
        )

    @provide(scope=Scope.APP)
    def rag_service(
        self,
        session_factory: async_sessionmaker,
        paperless_client: PaperlessClient,
        llm_service: LLMService,
    ) -> RAGService:
        return RAGServiceImpl(
            session_factory=session_factory,
            paperless_client=paperless_client,
            llm_service=llm_service,
        )

    @provide(scope=Scope.APP)
    def classifier_service(
        self,
        paperless_client: PaperlessClient,
        session_factory: async_sessionmaker,
        state: AutoClassifyState,
        llm_service: LLMService,
    ) -> DocumentClassifierService:
        return DocumentClassifierServiceImpl(
            paperless=paperless_client,
            session_factory=session_factory,
            state=state,
            llm_service=llm_service,
        )

    @provide(scope=Scope.APP)
    def duplicate_service(
        self,
        session_factory: async_sessionmaker,
        paperless_client: PaperlessClient,
        state: DuplicateScanState,
    ) -> DuplicateService:
        return DuplicateServiceImpl(
            session_factory=session_factory,
            paperless_client=paperless_client,
            state=state,
        )

    @provide(scope=Scope.APP)
    def cloud_import_service(
        self,
        session_factory: async_sessionmaker,
        state: CloudSyncState,
    ) -> CloudImportService:
        return CloudImportServiceImpl(session_factory=session_factory, state=state)

    @provide(scope=Scope.APP)
    def config_service(
        self,
        session_factory: async_sessionmaker,
    ) -> ConfigService:
        return ConfigServiceImpl(session_factory=session_factory)

    @provide(scope=Scope.APP)
    def tags_service(
        self,
        paperless_client: PaperlessClient,
        llm_service: LLMService,
        config_service: ConfigService,
        statistics_service: StatisticsService,
        session_factory: async_sessionmaker,
    ) -> TagsService:
        return TagsServiceImpl(
            paperless_client=paperless_client,
            llm_service=llm_service,
            config_service=config_service,
            statistics_service=statistics_service,
            session_factory=session_factory,
        )

    @provide(scope=Scope.APP)
    def correspondents_service(
        self,
        paperless_client: PaperlessClient,
        llm_service: LLMService,
        config_service: ConfigService,
        statistics_service: StatisticsService,
        session_factory: async_sessionmaker,
    ) -> CorrespondentsService:
        return CorrespondentsServiceImpl(
            paperless_client=paperless_client,
            llm_service=llm_service,
            config_service=config_service,
            statistics_service=statistics_service,
            session_factory=session_factory,
        )

    @provide(scope=Scope.APP)
    def document_types_service(
        self,
        paperless_client: PaperlessClient,
        llm_service: LLMService,
        config_service: ConfigService,
        statistics_service: StatisticsService,
        session_factory: async_sessionmaker,
    ) -> DocumentTypesService:
        return DocumentTypesServiceImpl(
            paperless_client=paperless_client,
            llm_service=llm_service,
            config_service=config_service,
            statistics_service=statistics_service,
            session_factory=session_factory,
        )

    @provide(scope=Scope.APP)
    def ignored_items_service(
        self,
        session_factory: async_sessionmaker,
    ) -> IgnoredItemsService:
        return IgnoredItemsServiceImpl(session_factory=session_factory)

    @provide(scope=Scope.APP)
    def api_keys_service(
        self,
        session_factory: async_sessionmaker,
    ) -> ApiKeysService:
        return ApiKeysServiceImpl(session_factory=session_factory)

    @provide(scope=Scope.APP)
    def auth_service(
        self,
        session_factory: async_sessionmaker,
    ) -> AuthService:
        return AuthServiceImpl(session_factory=session_factory)


# Module-level container singleton
container: AsyncContainer = make_async_container(AppProvider())
