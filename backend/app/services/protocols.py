"""Protocol definitions for all backend services.

These Protocols establish the contracts that routers depend on.
Implementation classes may be refactored in future plans; the Protocols
ensure routers always have a stable interface to code against.
"""

from typing import Protocol, runtime_checkable, Optional, List, Dict, Any, AsyncIterator


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


@runtime_checkable
class PaperlessClient(Protocol):
    """Protocol for Paperless-ngx API client."""

    async def test_connection(self) -> bool: ...

    async def get_correspondents(self, use_cache: bool = True) -> List[Dict]: ...

    async def get_tags(self, use_cache: bool = True) -> List[Dict]: ...

    async def get_document_types(self, use_cache: bool = True) -> List[Dict]: ...

    async def get_storage_paths(self, use_cache: bool = True) -> List[Dict]: ...

    async def get_custom_fields(self, use_cache: bool = True) -> List[Dict]: ...

    async def get_document(self, document_id: int) -> Optional[Dict]: ...

    async def get_document_count(self, **kwargs) -> int: ...

    async def bulk_update_documents(self, **kwargs) -> None: ...

    async def get_or_create_tag(self, name: str) -> Dict: ...

    async def get_document_preview_image(self, document_id: int) -> bytes: ...

    async def get_document_thumbnail_bytes(self, document_id: int) -> bytes: ...

    async def download_document_file(self, document_id: int) -> bytes: ...

    async def update_document(self, document_id: int, data: Dict) -> Dict: ...

    async def get_documents(self, **kwargs) -> List[Dict]: ...

    async def get_documents_by_correspondent(self, correspondent_id: int) -> List[Dict]: ...

    async def delete_correspondent(self, correspondent_id: int) -> bool: ...

    async def delete_tag(self, tag_id: int) -> bool: ...

    async def delete_document_type(self, document_type_id: int) -> bool: ...

    async def post(self, endpoint: str, **kwargs) -> Optional[Dict]: ...

    async def get(self, endpoint: str, **kwargs) -> Optional[Dict]: ...

    async def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]: ...


@runtime_checkable
class SimilarityService(Protocol):
    """Protocol for similarity detection service."""

    async def find_similar_correspondents(self, batch_size: int = 200) -> Dict: ...

    async def find_similar_tags(self, batch_size: int = 200) -> Dict: ...

    async def find_similar_document_types(self, batch_size: int = 200) -> Dict: ...

    async def find_nonsense_tags(self, batch_size: int = 300) -> Dict: ...

    async def find_tags_that_are_correspondents(self, batch_size: int = 300) -> Dict: ...

    async def find_tags_that_are_document_types(self, batch_size: int = 300) -> Dict: ...


@runtime_checkable
class MergeService(Protocol):
    """Protocol for merge service."""

    async def merge_correspondents(self, target_id: int, target_name: str, source_ids: List[int]) -> Dict: ...

    async def merge_tags(self, target_id: int, target_name: str, source_ids: List[int]) -> Dict: ...

    async def merge_document_types(self, target_id: int, target_name: str, source_ids: List[int]) -> Dict: ...

    async def get_history(self, entity_type: str = None) -> List[Dict]: ...


@runtime_checkable
class StatisticsService(Protocol):
    """Protocol for statistics service."""

    async def record_operation(
        self,
        entity_type: str,
        operation: str,
        items_affected: int,
        documents_affected: int = 0,
        items_before: int = 0,
        items_after: int = 0,
        details: dict = None,
    ) -> Any: ...

    async def get_total_stats(self) -> Dict: ...

    async def get_recent_operations(self, limit: int = 10) -> List[Dict]: ...

    async def get_daily_trend(self, days: int = 7) -> List[Dict]: ...


@runtime_checkable
class OcrService(Protocol):
    """Protocol for OCR service."""

    @property
    def model(self) -> str: ...

    async def watchdog_loop(self, paperless_client: PaperlessClient) -> None: ...

    async def batch_ocr(
        self,
        paperless_client: PaperlessClient,
        mode: str,
        document_ids: List[int] = None,
        set_finish_tag: bool = True,
        remove_runocr_tag: bool = True,
    ) -> None: ...

    async def ocr_document(
        self,
        paperless_client: PaperlessClient,
        document_id: int,
        force: bool = False,
        db_session: Any = None,
    ) -> Dict[str, Any]: ...

    async def test_connection(self) -> Dict[str, Any]: ...

    async def apply_ocr_result(
        self,
        paperless_client: PaperlessClient,
        document_id: int,
        content: str,
        set_finish_tag: bool = True,
    ) -> None: ...

    def get_stats(self) -> List[Dict[str, Any]]: ...

    def get_current_url(self) -> str: ...


@runtime_checkable
class RAGService(Protocol):
    """Protocol for RAG service."""

    @property
    def indexer(self) -> Any: ...

    async def chat_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        filters: Optional[Dict] = None,
    ) -> AsyncIterator[str]: ...

    async def search(self, query: str, limit: int = 5, filters: Optional[Dict] = None) -> List[Any]: ...

    async def get_sessions(self) -> List[Dict]: ...

    async def get_session(self, session_id: str) -> Optional[Dict]: ...

    async def delete_session(self, session_id: str) -> bool: ...

    async def get_config_dict(self) -> Dict: ...

    async def update_config(self, updates: Dict) -> Dict: ...


@runtime_checkable
class DocumentClassifierService(Protocol):
    """Protocol for document classifier service."""

    async def classify_document(self, document_id: int) -> Any: ...

    async def classify_document_auto(self, document_id: int, mode: str = "review") -> Dict[str, Any]: ...

    async def benchmark_document(self, document_id: int, slots: List[tuple]) -> Dict: ...

    async def apply_classification(self, document_id: int, classification: Dict) -> Dict: ...

    async def get_config(self) -> Any: ...

    async def save_config(self, updates: Dict[str, Any]) -> Any: ...

    async def get_storage_profiles(self) -> List[Any]: ...

    async def save_storage_profile(self, data: Dict[str, Any]) -> Any: ...

    async def get_custom_field_mappings(self) -> List[Any]: ...

    async def save_custom_field_mapping(self, data: Dict[str, Any]) -> Any: ...


@runtime_checkable
class DuplicateService(Protocol):
    """Protocol for duplicate detection service."""

    async def scan_all(self, modes: List[str], similarity_threshold: float = 0.92) -> None: ...


@runtime_checkable
class CloudImportService(Protocol):
    """Protocol for cloud import service."""

    async def sync_source(self, source: Any, pl_client: PaperlessClient, db: Any) -> Dict: ...

    async def test_connection(self, source: Any) -> Dict: ...

    async def sync_all_sources(self, pl_client: PaperlessClient) -> Dict: ...
