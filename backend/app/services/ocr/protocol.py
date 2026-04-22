"""Protocol for OCR service."""

from typing import Protocol, runtime_checkable, Any, List, Dict, Optional

# Import PaperlessClient from old path during Wave 1 (Plan 07 handles final path alignment)
from app.services.protocols import PaperlessClient


@runtime_checkable
class OcrService(Protocol):
    """Protocol for OCR service."""

    @property
    def model(self) -> str: ...

    @property
    def max_image_size(self) -> int: ...

    @staticmethod
    def get_model_params(model_name: str) -> dict: ...

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

    def _prepare_image_for_ollama(self, img: Any, max_size: int = None) -> bytes: ...

    async def _ocr_single_image(
        self,
        image_bytes: bytes,
        page_num: int = 0,
        total_pages: int = 0,
        timeout: float = 300.0,
    ) -> str: ...