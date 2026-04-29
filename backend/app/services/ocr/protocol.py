"""Protocol for OCR service."""

from typing import Protocol, runtime_checkable, Any, List, Dict

from app.services.paperless import PaperlessClient
from .state import OcrCompareSlot, OcrCompareState


@runtime_checkable
class OcrService(Protocol):
    """Protocol for OCR service."""

    @staticmethod
    def get_model_params(model_name: str) -> dict: ...

    async def processor_loop(self, paperless_client: PaperlessClient) -> None: ...

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

    async def run_compare_job(
        self,
        paperless_client: PaperlessClient,
        document_id: int,
        slots: list[OcrCompareSlot],
        target_page: int,
        compare_state: OcrCompareState,
    ) -> None: ...

    async def test_connection(self) -> Dict[str, Any]: ...

    async def apply_ocr_result(
        self,
        paperless_client: PaperlessClient,
        document_id: int,
        new_content: str,
        set_finish_tag: bool = True,
    ) -> Dict[str, Any]: ...

    def get_stats(self) -> List[Dict[str, Any]]: ...
