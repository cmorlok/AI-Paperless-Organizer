"""Protocol for OCR service."""

from typing import Protocol, runtime_checkable, Any, List, Dict

from app.services.paperless import PaperlessClient
from .state import OcrCompareSlot, OcrCompareState


@runtime_checkable
class OcrService(Protocol):
    """Protocol for OCR service."""

    @staticmethod
    def get_model_params(model: str) -> dict: ...

    async def processor_loop(self, paperless_client: PaperlessClient) -> None: ...

    async def batch_ocr(
        self,
        paperless_client: PaperlessClient,
        mode: str,
        document_ids: List[int] | None = None,
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

    async def get_ocr_status(
        self,
        paperless_client: PaperlessClient,
    ) -> Dict[str, Any]: ...

    async def apply_review_item(
        self,
        document_id: int,
        paperless_client: PaperlessClient,
    ) -> Dict[str, Any]: ...

    async def reset_all_review_items(
        self,
        paperless_client: PaperlessClient,
    ) -> Dict[str, Any]: ...

    async def keep_all_originals(
        self,
        paperless_client: PaperlessClient,
    ) -> Dict[str, Any]: ...

    async def add_to_ignore_list(
        self,
        document_id: int,
        paperless_client: PaperlessClient,
    ) -> Dict[str, Any]: ...

    async def remove_from_error_list(
        self,
        document_id: int,
        paperless_client: PaperlessClient,
    ) -> Dict[str, Any]: ...

    async def evaluate_ocr_results(
        self,
        document_title: str,
        results: List[dict],
        eval_provider: str,
        eval_model: str | None = None,
    ) -> Dict[str, Any]: ...
