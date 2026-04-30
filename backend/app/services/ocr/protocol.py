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

    # ── Extracted router business logic ─────────────────────────────────────

    def get_ocr_settings_with_state(self) -> dict: ...

    async def save_ocr_settings(
        self, model: str, max_image_size: int, smart_skip_enabled: bool, config_svc: Any,
    ) -> dict: ...

    async def persist_processor_enabled(self, enabled: bool, config_svc: Any) -> None: ...

    async def configure_processor(
        self, enabled: bool, interval_minutes: int, config_svc: Any, client: Any,
    ) -> dict: ...

    def get_processor_status_dict(self) -> dict: ...

    def pause_batch(self) -> dict: ...

    def resume_batch(self) -> dict: ...

    def stop_batch(self) -> dict: ...

    async def ocr_single_document_safe(
        self, client: Any, document_id: int, force: bool, db_session: Any,
    ) -> dict: ...

    def get_progress_dict(self, document_id: int) -> dict: ...

    async def apply_ocr_result_background(
        self, client: Any, document_id: int, content: str, set_finish_tag: bool,
    ) -> None: ...

    def check_batch_not_running(self) -> None: ...

    def get_batch_status_dict(self, llm_service: Any = None) -> dict: ...

    def dismiss_review_item_from_queue(self, document_id: int) -> dict: ...

    def ignore_review_item_permanently(self, document_id: int) -> dict: ...

    def get_error_list_with_counts(self) -> dict: ...

    def clear_all_errors(self) -> dict: ...

    def remove_from_ocr_ignore_list(self, document_id: int) -> None: ...

    async def get_preview_response(self, client: Any, document_id: int): ...

    async def get_thumbnail_response(self, client: Any, document_id: int): ...

    async def validate_and_start_compare(
        self, client: Any, document_id: int, slots: list, page: int, compare_state: Any,
    ) -> dict: ...

    def get_compare_status_dict(self, compare_state: Any) -> dict: ...
