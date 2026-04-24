"""OCR service: Ollama vision OCR, watchdog loop, batch processing."""

from app.services.ocr.protocol import OcrService
from app.services.ocr.state import OcrState, OcrCompareState
from app.services.ocr.service import (
    load_review_queue,
    save_review_queue,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OCR_MODEL,
    TAG_OCR_REVIEW,
    TAG_OCR_FINISH,
    TAG_OCR_ERROR,
)
from app.services.ocr.ignore import (
    load_ocr_ignore_list,
    save_ocr_ignore_list,
)
from app.services.ocr.error import (
    load_ocr_error_list,
    save_ocr_error_list,
    load_ocr_error_counts,
    save_ocr_error_counts,
)

__all__ = [
    "OcrService",
    "OcrState",
    "OcrCompareState",
    "load_review_queue",
    "save_review_queue",
    "load_ocr_ignore_list",
    "save_ocr_ignore_list",
    "load_ocr_error_list",
    "save_ocr_error_list",
    "load_ocr_error_counts",
    "save_ocr_error_counts",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_OCR_MODEL",
    "TAG_OCR_REVIEW",
    "TAG_OCR_FINISH",
    "TAG_OCR_ERROR",
]
