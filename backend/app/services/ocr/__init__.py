"""OCR service: Ollama vision OCR, watchdog loop, batch processing."""

from app.services.ocr.protocol import OcrService
from app.services.ocr.state import OcrState
from app.services.ocr.service import (
    load_review_queue,
    save_review_queue,
    load_ocr_ignore_list,
    save_ocr_ignore_list,
    load_ocr_error_list,
    save_ocr_error_list,
    load_ocr_error_counts,
    save_ocr_error_counts,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OCR_MODEL,
    TAG_OCR_REVIEW,
    TAG_OCR_FINISH,
    TAG_OCR_ERROR,
)

__all__ = [
    "OcrService",
    "OcrState",
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
