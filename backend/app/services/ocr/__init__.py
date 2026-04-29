"""OCR service: Vision OCR, processor loop, batch processing."""

from app.services.ocr.protocol import OcrService
from app.services.ocr.state import OcrState, OcrCompareState, OcrCompareSlot
from app.services.ocr.service import (
    load_review_queue,
    save_review_queue,
    DEFAULT_OCR_MODEL,
    TAG_OCR_REVIEW,
    TAG_OCR_FINISH,
    TAG_OCR_ERROR,
)
from app.services.ocr.ignore import (
    load_ocr_ignore_list,
    save_ocr_ignore_list,
    get_ocr_ignored_ids,
)
from app.services.ocr.error import (
    load_ocr_error_list,
    save_ocr_error_list,
    load_ocr_error_counts,
    save_ocr_error_counts,
    increment_ocr_error,
    reset_ocr_error,
)

__all__ = [
    "OcrService",
    "OcrState",
    "OcrCompareState",
    "OcrCompareSlot",
    "load_review_queue",
    "save_review_queue",
    "load_ocr_ignore_list",
    "save_ocr_ignore_list",
    "get_ocr_ignored_ids",
    "load_ocr_error_list",
    "save_ocr_error_list",
    "load_ocr_error_counts",
    "save_ocr_error_counts",
    "increment_ocr_error",
    "reset_ocr_error",
    "DEFAULT_OCR_MODEL",
    "TAG_OCR_REVIEW",
    "TAG_OCR_FINISH",
    "TAG_OCR_ERROR",
]
