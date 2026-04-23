"""OCR state and constants."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from app.services.base_state import BaseState, CancelMixin

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OCR_MODEL = "qwen2.5vl:7b"

TAG_RUN_OCR = "runocr"
TAG_OCR_FINISH = "ocrfinish"
TAG_OCR_REVIEW = "ocrpruefen"
TAG_OCR_ERROR = "ocrfehler"

REVIEW_QUEUE_FILE = Path("/app/data/ocr_review_queue.json")
OCR_IGNORE_FILE = Path("/app/data/ocr_ignore_list.json")
OCR_ERROR_COUNT_FILE = Path("/app/data/ocr_error_counts.json")
OCR_ERROR_FILE = Path("/app/data/ocr_error_list.json")

QUALITY_THRESHOLD = 0.5
MAX_ERROR_COUNT = 3


class PageProgress(BaseModel):
    page: int = Field(ge=1)
    status: str = "pending"
    chars: int = Field(ge=0, default=0)
    error: str | None = None


class OcrDocumentProgress(BaseModel):
    total_pages: int = Field(ge=0, default=0)
    done: int = Field(ge=0, default=0)
    errors: int = Field(ge=0, default=0)
    current_page: int = Field(ge=0, default=0)
    pages: list[PageProgress] = Field(default_factory=list)
    started_at: float = 0.0
    status: str = ""


class OcrBatchProgress(BaseModel):
    running: bool = False
    should_stop: bool = False
    total: int = Field(ge=0, default=0)
    processed: int = Field(ge=0, default=0)
    current_document: dict | None = None
    errors: list[str] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list)
    mode: str | None = None
    paused: bool = False


class OcrWatchdogProgress(BaseModel):
    enabled: bool = False
    running: bool = False
    interval_minutes: int = Field(ge=1, default=5)
    last_run: str | None = None
    task: Any = Field(default=None, exclude=True)


class OcrState(BaseState, CancelMixin):
    batch: OcrBatchProgress = Field(default_factory=OcrBatchProgress)
    watchdog: OcrWatchdogProgress = Field(default_factory=OcrWatchdogProgress)
    page_progress: dict[int, OcrDocumentProgress] = Field(default_factory=dict)
    lock_holder: str | None = None

    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    def _reset_fields(self) -> None:
        self.batch = OcrBatchProgress()
        self.watchdog = OcrWatchdogProgress()
        self.page_progress.clear()
        self.lock_holder = None
        self.clear_cancel()

    def acquire_lock(self, holder: str) -> bool:
        if self.lock_holder is not None:
            return False
        self.lock_holder = holder
        return True

    def release_lock(self) -> None:
        self.lock_holder = None

    def is_locked(self) -> bool:
        return self.lock_holder is not None

    def current_lock_holder(self) -> str | None:
        return self.lock_holder

    def cancel(self) -> None:
        self.request_cancel()
        self.batch.should_stop = True


__all__ = [
    "OcrState",
    "PageProgress",
    "OcrDocumentProgress",
    "OcrBatchProgress",
    "OcrWatchdogProgress",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_OCR_MODEL",
    "TAG_RUN_OCR",
    "TAG_OCR_FINISH",
    "TAG_OCR_REVIEW",
    "TAG_OCR_ERROR",
    "REVIEW_QUEUE_FILE",
    "OCR_IGNORE_FILE",
    "OCR_ERROR_COUNT_FILE",
    "OCR_ERROR_FILE",
    "QUALITY_THRESHOLD",
    "MAX_ERROR_COUNT",
]
