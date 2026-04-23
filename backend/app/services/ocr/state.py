"""OCR state and constants."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

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

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        setattr(self, key, value)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class OcrWatchdogProgress(BaseModel):
    enabled: bool = False
    running: bool = False
    interval_minutes: int = Field(ge=1, default=5)
    last_run: str | None = None
    task: Any = Field(default=None, exclude=True)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        setattr(self, key, value)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class OcrCompareState(BaseModel):
    """State for the OCR compare (Vergleich) feature."""
    running: bool = False
    phase: str = ""
    current_model: str = ""
    current_model_index: int = 0
    total_models: int = 0
    current_page: int = 0
    total_pages: int = 0
    compared_page: int = 0
    title: str = ""
    old_content: str = ""
    results: list = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    job_start: float = Field(default=0.0, alias="_job_start")
    models: list = Field(default_factory=list)
    document_id: int = 0
    error: str | None = None

    model_config = ConfigDict(populate_by_name=True)

    def reset(self) -> None:
        self.running = False
        self.phase = ""
        self.current_model = ""
        self.current_model_index = 0
        self.total_models = 0
        self.current_page = 0
        self.total_pages = 0
        self.compared_page = 0
        self.title = ""
        self.old_content = ""
        self.results = []
        self.elapsed_seconds = 0.0
        self.job_start = 0.0
        self.models = []
        self.document_id = 0
        self.error = None


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

    def cancel(self) -> None:
        """Cancel the OCR operation and signal batch should stop."""
        self.request_cancel()
        self.batch.should_stop = True

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


def _get_ocr_state_instance() -> OcrState:
    if not hasattr(_get_ocr_state_instance, "_instance"):
        _get_ocr_state_instance._instance = OcrState()
    return _get_ocr_state_instance._instance


class _DictProxy:
    def __init__(self, attr: str):
        self._attr = attr

    @property
    def _state(self):
        return _get_ocr_state_instance()

    def __getitem__(self, key: str):
        return getattr(getattr(self._state, self._attr), key)

    def __setitem__(self, key: str, value) -> None:
        setattr(getattr(self._state, self._attr), key, value)

    def get(self, key: str, default=None):
        return getattr(getattr(self._state, self._attr), key, default)


class _ModuleLevelProxy:
    def __getitem__(self, key: str):
        state = _get_ocr_state_instance()
        if key == "running":
            return state.batch.running
        if key == "enabled":
            return state.watchdog.enabled
        if key == "interval_minutes":
            return state.watchdog.interval_minutes
        if key == "last_run":
            return state.watchdog.last_run
        return getattr(state, key)

    def __setitem__(self, key: str, value) -> None:
        state = _get_ocr_state_instance()
        if key == "running":
            state.batch.running = value
        elif key == "enabled":
            state.watchdog.enabled = value
        elif key == "interval_minutes":
            state.watchdog.interval_minutes = value
        elif key == "last_run":
            state.watchdog.last_run = value
        else:
            setattr(state, key, value)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except (AttributeError, KeyError):
            return default


batch_state = _ModuleLevelProxy()
watchdog_state = _ModuleLevelProxy()
single_ocr_running = _DictProxy("batch")
