"""OCR state and constants."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

# Default OCR settings
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OCR_MODEL = "qwen2.5vl:7b"

# Tag names for OCR workflow
TAG_RUN_OCR = "runocr"
TAG_OCR_FINISH = "ocrfinish"
TAG_OCR_REVIEW = "ocrpruefen"
TAG_OCR_ERROR = "ocrfehler"

# Batch job state (in-memory, single instance)
batch_state: Dict[str, Any] = {
    "running": False,
    "should_stop": False,
    "total": 0,
    "processed": 0,
    "current_document": None,
    "errors": [],
    "log": [],
    "mode": None,
    "paused": False,
}

# Track single OCR to prevent watchdog conflicts
single_ocr_running: Dict[str, bool] = {"value": False}

# Live page-level progress for frontend polling
ocr_page_progress: Dict[int, Dict[str, Any]] = {}
# { document_id: { total_pages, done, errors, current_page, pages: [{page, status, chars}], started_at } }

# File paths
REVIEW_QUEUE_FILE = Path("/app/data/ocr_review_queue.json")
OCR_IGNORE_FILE = Path("/app/data/ocr_ignore_list.json")
OCR_ERROR_COUNT_FILE = Path("/app/data/ocr_error_counts.json")
OCR_ERROR_FILE = Path("/app/data/ocr_error_list.json")

# Quality threshold: if new text is less than this ratio of old text, flag for review
QUALITY_THRESHOLD = 0.5
# Max error count before a document is permanently tagged as ocrfehler
MAX_ERROR_COUNT = 3

# Watchdog state
watchdog_state: Dict[str, Any] = {
    "enabled": False,
    "running": False,
    "interval_minutes": 5,
    "last_run": None,
    "task": None,  # asyncio.Task
}
