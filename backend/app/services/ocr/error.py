"""OCR error tracking file operations."""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Set

from app.services.ocr.state import OCR_ERROR_COUNT_FILE, OCR_ERROR_FILE

logger = logging.getLogger(__name__)


def load_ocr_error_counts() -> Dict:
    """Load error counts per document ID."""
    try:
        if OCR_ERROR_COUNT_FILE.exists():
            return json.loads(OCR_ERROR_COUNT_FILE.read_text())
    except Exception:
        pass
    return {}


def save_ocr_error_counts(counts: Dict) -> None:
    """Save error counts per document ID."""
    try:
        OCR_ERROR_COUNT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OCR_ERROR_COUNT_FILE.write_text(json.dumps(counts, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error(f"Error saving OCR error counts: {e}")


def increment_ocr_error(document_id: int, title: str, error_msg: str) -> int:
    """Increment error count for a document. Returns new count."""
    counts = load_ocr_error_counts()
    key = str(document_id)
    entry = counts.get(key, {"count": 0, "title": title, "errors": []})
    entry["count"] += 1
    entry["title"] = title
    entry["errors"].append({"error": error_msg[:200], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
    # Keep only last 5 errors per doc
    entry["errors"] = entry["errors"][-5:]
    counts[key] = entry
    save_ocr_error_counts(counts)
    return entry["count"]


def reset_ocr_error(document_id: int) -> None:
    """Reset error count for a document (e.g. after successful OCR)."""
    counts = load_ocr_error_counts()
    key = str(document_id)
    if key in counts:
        del counts[key]
        save_ocr_error_counts(counts)


def load_ocr_error_list() -> List[Dict]:
    """Load OCR error list from file."""
    try:
        if OCR_ERROR_FILE.exists():
            return json.loads(OCR_ERROR_FILE.read_text())
    except Exception:
        pass
    return []


def save_ocr_error_list(error_list: List[Dict]) -> None:
    """Save OCR error list to file."""
    try:
        OCR_ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)
        OCR_ERROR_FILE.write_text(json.dumps(error_list, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error(f"Error saving OCR error list: {e}")


def get_ocr_error_ids() -> Set[int]:
    """Get set of document IDs that are permanently marked as error."""
    return {item["document_id"] for item in load_ocr_error_list()}
