"""OCR ignore list file operations."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Set

from app.services.ocr.state import OCR_IGNORE_FILE

logger = logging.getLogger(__name__)


def load_ocr_ignore_list() -> List[Dict]:
    """Load OCR ignore list from file."""
    try:
        if OCR_IGNORE_FILE.exists():
            return json.loads(OCR_IGNORE_FILE.read_text())
    except Exception:
        pass
    return []


def save_ocr_ignore_list(ignore_list: List[Dict]) -> None:
    """Save OCR ignore list to file."""
    try:
        OCR_IGNORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        OCR_IGNORE_FILE.write_text(json.dumps(ignore_list, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error(f"Error saving OCR ignore list: {e}")


def get_ocr_ignored_ids() -> Set[int]:
    """Get set of document IDs that should be skipped in OCR."""
    return {item["document_id"] for item in load_ocr_ignore_list()}
