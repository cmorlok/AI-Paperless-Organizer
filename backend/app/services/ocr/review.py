"""OCR review queue file operations."""

from __future__ import annotations

import json
import logging
from typing import Dict, List

from app.services.ocr.state import REVIEW_QUEUE_FILE

logger = logging.getLogger(__name__)


def load_review_queue() -> List[Dict]:
    """Load review queue from file."""
    try:
        if REVIEW_QUEUE_FILE.exists():
            return json.loads(REVIEW_QUEUE_FILE.read_text())
    except Exception:
        pass
    return []


def save_review_queue(queue: List[Dict]) -> None:
    """Save review queue to file."""
    try:
        REVIEW_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        REVIEW_QUEUE_FILE.write_text(json.dumps(queue, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error(f"Error saving review queue: {e}")
