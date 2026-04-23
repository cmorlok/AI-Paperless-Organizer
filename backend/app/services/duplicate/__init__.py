"""Duplicate detection service: exact, semantic, and invoice matching."""

from app.services.duplicate.protocol import DuplicateService
from app.services.duplicate.state import DuplicateScanState

__all__ = [
    "DuplicateService",
    "DuplicateScanState",
]
