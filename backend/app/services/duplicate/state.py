"""Duplicate detection state management.

Scan state is module-level for background job status visibility.
"""

from typing import Dict

_scan_state: Dict = {
    "running": False,
    "phase": "",       # "exact", "similar", "invoices", "done"
    "progress": 0,
    "total": 0,
    "results": {
        "exact": [],
        "similar": [],
        "invoices": [],
    },
    "error": None,
    "cancel_requested": False,
}


def get_scan_state() -> Dict:
    """Return a reference to the current scan state dict."""
    return _scan_state


def _is_cancelled() -> bool:
    """Check if cancellation was requested."""
    return _scan_state.get("cancel_requested", False)