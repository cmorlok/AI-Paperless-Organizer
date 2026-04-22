from __future__ import annotations

from typing import Dict

_cloud_sync_state: Dict = {
    "enabled": False,
    "running": False,
    "current_source_id": None,
    "current_source_name": None,
    "current_file": None,
    "task": None,
    "last_run": None,
    "files_imported_session": 0,
    "errors_session": 0,
}

_VALID_EXTENSIONS = {
    "pdf", "png", "jpg", "jpeg", "tiff", "tif", "heic",
    "docx", "doc", "odt", "txt", "eml",
}


def get_cloud_sync_state() -> Dict:
    return _cloud_sync_state
