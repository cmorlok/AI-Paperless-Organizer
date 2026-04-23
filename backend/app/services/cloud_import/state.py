"""Cloud import state management."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.services.base_state import BaseState, CancelMixin

_VALID_EXTENSIONS = {
    "pdf", "png", "jpg", "jpeg", "tiff", "tif", "heic",
    "docx", "doc", "odt", "txt", "eml",
}


class CloudSyncState(BaseState, CancelMixin):
    """State for the cloud import sync daemon."""

    current_source_id: int | None = None
    current_source_name: str | None = None
    current_file: str | None = None
    last_run: str | None = None
    files_imported_session: int = Field(ge=0, default=0)
    errors_session: int = Field(ge=0, default=0)

    task: Any = Field(default=None, exclude=True)

    def _reset_fields(self) -> None:
        self.current_source_id = None
        self.current_source_name = None
        self.current_file = None
        self.last_run = None
        self.files_imported_session = 0
        self.errors_session = 0
        self.clear_cancel()


__all__ = ["CloudSyncState", "_VALID_EXTENSIONS"]