"""Duplicate detection state management."""

from __future__ import annotations

from pydantic import Field, field_validator

from app.services.base_state import BaseState, CancelMixin


class DuplicateScanState(BaseState, CancelMixin):
    """State for duplicate detection scan job."""

    phase: str = ""
    progress: int = Field(ge=0, default=0)
    total: int = Field(ge=0, default=0)
    results: dict = Field(default_factory=lambda: {"exact": [], "similar": [], "invoices": []})
    error: str | None = None

    def _reset_fields(self) -> None:
        self.phase = ""
        self.progress = 0
        self.total = 0
        self.results = {"exact": [], "similar": [], "invoices": []}
        self.error = None
        self.clear_cancel()

    @field_validator("phase")
    @classmethod
    def validate_phase(cls, v: str) -> str:
        allowed = {"", "exact", "similar", "invoices", "done"}
        if v not in allowed:
            raise ValueError(f"phase must be one of {allowed}")
        return v

    def __setitem__(self, key: str, value) -> None:
        if key == "cancel_requested":
            if value:
                self.request_cancel()
            else:
                self.clear_cancel()
            return
        object.__setattr__(self, key, value)

    def __getitem__(self, key: str):
        if key == "cancel_requested":
            return self.is_cancelled()
        return getattr(self, key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except (AttributeError, KeyError):
            return default

    def start_scan(self) -> None:
        self.running = True
        self.phase = ""
        self.progress = 0
        self.total = 0
        self.results = {"exact": [], "similar": [], "invoices": []}
        self.error = None

    def update_progress(self, phase: str, progress: int, total: int) -> None:
        self.phase = phase
        self.progress = progress
        self.total = total

    def complete(self) -> None:
        self.phase = "done"
        self.running = False


__all__ = ["DuplicateScanState"]