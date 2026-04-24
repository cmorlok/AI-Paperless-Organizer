"""Auto-classify state for background classification loop."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from app.services.base_state import BaseState, CancelMixin


class AutoClassifyState(BaseState, CancelMixin):
    """State for the auto-classify background loop."""

    processed: int = Field(ge=0, default=0)
    errors: int = Field(ge=0, default=0)
    reviewed: int = Field(ge=0, default=0)
    current_doc: int | None = None
    last_run: str | None = None

    _task: Any = PrivateAttr(default=None)

    def _reset_fields(self) -> None:
        self.processed = 0
        self.errors = 0
        self.reviewed = 0
        self.current_doc = None
        self.last_run = None
        self.clear_cancel()

    @property
    def task(self) -> Any:
        return self._task

    @task.setter
    def task(self, value: Any) -> None:
        self._task = value


__all__ = ["AutoClassifyState"]