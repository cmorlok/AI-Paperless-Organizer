"""Tests for BaseState and CancelMixin — TDD RED phase."""

from __future__ import annotations

import asyncio

import pytest

from app.services.base_state import BaseState, CancelMixin


class TestBaseState:
    """Tests for BaseState base class."""

    def test_default_fields(self) -> None:
        """Test 1: BaseState() creates instance with running=False, enabled=False."""
        state = BaseState()
        assert state.running is False
        assert state.enabled is False

    def test_reset_preserves_identity(self) -> None:
        """Test 2: BaseState.reset() clears running to False without creating new object."""
        state = BaseState()
        state.running = True
        original_id = id(state)
        state.reset()
        assert state.running is False
        assert id(state) == original_id

    def test_model_dump_excludes_private_fields(self) -> None:
        """Test 3: model_dump() excludes fields marked exclude=True and fields starting with _."""
        state = BaseState()
        dumped = state.model_dump()
        # _lock is excluded (it starts with _ and has exclude=True)
        assert "_lock" not in dumped
        # _cancelled is not on BaseState itself, but the pattern should work
        # running and enabled should be present
        assert "running" in dumped
        assert "enabled" in dumped

    def test_lock_is_shared_classvar(self) -> None:
        """Test 6: BaseState._lock is a shared asyncio.Lock (ClassVar) — same lock across instances."""
        state1 = BaseState()
        state2 = BaseState()
        # Both instances share the same lock object
        assert state1._lock is state2._lock
        assert isinstance(state1._lock, asyncio.Lock)

    def test_is_running_and_is_enabled(self) -> None:
        """BaseState.is_running() and is_enabled() reflect field values."""
        state = BaseState()
        assert state.is_running() is False
        assert state.is_enabled() is False
        state.running = True
        state.enabled = True
        assert state.is_running() is True
        assert state.is_enabled() is True


class SubState(BaseState):
    """Test subclass with custom fields for reset hook testing."""

    phase: str = ""
    progress: int = 0

    def _reset_fields(self) -> None:
        self.phase = ""
        self.progress = 0


class TestBaseStateSubclass:
    """Tests for subclass behavior."""

    def test_subclass_reset_clears_custom_fields(self) -> None:
        """Test 5: Subclass with custom fields resets correctly (calls _reset_fields hook)."""
        state = SubState()
        state.running = True
        state.phase = "scanning"
        state.progress = 75
        original_id = id(state)
        state.reset()
        assert state.running is False
        assert state.phase == ""
        assert state.progress == 0
        assert id(state) == original_id

    def test_subclass_model_dump_includes_custom_fields(self) -> None:
        """Subclass fields are included in model_dump (unless excluded)."""
        state = SubState()
        state.phase = "scanning"
        state.progress = 50
        dumped = state.model_dump()
        assert "phase" in dumped
        assert "progress" in dumped
        assert dumped["phase"] == "scanning"
        assert dumped["progress"] == 50


class CancelState(CancelMixin, BaseState):
    """Test class combining CancelMixin with BaseState."""

    pass


class TestCancelMixin:
    """Tests for CancelMixin."""

    def test_request_cancel_and_is_cancelled(self) -> None:
        """Test 4: CancelMixin.request_cancel() + is_cancelled() + clear_cancel() works."""
        state = CancelState()
        assert state.is_cancelled() is False
        state.request_cancel()
        assert state.is_cancelled() is True
        state.clear_cancel()
        assert state.is_cancelled() is False

    def test_cancel_excluded_from_dump(self) -> None:
        """_cancelled field is excluded from model_dump."""
        state = CancelState()
        state.request_cancel()
        dumped = state.model_dump()
        assert "_cancelled" not in dumped
        assert "cancelled" not in dumped

    def test_cancel_does_not_affect_reset(self) -> None:
        """CancelMixin state is independent of BaseState reset."""
        state = CancelState()
        state.running = True
        state.request_cancel()
        state.reset()
        # reset() clears running but not _cancelled
        assert state.running is False
        assert state.is_cancelled() is True
