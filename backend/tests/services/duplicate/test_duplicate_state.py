"""Tests for DuplicateScanState."""

import pytest

from app.services.duplicate.state import DuplicateScanState


class TestFields:
    def test_has_required_fields(self):
        state = DuplicateScanState()
        assert hasattr(state, "phase")
        assert hasattr(state, "progress")
        assert hasattr(state, "total")
        assert hasattr(state, "results")
        assert hasattr(state, "error")


class TestReset:
    def test_reset_clears_all_fields(self):
        state = DuplicateScanState()
        state.phase = "exact"
        state.progress = 50
        state.total = 100
        state.results = {"exact": [1, 2, 3]}
        state.error = "some error"
        original_id = id(state)
        state.reset()
        assert state.phase == ""
        assert state.progress == 0
        assert state.total == 0
        assert state.results == {"exact": [], "similar": [], "invoices": []}
        assert state.error is None
        assert id(state) == original_id


class TestPhaseValidator:
    def test_valid_phases(self):
        for phase in ("", "exact", "similar", "invoices", "done"):
            state = DuplicateScanState(phase=phase)
            assert state.phase == phase

    def test_invalid_phase_rejected(self):
        with pytest.raises(Exception):
            DuplicateScanState(phase="invalid")


class TestStateTransitions:
    def test_start_scan(self):
        state = DuplicateScanState()
        state.start_scan()
        assert state.running is True
        assert state.phase == ""
        assert state.progress == 0

    def test_update_progress(self):
        state = DuplicateScanState()
        state.update_progress("exact", 25, 100)
        assert state.phase == "exact"
        assert state.progress == 25
        assert state.total == 100

    def test_complete(self):
        state = DuplicateScanState()
        state.complete()
        assert state.phase == "done"
        assert state.running is False


class TestDictAccess:
    def test_dict_getitem(self):
        state = DuplicateScanState()
        state.phase = "exact"
        assert state["phase"] == "exact"

    def test_dict_setitem(self):
        state = DuplicateScanState()
        state["phase"] = "similar"
        assert state.phase == "similar"

    def test_get_with_default(self):
        state = DuplicateScanState()
        assert state.get("nonexistent", "default") == "default"

    def test_cancel_key_setitem(self):
        state = DuplicateScanState()
        state["cancel_requested"] = True
        assert state.is_cancelled() is True

    def test_cancel_key_getitem(self):
        state = DuplicateScanState()
        state.request_cancel()
        assert state["cancel_requested"] is True


class TestValidators:
    def test_negative_progress_rejected(self):
        with pytest.raises(Exception):
            DuplicateScanState(progress=-1)

    def test_negative_total_rejected(self):
        with pytest.raises(Exception):
            DuplicateScanState(total=-1)