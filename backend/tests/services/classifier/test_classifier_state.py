"""Tests for AutoClassifyState."""

import pytest

from app.services.classifier.state import AutoClassifyState


class TestAutoClassifyStateFields:
    def test_has_required_fields(self):
        state = AutoClassifyState()
        assert hasattr(state, "processed")
        assert hasattr(state, "errors")
        assert hasattr(state, "reviewed")
        assert hasattr(state, "current_doc")
        assert hasattr(state, "last_run")


class TestReset:
    def test_reset_preserves_identity_and_clears_counters(self):
        state = AutoClassifyState()
        state.processed = 10
        state.errors = 2
        state.reviewed = 5
        state.current_doc = 42
        state.last_run = "2024-01-01"
        original_id = id(state)
        state.reset()
        assert state.processed == 0
        assert state.errors == 0
        assert state.reviewed == 0
        assert state.current_doc is None
        assert state.last_run is None
        assert id(state) == original_id

    def test_reset_clears_cancellation(self):
        state = AutoClassifyState()
        state.request_cancel()
        assert state.is_cancelled()
        state.reset()
        assert not state.is_cancelled()


class TestModelDump:
    def test_model_dump_excludes_task(self):
        state = AutoClassifyState()
        state._task = "some_task"
        dumped = state.model_dump()
        assert "_task" not in dumped


class TestValidators:
    def test_negative_processed_rejected(self):
        with pytest.raises(Exception):
            AutoClassifyState(processed=-1)

    def test_negative_errors_rejected(self):
        with pytest.raises(Exception):
            AutoClassifyState(errors=-1)

    def test_negative_reviewed_rejected(self):
        with pytest.raises(Exception):
            AutoClassifyState(reviewed=-1)

    def test_zero_processed_allowed(self):
        state = AutoClassifyState(processed=0)
        assert state.processed == 0