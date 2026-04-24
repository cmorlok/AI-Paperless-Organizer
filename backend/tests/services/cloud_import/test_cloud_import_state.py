"""Tests for CloudSyncState."""

import pytest

from app.services.cloud_import.state import CloudSyncState


class TestFields:
    def test_has_required_fields(self):
        state = CloudSyncState()
        assert hasattr(state, "current_source_id")
        assert hasattr(state, "current_source_name")
        assert hasattr(state, "current_file")
        assert hasattr(state, "last_run")
        assert hasattr(state, "files_imported_session")
        assert hasattr(state, "errors_session")


class TestReset:
    def test_reset_preserves_identity_and_clears(self):
        state = CloudSyncState()
        state.current_source_id = 42
        state.current_source_name = "test"
        state.files_imported_session = 5
        state.errors_session = 2
        original_id = id(state)
        state.reset()
        assert state.current_source_id is None
        assert state.current_source_name is None
        assert state.files_imported_session == 0
        assert state.errors_session == 0
        assert id(state) == original_id


class TestModelDump:
    def test_excludes_task(self):
        state = CloudSyncState()
        state.task = "some_task"
        dumped = state.model_dump()
        assert "task" not in dumped


class TestValidators:
    def test_negative_files_imported_rejected(self):
        with pytest.raises(Exception):
            CloudSyncState(files_imported_session=-1)

    def test_negative_errors_rejected(self):
        with pytest.raises(Exception):
            CloudSyncState(errors_session=-1)

    def test_zero_values_allowed(self):
        state = CloudSyncState(files_imported_session=0, errors_session=0)
        assert state.files_imported_session == 0
        assert state.errors_session == 0