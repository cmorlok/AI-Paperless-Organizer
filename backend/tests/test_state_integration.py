"""Integration tests for state class consolidation (STATE-08, STATE-09).

Verifies:
- All four state classes are importable from their respective modules
- AppProvider has providers for all four state classes
- reset() preserves object identity for each state class
- model_dump() returns JSON-serializable dicts (no asyncio objects)
- No legacy dict attributes in routers (no _scan_state, _cloud_sync_state, etc.)
- Accessor functions removed from services (no get_scan_state, get_cloud_sync_state)
"""

from __future__ import annotations

import json

from app.services.ocr.state import OcrState
from app.services.classifier.state import AutoClassifyState
from app.services.cloud_import.state import CloudSyncState
from app.services.duplicate.state import DuplicateScanState
from app.container import AppProvider


class TestStateClassesImportable:
    def test_ocr_state_importable(self):
        assert OcrState is not None

    def test_auto_classify_state_importable(self):
        assert AutoClassifyState is not None

    def test_cloud_sync_state_importable(self):
        assert CloudSyncState is not None

    def test_duplicate_scan_state_importable(self):
        assert DuplicateScanState is not None


class TestContainerProviders:
    def test_app_provider_has_ocr_state_provider(self):
        provider = AppProvider()
        # Each provider method is registered via @provide decorator
        # The method name maps to the type it provides
        assert hasattr(provider, "ocr_state")

    def test_app_provider_has_auto_classify_state_provider(self):
        provider = AppProvider()
        assert hasattr(provider, "auto_classify_state")

    def test_app_provider_has_cloud_sync_state_provider(self):
        provider = AppProvider()
        assert hasattr(provider, "cloud_sync_state")

    def test_app_provider_has_duplicate_scan_state_provider(self):
        provider = AppProvider()
        assert hasattr(provider, "duplicate_scan_state")

    def test_ocr_state_provider_returns_ocr_state_instance(self):
        provider = AppProvider()
        result = provider.ocr_state()
        assert isinstance(result, OcrState)

    def test_auto_classify_state_provider_returns_auto_classify_state_instance(self):
        provider = AppProvider()
        result = provider.auto_classify_state()
        assert isinstance(result, AutoClassifyState)

    def test_cloud_sync_state_provider_returns_cloud_sync_state_instance(self):
        provider = AppProvider()
        result = provider.cloud_sync_state()
        assert isinstance(result, CloudSyncState)

    def test_duplicate_scan_state_provider_returns_duplicate_scan_state_instance(self):
        provider = AppProvider()
        result = provider.duplicate_scan_state()
        assert isinstance(result, DuplicateScanState)


class TestResetPreservesObjectIdentity:
    def test_ocr_state_reset_preserves_identity(self):
        state = OcrState()
        state.running = True
        original_id = id(state)
        state.reset()
        assert state.running is False
        assert id(state) == original_id

    def test_auto_classify_state_reset_preserves_identity(self):
        state = AutoClassifyState()
        state.running = True
        state.processed = 5
        original_id = id(state)
        state.reset()
        assert state.running is False
        assert state.processed == 0
        assert id(state) == original_id

    def test_cloud_sync_state_reset_preserves_identity(self):
        state = CloudSyncState()
        state.running = True
        state.files_imported_session = 3
        original_id = id(state)
        state.reset()
        assert state.running is False
        assert state.files_imported_session == 0
        assert id(state) == original_id

    def test_duplicate_scan_state_reset_preserves_identity(self):
        state = DuplicateScanState()
        state.running = True
        state.progress = 50
        original_id = id(state)
        state.reset()
        assert state.running is False
        assert state.progress == 0
        assert id(state) == original_id


class TestModelDumpJsonSerializable:
    def _assert_json_serializable(self, data: dict) -> None:
        """Assert that data can be serialized to JSON without errors."""
        serialized = json.dumps(data)
        assert isinstance(serialized, str)

    def test_ocr_state_model_dump_is_json_serializable(self):
        state = OcrState()
        dumped = state.model_dump()
        self._assert_json_serializable(dumped)

    def test_ocr_state_model_dump_excludes_lock(self):
        state = OcrState()
        dumped = state.model_dump()
        assert "_lock" not in dumped
        assert "lock" not in dumped

    def test_auto_classify_state_model_dump_is_json_serializable(self):
        state = AutoClassifyState()
        dumped = state.model_dump()
        self._assert_json_serializable(dumped)

    def test_auto_classify_state_model_dump_excludes_task(self):
        state = AutoClassifyState()
        dumped = state.model_dump()
        # _task is a PrivateAttr — should not appear in dump
        assert "_task" not in dumped
        assert "task" not in dumped

    def test_cloud_sync_state_model_dump_is_json_serializable(self):
        state = CloudSyncState()
        dumped = state.model_dump()
        self._assert_json_serializable(dumped)

    def test_cloud_sync_state_model_dump_excludes_task(self):
        state = CloudSyncState()
        dumped = state.model_dump()
        # task field has exclude=True
        assert "task" not in dumped

    def test_duplicate_scan_state_model_dump_is_json_serializable(self):
        state = DuplicateScanState()
        dumped = state.model_dump()
        self._assert_json_serializable(dumped)

    def test_auto_classify_state_model_dump_excludes_cancelled(self):
        state = AutoClassifyState()
        state.request_cancel()
        dumped = state.model_dump()
        assert "_cancelled" not in dumped
        assert "cancelled" not in dumped


class TestNoLegacyDictAttributesInRouters:
    def test_duplicates_router_has_no_scan_state_attribute(self):
        import app.routers.duplicates as duplicates_module
        assert not hasattr(duplicates_module, "_scan_state"), \
            "_scan_state dict still present in routers.duplicates"

    def test_cloud_import_router_has_no_cloud_sync_state_attribute(self):
        import app.routers.cloud_import as cloud_import_module
        assert not hasattr(cloud_import_module, "_cloud_sync_state"), \
            "_cloud_sync_state dict still present in routers.cloud_import"

    def test_classifier_router_has_no_auto_classify_state_attribute(self):
        import app.routers.classifier as classifier_module
        assert not hasattr(classifier_module, "_auto_classify_state"), \
            "_auto_classify_state dict still present in routers.classifier"

    def test_ocr_router_has_no_batch_state_attribute(self):
        import app.routers.ocr as ocr_module
        assert not hasattr(ocr_module, "batch_state"), \
            "batch_state still present in routers.ocr"

    def test_ocr_router_has_no_watchdog_state_attribute(self):
        import app.routers.ocr as ocr_module
        assert not hasattr(ocr_module, "watchdog_state"), \
            "watchdog_state still present in routers.ocr"

    def test_ocr_router_has_no_single_ocr_running_attribute(self):
        import app.routers.ocr as ocr_module
        assert not hasattr(ocr_module, "single_ocr_running"), \
            "single_ocr_running still present in routers.ocr"

    def test_ocr_router_has_no_ocr_page_progress_attribute(self):
        import app.routers.ocr as ocr_module
        assert not hasattr(ocr_module, "ocr_page_progress"), \
            "ocr_page_progress still present in routers.ocr"


class TestAccessorFunctionsRemoved:
    def test_duplicate_state_has_no_get_scan_state_function(self):
        import app.services.duplicate.state as dup_state_module
        assert not hasattr(dup_state_module, "get_scan_state"), \
            "get_scan_state accessor function still present in services.duplicate.state"

    def test_cloud_import_state_has_no_get_cloud_sync_state_function(self):
        import app.services.cloud_import.state as cloud_state_module
        assert not hasattr(cloud_state_module, "get_cloud_sync_state"), \
            "get_cloud_sync_state accessor function still present in services.cloud_import.state"
