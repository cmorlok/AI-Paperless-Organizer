"""Tests for the GET /api/debug/services endpoint (STATE-08).

Verifies:
- The get_service_statuses handler function returns the expected schema
- Response has "services" key containing a list of exactly 4 items
- Each item has keys: name, label, enabled, running, current_op, detail
- Service names are: ocr, classifier, cloud_import, duplicate
- German labels are present: OCR, Klassifizierung, Cloud-Import, Duplikat-Scan
- duplicate service has enabled=False (on-demand, not a daemon)
- model_dump() detail fields contain no asyncio objects (JSON-serializable)
- _ocr_current_op helper returns correct strings for various states
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.services.ocr.state import OcrState
from app.services.classifier.state import AutoClassifyState
from app.services.cloud_import.state import CloudSyncState
from app.services.duplicate.state import DuplicateScanState
from app.routers.debug import _ocr_current_op, get_service_statuses


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _call_handler(
    ocr_state: OcrState | None = None,
    classify_state: AutoClassifyState | None = None,
    cloud_state: CloudSyncState | None = None,
    dup_state: DuplicateScanState | None = None,
) -> dict:
    """Call the endpoint handler directly with state instances (bypass Dishka).

    Uses __dishka_orig_func__ to bypass the @inject decorator which requires
    a FastAPI request context with a live Dishka container.
    """
    ocr_state = ocr_state or OcrState()
    classify_state = classify_state or AutoClassifyState()
    cloud_state = cloud_state or CloudSyncState()
    dup_state = dup_state or DuplicateScanState()
    orig_func = getattr(get_service_statuses, "__dishka_orig_func__", get_service_statuses)
    return await orig_func(
        ocr_state=ocr_state,
        classify_state=classify_state,
        cloud_state=cloud_state,
        dup_state=dup_state,
    )


# ── Tests for response structure ──────────────────────────────────────────────

class TestResponseStructure:
    @pytest.mark.asyncio
    async def test_response_has_services_key(self):
        result = await _call_handler()
        assert "services" in result

    @pytest.mark.asyncio
    async def test_services_is_a_list(self):
        result = await _call_handler()
        assert isinstance(result["services"], list)

    @pytest.mark.asyncio
    async def test_services_list_has_exactly_4_items(self):
        result = await _call_handler()
        assert len(result["services"]) == 4

    @pytest.mark.asyncio
    async def test_each_service_has_required_keys(self):
        result = await _call_handler()
        required_keys = {"name", "label", "enabled", "running", "current_op", "detail"}
        for service in result["services"]:
            assert required_keys.issubset(service.keys()), \
                f"Service {service.get('name')} missing keys: {required_keys - service.keys()}"


class TestServiceNames:
    @pytest.mark.asyncio
    async def test_service_names_are_correct(self):
        result = await _call_handler()
        names = [s["name"] for s in result["services"]]
        assert names == ["ocr", "classifier", "cloud_import", "duplicate"]

    @pytest.mark.asyncio
    async def test_ocr_service_name(self):
        result = await _call_handler()
        ocr = next(s for s in result["services"] if s["name"] == "ocr")
        assert ocr["name"] == "ocr"

    @pytest.mark.asyncio
    async def test_classifier_service_name(self):
        result = await _call_handler()
        svc = next(s for s in result["services"] if s["name"] == "classifier")
        assert svc["name"] == "classifier"

    @pytest.mark.asyncio
    async def test_cloud_import_service_name(self):
        result = await _call_handler()
        svc = next(s for s in result["services"] if s["name"] == "cloud_import")
        assert svc["name"] == "cloud_import"

    @pytest.mark.asyncio
    async def test_duplicate_service_name(self):
        result = await _call_handler()
        svc = next(s for s in result["services"] if s["name"] == "duplicate")
        assert svc["name"] == "duplicate"


class TestGermanLabels:
    @pytest.mark.asyncio
    async def test_ocr_label_is_german(self):
        result = await _call_handler()
        ocr = next(s for s in result["services"] if s["name"] == "ocr")
        assert ocr["label"] == "OCR"

    @pytest.mark.asyncio
    async def test_classifier_label_is_german(self):
        result = await _call_handler()
        svc = next(s for s in result["services"] if s["name"] == "classifier")
        assert svc["label"] == "Klassifizierung"

    @pytest.mark.asyncio
    async def test_cloud_import_label_is_german(self):
        result = await _call_handler()
        svc = next(s for s in result["services"] if s["name"] == "cloud_import")
        assert svc["label"] == "Cloud-Import"

    @pytest.mark.asyncio
    async def test_duplicate_label_is_german(self):
        result = await _call_handler()
        svc = next(s for s in result["services"] if s["name"] == "duplicate")
        assert svc["label"] == "Duplikat-Scan"


class TestDuplicateServiceIsOnDemand:
    @pytest.mark.asyncio
    async def test_duplicate_enabled_is_false(self):
        """Duplicate scan is on-demand, not a daemon — enabled must always be False."""
        dup_state = DuplicateScanState()
        dup_state.running = True  # Even when running, enabled is hardcoded False
        result = await _call_handler(dup_state=dup_state)
        svc = next(s for s in result["services"] if s["name"] == "duplicate")
        assert svc["enabled"] is False

    @pytest.mark.asyncio
    async def test_duplicate_running_reflects_state(self):
        dup_state = DuplicateScanState()
        dup_state.running = True
        result = await _call_handler(dup_state=dup_state)
        svc = next(s for s in result["services"] if s["name"] == "duplicate")
        assert svc["running"] is True


class TestDetailFieldsJsonSerializable:
    @pytest.mark.asyncio
    async def test_all_detail_fields_are_json_serializable(self):
        result = await _call_handler()
        for service in result["services"]:
            detail = service["detail"]
            try:
                json.dumps(detail)
            except (TypeError, ValueError) as e:
                pytest.fail(
                    f"Service '{service['name']}' detail is not JSON-serializable: {e}"
                )

    @pytest.mark.asyncio
    async def test_ocr_detail_has_no_asyncio_lock(self):
        result = await _call_handler()
        ocr = next(s for s in result["services"] if s["name"] == "ocr")
        detail_str = json.dumps(ocr["detail"])
        assert "Lock" not in detail_str
        assert "asyncio" not in detail_str

    @pytest.mark.asyncio
    async def test_classifier_detail_has_no_asyncio_task(self):
        classify_state = AutoClassifyState()
        # Assign a real asyncio Task to verify it's excluded
        loop = asyncio.get_event_loop()
        async def _noop():
            pass
        task = loop.create_task(_noop())
        classify_state.task = task
        result = await _call_handler(classify_state=classify_state)
        svc = next(s for s in result["services"] if s["name"] == "classifier")
        detail_str = json.dumps(svc["detail"])
        assert "Task" not in detail_str
        assert "asyncio" not in detail_str
        task.cancel()


# ── Tests for _ocr_current_op helper ─────────────────────────────────────────

class TestOcrCurrentOp:
    def test_returns_none_when_idle(self):
        state = OcrState()
        result = _ocr_current_op(state)
        assert result is None

    def test_returns_batch_info_when_batch_running_with_document(self):
        state = OcrState()
        state.batch.running = True
        state.batch.current_document = {"title": "Rechnung 2024"}
        result = _ocr_current_op(state)
        assert result == "Batch: Rechnung 2024"

    def test_returns_none_when_batch_running_but_no_document(self):
        """An empty dict is falsy — no current_op when no document is set."""
        state = OcrState()
        state.batch.running = True
        state.batch.current_document = None
        result = _ocr_current_op(state)
        assert result is None

    def test_returns_watchdog_info_when_watchdog_running(self):
        state = OcrState()
        state.watchdog.running = True
        state.watchdog.interval_minutes = 10
        result = _ocr_current_op(state)
        assert result == "Watchdog (Intervall: 10min)"

    def test_returns_single_ocr_info_when_locked(self):
        state = OcrState()
        state.lock_holder = "batch"
        result = _ocr_current_op(state)
        assert result == "Einzel-OCR (batch)"

    def test_batch_takes_precedence_over_watchdog(self):
        """When batch is running with a document, that is shown first."""
        state = OcrState()
        state.batch.running = True
        state.batch.current_document = {"title": "Doc"}
        state.watchdog.running = True
        result = _ocr_current_op(state)
        assert result is not None
        assert result.startswith("Batch:")


class TestCurrentOpForAllServices:
    @pytest.mark.asyncio
    async def test_ocr_current_op_is_none_when_idle(self):
        result = await _call_handler()
        ocr = next(s for s in result["services"] if s["name"] == "ocr")
        assert ocr["current_op"] is None

    @pytest.mark.asyncio
    async def test_classifier_current_op_is_none_when_no_doc(self):
        result = await _call_handler()
        svc = next(s for s in result["services"] if s["name"] == "classifier")
        assert svc["current_op"] is None

    @pytest.mark.asyncio
    async def test_classifier_current_op_shows_doc_id_when_processing(self):
        classify_state = AutoClassifyState()
        classify_state.current_doc = 42
        result = await _call_handler(classify_state=classify_state)
        svc = next(s for s in result["services"] if s["name"] == "classifier")
        assert svc["current_op"] == "Dokument 42"

    @pytest.mark.asyncio
    async def test_cloud_import_current_op_shows_filename(self):
        cloud_state = CloudSyncState()
        cloud_state.current_file = "dokument.pdf"
        result = await _call_handler(cloud_state=cloud_state)
        svc = next(s for s in result["services"] if s["name"] == "cloud_import")
        assert svc["current_op"] == "dokument.pdf"

    @pytest.mark.asyncio
    async def test_duplicate_current_op_is_none_when_not_running(self):
        dup_state = DuplicateScanState()
        dup_state.phase = "exact"
        dup_state.running = False
        result = await _call_handler(dup_state=dup_state)
        svc = next(s for s in result["services"] if s["name"] == "duplicate")
        assert svc["current_op"] is None

    @pytest.mark.asyncio
    async def test_duplicate_current_op_shows_phase_when_running(self):
        dup_state = DuplicateScanState()
        dup_state.running = True
        dup_state.phase = "exact"
        result = await _call_handler(dup_state=dup_state)
        svc = next(s for s in result["services"] if s["name"] == "duplicate")
        assert svc["current_op"] == "exact"
