"""Tests for OcrState and nested models."""

import asyncio

import pytest

from app.services.ocr.state import (
    OcrState,
    OcrBatchProgress,
    OcrDocumentProgress,
    OcrProcessorProgress,
    PageProgress,
)


class TestOcrStateFields:
    def test_ocr_state_has_batch_processor_page_progress(self):
        state = OcrState()
        assert hasattr(state, "batch")
        assert hasattr(state, "processor")
        assert hasattr(state, "page_progress")
        assert isinstance(state.batch, OcrBatchProgress)
        assert isinstance(state.processor, OcrProcessorProgress)
        assert isinstance(state.page_progress, dict)


class TestOcrStateReset:
    def test_reset_clears_batch_and_processor_preserves_identity(self):
        state = OcrState()
        state.batch.running = True
        state.batch.total = 10
        state.batch.processed = 5
        state.processor.enabled = True
        state.processor.running = True
        state.page_progress[1] = OcrDocumentProgress(total_pages=3, done=1)
        original_id = id(state)
        state.reset()
        assert state.batch.running is False
        assert state.batch.total == 0
        assert state.batch.processed == 0
        assert state.processor.enabled is False
        assert state.processor.running is False
        assert len(state.page_progress) == 0
        assert id(state) == original_id

    def test_reset_clears_cancellation(self):
        state = OcrState()
        state.request_cancel()
        assert state.is_cancelled()
        state.reset()
        assert not state.is_cancelled()


class TestPageProgressValidation:
    def test_page_must_be_ge_1(self):
        with pytest.raises(Exception):
            PageProgress(page=0)

    def test_page_ge_1_ok(self):
        p = PageProgress(page=1)
        assert p.page == 1

    def test_chars_must_be_ge_0(self):
        with pytest.raises(Exception):
            PageProgress(page=1, chars=-1)

    def test_chars_ge_0_ok(self):
        p = PageProgress(page=1, chars=0)
        assert p.chars == 0


class TestOcrDocumentProgressValidation:
    def test_done_and_total_pages_ge_0(self):
        dp = OcrDocumentProgress(total_pages=0, done=0)
        assert dp.total_pages == 0
        assert dp.done == 0

    def test_negative_total_pages_rejected(self):
        with pytest.raises(Exception):
            OcrDocumentProgress(total_pages=-1)


class TestModelDumpExclusion:
    def test_model_dump_excludes_lock(self):
        state = OcrState()
        dumped = state.model_dump()
        assert "_lock" not in dumped

    def test_model_dump_excludes_processor_task(self):
        state = OcrState()
        dumped = state.model_dump()
        processor_dumped = dumped.get("processor", {})
        assert "task" not in processor_dumped


class TestOcrStateCancel:
    def test_cancel_sets_cancellation_and_batch_should_stop(self):
        state = OcrState()
        state.cancel()
        assert state.is_cancelled()
        assert state.batch.should_stop is True
