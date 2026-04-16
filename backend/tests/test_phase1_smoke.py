"""Phase 1 LiteLLM Migration — Smoke Tests

Verifies all consumer imports resolve, API endpoints still work, and
custom provider files have been deleted. Run after Plan 01-05 execution.
"""
import pytest
import os
import sys

# Ensure backend is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestPhase1Smoke:
    """Smoke tests for Phase 1 LiteLLM migration completion."""

    def test_llm_provider_deleted(self):
        """LLM-06: llm_provider.py must not exist."""
        import importlib
        with pytest.raises(ImportError):
            importlib.import_module("app.services.llm_provider")

    def test_openai_provider_deleted(self):
        """LLM-06: openai_provider.py must not exist in classifier."""
        from pathlib import Path
        p = Path(__file__).parent.parent / "app" / "services" / "classifier" / "openai_provider.py"
        assert not p.exists(), f"openai_provider.py still exists at {p}"

    def test_ollama_provider_deleted(self):
        """LLM-06: ollama_provider.py must not exist in classifier."""
        from pathlib import Path
        p = Path(__file__).parent.parent / "app" / "services" / "classifier" / "ollama_provider.py"
        assert not p.exists(), f"ollama_provider.py still exists at {p}"

    def test_litellm_service_importable(self):
        """LLM-01: litellm_service.py must be importable with core methods."""
        from app.services.litellm_service import LitellmService, get_llm_service

        # Verify core methods exist
        assert hasattr(LitellmService, "complete")
        assert hasattr(LitellmService, "test_connection")
        assert hasattr(LitellmService, "estimate_tokens")
        assert hasattr(LitellmService, "analyze_for_similarity")

        # Verify estimate_tokens works
        s = LitellmService()
        tokens = s.estimate_tokens("hello world test string")
        assert isinstance(tokens, int)
        assert tokens > 0

        print("PASS: LitellmService importable with core methods")

    def test_all_6_consumers_updated(self):
        """LLM-01: All 6 consumer files must import from litellm_service."""
        consumer_files = [
            "app/routers/llm.py",
            "app/routers/tags.py",
            "app/routers/correspondents.py",
            "app/routers/document_types.py",
            "app/routers/ocr.py",
            "app/services/similarity.py",
        ]
        from pathlib import Path
        backend_root = Path(__file__).parent.parent

        for rel_path in consumer_files:
            file_path = backend_root / rel_path
            content = file_path.read_text()
            assert "from app.services.litellm_service import" in content, \
                f"{rel_path} does not import from litellm_service"
            assert "from app.services.llm_provider import" not in content, \
                f"{rel_path} still imports from llm_provider"

        print(f"PASS: All {len(consumer_files)} consumer files updated")

    def test_all_routers_importable(self):
        """All routers must import without error."""
        from app.routers.llm import router as llm_router
        from app.routers.tags import router as tags_router
        from app.routers.correspondents import router as corr_router
        from app.routers.document_types import router as dtype_router
        from app.routers.ocr import router as ocr_router

        assert llm_router is not None
        assert tags_router is not None
        assert corr_router is not None
        assert dtype_router is not None
        assert ocr_router is not None

        print("PASS: All 5 routers importable")

    def test_classifier_service_importable(self):
        """Classifier service must import without error (uses litellm_provider)."""
        from app.services.classifier.service import DocumentClassifierService
        assert DocumentClassifierService is not None
        print("PASS: DocumentClassifierService importable")

    def test_rag_service_importable(self):
        """RAG service must import without error (uses litellm)."""
        from app.services.rag.service import RAGService
        assert RAGService is not None
        print("PASS: RAGService importable")

    def test_duplicate_service_importable(self):
        """Duplicate service must import without error (uses litellm)."""
        from app.services.duplicate_service import DuplicateService
        assert DuplicateService is not None
        print("PASS: DuplicateService importable")

    def test_ocr_service_importable(self):
        """OCR service must import without error (uses litellm)."""
        from app.services.ocr_service import batch_state
        assert batch_state is not None
        print("PASS: OCR service importable")

    def test_app_main_loads(self):
        """Full app must load without error."""
        from app.main import app
        assert app is not None
        print("PASS: Full app loads")

    def test_no_direct_openai_sdk_imports_in_services(self):
        """Verify no direct 'from openai import' in the 6 migrated consumer files.

        Note: Other files (routers/classifier.py, services/rag/embedding_service.py) are
        out of scope for this plan — they'll be handled in subsequent plans.
        """
        from pathlib import Path
        import subprocess

        # Only check the 6 files migrated in this plan
        migrated_files = [
            "app/routers/llm.py",
            "app/routers/tags.py",
            "app/routers/correspondents.py",
            "app/routers/document_types.py",
            "app/routers/ocr.py",
            "app/services/similarity.py",
        ]
        backend_root = Path(__file__).parent.parent
        lines = []
        for pattern in ["from openai import", "from anthropic import"]:
            for rel_path in migrated_files:
                abs_path = backend_root / rel_path
                if abs_path.exists():
                    content = abs_path.read_text()
                    for line_num, line in enumerate(content.splitlines(), 1):
                        if pattern in line and "__pycache__" not in line:
                            lines.append(f"{rel_path}:{line_num}: {line.strip()}")

        assert len(lines) == 0, f"Found direct SDK import in migrated files: {lines}"
        print("PASS: No direct openai/anthropic SDK imports in migrated files")

    def test_litellm_in_requirements(self):
        """Verify litellm is in requirements.txt."""
        from pathlib import Path
        req_file = Path(__file__).parent.parent / "requirements.txt"
        content = req_file.read_text()
        assert "litellm" in content, "litellm not found in requirements.txt"
        print("PASS: litellm in requirements.txt")
