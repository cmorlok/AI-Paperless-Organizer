import ast
import inspect
from pathlib import Path

import pytest


class TestDuplicateServiceDIGaps:
    def test_no_internal_get_paperless_client(self):
        """Verify _get_paperless_client is removed from DuplicateService."""
        src = Path(__file__).parent.parent / "app" / "services" / "duplicate_service.py"
        text = src.read_text()
        assert "_get_paperless_client" not in text
        assert "PaperlessClient(base_url=" not in text

    def test_duplicate_service_accepts_paperless_client(self):
        """Verify DuplicateService.__init__ takes paperless_client."""
        from app.services.duplicate_service import DuplicateService
        sig = inspect.signature(DuplicateService.__init__)
        params = list(sig.parameters.keys())
        assert "paperless_client" in params
        assert "session_factory" in params

    def test_duplicate_service_stores_paperless_client(self):
        """Verify DuplicateService stores paperless_client on self."""
        from app.services.duplicate_service import DuplicateService
        src = inspect.getsource(DuplicateService)
        assert "self.paperless_client" in src


class TestRAGServiceDIGaps:
    def test_no_internal_get_paperless_client_in_indexer(self):
        """Verify _get_paperless_client is removed from Indexer."""
        src = Path(__file__).parent.parent / "app" / "services" / "rag" / "indexer.py"
        text = src.read_text()
        assert "_get_paperless_client" not in text
        assert "PaperlessClient(base_url=" not in text

    def test_indexer_accepts_paperless_client(self):
        """Verify Indexer.__init__ takes paperless_client."""
        from app.services.rag.indexer import Indexer
        sig = inspect.signature(Indexer.__init__)
        params = list(sig.parameters.keys())
        assert "paperless_client" in params
        assert "search_engine" in params

    def test_rag_service_accepts_paperless_client(self):
        """Verify RAGService.__init__ takes paperless_client."""
        from app.services.rag.service import RAGService
        sig = inspect.signature(RAGService.__init__)
        params = list(sig.parameters.keys())
        assert "paperless_client" in params
        assert "session_factory" in params

    def test_rag_service_passes_paperless_client_to_indexer(self):
        """Verify RAGService forwards paperless_client to Indexer."""
        from app.services.rag.service import RAGService
        src = inspect.getsource(RAGService)
        assert "Indexer(self.search_engine, paperless_client=paperless_client)" in src


class TestContainerDIGaps:
    def test_duplicate_service_provider_passes_paperless_client(self):
        """Verify container provider for duplicate_service passes paperless_client."""
        src = Path(__file__).parent.parent / "app" / "container.py"
        text = src.read_text()
        # Locate the duplicate_service method and verify it passes paperless_client
        assert "def duplicate_service(" in text
        assert "DuplicateServiceImpl(" in text
        assert "paperless_client=paperless_client" in text

    def test_rag_service_provider_passes_paperless_client(self):
        """Verify container provider for rag_service passes paperless_client."""
        src = Path(__file__).parent.parent / "app" / "container.py"
        text = src.read_text()
        # Locate the rag_service method and verify it passes paperless_client
        assert "def rag_service(" in text
        assert "RAGServiceImpl(" in text
        assert "paperless_client=paperless_client" in text
