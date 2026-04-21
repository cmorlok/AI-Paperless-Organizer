"""Phase 2 Dependency Injection -- Smoke Tests"""
import pytest
import os
import sys
import importlib
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDIFoundation:
    def test_app_imports_cleanly(self):
        from app.main import app
        assert app is not None
        assert hasattr(app, 'container')

    def test_container_has_all_providers(self):
        from app.container import AppProvider
        provider = AppProvider()
        # Count @provide-decorated methods
        import inspect
        methods = [m for m in dir(provider) if not m.startswith('_')]
        assert len(methods) >= 11  # 10 services + session_factory

    def test_no_llm_factory_function(self):
        from app.services import llm_service
        assert not hasattr(llm_service, 'get_llm_service'), "get_llm_service still exists"

    def test_no_paperless_client_factory(self):
        from app.services import paperless_client
        assert not hasattr(paperless_client, 'get_paperless_client'), "get_paperless_client still exists"

    def test_no_similarity_factory(self):
        from app.services import similarity
        assert not hasattr(similarity, 'get_similarity_service'), "get_similarity_service still exists"

    def test_no_merge_factory(self):
        from app.services import merge
        assert not hasattr(merge, 'get_merge_service'), "get_merge_service still exists"

    def test_no_statistics_factory(self):
        from app.services import statistics
        assert not hasattr(statistics, 'get_statistics_service'), "get_statistics_service still exists"

    def test_no_cloud_import_factory(self):
        from app.services import cloud_import_service
        assert not hasattr(cloud_import_service, 'get_cloud_import_service'), "get_cloud_import_service still exists"

    def test_all_routers_importable(self):
        from app.routers import paperless, correspondents, tags, document_types
        from app.routers import settings, llm, debug, statistics, ignored_items
        from app.routers import api_keys, ocr, cleanup, classifier, rag
        from app.routers import cloud_import, duplicates
        assert all(r.router is not None for r in [
            paperless, correspondents, tags, document_types, settings,
            llm, debug, statistics, ignored_items, api_keys, ocr,
            cleanup, classifier, rag, cloud_import, duplicates
        ])

    def test_services_implement_protocols(self):
        from app.services.protocols import LLMService, PaperlessClient
        from app.services.llm_service import LitellmService
        from app.services.paperless_client import PaperlessClient as PC
        assert hasattr(LitellmService, 'complete')
        assert hasattr(PC, 'test_connection')

    def test_no_factory_functions_remain(self):
        """Scan all service files for remaining get_* factory functions."""
        import ast
        backend_dir = Path(__file__).parent.parent / "app"
        factory_patterns = [
            "get_llm_service", "get_paperless_client", "get_similarity_service",
            "get_merge_service", "get_statistics_service", "get_cloud_import_service",
            "get_ocr_service", "get_rag_service"
        ]
        violations = []
        for py_file in backend_dir.rglob("*.py"):
            if py_file.name.startswith("test_"):
                continue
            try:
                tree = ast.parse(py_file.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name in factory_patterns:
                        violations.append(f"{py_file.relative_to(backend_dir)}:{node.name}")
            except Exception:
                pass
        assert not violations, f"Factory functions still exist: {violations}"
