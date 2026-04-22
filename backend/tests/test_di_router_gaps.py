"""Phase 2 Dependency Injection -- Router Gap Closure Tests"""
import ast
from pathlib import Path

import pytest


def _get_function_source(text: str, func_name: str) -> str:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            lines = text.splitlines()
            # Include decorators above the function definition
            start = node.lineno - 1
            for decorator in node.decorator_list:
                start = min(start, decorator.lineno - 1)
            end = node.end_lineno
            return "\n".join(lines[start:end])
    raise ValueError(f"Function {func_name} not found")


class TestCloudImportDIGaps:
    def test_no_direct_paperless_client_instantiation(self):
        """Verify no PaperlessClient(base_url=...) remains in cloud_import.py."""
        src = Path(__file__).parent.parent / "app" / "routers" / "cloud_import.py"
        text = src.read_text()
        assert "PaperlessClient(base_url=" not in text
        assert "PaperlessSettings" not in text

    def test_paperless_tags_uses_inject(self):
        """Verify /paperless/tags uses @inject + FromDishka[PaperlessClient]."""
        src = Path(__file__).parent.parent / "app" / "routers" / "cloud_import.py"
        text = src.read_text()
        func_src = _get_function_source(text, "get_paperless_tags")
        assert "@inject" in func_src
        assert "FromDishka[PaperlessClient]" in func_src
        assert "PaperlessClient(base_url=" not in func_src

    def test_paperless_correspondents_uses_inject(self):
        """Verify /paperless/correspondents uses @inject + FromDishka[PaperlessClient]."""
        src = Path(__file__).parent.parent / "app" / "routers" / "cloud_import.py"
        text = src.read_text()
        func_src = _get_function_source(text, "get_paperless_correspondents")
        assert "@inject" in func_src
        assert "FromDishka[PaperlessClient]" in func_src
        assert "PaperlessClient(base_url=" not in func_src

    def test_paperless_document_types_uses_inject(self):
        """Verify /paperless/document-types uses @inject + FromDishka[PaperlessClient]."""
        src = Path(__file__).parent.parent / "app" / "routers" / "cloud_import.py"
        text = src.read_text()
        func_src = _get_function_source(text, "get_paperless_document_types")
        assert "@inject" in func_src
        assert "FromDishka[PaperlessClient]" in func_src
        assert "PaperlessClient(base_url=" not in func_src


class TestDuplicatesDIGaps:
    def test_no_direct_duplicate_service_instantiation(self):
        """Verify no DuplicateService() remains in duplicates.py."""
        src = Path(__file__).parent.parent / "app" / "routers" / "duplicates.py"
        text = src.read_text()
        assert "DuplicateService()" not in text

    def test_start_scan_uses_inject(self):
        """Verify /scan uses @inject + FromDishka[DuplicateService]."""
        src = Path(__file__).parent.parent / "app" / "routers" / "duplicates.py"
        text = src.read_text()
        func_src = _get_function_source(text, "start_scan")
        assert "@inject" in func_src
        assert "FromDishka[DuplicateService]" in func_src
        assert "DuplicateService()" not in func_src
