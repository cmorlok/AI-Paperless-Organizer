"""Test that no module outside container.py imports concrete service classes."""

import ast
import pytest
from pathlib import Path

# Map of concrete modules -> banned class names
BANNED_IMPORTS = {
    "app.services.paperless_client": ["PaperlessClient"],
    "app.services.ocr_service": ["OcrService"],
    "app.services.llm_service": ["LitellmService"],
    "app.services.classifier.service": ["DocumentClassifierService"],
    "app.services.rag.service": ["RAGService"],
    "app.services.duplicate_service": ["DuplicateService"],
    "app.services.cloud_import_service": ["CloudImportService"],
    "app.services.similarity": ["SimilarityService"],
    "app.services.merge": ["MergeService"],
    "app.services.statistics": ["StatisticsService"],
}

ALLOWED_FILES = {
    # The container is the ONLY place allowed to import concrete implementations
    "backend/app/container.py",
}


def _find_source_files():
    root = Path(__file__).parent.parent / "app"
    for path in root.rglob("*.py"):
        rel = path.relative_to(Path(__file__).parent.parent.parent)
        if str(rel) in ALLOWED_FILES:
            continue
        if "test_" in path.name or path.name.startswith("test"):
            continue
        yield path


def _get_imported_names(tree: ast.AST) -> list[tuple[str, str]]:
    """Return list of (module, name) for all imports in the AST."""
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                results.append((module, alias.name))
    return results


@pytest.mark.parametrize("source_file", list(_find_source_files()))
def test_no_concrete_service_imports(source_file: Path):
    tree = ast.parse(source_file.read_text())
    imports = _get_imported_names(tree)

    violations = []
    for module, name in imports:
        for banned_module, banned_names in BANNED_IMPORTS.items():
            if module == banned_module and name in banned_names:
                violations.append(f"{source_file.name}: imports {name} from {module}")

    assert not violations, "Concrete service imports found:\n" + "\n".join(violations)


def test_routers_import_from_protocols():
    """All routers must import service types from protocol files.

    Protocols are distributed to service sub-packages (e.g., app.services.paperless.protocol).
    """
    routers_dir = Path(__file__).parent.parent / "app" / "routers"
    for router_file in routers_dir.glob("*.py"):
        if router_file.name == "__init__.py":
            continue
        tree = ast.parse(router_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("app.services.") and not module.endswith(".protocol"):
                    # Allow protocol imports from sub-packages
                    for alias in node.names:
                        if alias.name in sum(BANNED_IMPORTS.values(), []):
                            pytest.fail(
                                f"{router_file.name}: imports {alias.name} from {module} — "
                                f"must import from protocol file (e.g., app.services.X.protocol)"
                            )
