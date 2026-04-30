"""Test that no module outside container.py imports concrete service classes."""

import ast
import pytest
from pathlib import Path

# Map of concrete modules -> banned class names
# After Phase 04, concrete implementations live in *.service sub-modules.
# Package-level imports (e.g., from app.services.similarity import SimilarityService)
# are VALID protocol imports via __init__.py re-exports.
BANNED_IMPORTS = {
    "app.services.paperless.service": ["PaperlessClient"],
    "app.services.ocr.service": ["OcrService"],
    "app.services.llm.service": ["LitellmService"],
    "app.services.classifier.service": ["DocumentClassifierService"],
    "app.services.rag.service": ["RAGService"],
    "app.services.duplicate.service": ["DuplicateService"],
    "app.services.cloud_import.service": ["CloudImportService"],
    "app.services.similarity.service": ["SimilarityService"],
    "app.services.merge.service": ["MergeService"],
    "app.services.statistics.service": ["StatisticsService"],
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
    """All routers must import service types from protocol files or package-level __init__.

    After Phase 04, services export from __init__.py (e.g., app.services.classifier),
    which re-exports from protocol.py. Both patterns are valid.
    """
    routers_dir = Path(__file__).parent.parent / "app" / "routers"
    for router_file in routers_dir.glob("*.py"):
        if router_file.name == "__init__.py":
            continue
        tree = ast.parse(router_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # Only flag direct .service imports (not package-level imports)
                if module.endswith(".service") and module.startswith("app.services."):
                    for alias in node.names:
                        if alias.name in sum(BANNED_IMPORTS.values(), []):
                            pytest.fail(
                                f"{router_file.name}: imports {alias.name} from {module} — "
                                f"must import from package level (e.g., app.services.X) or protocol"
                            )
