"""Protocol for cloud import service."""

from typing import Protocol, runtime_checkable, Any, Dict

# Import PaperlessClient from old path during Wave 1 (Plan 07 handles final path alignment)
from app.services.protocols import PaperlessClient


@runtime_checkable
class CloudImportService(Protocol):
    """Protocol for cloud import service."""

    async def sync_source(self, source: Any, pl_client: PaperlessClient, db: Any) -> Dict: ...

    async def test_connection(self, source: Any) -> Dict: ...

    async def sync_all_sources(self, pl_client: PaperlessClient) -> Dict: ...