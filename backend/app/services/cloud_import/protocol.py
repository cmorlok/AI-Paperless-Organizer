"""Protocol for cloud import service."""

from typing import Protocol, runtime_checkable, Any, Dict

# Import PaperlessClient — cross-service, use package-level
from app.services.paperless import PaperlessClient


@runtime_checkable
class CloudImportService(Protocol):
    """Protocol for cloud import service."""

    async def sync_source(self, source: Any, pl_client: PaperlessClient, db: Any) -> Dict: ...

    async def test_connection(self, source: Any) -> Dict: ...

    async def list_files(self, source: Any) -> list[Dict]: ...