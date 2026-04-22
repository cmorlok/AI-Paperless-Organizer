"""Protocol for statistics service."""

from typing import Protocol, runtime_checkable, Any, Dict, List


@runtime_checkable
class StatisticsService(Protocol):
    """Protocol for statistics service."""

    async def record_operation(
        self,
        entity_type: str,
        operation: str,
        items_affected: int,
        documents_affected: int = 0,
        items_before: int = 0,
        items_after: int = 0,
        details: dict = None,
    ) -> Any: ...

    async def get_total_stats(self) -> Dict: ...

    async def get_recent_operations(self, limit: int = 10) -> List[Dict]: ...

    async def get_daily_trend(self, days: int = 7) -> List[Dict]: ...