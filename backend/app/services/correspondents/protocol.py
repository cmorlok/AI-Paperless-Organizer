"""Protocol for correspondents business-logic service."""

from typing import Dict, Protocol, runtime_checkable


@runtime_checkable
class CorrespondentsService(Protocol):
    """Protocol for correspondents business-logic service."""

    async def estimate_correspondents(self) -> Dict: ...

    async def save_similarity_analysis(self, result: Dict) -> None: ...

    async def delete_empty_correspondents(self) -> Dict: ...
