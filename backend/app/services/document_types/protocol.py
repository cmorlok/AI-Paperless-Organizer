"""Protocol for document types business-logic service."""

from typing import Dict, Protocol, runtime_checkable


@runtime_checkable
class DocumentTypesService(Protocol):
    """Protocol for document types business-logic service."""

    async def estimate_document_types(self) -> Dict: ...

    async def save_similarity_analysis(self, result: Dict) -> None: ...

    async def delete_empty_document_types(self) -> Dict: ...
