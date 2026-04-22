"""Protocol for similarity detection service."""

from typing import Protocol, runtime_checkable, Dict


@runtime_checkable
class SimilarityService(Protocol):
    """Protocol for similarity detection service."""

    async def find_similar_correspondents(self, batch_size: int = 200) -> Dict: ...

    async def find_similar_tags(self, batch_size: int = 200) -> Dict: ...

    async def find_similar_document_types(self, batch_size: int = 200) -> Dict: ...

    async def find_nonsense_tags(self, batch_size: int = 300) -> Dict: ...

    async def find_tags_that_are_correspondents(self, batch_size: int = 300) -> Dict: ...

    async def find_tags_that_are_document_types(self, batch_size: int = 300) -> Dict: ...