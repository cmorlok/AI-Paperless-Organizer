"""Protocol for RAG service."""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Any, Optional, Dict, List, AsyncIterator, AsyncGenerator


@runtime_checkable
class RAGService(Protocol):
    """Protocol for RAG service."""

    @property
    def indexer(self) -> Any: ...

    def chat_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        filters: Optional[Dict] = None,
    ) -> AsyncGenerator[str, None]: ...

    async def search(self, query: str, limit: int = 5, filters: Optional[Dict] = None) -> List[Any]: ...

    async def get_sessions(self) -> List[Dict]: ...

    async def get_session(self, session_id: str) -> Optional[Dict]: ...

    async def delete_session(self, session_id: str) -> bool: ...

    async def get_config_dict(self) -> Dict: ...

    async def update_config(self, updates: Dict) -> Dict: ...