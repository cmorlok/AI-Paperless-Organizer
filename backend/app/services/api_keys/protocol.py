"""Protocol for API keys service."""

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class ApiKeysService(Protocol):
    """Protocol for API keys service."""

    async def list_keys(self) -> list: ...

    async def create_key(self, name: str) -> dict: ...

    async def delete_key(self, key_id: int) -> None: ...

    async def toggle_key(self, key_id: int) -> dict: ...

    async def validate_key(self, token: str) -> Optional[dict]: ...
