"""Protocol for configuration service."""

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class ConfigService(Protocol):
    """Protocol for configuration service — reads/writes AppSettings KV store."""

    async def get(self, key: str) -> Optional[str]:
        """Get a setting value by key. Returns None if not found."""
        ...

    async def set(self, key: str, value: str, value_type: str = "str") -> None:
        """Set a setting value. Creates new row if key doesn't exist, updates if it does."""
        ...
