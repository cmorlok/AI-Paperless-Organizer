"""ConfigService — reads/writes AppSettings KV store."""

from app.services.config.protocol import ConfigService
from app.services.config.service import ConfigServiceImpl

__all__ = ["ConfigService", "ConfigServiceImpl"]
