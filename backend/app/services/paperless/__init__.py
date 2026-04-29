"""Paperless-ngx API client service."""

from app.services.paperless.protocol import PaperlessClient
from app.services.paperless.service import CACHE_TTL
from app.services.paperless.exceptions import (
    PaperlessError,
    PaperlessNotFoundError,
    PaperlessAuthError,
    PaperlessServerError,
    PaperlessConnectionError,
)

__all__ = [
    "PaperlessClient",
    "CACHE_TTL",
    "PaperlessError",
    "PaperlessNotFoundError",
    "PaperlessAuthError",
    "PaperlessServerError",
    "PaperlessConnectionError",
]
