"""Paperless-ngx API client service."""

from app.services.paperless.protocol import PaperlessClient
from app.services.paperless.exceptions import (
    PaperlessError,
    PaperlessNotFoundError,
    PaperlessAuthError,
    PaperlessServerError,
    PaperlessConnectionError,
)

__all__ = [
    "PaperlessClient",
    "PaperlessError",
    "PaperlessNotFoundError",
    "PaperlessAuthError",
    "PaperlessServerError",
    "PaperlessConnectionError",
]
