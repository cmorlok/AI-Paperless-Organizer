"""Paperless API exception hierarchy."""


class PaperlessError(Exception):
    """Base exception for Paperless API errors."""
    pass


class PaperlessNotFoundError(PaperlessError):
    """Document/tag/correspondent not found (HTTP 404)."""
    pass


class PaperlessAuthError(PaperlessError):
    """Authentication failed (HTTP 401/403)."""
    pass


class PaperlessServerError(PaperlessError):
    """Paperless server error (HTTP 5xx) or unmapped 4xx (including 422)."""
    pass


class PaperlessConnectionError(PaperlessError):
    """Connection to Paperless failed (timeouts, connection refused)."""
    pass
