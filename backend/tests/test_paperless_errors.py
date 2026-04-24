"""Tests for PaperlessClient error handling."""

import pytest
from unittest.mock import MagicMock

from app.services.paperless.service import PaperlessClient
from app.services.paperless.exceptions import (
    PaperlessError,
    PaperlessNotFoundError,
    PaperlessAuthError,
    PaperlessServerError,
    PaperlessConnectionError,
)


class TestHandleResponseErrors:
    """Test _handle_response_errors method."""

    @pytest.fixture
    def client(self):
        return PaperlessClient(base_url="http://localhost:8000", api_token="test")

    def test_404_raises_not_found_error(self, client):
        """HTTP 404 should raise PaperlessNotFoundError."""
        response = MagicMock()
        response.status_code = 404
        response.url = "http://localhost:8000/api/documents/999"

        with pytest.raises(PaperlessNotFoundError) as exc_info:
            client._handle_response_errors(response)

        assert "Resource not found" in str(exc_info.value)

    def test_401_raises_auth_error(self, client):
        """HTTP 401 should raise PaperlessAuthError."""
        response = MagicMock()
        response.status_code = 401
        response.url = "http://localhost:8000/api/documents/"

        with pytest.raises(PaperlessAuthError) as exc_info:
            client._handle_response_errors(response)

        assert "auth error" in str(exc_info.value)

    def test_403_raises_auth_error(self, client):
        """HTTP 403 should raise PaperlessAuthError."""
        response = MagicMock()
        response.status_code = 403
        response.url = "http://localhost:8000/api/documents/"

        with pytest.raises(PaperlessAuthError) as exc_info:
            client._handle_response_errors(response)

        assert "auth error" in str(exc_info.value)

    def test_422_raises_server_error(self, client):
        """HTTP 422 should raise PaperlessServerError (not httpx.HTTPStatusError)."""
        response = MagicMock()
        response.status_code = 422
        response.url = "http://localhost:8000/api/documents/"
        response.reason_phrase = "Unprocessable Entity"
        # Note: raise_for_status would raise httpx.HTTPStatusError, but we intercept first

        with pytest.raises(PaperlessServerError) as exc_info:
            client._handle_response_errors(response)

        assert "422" in str(exc_info.value)

    def test_500_raises_server_error(self, client):
        """HTTP 500 should raise PaperlessServerError."""
        response = MagicMock()
        response.status_code = 500
        response.url = "http://localhost:8000/api/documents/"
        response.reason_phrase = "Internal Server Error"

        with pytest.raises(PaperlessServerError) as exc_info:
            client._handle_response_errors(response)

        assert "500" in str(exc_info.value)

    def test_502_raises_server_error(self, client):
        """HTTP 502 should raise PaperlessServerError."""
        response = MagicMock()
        response.status_code = 502
        response.url = "http://localhost:8000/api/documents/"
        response.reason_phrase = "Bad Gateway"

        with pytest.raises(PaperlessServerError) as exc_info:
            client._handle_response_errors(response)

        assert "502" in str(exc_info.value)

    def test_503_raises_server_error(self, client):
        """HTTP 503 should raise PaperlessServerError."""
        response = MagicMock()
        response.status_code = 503
        response.url = "http://localhost:8000/api/documents/"
        response.reason_phrase = "Service Unavailable"

        with pytest.raises(PaperlessServerError) as exc_info:
            client._handle_response_errors(response)

        assert "503" in str(exc_info.value)

    def test_400_raises_server_error(self, client):
        """HTTP 400 should raise PaperlessServerError."""
        response = MagicMock()
        response.status_code = 400
        response.url = "http://localhost:8000/api/documents/"
        response.reason_phrase = "Bad Request"

        with pytest.raises(PaperlessServerError) as exc_info:
            client._handle_response_errors(response)

        assert "400" in str(exc_info.value)

    def test_200_does_not_raise(self, client):
        """HTTP 200 should not raise any exception."""
        response = MagicMock()
        response.status_code = 200

        # Should not raise
        client._handle_response_errors(response)

    def test_201_does_not_raise(self, client):
        """HTTP 201 should not raise any exception."""
        response = MagicMock()
        response.status_code = 201

        # Should not raise
        client._handle_response_errors(response)


class TestExceptionHierarchy:
    """Test exception hierarchy relationships."""

    def test_not_found_is_paperless_error(self):
        """PaperlessNotFoundError should be a PaperlessError."""
        assert issubclass(PaperlessNotFoundError, PaperlessError)

    def test_auth_is_paperless_error(self):
        """PaperlessAuthError should be a PaperlessError."""
        assert issubclass(PaperlessAuthError, PaperlessError)

    def test_server_is_paperless_error(self):
        """PaperlessServerError should be a PaperlessError."""
        assert issubclass(PaperlessServerError, PaperlessError)

    def test_connection_is_paperless_error(self):
        """PaperlessConnectionError should be a PaperlessError."""
        assert issubclass(PaperlessConnectionError, PaperlessError)

    def test_can_catch_all_with_base(self):
        """All specific exceptions can be caught with PaperlessError."""
        with pytest.raises(PaperlessError):
            raise PaperlessNotFoundError("test")

        with pytest.raises(PaperlessError):
            raise PaperlessAuthError("test")

        with pytest.raises(PaperlessError):
            raise PaperlessServerError("test")

        with pytest.raises(PaperlessError):
            raise PaperlessConnectionError("test")
