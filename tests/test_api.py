"""
Tests for the API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Import app
from src.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_structure(self, client):
        """Test health endpoint returns expected structure."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "mongodb_connected" in data
        assert "fireworks_configured" in data
        assert "timestamp" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestQuoteEndpoints:
    """Tests for quote processing endpoints."""

    def test_quote_endpoint_validation(self, client):
        """Test that short emails are rejected."""
        response = client.post(
            "/api/quotes/process",
            json={"email_content": "Too short"}
        )
        assert response.status_code == 422  # Validation error

    def test_quote_endpoint_requires_content(self, client):
        """Test that email_content is required."""
        response = client.post(
            "/api/quotes/process",
            json={}
        )
        assert response.status_code == 422

    def test_file_upload_wrong_type(self, client):
        """Test that wrong file types are rejected."""
        # Create a fake PDF file
        response = client.post(
            "/api/quotes/process-file",
            files={"file": ("test.pdf", b"fake pdf content", "application/pdf")}
        )
        assert response.status_code == 400

    def test_get_nonexistent_quote(self, client):
        """Test getting a quote that doesn't exist."""
        with patch('src.api.routes.get_collection') as mock_get_coll:
            mock_collection = MagicMock()
            mock_collection.find_one.return_value = None
            mock_get_coll.return_value = mock_collection

            response = client.get("/api/quotes/Q-NONEXISTENT")
            assert response.status_code == 404


class TestAPIDocumentation:
    """Tests for API documentation."""

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

    def test_docs_available(self, client):
        """Test Swagger docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, client):
        """Test ReDoc is available."""
        response = client.get("/redoc")
        assert response.status_code == 200


class TestResponseSchemas:
    """Tests for response schema validation."""

    def test_quote_response_fields(self):
        """Test QuoteResponse has all expected fields."""
        from src.api.schemas import QuoteResponse

        response = QuoteResponse(
            success=True,
            quote_id="Q-TEST-123",
            client_name="Test Corp",
            total_premium=10000.0,
            risk_level="MEDIUM",
            risk_score=45.0
        )

        assert response.success is True
        assert response.quote_id == "Q-TEST-123"
        assert response.total_premium == 10000.0

    def test_error_response(self):
        """Test ErrorResponse model."""
        from src.api.schemas import ErrorResponse

        error = ErrorResponse(
            error="Something went wrong",
            detail="More details here"
        )

        assert error.success is False
        assert error.error == "Something went wrong"
