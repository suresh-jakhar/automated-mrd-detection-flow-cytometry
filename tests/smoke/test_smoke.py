"""
Smoke tests for quick validation after deployment.
Run these tests to verify basic functionality.
"""

import requests
import pytest
import os


BASE_URL = os.getenv("API_URL", "http://localhost:8000")


@pytest.mark.smoke
class TestSmokeTests:
    """Quick smoke tests for deployment validation."""

    def test_health_check(self):
        """Verify API is running."""
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_info_endpoint(self):
        """Verify model info is accessible."""
        response = requests.get(f"{BASE_URL}/info", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["input_dim"] == 14
        assert data["latent_dim"] == 4

    def test_docs_accessible(self):
        """Verify API documentation is accessible."""
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        assert response.status_code == 200

    def test_root_accessible(self):
        """Verify root endpoint returns HTML."""
        response = requests.get(f"{BASE_URL}/", timeout=5)
        assert response.status_code == 200
        assert "html" in response.headers.get("content-type", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "smoke"])
