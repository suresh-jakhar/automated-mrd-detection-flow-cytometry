"""
Integration tests for the FastAPI application.
"""

import pytest
import io
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
from model_wrapper import VarAutoEncoder


@pytest.fixture(scope="module")
def setup_test_model(tmp_path_factory):
    """Create a temporary model file for testing."""
    # Create temporary directory for model
    tmp_dir = tmp_path_factory.mktemp("model")
    model_path = tmp_dir / "vae_4dim_6_final.pth"
    
    # Create and save a test model
    model = VarAutoEncoder(input_dim=14, latent_dim=4)
    torch.save(model.state_dict(), model_path)
    
    # Set environment variable for the app to use this model
    os.environ["MODEL_PATH"] = str(model_path)
    
    yield model_path
    
    # Cleanup
    if "MODEL_PATH" in os.environ:
        del os.environ["MODEL_PATH"]


@pytest.fixture
def client(setup_test_model):
    """Create a test client with model setup."""
    # Import app after MODEL_PATH is set
    from app import app
    return TestClient(app)


@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file in memory."""
    data = np.random.rand(1000, 14)
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(14)])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()


class TestHealthEndpoints:
    """Test health and info endpoints."""

    def test_health_endpoint(self, client):
        """Test /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"

    def test_info_endpoint(self, client):
        """Test /info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "input_dim" in data
        assert "latent_dim" in data
        assert "threshold" in data
        assert data["input_dim"] == 14
        assert data["latent_dim"] == 4


class TestPredictEndpoint:
    """Test prediction endpoints."""

    def test_predict_single_file(self, client, sample_csv_file):
        """Test prediction with single file."""
        files = {"files": ("test.csv", sample_csv_file, "text/csv")}
        response = client.post("/predict", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "mrd_pct" in data
        assert isinstance(data["mrd_pct"], float)
        assert 0.0 <= data["mrd_pct"] <= 100.0

    def test_predict_with_details(self, client, sample_csv_file):
        """Test prediction with detailed output."""
        files = {"files": ("test.csv", sample_csv_file, "text/csv")}
        params = {"return_details": True}
        response = client.post("/predict", files=files, params=params)

        assert response.status_code == 200
        data = response.json()
        assert "mrd_pct" in data
        assert "abnormal_count" in data
        assert "total" in data
        assert "mean_error" in data
        assert "std_error" in data
        assert "processing_time_seconds" in data

    def test_predict_multiple_files(self, client, sample_csv_file):
        """Test prediction with multiple files."""
        files = [
            ("files", ("test1.csv", sample_csv_file, "text/csv")),
            ("files", ("test2.csv", sample_csv_file, "text/csv")),
            ("files", ("test3.csv", sample_csv_file, "text/csv")),
        ]
        response = client.post("/predict", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["num_files"] == 3
        assert len(data["file_names"]) == 3

    def test_predict_custom_threshold(self, client, sample_csv_file):
        """Test prediction with custom threshold."""
        files = {"files": ("test.csv", sample_csv_file, "text/csv")}
        params = {"threshold": 0.05, "return_details": True}
        response = client.post("/predict", files=files, params=params)

        assert response.status_code == 200
        data = response.json()
        assert data["threshold"] == 0.05

    def test_predict_no_files(self, client):
        """Test prediction without files."""
        response = client.post("/predict")
        assert response.status_code == 422  # Unprocessable Entity

    def test_predict_invalid_file_type(self, client):
        """Test prediction with invalid file type."""
        files = {"files": ("test.txt", "invalid content", "text/plain")}
        response = client.post("/predict", files=files)
        assert response.status_code == 400

    def test_predict_wrong_dimensions(self, client):
        """Test prediction with wrong number of features."""
        # Create CSV with wrong number of features
        data = np.random.rand(100, 10)  # Only 10 features instead of 14
        df = pd.DataFrame(data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        files = {"files": ("test.csv", csv_buffer.getvalue(), "text/csv")}
        response = client.post("/predict", files=files)
        assert response.status_code == 400

    def test_predict_with_time_column(self, client):
        """Test that Time column is properly dropped."""
        data = np.random.rand(100, 14)
        df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(14)])
        df["Time"] = np.arange(100)

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        files = {"files": ("test.csv", csv_buffer.getvalue(), "text/csv")}
        response = client.post("/predict", files=files)

        assert response.status_code == 200


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_returns_html(self, client):
        """Test that root endpoint returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestCORS:
    """Test CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present."""
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert "access-control-allow-origin" in response.headers


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_csv_format(self, client):
        """Test with invalid CSV format."""
        invalid_csv = "invalid,csv,data\n1,2"
        files = {"files": ("test.csv", invalid_csv, "text/csv")}
        response = client.post("/predict", files=files)
        assert response.status_code in [400, 500]

    def test_empty_file(self, client):
        """Test with empty file."""
        files = {"files": ("test.csv", "", "text/csv")}
        response = client.post("/predict", files=files)
        assert response.status_code in [400, 500]


class TestPerformance:
    """Test performance characteristics."""

    def test_large_file_handling(self, client):
        """Test handling of large file."""
        # Create a large CSV (10k rows)
        data = np.random.rand(10000, 14)
        df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(14)])
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        files = {"files": ("large_test.csv", csv_buffer.getvalue(), "text/csv")}
        response = client.post("/predict", files=files, params={"return_details": True})

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 10000

    def test_concurrent_requests(self, client, sample_csv_file):
        """Test that multiple requests can be handled."""
        files = {"files": ("test.csv", sample_csv_file, "text/csv")}

        # Make multiple requests
        responses = []
        for _ in range(5):
            response = client.post("/predict", files=files)
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == 200
