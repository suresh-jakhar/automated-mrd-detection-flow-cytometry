"""
Unit tests for the inference module.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from inference import MRDPredictor


class TestMRDPredictor:
    """Test suite for MRDPredictor class."""

    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """Create a temporary model file."""
        model_path = tmp_path / "test_model.pth"
        # Create a dummy model
        from model_wrapper import VarAutoEncoder

        model = VarAutoEncoder(input_dim=14, latent_dim=4)
        torch.save(model.state_dict(), model_path)
        return str(model_path)

    @pytest.fixture
    def predictor(self, mock_model_path):
        """Create a predictor instance for testing."""
        return MRDPredictor(
            model_path=mock_model_path,
            input_dim=14,
            latent_dim=4,
            threshold=0.023,
            device="cpu",
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        return np.random.rand(1000, 14).astype(np.float32)

    def test_predictor_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor is not None
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.input_dim == 14
        assert predictor.latent_dim == 4
        assert predictor.threshold == 0.023

    def test_preprocess_fits_scaler(self, predictor, sample_data):
        """Test that preprocess fits the scaler on first call."""
        assert not predictor.scaler_fitted

        preprocessed = predictor.preprocess(sample_data)

        assert predictor.scaler_fitted
        assert preprocessed.shape == sample_data.shape
        assert preprocessed.min() >= 0.0
        assert preprocessed.max() <= 1.001  # Allow for float32 precision

    def test_preprocess_consistent(self, predictor, sample_data):
        """Test that preprocess is consistent after fitting."""
        result1 = predictor.preprocess(sample_data)
        result2 = predictor.preprocess(sample_data)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_compute_reconstruction_errors_shape(self, predictor, sample_data):
        """Test reconstruction errors output shape."""
        preprocessed = predictor.preprocess(sample_data)
        errors = predictor.compute_reconstruction_errors(preprocessed, batch_size=256)

        assert errors.shape == (1000,)
        assert errors.dtype in [np.float32, np.float64]  # Accept both float types

    def test_compute_reconstruction_errors_values(self, predictor, sample_data):
        """Test reconstruction errors are non-negative."""
        preprocessed = predictor.preprocess(sample_data)
        errors = predictor.compute_reconstruction_errors(preprocessed, batch_size=256)

        assert (errors >= 0).all()
        assert not np.isnan(errors).any()
        assert np.isfinite(errors).all()

    def test_predict_mrd_basic(self, predictor, sample_data):
        """Test basic MRD prediction."""
        preprocessed = predictor.preprocess(sample_data)
        mrd_pct = predictor.predict_mrd(preprocessed, batch_size=256)

        assert isinstance(mrd_pct, float)
        assert 0.0 <= mrd_pct <= 100.0

    def test_predict_mrd_with_details(self, predictor, sample_data):
        """Test MRD prediction with detailed output."""
        preprocessed = predictor.preprocess(sample_data)
        result = predictor.predict_mrd(
            preprocessed, batch_size=256, return_details=True
        )

        assert isinstance(result, dict)
        assert "mrd_pct" in result
        assert "abnormal_count" in result
        assert "total" in result
        assert "threshold" in result
        assert "mean_error" in result
        assert "std_error" in result

        assert result["total"] == 1000
        assert result["abnormal_count"] >= 0
        assert result["abnormal_count"] <= 1000
        assert 0.0 <= result["mrd_pct"] <= 100.0

    def test_predict_mrd_custom_threshold(self, predictor, sample_data):
        """Test MRD prediction with custom threshold."""
        preprocessed = predictor.preprocess(sample_data)

        result_low = predictor.predict_mrd(
            preprocessed, threshold=0.001, return_details=True
        )
        result_high = predictor.predict_mrd(
            preprocessed, threshold=0.1, return_details=True
        )

        # Lower threshold should result in more abnormal cells
        assert result_low["abnormal_count"] >= result_high["abnormal_count"]

    def test_predict_mrd_batch_size(self, predictor, sample_data):
        """Test that different batch sizes give same results."""
        preprocessed = predictor.preprocess(sample_data)

        result1 = predictor.predict_mrd(
            preprocessed, batch_size=128, return_details=True
        )
        result2 = predictor.predict_mrd(
            preprocessed, batch_size=512, return_details=True
        )

        assert result1["mrd_pct"] == result2["mrd_pct"]
        assert result1["abnormal_count"] == result2["abnormal_count"]

    def test_get_latent_embeddings_shape(self, predictor, sample_data):
        """Test latent embeddings output shape."""
        preprocessed = predictor.preprocess(sample_data)
        embeddings = predictor.get_latent_embeddings(preprocessed, batch_size=256)

        assert embeddings.shape == (1000, 4)
        assert not np.isnan(embeddings).any()

    def test_get_latent_embeddings_deterministic(self, predictor, sample_data):
        """Test that embeddings are deterministic."""
        preprocessed = predictor.preprocess(sample_data)

        emb1 = predictor.get_latent_embeddings(preprocessed, batch_size=256)
        emb2 = predictor.get_latent_embeddings(preprocessed, batch_size=256)

        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_large_dataset(self, predictor):
        """Test handling of large datasets."""
        large_data = np.random.rand(100000, 14).astype(np.float32)
        preprocessed = predictor.preprocess(large_data)

        result = predictor.predict_mrd(
            preprocessed, batch_size=4096, return_details=True
        )

        assert result["total"] == 100000
        assert isinstance(result["mrd_pct"], float)

    def test_edge_case_single_sample(self, predictor):
        """Test with single sample."""
        single_sample = np.random.rand(1, 14).astype(np.float32)
        preprocessed = predictor.preprocess(single_sample)

        result = predictor.predict_mrd(preprocessed, return_details=True)

        assert result["total"] == 1
        assert result["abnormal_count"] in [0, 1]

    def test_edge_case_all_identical(self, predictor):
        """Test with all identical samples."""
        identical_data = np.ones((100, 14), dtype=np.float32)
        preprocessed = predictor.preprocess(identical_data)

        errors = predictor.compute_reconstruction_errors(preprocessed)

        # All reconstruction errors should be very similar (allowing for VAE stochasticity)
        assert errors.std() < 0.02  # Relaxed threshold for stochastic VAE behavior

    def test_device_handling(self, mock_model_path):
        """Test device handling (CPU/GPU)."""
        predictor_cpu = MRDPredictor(model_path=mock_model_path, device="cpu")
        assert str(predictor_cpu.device) == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            predictor_gpu = MRDPredictor(model_path=mock_model_path, device="cuda")
            assert "cuda" in str(predictor_gpu.device)
