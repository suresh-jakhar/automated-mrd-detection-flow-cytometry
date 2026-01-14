"""
Unit tests for the VAE model wrapper.
"""

import pytest
import torch
import numpy as np
from model_wrapper import VarAutoEncoder


class TestVarAutoEncoder:
    """Test suite for VarAutoEncoder model."""

    @pytest.fixture
    def model(self):
        """Create a model instance for testing."""
        return VarAutoEncoder(input_dim=14, latent_dim=4)

    @pytest.fixture
    def sample_input(self):
        """Create sample input data."""
        return torch.randn(32, 14)  # batch_size=32, input_dim=14

    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "fc_mu")
        assert hasattr(model, "fc_log_var")

    def test_forward_pass_shape(self, model, sample_input):
        """Test forward pass output shapes."""
        model.eval()
        with torch.no_grad():
            decoded, mu, log_var = model(sample_input)

        assert decoded.shape == sample_input.shape
        assert mu.shape == (32, 4)  # batch_size=32, latent_dim=4
        assert log_var.shape == (32, 4)

    def test_forward_pass_values(self, model, sample_input):
        """Test forward pass produces valid values."""
        model.eval()
        with torch.no_grad():
            decoded, mu, log_var = model(sample_input)

        assert not torch.isnan(decoded).any()
        assert not torch.isnan(mu).any()
        assert not torch.isnan(log_var).any()
        assert torch.isfinite(decoded).all()

    def test_encode_method(self, model, sample_input):
        """Test encode method."""
        model.eval()
        with torch.no_grad():
            mu = model.encode(sample_input)

        assert mu.shape == (32, 4)
        assert not torch.isnan(mu).any()

    def test_decode_method(self, model):
        """Test decode method."""
        model.eval()
        latent = torch.randn(32, 4)
        with torch.no_grad():
            decoded = model.decode(latent)

        assert decoded.shape == (32, 14)
        assert not torch.isnan(decoded).any()

    def test_reparameterize(self, model):
        """Test reparameterization trick."""
        mu = torch.zeros(32, 4)
        log_var = torch.zeros(32, 4)

        z = model.reparameterize(mu, log_var)

        assert z.shape == (32, 4)
        # With log_var=0, std=1, so z should be normally distributed around mu
        assert z.mean().abs() < 0.5  # Approximately centered at 0

    def test_model_parameters_count(self, model):
        """Test total number of parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == 894  # Expected from documentation

    def test_batch_size_flexibility(self, model):
        """Test model works with different batch sizes."""
        model.eval()
        batch_sizes = [1, 16, 64, 128]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 14)
            with torch.no_grad():
                decoded, mu, log_var = model(x)

            assert decoded.shape == (batch_size, 14)
            assert mu.shape == (batch_size, 4)

    def test_model_save_load(self, model, tmp_path):
        """Test model save and load."""
        # Save model
        save_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), save_path)

        # Load model
        new_model = VarAutoEncoder(input_dim=14, latent_dim=4)
        new_model.load_state_dict(torch.load(save_path))

        # Verify that all parameters match exactly
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)

        # Verify state dicts are identical
        assert set(model.state_dict().keys()) == set(new_model.state_dict().keys())

    def test_deterministic_encoding(self, model, sample_input):
        """Test that encoding is deterministic in eval mode."""
        model.eval()

        with torch.no_grad():
            mu1 = model.encode(sample_input)
            mu2 = model.encode(sample_input)

        assert torch.allclose(mu1, mu2)

    def test_gradient_flow(self, model, sample_input):
        """Test gradients flow through the model."""
        model.train()

        decoded, mu, log_var = model(sample_input)
        loss = decoded.sum() + mu.sum() + log_var.sum()
        loss.backward()

        # Check that gradients exist and are non-zero
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad))
