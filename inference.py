"""
Inference module for MRD detection.
Handles batch processing, preprocessing, and MRD% calculation.
"""

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import logging
from model_wrapper import VarAutoEncoder

logger = logging.getLogger(__name__)


class MRDPredictor:
    """
    MRD predictor using the trained VAE model.
    Handles preprocessing, batch inference, and MRD% calculation.
    """

    def __init__(
        self,
        model_path,
        scaler_path=None,
        input_dim=14,
        latent_dim=4,
        threshold=0.02310833333333333,
        device="cpu",
    ):
        """
        Initialize the MRD predictor.

        Args:
            model_path: Path to the trained model checkpoint (.pth file)
            scaler_path: Path to the fitted MinMaxScaler (optional, for reproducibility)
            input_dim: Number of input features
            latent_dim: Latent dimension
            threshold: Reconstruction error threshold for MRD detection (default from training)
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.threshold = threshold
        self.scaler_fitted = False

        # Initialize model
        self.model = VarAutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(
            self.device
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Initialize scaler (if provided, otherwise create default MinMaxScaler)
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self.scaler_fitted = True
            logger.info(f"Loaded fitted scaler from {scaler_path}")
        else:
            self.scaler = MinMaxScaler()
            # Scaler will be fit on first batch of data
            logger.info(
                "MinMaxScaler initialized. Will be fitted on first batch of data."
            )

    def preprocess(self, X):
        """
        Preprocess input data using MinMaxScaler.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            Preprocessed numpy array
        """
        # If scaler is not fitted yet, fit it on this batch
        if not self.scaler_fitted and self.scaler:
            self.scaler.fit(X)
            self.scaler_fitted = True
            logger.info("MinMaxScaler fitted on input data")

        if self.scaler and self.scaler_fitted:
            return self.scaler.transform(X)
        return X

    def compute_reconstruction_errors(self, X, batch_size=4096):
        """
        Compute per-cell reconstruction errors.

        Args:
            X: numpy array of shape (n_samples, n_features)
            batch_size: Batch size for processing

        Returns:
            numpy array of reconstruction errors (MSE per sample)
        """
        n_samples = X.shape[0]
        errors = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = X[i : i + batch_size]
                x_batch = torch.from_numpy(batch).float().to(self.device)

                # Forward pass
                recon, _, _ = self.model(x_batch)

                # Compute MSE per sample
                mse = ((x_batch - recon) ** 2).mean(dim=1).cpu().numpy()
                errors.extend(mse)

        return np.array(errors)

    def predict_mrd(self, X, batch_size=4096, threshold=None, return_details=False):
        """
        Predict MRD% from input data.

        Args:
            X: numpy array of preprocessed features (n_samples, n_features)
            batch_size: Batch size for processing
            threshold: Override the default threshold if needed
            return_details: If True, return detailed stats (abnormal count, total, threshold used, etc.)

        Returns:
            float: MRD% (percentage of cells above threshold)
            OR dict: {'mrd_pct': float, 'abnormal_count': int, 'total': int, 'threshold': float, ...}
        """
        threshold = threshold or self.threshold

        # Compute reconstruction errors
        errors = self.compute_reconstruction_errors(X, batch_size=batch_size)

        # Count cells above threshold
        abnormal_mask = errors > threshold
        abnormal_count = abnormal_mask.sum()
        total_count = len(errors)
        mrd_pct = 100.0 * abnormal_count / total_count if total_count > 0 else 0.0

        if return_details:
            return {
                "mrd_pct": float(mrd_pct),
                "abnormal_count": int(abnormal_count),
                "total": int(total_count),
                "threshold": float(threshold),
                "mean_error": float(errors.mean()),
                "std_error": float(errors.std()),
                "min_error": float(errors.min()),
                "max_error": float(errors.max()),
            }

        return mrd_pct

    def get_latent_embeddings(self, X, batch_size=4096):
        """
        Get latent space embeddings (μ) for each cell.

        Args:
            X: numpy array of preprocessed features
            batch_size: Batch size for processing

        Returns:
            numpy array of shape (n_samples, latent_dim)
        """
        n_samples = X.shape[0]
        embeddings = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = X[i : i + batch_size]
                x_batch = torch.from_numpy(batch).float().to(self.device)
                z = self.model.encode(x_batch)
                embeddings.append(z.cpu().numpy())

        return np.vstack(embeddings)
