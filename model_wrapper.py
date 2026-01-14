"""
Model wrapper for the trained VAE model.
Matches the exact architecture from training.ipynb
"""

import torch
import torch.nn as nn


class VarAutoEncoder(nn.Module):
    """Variational Autoencoder matching the training pipeline."""
    
    def __init__(self, input_dim, latent_dim=4):
        super().__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ELU(),
            nn.Linear(16, 8),
            nn.ELU(),
            nn.Linear(8, 4),
            nn.ELU(),
        )

        # Latent space heads
        self.fc_mu = nn.Linear(4, latent_dim)
        self.fc_log_var = nn.Linear(4, latent_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.ELU(),
            nn.Linear(4, 8),
            nn.ELU(),
            nn.Linear(8, 16),
            nn.ELU(),
            nn.Linear(16, input_dim),
            nn.ELU()
        )

    def reparameterize(self, mu, log_var):
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass through encoder-decoder."""
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var
    
    def encode(self, x):
        """Encode input to latent space (mu only, deterministic at eval time)."""
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        return mu
    
    def decode(self, z):
        """Decode latent vector back to input space."""
        return self.decoder(z)
