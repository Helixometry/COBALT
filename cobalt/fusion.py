"""Hyperbolic fusion module for COBALT."""

import torch
import torch.nn as nn


class HyperbolicFusion(nn.Module):
    """Hyperbolic geometry based fusion of spectral and foundation features."""

    def __init__(self, hidden_dim: int, num_prototypes: int = 32, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes

        # Codebook prototypes
        self.spectral_prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dim))
        self.foundation_prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dim))

        # Bandit weighter
        from .bandit import BanditWeighter
        self.bandit = BanditWeighter(hidden_dim, num_prototypes)

        # Layer norm and dropout
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, spectral: torch.Tensor, foundation: torch.Tensor) -> torch.Tensor:
        """Fuse spectral and foundation features in hyperbolic space."""
        # Average pooling
        spec_pool = spectral.mean(dim=1)
        found_pool = foundation.mean(dim=1)

        # Bandit weighting
        spec_weight, found_weight = self.bandit(spec_pool, found_pool)

        # Weighted fusion
        fused = spec_weight * spec_pool + found_weight * found_pool
        fused = self.norm(fused)

        return self.dropout(fused)
