"""Bandit weighting module for adaptive modality fusion."""

import torch
import torch.nn as nn


class BanditWeighter(nn.Module):
    """Estimate reliability weights using bandit approach."""

    def __init__(self, hidden_dim: int, num_prototypes: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes

        # Confidence estimators
        self.spectral_confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.foundation_confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, spectral: torch.Tensor, foundation: torch.Tensor):
        """Compute adaptive weights for modalities."""
        spec_conf = self.spectral_confidence(spectral)
        found_conf = self.foundation_confidence(foundation)

        # Normalize to weights
        total_conf = spec_conf + found_conf + 1e-8
        spec_weight = spec_conf / total_conf
        found_weight = found_conf / total_conf

        return spec_weight, found_weight
