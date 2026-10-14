"""COBALT model for TB detection from cough audio."""

import torch
import torch.nn as nn
from typing import Dict, Optional


class COBALT(nn.Module):
    """
    COBALT: Cough-based TB Detection with Hyperbolic Prototypes.

    Fuses spectral and foundation model features using hyperbolic geometry
    and bandit-style reliability weighting for TB screening.

    Args:
        spectral_dim (int): Spectral feature dimension
        foundation_dim (int): Foundation model feature dimension
        hidden_dim (int): Fusion hidden dimension
        num_prototypes (int): Number of codebook prototypes
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        spectral_dim: int = 13,
        foundation_dim: int = 768,
        hidden_dim: int = 256,
        num_prototypes: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.spectral_dim = spectral_dim
        self.foundation_dim = foundation_dim
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes

        # Projection layers
        self.spectral_proj = nn.Linear(spectral_dim, hidden_dim)
        self.foundation_proj = nn.Linear(foundation_dim, hidden_dim)

        # Hyperbolic fusion
        from .fusion import HyperbolicFusion
        self.fusion = HyperbolicFusion(hidden_dim, num_prototypes, dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # Binary: TB+/TB-
        )

    def forward(
        self,
        spectral_features: torch.Tensor,
        foundation_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of COBALT."""
        # Project to common space
        spectral_proj = self.spectral_proj(spectral_features)
        foundation_proj = self.foundation_proj(foundation_features)

        # Hyperbolic fusion
        fused = self.fusion(spectral_proj, foundation_proj)

        # Classification
        logits = self.classifier(fused)
        probs = torch.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "healthy_prob": probs[:, 0],
            "tb_prob": probs[:, 1],
            "features": fused,
        }

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "COBALT":
        """Load pretrained COBALT model."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def save(self, checkpoint_path: str, config: Optional[Dict] = None) -> None:
        """Save COBALT model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {
                "spectral_dim": self.spectral_dim,
                "foundation_dim": self.foundation_dim,
                "hidden_dim": self.hidden_dim,
                "num_prototypes": self.num_prototypes,
            },
        }
        torch.save(checkpoint, checkpoint_path)
