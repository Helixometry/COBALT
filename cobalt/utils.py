"""Utility functions."""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional


def get_device() -> torch.device:
    """Get device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42) -> None:
    """Set random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
