"""Training script for COBALT."""

import argparse
import torch
import yaml
from pathlib import Path
from cobalt.model import COBALT
from cobalt.utils import seed_everything, get_device

def main():
    parser = argparse.ArgumentParser(description="Train COBALT")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed_everything(config.get('seed', 42))
    device = get_device()

    model = COBALT(**config['model']).to(device)
    print(f"Training COBALT on {device}")

if __name__ == "__main__":
    main()
