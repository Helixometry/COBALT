"""Inference script for COBALT."""

import argparse
from pathlib import Path
from cobalt.model import COBALT
from cobalt.utils import get_device

def main():
    parser = argparse.ArgumentParser(description="COBALT inference")
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    args = parser.parse_args()
    device = get_device()
    model = COBALT.from_pretrained(args.checkpoint).to(device)
    print(f"\nCOBALT TB Detection Results")
    print("=" * 60)
    print(f"Audio: {Path(args.audio).name}")
    print(f"\n(Placeholder inference)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
