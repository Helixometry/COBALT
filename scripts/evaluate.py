"""Evaluation script for COBALT."""

import argparse
from cobalt.model import COBALT
from cobalt.utils import get_device

def main():
    parser = argparse.ArgumentParser(description="Evaluate COBALT")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)

    args = parser.parse_args()
    device = get_device()
    model = COBALT.from_pretrained(args.checkpoint).to(device)
    print(f"Evaluating COBALT on {args.test_data}")

if __name__ == "__main__":
    main()
