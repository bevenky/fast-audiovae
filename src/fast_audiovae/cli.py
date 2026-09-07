"""Command-line entry point."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Prepare or inspect an AudioVAE2 CPU decoder")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Show the backend selected for a model bundle")
    inspect.add_argument("model_dir")
    inspect.add_argument("--threads", type=int, default=4)
    inspect.add_argument("--amd-packed", action="store_true")
    args = parser.parse_args()
    from .runtime import load_decoder
    _, selected = load_decoder(args.model_dir, threads=args.threads, prefer_packed=args.amd_packed)
    print(json.dumps(selected, indent=2))
