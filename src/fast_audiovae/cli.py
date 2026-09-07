"""Command-line entry point."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Prepare or inspect an AudioVAE2 CPU decoder")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Download the pinned weights and prepare a CPU model bundle")
    prepare.add_argument("--output", default="artifacts")
    prepare.add_argument("--source", help="Use a local copy of the pinned ONNX export")
    prepare.add_argument("--native-build", help="Native build.json; otherwise use the local platform build")
    prepare.add_argument("--amd-build", help="Optional AMD packed-matrix build.json")
    prepare.add_argument("--block-fusion", choices=("none", "adds", "chain", "both"), default="none",
                         help="Explicit experimental residual-block fusion; requires a matching new native build")
    prepare.add_argument("--native-backend", choices=("auto", "avx2", "avx512"), default="auto",
                         help="Explicit Linux x86 custom-kernel backend; unsupported CPUs use standard ONNX at load time")
    inspect = commands.add_parser("inspect", help="Show the backend selected for a model bundle")
    inspect.add_argument("model_dir")
    inspect.add_argument("--threads", type=int, help="CPU workers; default is up to four visible CPUs")
    inspect.add_argument("--amd-packed", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare as prepare_bundle
        selected = prepare_bundle(args.output, source=args.source, native_build=args.native_build,
                                  amd_build=args.amd_build, block_fusion=args.block_fusion,
                                  native_backend=args.native_backend)
    else:
        from .runtime import load_decoder
        _, selected = load_decoder(args.model_dir, threads=args.threads, prefer_packed=args.amd_packed)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
