"""Run the bounded comparison using only the previous run's verified targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from .cached_comparison_data import load_parent_crops
from .objective_comparison import ObjectiveComparisonConfig, run_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--run-name", default="objective-comparison-v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args(argv)
    if args.parent_checkpoint.is_symlink():
        raise ValueError("The parent checkpoint must be an original immutable file")
    actual = hashlib.sha256(args.parent_checkpoint.read_bytes()).hexdigest()
    if actual != args.parent_sha256:
        raise ValueError("Parent checkpoint checksum mismatch")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The requested training GPU is unavailable")
    parent = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
    unchanged = ("model.py", "optimizers.py", "cache.py", "source_corpus.py",
                 "losses_distillation.py", "gradient_balancer.py", "discriminators.py")
    for name in unchanged:
        digest = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        if digest != parent["identity"]["implementation"][name]:
            raise ValueError(f"Controlled comparison changed a held-constant implementation: {name}")
    diagnostic, sentinel, evidence = load_parent_crops(parent, args.cache_dir)
    print(json.dumps({"event": "inputs_verified", "parent_sha256": actual,
        "parent_step": parent["engine"]["step"], "diagnostic_crops": len(diagnostic),
        "sentinel_crops": len(sentinel), "exact_parent_latents_and_targets": True,
        "teacher_inference_calls": 0, "held_constant_sources": list(unchanged)}), flush=True)
    result = run_comparison(parent, diagnostic, sentinel, args.output_dir,
        parent_checkpoint_sha256=actual, data_identity=evidence, device=args.device,
        config=ObjectiveComparisonConfig(), log_dir=args.log_dir, run_name=args.run_name)
    print(json.dumps({"event": "comparison_finished", "output_dir": str(args.output_dir),
                      "state": result.get("state"), "main_training_started": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
