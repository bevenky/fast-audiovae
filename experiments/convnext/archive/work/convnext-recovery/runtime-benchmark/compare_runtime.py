"""Compare matched bounded runtime trials; never infer quality from speed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_runtime import atomic_json


def compare_reports(old, new):
    required = ("checkpoint_sha256", "target_cache_sha256", "source_engine_sha256", "training_crops",
                "training_tensor_sha256", "training_policy", "restored_global_rng_sha256")
    for key in required:
        if old[key] != new[key]:
            raise ValueError("Unmatched experiment input: " + key)
    for key in ("input_sha256", "teacher_state_sha256", "input_ids", "input_lengths", "padded_shape", "api"):
        if old["teacher"][key] != new["teacher"][key]:
            raise ValueError("Unmatched teacher input: " + key)
    if old["runtime"]["script_sha256"] != new["runtime"]["script_sha256"]:
        raise ValueError("Different benchmark source")
    for left, right in zip(old["training_steps"], new["training_steps"], strict=True):
        for key in ("index", "phase", "padded_latent_frames", "D_views"):
            if left[key] != right[key]:
                raise ValueError("Mismatched training step/view schedule")
    rows = {}
    measurements = {"training_step": (old["training_summary"], new["training_summary"])}
    for name in ("encode", "decode"):
        measurements["teacher_" + name] = (old["teacher"]["calls"][name]["summary"], new["teacher"]["calls"][name]["summary"])
    for name, (before, after) in measurements.items():
        rows[name] = {"old_mean_seconds": before["mean_seconds"], "new_mean_seconds": after["mean_seconds"],
            "speed_ratio_old_over_new": before["mean_seconds"] / after["mean_seconds"],
            "wall_time_reduction_percent": 100 * (1 - after["mean_seconds"] / before["mean_seconds"]),
            "old_cv": before["coefficient_of_variation"], "new_cv": after["coefficient_of_variation"],
            "old_median_seconds": before["median_seconds"], "new_median_seconds": after["median_seconds"]}
    return {"format_version": 1, "inputs_matched": True, "runtimes": {"old": old["runtime"], "new": new["runtime"]},
        "timings": rows, "resume": {"old_final_step": old["training_summary"]["end_step"],
            "new_final_step": new["training_summary"]["end_step"], "all_recorded_losses_finite": True},
        "parameter_delta": {name: {k: report["student_parameter_delta"][k] for k in
            ("update_norm", "update_rms", "relative_update_l2", "final_postclip_gradient_norm")}
            for name, report in (("old", old), ("new", new))},
        "limits": ["Ten timed steps in one trial per runtime; elapsed-time variation and longer-run overhead remain unmeasured.",
            "Identical inputs and starting state do not require bitwise-identical evolving optimizer trajectories across versions.",
            "No perceptual quality or convergence conclusion follows from finite losses, parameter deltas or faster throughput.",
            "CPU inference RTF is not measured by this H100 training and target-preparation benchmark."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = compare_reports(json.loads(Path(args.old).read_text()), json.loads(Path(args.new).read_text()))
    atomic_json(args.out, result)
    print(json.dumps(result["timings"], indent=2))


if __name__ == "__main__":
    main()
