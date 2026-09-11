"""Audit every archived z -> T(z) pair without changing caches or weights.

Run explicitly in the qualified CUDA environment. Importing or compiling this
file does not import torch, initialize CUDA, load models, or run inference.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

VERSION = "conditional-pair-audit-v1"
BASE = Path("/workspace/fast-audiovae-convnext-20260909-r9/remediation/corrected-screen")
ASSETS = Path("/workspace/fast-audiovae-convnext-20260908-r1/assets")
EXPECTED_DATA_SHA = "90632c05076e51cb2bca78ae7cd25ef0a72440d34f51dcabf18a5dddbd918d5a"
EXPECTED_RECEIPT_SHA = "406745cdc118f3a6c68114dbbd20f2749e5050fd2550ee771a21aabba4616648"
EXPECTED_CACHE_SHA = "9acf8808622c86eb6ceaf974571706d7868396edcb3583c12543399da281a883"
EXPECTED_TEACHER_STATE = "8da1691a055ac7eee3d06750846fc1ad1f887adfec13c31340cab7397fde1501"
EXPECTED_POOLS = {"discriminator_warmup": 2048, "gradient_calibration": 1024,
                  "regular_generator": 12800, "targeted_generator": 12800}
GEOMETRY = ("source_id", "start_frame", "context_start_frame", "context_frames",
            "scored_frames", "valid_scored_samples", "cache_key")
MAX_ABS = 1e-5
RMS = 1e-6
HOP = 1920


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False) + "\n").encode()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def reset_affected_prefix():
    """Exact support bound from the pinned decoder's causal local layers."""
    length = 6  # Initial latent kernel7; following kernel1 is pointwise.
    stages = []
    for stride in (8, 6, 5, 2, 2, 2):
        # Transpose kernel2s, stride s, right trim s. Then residual kernels7
        # with dilation1,3,9. Sample-rate conditioning is pointwise.
        length = (length + 1) * stride + 6 * (1 + 3 + 9)
        stages.append(length)
    return {"stage_prefix_samples": stages, "final_prefix_samples": length + 6,
            "minimum_full_latent_context": math.ceil((length + 6) / HOP),
            "derivation": "L=6; for each stride s: L=(L+1)*s+78; final kernel7: L+=6"}


def score_bounds(crop):
    start = crop.context_frames * HOP + (6 if crop.context_start_frame > 0 else 0)
    stop = crop.context_frames * HOP + crop.valid_scored_samples
    if not 0 <= start < stop <= crop.teacher_audio.shape[-1]:
        raise ValueError("Empty or invalid historical score mask")
    if crop.context_start_frame + crop.context_frames != crop.start_frame:
        raise ValueError("Inconsistent absolute crop coordinates")
    if crop.context_start_frame > 0 and start < reset_affected_prefix()["final_prefix_samples"]:
        raise ValueError("Insufficient real latent context for teacher decoder reset")
    return start, stop


def metric(actual, expected):
    import torch
    if actual.shape != expected.shape or actual.numel() == 0:
        raise ValueError("Shape mismatch or empty comparison")
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        return {"samples": actual.numel(), "finite": False, "max_abs": None,
                "rms": None, "squared_error_sum": None, "bitwise_equal": False, "passed": False}
    error = actual.double() - expected.double()
    maximum = float(error.abs().max())
    square_sum = float(error.square().sum())
    rms = math.sqrt(square_sum / error.numel())
    return {"samples": error.numel(), "finite": True, "max_abs": maximum,
            "rms": rms, "squared_error_sum": square_sum,
            "bitwise_equal": bool(actual.equal(expected)), "passed": maximum <= MAX_ABS and rms <= RMS}


def aggregate(rows):
    finite = [r for r in rows if r["comparison"]["finite"]]
    samples = sum(r["comparison"]["samples"] for r in finite)
    return {"crops": len(rows), "sources": len({r["source_id"] for r in rows}),
            "failed_crops": sum(not r["comparison"]["passed"] for r in rows),
            "nonfinite_crops": len(rows) - len(finite), "finite_scored_samples": samples,
            "max_abs": max((r["comparison"]["max_abs"] for r in finite), default=None),
            "pooled_rms": math.sqrt(sum(r["comparison"]["squared_error_sum"] for r in finite) / samples) if samples else None,
            "max_crop_rms": max((r["comparison"]["rms"] for r in finite), default=None),
            "bitwise_equal_crops": sum(r["comparison"]["bitwise_equal"] for r in rows)}


def collect(saved, data, torch):
    from audiovae_student.cache import TrainingCrop
    from audiovae_student.batching import _validate_crop
    if {name: len(pool) for name, pool in saved["pools"].items()} != EXPECTED_POOLS:
        raise ValueError("Original pool inventory changed")
    if {name: len(pool) for name, pool in data["pools"].items()} != EXPECTED_POOLS:
        raise ValueError("Original planned pool inventory changed")
    unique = {}; membership = defaultdict(set); signatures = {}; aliases = 0
    for name, rows in list(saved["pools"].items()) + [("historical_heldout", saved["heldout"])]:
        for index, raw in enumerate(rows):
            crop = TrainingCrop(**raw)
            _validate_crop(crop)
            score_bounds(crop)
            for value in (crop.latents, crop.teacher_audio):
                if value.dtype != torch.float32 or value.device.type != "cpu" or value.requires_grad:
                    raise ValueError("Archive must contain detached CPU FP32 targets")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("Nonfinite archived tensor")
            if name != "historical_heldout":
                window = data["pools"][name][index]["window"]
                if (crop.source_id, crop.start_frame, crop.valid_scored_samples) != (
                        window["source_id"], window["start_frame"], window["valid_output_samples48k"]):
                    raise ValueError("Cached crop differs from its planned source window")
            key = (crop.source_id, crop.start_frame, crop.valid_scored_samples)
            signature = {**{k: getattr(crop, k) for k in GEOMETRY},
                         "latent_sha256": tensor_sha(crop.latents), "target_sha256": tensor_sha(crop.teacher_audio)}
            if key in unique:
                if signature != signatures[key]:
                    raise ValueError("Duplicate window has conflicting geometry or tensors")
                aliases += 1
            else:
                unique[key] = crop; signatures[key] = signature
            membership[key].add(name)
    training = [key for key in unique if "historical_heldout" not in membership[key]]
    heldout = [key for key in unique if "historical_heldout" in membership[key]]
    if len(training) != 17328 or len({key[0] for key in training}) != 4749:
        raise ValueError("Missing unique training crops or sources")
    if len(heldout) != 285 or len({key[0] for key in heldout}) != 147:
        raise ValueError("Missing historical heldout crops or sources")
    if {key[0] for key in training} & {key[0] for key in heldout}:
        raise ValueError("Training and historical heldout source overlap")
    return unique, signatures, membership, aliases


def main(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from audiovae_student.teacher import FrozenAudioVAE2
    from audiovae_student.objective_comparison import state_fingerprint
    from audiovae_student.restart_data import digest as plan_digest
    if args.batch_size != 32:
        raise ValueError("The registered maximum batch size is32")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    if sha(BASE / "targeted-data.json") != EXPECTED_DATA_SHA or sha(BASE / "target-cache.json") != EXPECTED_RECEIPT_SHA:
        raise ValueError("Historical plan or cache receipt changed")
    receipt = json.loads((BASE / "target-cache.json").read_text())
    cache = Path(receipt["path"])
    if receipt["sha256"] != EXPECTED_CACHE_SHA or sha(cache) != EXPECTED_CACHE_SHA:
        raise ValueError("Historical target cache bytes changed")
    data = json.loads((BASE / "targeted-data.json").read_text())
    if plan_digest({k: v for k, v in data.items() if k != "identity_sha256"}) != data["identity_sha256"]:
        raise ValueError("Historical data identity changed")
    saved = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    if saved["identity"] != receipt["identity"]:
        raise ValueError("Cache identity differs from archived receipt")
    unique, signatures, membership, aliases = collect(saved, data, torch)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.version() != 92501:
        raise ValueError("This audit requires the corrected cuDNN9.25.1 runtime")
    teacher = FrozenAudioVAE2.from_files(ASSETS / "audio_vae_v2.py", ASSETS / "audiovae.pth", device="cuda")
    before = state_fingerprint(teacher.model.state_dict())
    if before != EXPECTED_TEACHER_STATE or before != saved["identity"]["teacher_state_sha256"]:
        raise ValueError("Teacher does not match the archived frozen weights")
    record = {"version": VERSION, "parameter_updates": 0, "encoder_calls": 0,
              "target_policy": "Archived conditional latent-to-teacher-waveform pairs; not canonical encoder output",
              "scope": "All17328 unique training pool crops and285 historical heldout crops",
              "cache_path": str(cache), "cache_sha256": EXPECTED_CACHE_SHA,
              "data_plan_sha256": EXPECTED_DATA_SHA, "teacher_state_sha256": before,
              "teacher": teacher.provenance, "runtime": {"torch": str(torch.__version__),
              "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "tf32": False,
              "deterministic": True, "cudnn_benchmark": False}, "context_bound": reset_affected_prefix(),
              "tolerance": {"max_abs": MAX_ABS, "per_crop_rms": RMS, "relative_tolerance": 0},
              "maximum_batch_size": 32, "extra_batch_padding": False, "duplicate_aliases_verified": aliases,
              "unique_crops": len(unique), "manifest_sha256": digest(list(signatures.values())),
              "mask_policy": "Only valid scored samples; exclude first6 scored samples for interior-context crops",
              "script_sha256": sha(__file__), "complete": False, "passed": False}
    write_json(args.out / "inventory.json", {**record, "crops": list(signatures.values())})
    buckets = defaultdict(list)
    for key, crop in unique.items():
        buckets[crop.latents.shape[-1]].append(key)
    rows = []; shape_checks = []; seen_shapes = set(); replays = 0; batch_index = 0
    metadata = dict(data["rows"])
    metadata.update(saved["metadata"])
    with torch.no_grad(), (args.out / "rows.jsonl").open("x") as rows_file:
        for frames in sorted(buckets):
            keys = buckets[frames]
            for offset in range(0, len(keys), args.batch_size):
                selected = keys[offset:offset + args.batch_size]
                z = torch.cat([unique[key].latents for key in selected], dim=0).to(teacher.device)
                input_sha = tensor_sha(z)
                # First call for a shape is deliberately batched. Sentinel
                # qualification must not accidentally warm the singleton first.
                prediction = teacher.decode(z)
                if tensor_sha(z) != input_sha:
                    raise ValueError("Teacher decode mutated the archived latent input")
                if prediction.shape != (len(selected), 1, frames * HOP):
                    raise ValueError("Teacher output shape changed")
                shape = (len(selected), frames)
                if shape not in seen_shapes:
                    singleton = teacher.decode(z[:1].contiguous())
                    a, b = score_bounds(unique[selected[0]])
                    check = metric(prediction[:1, :, a:b].cpu(), singleton[..., a:b].cpu())
                    shape_checks.append({"batch": shape[0], "frames": frames, "source_id": selected[0][0], "comparison": check})
                    seen_shapes.add(shape)
                for index, key in enumerate(selected):
                    crop = unique[key]; a, b = score_bounds(crop)
                    actual = prediction[index:index + 1, :, a:b].cpu()
                    comparison = metric(actual, crop.teacher_audio[..., a:b])
                    meta = metadata.get(crop.source_id, {})
                    row = {**signatures[key], "pools": sorted(membership[key]),
                           "language": meta.get("language", "unknown"), "condition": meta.get("condition", "unknown"),
                           "batch_index": batch_index, "score_start": a, "score_stop": b, "comparison": comparison}
                    if not comparison["passed"] and replays < 8:
                        one = teacher.decode(z[index:index + 1].contiguous()).cpu()[..., a:b]
                        row["singleton_vs_archived"] = metric(one, crop.teacher_audio[..., a:b])
                        row["batch_vs_singleton"] = metric(actual, one)
                        replays += 1
                    rows.append(row)
                    rows_file.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                batch_index += 1
                if batch_index % 10 == 0:
                    rows_file.flush()
                    progress = {"completed": len(rows), "expected": len(unique),
                                "failed": sum(not r["comparison"]["passed"] for r in rows),
                                "batches": batch_index, "elapsed_seconds": time.time() - started}
                    write_json(args.out / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
    cohorts = {"all": rows}
    for name in list(EXPECTED_POOLS) + ["historical_heldout"]:
        cohorts["pool/" + name] = [r for r in rows if name in r["pools"]]
    cohorts["unique_training"] = [r for r in rows if "historical_heldout" not in r["pools"]]
    for field in ("language", "condition"):
        for value in sorted({str(r[field]) for r in rows}):
            cohorts[field + "/" + value] = [r for r in rows if str(r[field]) == value]
    after = state_fingerprint(teacher.model.state_dict())
    preserved = after == before and sha(cache) == EXPECTED_CACHE_SHA
    preserved = preserved and sha(BASE / "targeted-data.json") == EXPECTED_DATA_SHA
    preserved = preserved and sha(BASE / "target-cache.json") == EXPECTED_RECEIPT_SHA
    frozen = not any(p.requires_grad for p in teacher.model.parameters()) and not any(m.training for m in teacher.model.modules())
    record.update({"complete": len(rows) == len(unique), "cohorts": {k: aggregate(v) for k, v in cohorts.items()},
                   "shape_sentinel_checks": shape_checks, "bounded_failed_singleton_replays": replays,
                   "teacher_state_after_sha256": after, "archived_files_unchanged": preserved, "teacher_remained_frozen": frozen,
                   "rows_sha256": sha(args.out / "rows.jsonl"), "elapsed_seconds": time.time() - started,
                   "passed": preserved and frozen and len(rows) == len(unique) and all(r["comparison"]["passed"] for r in rows)
                   and all(r["comparison"]["passed"] for r in shape_checks),
                   "interpretation": "A pass supports this fixed conditional training distribution only. It does not validate old encoder outputs or imply perceptual equivalence. Both canonical and historical validation gates remain required."})
    write_json(args.out / "receipt.json", record)
    print(json.dumps({"complete": record["complete"], "passed": record["passed"], "output": str(args.out)}, sort_keys=True), flush=True)
    return 0 if record["passed"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    import fcntl
    lock = Path("/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock")
    with lock.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(main(args))
