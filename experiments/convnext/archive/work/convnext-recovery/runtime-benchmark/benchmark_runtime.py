"""Bounded matched H100 training/teacher throughput with disposable engines.

Run once per isolated runtime. The first three distinct batches warm up a
disposable exact resume; the next ten distinct batches are timed. No checkpoint
or tensor artifact is written. Original files are fingerprinted before/after.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

WARMUP, TIMED, BATCH = 3, 10, 32


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def timing_summary(seconds):
    if len(seconds) < 2 or any(not math.isfinite(s) or s <= 0 for s in seconds):
        raise ValueError("At least two positive finite times required")
    mean = statistics.mean(seconds)
    return {"count": len(seconds), "mean_seconds": mean, "median_seconds": statistics.median(seconds),
            "stdev_seconds": statistics.stdev(seconds), "coefficient_of_variation": statistics.stdev(seconds) / mean,
            "min_seconds": min(seconds), "max_seconds": max(seconds), "total_seconds": sum(seconds)}


def unique_batch_identity(crops):
    """Scored audio is unique; reused preceding causal context is permitted."""
    intervals = {}
    identities = []
    for crop in crops:
        start, end = crop.start_frame * 1920, crop.start_frame * 1920 + crop.valid_scored_samples
        if end <= start:
            raise ValueError("Empty scored interval")
        for prior_start, prior_end in intervals.setdefault(crop.source_id, []):
            if start < prior_end and end > prior_start:
                raise ValueError("Repeated/overlapping scored training crop")
        intervals[crop.source_id].append((start, end))
        identities.append({"source_id": crop.source_id, "start_frame": crop.start_frame,
            "context_start_frame": crop.context_start_frame, "latent_frames": crop.latents.shape[-1],
            "valid_scored_samples": crop.valid_scored_samples})
    return identities


def distribution_versions():
    wanted = {"torch", "torchaudio", "torchvision", "numpy", "scipy", "soundfile", "librosa", "pydantic",
              "triton", "tensorboard", "safetensors", "einops", "onnx", "onnxruntime"}
    result = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name", "").lower().replace("_", "-")
        if name in wanted or name.startswith("nvidia-"):
            result[name] = dist.version
    return result


def finite_metrics(metrics):
    if not metrics or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError("Nonfinite or invalid training metrics")


def state_delta(model, initial):
    import torch
    squared_delta = squared_initial = squared_grad = 0.
    count = 0
    rows = {}
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().double()
        previous = initial[name].double()
        difference = value - previous
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Nonfinite parameter after disposable steps")
        gradient = parameter.grad.detach().cpu().double() if parameter.grad is not None else None
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Missing/nonfinite final gradient")
        d2, v2, g2 = (float(x.square().sum()) for x in (difference, previous, gradient))
        squared_delta += d2; squared_initial += v2; squared_grad += g2; count += value.numel()
        rows[name] = {"update_norm": math.sqrt(d2), "update_max_abs": float(difference.abs().max()),
                      "final_postclip_gradient_norm": math.sqrt(g2)}
    return {"scope": "All13 disposable updates, including3warmups; no quality claim",
            "parameter_count": count, "update_norm": math.sqrt(squared_delta),
            "update_rms": math.sqrt(squared_delta / count),
            "relative_update_l2": math.sqrt(squared_delta / squared_initial),
            "final_postclip_gradient_norm": math.sqrt(squared_grad), "parameters": rows}


def synchronized_call(call):
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def run(args):
    import torch
    import torch.nn.functional as F
    from diagnostic_common import load_context, status
    from audiovae_student.objective_comparison import state_fingerprint
    from audiovae_student.training import _restore_rng, _rng_state
    from run_corrected_screen import bind_views
    from repair_natural_history import source_rows, authentic_audio, tensor_hash
    from cache_probe import IDS, LENGTHS

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "runtime-benchmark.json"
    if report_path.exists() or (out / "started.json").exists():
        raise FileExistsError("Use a fresh output directory; refusing to replay completed/started benchmark")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if (torch.backends.cudnn.version() or 0) < args.minimum_cudnn:
        raise RuntimeError("Correctness-matched baseline requires the validated fixed cuDNN library, not9.19")
    runtime = {"label": args.label, "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
        "packages": distribution_versions(), "device": "cuda", "dtype": "float32", "autocast": False,
        "tf32": False, "cudnn_enabled": True, "cudnn_benchmark": False, "deterministic": True,
        "torch_num_threads": torch.get_num_threads(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_json(out / "started.json", runtime)
    ctx = load_context()
    engine = ctx.engine("targeted")
    expected = ctx.checkpoints["targeted"]["engine"]
    actual = engine.state_dict()
    restoration = {key: {"saved_sha256": state_fingerprint(expected[key]), "restored_sha256": state_fingerprint(actual[key])}
        for key in ("model", "discriminators", "optimizer", "discriminator_optimizer", "balancer", "crop_rng")}
    if any(row["saved_sha256"] != row["restored_sha256"] for row in restoration.values()):
        raise RuntimeError("Optimizer/model/EMA resume changed saved raw state")
    if engine.step != 8490 or engine.recipe.total_steps < engine.step + WARMUP + TIMED:
        raise ValueError("Wrong parent or insufficient authorized disposable step budget")
    crops = ctx.pools["gradient_calibration"][:(WARMUP + TIMED) * BATCH]
    if len(crops) != (WARMUP + TIMED) * BATCH:
        raise ValueError("Calibration pool lacks416unique crops")
    crop_identity = unique_batch_identity(crops)
    bind_views(engine, ctx.data["pools"]["gradient_calibration"])
    initial = {name: p.detach().cpu().clone() for name, p in engine.model.named_parameters()}
    source_engine_hash = state_fingerprint(actual)
    del actual
    result = {"format_version": 1, "runtime": runtime, "checkpoint_sha256": ctx.expected_hashes["targeted"],
        "target_cache_sha256": ctx.receipt["target_cache_sha256"], "source_engine_sha256": source_engine_hash,
        "restoration": restoration, "training_crops": crop_identity,
        "training_tensor_sha256": state_fingerprint([{"z": c.latents, "target": c.teacher_audio} for c in crops]),
        "training_policy": {"warmup_updates": WARMUP, "timed_updates": TIMED, "batch": BATCH,
            "unique_scored_crops_within_trial": True, "parent_step": engine.step,
            "rng_policy": "Restore saved parent global RNG; retain exact restored parent crop RNG, with matched planned discriminator views",
            "reuse": "Earlier training/calibration data reused for authorized debugging only",
            "CUDA_synchronization": "Before and after each measured call",
            "timed_work": "Full train_step including CPU-to-GPU batch preparation, G/D forwards/backwards, EMA, clipping and both optimizers",
            "excluded": "Checkpoint loading, source validation, checkpoint writing, full validation, downloads and worker startup"},
        "training_steps": []}
    _restore_rng(ctx.checkpoints["targeted"]["rng"])
    result["restored_global_rng_sha256"] = state_fingerprint(_rng_state())
    torch.cuda.reset_peak_memory_stats()
    for index in range(WARMUP + TIMED):
        selected = crops[index * BATCH:(index + 1) * BATCH]
        metrics, seconds = synchronized_call(lambda: engine.train_step(selected))
        finite_metrics(metrics)
        phase = "warmup" if index < WARMUP else "timed"
        result["training_steps"].append({"index": index, "phase": phase, "seconds": seconds,
            "metrics": metrics, "padded_latent_frames": max(c.latents.shape[-1] for c in selected),
            "D_views": engine.screen_last_views.copy()})
        atomic_json(out / "progress.json", {"stage": "training_microbenchmark", "index": index, "phase": phase,
                                            "seconds": seconds, "step": engine.step})
        status("runtime_training_step", label=args.label, index=index, phase=phase, seconds=seconds)
    result["training_summary"] = timing_summary([r["seconds"] for r in result["training_steps"] if r["phase"] == "timed"])
    timed_samples = sum(r["metrics"]["scored_samples"] for r in result["training_steps"] if r["phase"] == "timed")
    result["training_summary"].update(scored_audio_seconds=timed_samples / 48000,
        scored_audio_seconds_per_wall_second=(timed_samples / 48000) / result["training_summary"]["total_seconds"],
        end_step=engine.step, max_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    result["student_parameter_delta"] = state_delta(engine.model, initial)
    result["end_optimizer_sha256"] = state_fingerprint(engine.optimizer.state_dict())
    result["end_discriminator_optimizer_sha256"] = state_fingerprint(engine.discriminator_optimizer.state_dict())
    result["end_ema_sha256"] = state_fingerprint(engine.balancer.state_dict())
    del engine, initial
    gc.collect(); torch.cuda.empty_cache()

    rows, pins = source_rows(ctx)
    audios, sources = zip(*(authentic_audio(rows[sid]) for sid in IDS))
    if [a.shape[-1] for a in audios] != LENGTHS:
        raise ValueError("Teacher source audio differs from the fixed historical batch")
    teacher = ctx.teacher()
    teacher_state = state_fingerprint(teacher.model.state_dict())
    x = torch.cat([F.pad(audio, (0, 93440 - audio.shape[-1])) for audio in audios]).to(teacher.device)
    cond = torch.full((8,), teacher.sr_cond, dtype=torch.int32, device=teacher.device)
    teacher_report = {"input_ids": IDS, "input_lengths": LENGTHS, "padded_shape": list(x.shape),
        "input_sha256": tensor_hash(x), "source_manifest_pins": pins, "sources": list(sources),
        "teacher_state_sha256": teacher_state, "calls": {},
        "api": "Pinned frozen upstream model.encode / model.decode; cuDNN enabled with validated>=9.25 backend",
        "scope": "H100 batched target preparation, not CPU inference RTF; repeated inputs are timing warmup only"}
    with torch.no_grad():
        # Fixed z is generated once for decode timing, independently of the train engine.
        z = teacher.model.encode(x, teacher.sample_rate_in)
        if z.shape != (8, 64, 146) or z.dtype != torch.float32 or not bool(torch.isfinite(z).all()):
            raise RuntimeError("Teacher encoder output contract failed")
        teacher_report["latent_sha256"] = tensor_hash(z)
        teacher_report["latent_mean"] = float(z.double().mean())
        teacher_report["latent_rms"] = float(z.double().square().mean().sqrt())
        for name, call in (("encode", lambda: teacher.model.encode(x, teacher.sample_rate_in)),
                           ("decode", lambda: teacher.model.decode(z, sr_cond=cond))):
            times = []
            for i in range(WARMUP + TIMED):
                value, seconds = synchronized_call(call)
                if not bool(torch.isfinite(value).all()) or value.dtype != torch.float32:
                    raise FloatingPointError("Nonfinite/non-FP32 teacher output")
                expected_shape = (8, 64, 146) if name == "encode" else (8, 1, 146 * 1920)
                if tuple(value.shape) != expected_shape:
                    raise ValueError("Teacher sample accounting changed")
                times.append({"phase": "warmup" if i < WARMUP else "timed", "seconds": seconds})
            teacher_report["calls"][name] = {"measurements": times,
                "summary": timing_summary([v["seconds"] for v in times if v["phase"] == "timed"]),
                "output_sha256": tensor_hash(value), "output_rms": float(value.double().square().mean().sqrt()),
                "output_max_abs": float(value.abs().max())}
            status("runtime_teacher_complete", label=args.label, component=name,
                   mean_seconds=teacher_report["calls"][name]["summary"]["mean_seconds"])
    if state_fingerprint(teacher.model.state_dict()) != teacher_state:
        raise RuntimeError("Frozen teacher state changed")
    result["teacher"] = teacher_report
    result["checkpoint_preservation"] = ctx.verify_files()
    result["saved_checkpoints"] = 0
    result["retained_training_updates"] = 0
    result["disposable_training_updates"] = WARMUP + TIMED
    result["limits"] = ["One short throughput trial per runtime;10steps do not estimate full-run time or tail latency.",
        "Same data/order/starting state, but floating point differences may make the disposable trajectories diverge.",
        "Parameter deltas and finite losses verify basic resume behavior, not convergence or perceptual quality.",
        "Teacher timing bypasses validation-wrapper overhead using the same frozen upstream operations and weights."]
    atomic_json(report_path, result)
    status("runtime_benchmark_complete", label=args.label, path=str(report_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--minimum-cudnn", type=int, default=92500)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
