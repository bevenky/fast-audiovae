"""Fresh-process, no-update cuDNN parity audit of the retained student and D.

Worker results contain forward tensors and gradients, never a checkpoint. Both
workers see the same saved FP32 latent/target pairs. Their discriminator-only
inputs are constructed on CPU, independently of student output. This is a
bounded backend diagnostic, not proof about every earlier training step.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_tensor(a, b):
    """Raw differences plus a screening flag, not a universal quality bound."""
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"compatible": False, "left_shape": list(a.shape), "right_shape": list(b.shape),
                "left_dtype": str(a.dtype), "right_dtype": str(b.dtype)}
    a, b = a.detach().cpu(), b.detach().cpu()
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not bool(finite.all()):
        return {"compatible": True, "finite": False, "nonfinite_elements": int((~finite).sum())}
    delta = a.double() - b.double()
    norm_a, norm_b, norm_delta = (float(x.double().norm()) for x in (a, b, delta))
    max_abs = float(delta.abs().max()) if delta.numel() else 0.
    relative = norm_delta / max(norm_a, norm_b) if max(norm_a, norm_b) else 0.
    flat_index = int(delta.abs().argmax()) if delta.numel() else 0
    index, remaining = [], flat_index
    for size in reversed(delta.shape):
        index.append(remaining % size)
        remaining //= size
    return {"compatible": True, "finite": True, "shape": list(a.shape),
            "dtype": str(a.dtype), "exact_equal": bool(torch.equal(a, b)),
            "max_abs": max_abs, "rms": float(delta.square().mean().sqrt()) if delta.numel() else 0.,
            "relative_l2_symmetric": relative, "left_norm": norm_a, "right_norm": norm_b,
            "max_abs_index": list(reversed(index)),
            "screen_within_1e-5_abs_plus_relative": bool(torch.allclose(a, b, atol=1e-5, rtol=1e-5)),
            "channel_max_abs": delta.abs().flatten(2).amax((0, 2)).tolist() if delta.ndim >= 3 else None}


def compare_maps(left, right):
    if set(left) != set(right):
        raise ValueError("Tensor evidence keys differ")
    rows = {key: compare_tensor(left[key], right[key]) for key in left}
    suspects = [key for key, row in rows.items()
                if not row.get("screen_within_1e-5_abs_plus_relative", False)]
    return {"tensors": rows, "tensor_count": len(rows), "outside_screen": suspects,
            "all_finite_compatible": all(r.get("compatible") and r.get("finite") for r in rows.values()),
            "max_relative_l2": max((r.get("relative_l2_symmetric", 0.) for r in rows.values()), default=0.)}


def _copy(tensor):
    return tensor.detach().cpu().clone()


def _grad_map(named, values):
    import torch
    return {name: _copy(torch.zeros_like(param) if value is None else value)
            for (name, param), value in zip(named, values)}


def run_worker(args):
    # This branch runs in an otherwise fresh interpreter. No CUDA convolution
    # or context/model import has happened before the backend flag is set.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.backends.cudnn.enabled = args.worker == "on"
    from diagnostic_common import load_context, status
    from diagnose_objectives_updates import _losses_for_prediction, _preserved_attributes
    from audiovae_student.corrected_calibration import _fixed_engine, _scheduled_weights
    from audiovae_student.objective_comparison import state_fingerprint
    from audiovae_student.recipe_v2 import calibration_batch, scored_batch_v2
    from audiovae_student.discriminators import discriminator_loss
    from run_corrected_screen import bind_views

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / (args.worker + ".pt")).exists():
        raise FileExistsError("Fresh evidence directory required; refusing to overwrite worker payload")
    start = time.monotonic()
    ctx = load_context()
    engine = ctx.engine(args.checkpoint)
    crops = ctx.pools[args.pool][:32]
    if len(crops) != 32:
        raise ValueError("Audit requires 32 fixed crops")
    bind_views(engine, ctx.data["pools"][args.pool])
    engine_before = state_fingerprint(engine.state_dict())
    z_cpu, mask_cpu = calibration_batch(crops, "cpu")
    if z_cpu.dtype != torch.float32:
        raise ValueError("Expected preserved FP32 latent cache")
    # Same padded first rows for shape comparisons; padding is not recomputed.
    z, latent_mask = z_cpu.cuda(), mask_cpu.cuda()
    tensor_sets = {}
    metadata = {"format_version": 1, "checkpoint": args.checkpoint,
        "checkpoint_sha256": ctx.expected_hashes[args.checkpoint],
        "checkpoint_step": engine.step, "target_cache_sha256": ctx.receipt["target_cache_sha256"],
        "pool": args.pool, "source_crops": [{"source_id": c.source_id, "start_frame": c.start_frame,
            "context_start_frame": c.context_start_frame, "valid_scored_samples": c.valid_scored_samples} for c in crops],
        "input_identity": state_fingerprint({"latents": z_cpu, "mask": mask_cpu,
            "targets": [c.teacher_audio for c in crops]}),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "gpu": torch.cuda.get_device_name(), "seed": 1861,
        "script_sha256": file_sha(__file__),
        "environment": {key: os.environ.get(key) for key in ("TORCH_CUDNN_V8_API_DISABLED", "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT", "CUBLAS_WORKSPACE_CONFIG")},
        "optimizer_steps": 0, "retained_model_updates": 0,
        "D_state_policy": "Fixed saved discriminator, no D update; not an exact D-then-G training step",
        "limits": ["Representative fixed checkpoint/batches only; does not certify every historical shape or call order.",
            "cuDNN off is a comparison backend, not assumed exact arithmetic.",
            "1e-5 absolute-plus-relative is an anomaly screen; raw norms and errors are authoritative.",
            "D feature evidence retains first eight rows at every layer; all32 output logits and all parameter gradients are retained."]}

    with _fixed_engine(engine, 1861, _preserved_attributes(engine)) as guard:
        # Forward checks precede gradients, so a first-call shape effect is seen.
        with torch.no_grad():
            for label, zz, mm in (("B32_first", z, latent_mask), ("B1", z[:1], latent_mask[:1]),
                                  ("B8", z[:8], latent_mask[:8]),
                                  ("B32_repeat", z, latent_mask),
                                  ("B32_short", z[..., :-1], latent_mask[..., :-1]),
                                  ("B32_after_shape_change", z, latent_mask)):
                if zz.shape[-1] < 2 or not bool(mm.any(dim=1).all()):
                    raise ValueError("Second-shape test cannot drop all valid frames")
                tensor_sets["student_forward/" + label] = {"waveform": _copy(engine.model(zz, scored_latent_mask=mm))}
                status("student_backend_forward", backend=args.worker, shape=label)

        # CPU-defined target views and perturbation are identical in both workers.
        size = engine.recipe.adversarial_samples
        target_cpu = torch.cat([c.teacher_audio[..., c.scored_slice][..., :size] for c in crops])
        if target_cpu.shape != (32, 1, size) or target_cpu.dtype != torch.float32:
            raise ValueError("Every crop must contain the same full discriminator view")
        noise = torch.randn(target_cpu.shape, generator=torch.Generator().manual_seed(1862)) * .01
        fake_cpu = target_cpu + noise
        metadata["D_input_identity"] = state_fingerprint({"target": target_cpu, "fake": fake_cpu})
        fake, target = fake_cpu.cuda(), target_cpu.cuda()

        def d_evidence(audio):
            evidence = {}
            for i, head in enumerate(engine.discriminators(audio)):
                evidence[f"head{i}/logits"] = _copy(head.logits)
                for j, feature in enumerate(head.features):
                    evidence[f"head{i}/feature{j}/first8"] = _copy(feature[:8])
            return evidence

        with torch.no_grad():
            for label, audio in (("B32_first", fake), ("B1", fake[:1]), ("B8", fake[:8]),
                                 ("B32_repeat", fake), ("B32_short", fake[..., :-480]),
                                 ("B32_after_shape_change", fake)):
                tensor_sets["D_forward/" + label] = d_evidence(audio)
                status("D_backend_forward", backend=args.worker, shape=label)

        d_named = tuple((name, p) for name, p in engine.discriminators.named_parameters() if p.requires_grad)
        d_loss = discriminator_loss(engine.discriminators, fake, target,
             example_weights=fake.new_tensor([c.valid_scored_samples for c in crops]))
        d_grads = torch.autograd.grad(d_loss, tuple(p for _, p in d_named), allow_unused=True)
        tensor_sets["D_parameter_gradient"] = _grad_map(d_named, d_grads)
        metadata["D_loss_on_fixed_inputs"] = float(d_loss.detach())
        del d_grads, d_loss
        status("D_backend_gradient", backend=args.worker)

        batch, _ = scored_batch_v2(engine.model, crops, engine.reconstruction, engine.device)
        g_named = tuple((name, p) for name, p in engine.model.named_parameters() if p.requires_grad)
        params = tuple(p for _, p in g_named)
        # Fixed independent cotangent isolates the student backward kernel from D.
        cotangent_cpu = torch.randn(batch.prediction.shape, generator=torch.Generator().manual_seed(1863)) * 1e-4
        cotangent = cotangent_cpu.cuda().masked_fill(~batch.score_mask, 0)
        g_probe = torch.autograd.grad(batch.prediction, params, grad_outputs=cotangent,
                                     retain_graph=True, allow_unused=True)
        tensor_sets["student_fixed_cotangent_parameter_gradient"] = _grad_map(g_named, g_probe)
        metadata["fixed_cotangent_identity"] = state_fingerprint(cotangent_cpu)
        del g_probe
        rng = engine.crop_generator.get_state().clone()
        batch, losses = _losses_for_prediction(engine, batch, batch.prediction, crop_rng=rng)
        balancer = copy.deepcopy(engine.balancer)
        balancer.weights = _scheduled_weights(engine)
        balanced = balancer.combine(losses, batch.prediction, valid_mask=batch.score_mask)
        grads = torch.autograd.grad(batch.prediction, params, grad_outputs=balanced.gradient,
                                    allow_unused=True)
        tensor_sets["G_balanced_parameter_gradient"] = _grad_map(g_named, grads)
        tensor_sets["G_output"] = {"waveform": _copy(batch.prediction), "gradient": _copy(balanced.gradient)}
        metadata["G_raw_losses"] = {name: float(value.detach()) for name, value in losses.items()}
        metadata["G_balancer_metrics"] = balanced.metrics
        metadata["G_balancer_clone_state"] = balancer.state_dict()
        metadata["D_views_for_G"] = copy.deepcopy(engine.screen_last_views)
        status("G_backend_gradient", backend=args.worker)
        del grads, batch, losses, balanced
    metadata["immutable_engine_guard"] = guard
    engine_after = state_fingerprint(engine.state_dict())
    if engine_before != engine_after:
        raise RuntimeError("Backend audit changed engine state, including optimizer, EMA or crop RNG")
    metadata["complete_engine_before_sha256"] = engine_before
    metadata["complete_engine_after_sha256"] = engine_after
    metadata.update(ctx.verify_files())
    metadata["wall_seconds"] = time.monotonic() - start
    metadata["max_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    torch.save({"metadata": metadata, "tensor_sets": tensor_sets}, out / (args.worker + ".pt"))
    atomic_json(out / (args.worker + ".json"), metadata)
    status("backend_worker_complete", backend=args.worker, wall_seconds=metadata["wall_seconds"])


def report(args):
    import torch
    out = Path(args.out)
    left = torch.load(out / "on.pt", map_location="cpu", weights_only=True, mmap=True)
    right = torch.load(out / "off.pt", map_location="cpu", weights_only=True, mmap=True)
    required = ("checkpoint_sha256", "target_cache_sha256", "input_identity", "D_input_identity",
                "fixed_cotangent_identity", "D_views_for_G", "checkpoint_step", "script_sha256",
                "complete_engine_before_sha256")
    for key in required:
        if left["metadata"][key] != right["metadata"][key]:
            raise ValueError("Worker inputs or source differ: " + key)
    if set(left["tensor_sets"]) != set(right["tensor_sets"]):
        raise ValueError("Worker tensor sections differ")
    comparisons = {key: compare_maps(left["tensor_sets"][key], right["tensor_sets"][key])
                   for key in left["tensor_sets"]}
    within = {}
    for mode, data in (("on", left), ("off", right)):
        within[mode] = {}
        for component in ("student_forward", "D_forward"):
            first = data["tensor_sets"][component + "/B32_first"]
            for label, count in (("B1", 1), ("B8", 8), ("B32_repeat", 32), ("B32_after_shape_change", 32)):
                current = data["tensor_sets"][component + "/" + label]
                reference = {key: value[:count] for key, value in first.items()}
                within[mode][component + "/" + label] = compare_maps(reference, current)
    result = {"format_version": 1, "metadata": {"on": left["metadata"], "off": right["metadata"]},
              "on_vs_off": comparisons, "same_backend_batch_and_repeat": within,
              "interpretation": "Forward and parameter-gradient parity at these shapes can localize a backend discrepancy. Passing cannot certify all past updates; failing does not alone quantify historical model-quality damage.",
              "optimizer_steps": 0, "retained_model_updates": 0}
    atomic_json(out / "comparison.json", result)
    print(json.dumps({"comparison": str(out / "comparison.json"),
        "on_vs_off_sections_outside_screen": [k for k, v in comparisons.items() if v["outside_screen"]],
        "within_backend_sections_outside_screen": {mode: [k for k, v in rows.items() if v["outside_screen"]] for mode, rows in within.items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", choices=("parent", "targeted", "complex"), default="targeted")
    parser.add_argument("--pool", default="gradient_calibration")
    parser.add_argument("--worker", choices=("on", "off"))
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
    else:
        if not args.compare_only:
            for mode in ("on", "off"):
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--out", args.out,
                    "--checkpoint", args.checkpoint, "--pool", args.pool, "--worker", mode], check=True,
                    env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        report(args)


if __name__ == "__main__":
    main()
