"""Record cold/repeated full-source teacher execution without relaxing parity.

Every invocation remains in the report. No warmup result is silently discarded,
no close latent is accepted, and no cached target or retained model is changed.
"""
import argparse
import fcntl
import json
from pathlib import Path
import time

import torch

from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.restart_data import file_sha
from diagnostic_common import load_context, atomic_json, status
from full_source_teacher_head import (authenticated_source, crop_window, full_source_cache_key,
                                      preserved_teacher, tensor_bytes_sha)
from native_chain import QUARTER_SHA
from training_overlay import load_training_overlay, verify_overlay_files


def difference(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Repeat probe comparison shape/dtype mismatch")
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise ValueError("Repeat probe encountered nonfinite values")
    delta = actual.double() - expected.double()
    return {"exact": torch.equal(actual, expected), "max_abs": float(delta.abs().max()),
            "rms": float(delta.square().mean().sqrt()), "nonzero": int((delta != 0).sum()),
            "elements": delta.numel(), "gt_1e_7": int((delta.abs() > 1e-7).sum()),
            "gt_1e_6": int((delta.abs() > 1e-6).sum()), "gt_1e_5": int((delta.abs() > 1e-5).sum())}


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if file_sha(args.checkpoint) != QUARTER_SHA:
        raise ValueError("Wrong preserved quarter checkpoint")
    selection_path = Path(args.selection).resolve(strict=True)
    selection_sha = file_sha(selection_path)
    selection = json.loads(selection_path.read_text())
    positions = selection["splits"]["fit"]["indices"]
    if not 1 <= args.fit_position <= len(positions):
        raise ValueError("Fit position outside the original selection")
    ctx = load_context()
    overlay = load_training_overlay(ctx, args.training_receipt, required_counts={"targeted_generator": 12800})
    crop = overlay["pools"]["targeted_generator"][positions[args.fit_position - 1]]
    if args.fit_position == 483 and (crop.source_id, crop.start_frame) != ("freesound:179332", 64):
        raise ValueError("Position 483 is not the failing crop: " + str((crop.source_id, crop.start_frame)))
    teacher = ctx.teacher()
    teacher_sha = ctx.parent["identity"]["data"]["teacher_state_sha256"]
    if state_fingerprint(teacher.model.state_dict()) != teacher_sha:
        raise ValueError("Frozen teacher state differs")
    audio, inventory = authenticated_source(ctx.data, crop)
    x = audio.to(teacher.device)
    z_target = crop.latents.to(teacher.device)
    y_target = crop.teacher_audio.to(teacher.device)
    n = crop.latents.shape[-1]
    identity = {"fit_position_one_based": args.fit_position, "pool_index": positions[args.fit_position - 1],
        "source_id": crop.source_id, "start_frame": crop.start_frame, "context_start_frame": crop.context_start_frame,
        "crop_latent_shape": list(crop.latents.shape), "source_inventory": inventory,
        "checkpoint_sha256": QUARTER_SHA, "selection_sha256": selection_sha,
        "overlay": overlay["identity"], "teacher_state_sha256": teacher_sha,
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "script_sha256": file_sha(Path(__file__)),
        "helper_sha256": file_sha(Path(__file__).parent / "full_source_teacher_head.py"),
        "encoder_calls": 4, "decoder_calls_per_unique_full_latent": 4,
        "policy": "All calls measured in order. Exact cache and original full-source key required. No tolerance relaxation."}
    atomic_json(out / "identity.json", identity)
    report = {"identity": identity, "encoder_calls": [], "decoder_groups": []}
    started = time.monotonic()
    unique_z = {}
    previous_z = None
    with preserved_teacher(teacher.model), torch.no_grad(), torch.autocast(device_type=teacher.device.type, enabled=False):
        for call in range(1, 5):
            status("repeat_reference_encoder", call=call, source_id=crop.source_id)
            with torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=True, allow_tf32=False):
                z = teacher.model.encode(x, 16000)
            z_crop = crop_window(z, crop.context_start_frame, n)
            full_sha = tensor_bytes_sha(z)
            row = {"call_one_based": call, "full_shape": list(z.shape), "full_latent_sha256": full_sha,
                   "cached_crop": difference(z_crop, z_target),
                   "same_as_previous_full_output": torch.equal(z, previous_z) if previous_z is not None else None,
                   "previous_full_output_difference": difference(z, previous_z) if previous_z is not None else None}
            report["encoder_calls"].append(row)
            if full_sha not in unique_z:
                unique_z[full_sha] = {"latents": z.detach().clone(), "first_encoder_call": call,
                                      "cached_crop_exact": row["cached_crop"]["exact"]}
            previous_z = z.detach().clone()
            atomic_json(out / "progress.json", report)
        last = teacher.model.decoder.model[-2]
        if not isinstance(last, torch.nn.Conv1d) or not isinstance(teacher.model.decoder.model[-1], torch.nn.Tanh):
            raise ValueError("Unexpected teacher terminal operations")
        for full_sha, entry in unique_z.items():
            z = entry["latents"]
            group = {"full_latent_sha256": full_sha, "first_encoder_call": entry["first_encoder_call"],
                     "cached_crop_exact": entry["cached_crop_exact"], "calls": []}
            report["decoder_groups"].append(group)
            previous_y = None
            for call in range(1, 5):
                status("repeat_full_source_decoder", latent_first_call=entry["first_encoder_call"], call=call)
                captured = []
                handle = last.register_forward_hook(lambda _m, _a, v: captured.append(v.detach().clone()))
                try:
                    y = teacher.decode(z)
                finally:
                    handle.remove()
                if len(captured) != 1:
                    raise RuntimeError("Expected exactly one terminal convolution call")
                pre = captured[0]
                y_crop = crop_window(y, crop.context_start_frame * 1920, n * 1920)
                pre_crop = crop_window(pre, crop.context_start_frame * 1920, n * 1920)
                key = full_source_cache_key(inventory, teacher_sha, z, y)
                row = {"call_one_based": call, "full_waveform_sha256": tensor_bytes_sha(y),
                       "full_pre_tanh_sha256": tensor_bytes_sha(pre), "cached_crop": difference(y_crop, y_target),
                       "native_tanh_of_capture": difference(torch.tanh(pre), y),
                       "cropped_tanh_of_capture_vs_cache": difference(torch.tanh(pre_crop), y_target),
                       "original_full_source_cache_key_exact": key == crop.cache_key,
                       "recreated_full_source_cache_key": key,
                       "same_as_previous_full_output": torch.equal(y, previous_y) if previous_y is not None else None,
                       "previous_full_output_difference": difference(y, previous_y) if previous_y is not None else None}
                group["calls"].append(row)
                previous_y = y.detach().clone()
                atomic_json(out / "progress.json", report)
    if state_fingerprint(teacher.model.state_dict()) != teacher_sha:
        raise RuntimeError("Frozen teacher state changed")
    matches = []
    for group in report["decoder_groups"]:
        last_two = group["calls"][-2:]
        if (group["cached_crop_exact"] and last_two[-1]["same_as_previous_full_output"]
                and all(r["original_full_source_cache_key_exact"] and r["cached_crop"]["exact"]
                        and r["native_tanh_of_capture"]["exact"] for r in last_two)):
            matches.append(group["full_latent_sha256"])
    report.update(complete=True, elapsed_seconds=time.monotonic() - started,
        exact_stable_original_geometry_found=bool(matches), matching_full_latent_hashes=matches,
        encoder_final_two_stable=report["encoder_calls"][-1]["same_as_previous_full_output"],
        encoder_final_two_cached_crop_exact=all(r["cached_crop"]["exact"] for r in report["encoder_calls"][-2:]),
        teacher_state_preserved=True, targets_changed=False, tolerance_relaxed=False,
        fresh_files=verify_overlay_files(overlay), original_files=ctx.verify_files(),
        qualification="Diagnostic evidence only. No candidate weights, replacement targets or new training run.")
    if file_sha(selection_path) != selection_sha or file_sha(args.checkpoint) != QUARTER_SHA:
        raise RuntimeError("Selection or preserved checkpoint changed")
    atomic_json(out / "complete.json", report)
    status("source_repeats_complete", exact_stable_original_geometry_found=bool(matches), out=str(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "training-receipt", "selection", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--fit-position", type=int, default=483)
    args = parser.parse_args()
    lock = Path("/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock")
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
