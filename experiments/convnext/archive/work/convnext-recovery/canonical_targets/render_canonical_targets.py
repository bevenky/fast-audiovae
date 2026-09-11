"""Versioned, read-only-source reconstruction of the existing 285-crop panel.

Default command inventories source bytes only. --render explicitly runs the
frozen teacher. Nothing writes the old targets, checkpoints, or evaluation code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

FIXTURES = {"encoded_zero", "encoded_quiet_noise", "encoded_fade"}
GEOMETRY = ("source_id", "start_frame", "context_start_frame", "context_frames",
            "scored_frames", "valid_scored_samples")
VERSION = "canonical-encoder-cudnn-disabled-v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def source_registry(ctx):
    """Add the pinned Yell row omitted by the older natural-history registry."""
    from repair_natural_history import source_rows
    rows, pins = source_rows(ctx)
    base = ctx.out.parent
    addon = base / "corrected-screen/addon-validation.json"
    expected = ctx.identity["targets"]["heldout"].get("addon_validation_sha256")
    if not expected:
        raise ValueError("Original target identity does not pin the additional Yell manifest")
    if sha(addon) != expected:
        raise ValueError("Additional heldout source manifest bytes changed")
    pins[str(addon)] = expected
    for entry in json.loads(addon.read_text())["entries"]:
        row = entry["row"]
        sid = row["source_id"]
        if sid in rows and rows[sid] != row:
            raise ValueError("Conflicting source rows: " + sid)
        rows[sid] = row
    return rows, pins


def source_input(ctx, sid, rows):
    import torch
    from repair_natural_history import authentic_audio
    if sid not in FIXTURES:
        if sid not in rows:
            raise ValueError("No authentic source row for " + sid)
        audio, proof = authentic_audio(rows[sid])
        return audio, {**proof, "source_kind": "natural", "manifest_row": rows[sid],
                       "prepared_audio_sha256": tensor_sha(audio)}
    crops = [c for c in ctx.heldout if c.source_id == sid]
    if len(crops) != 1:
        raise ValueError("Fixture must have exactly one complete historical crop")
    crop = crops[0]
    if (crop.context_start_frame, crop.start_frame, crop.context_frames,
            crop.scored_frames, crop.valid_scored_samples) != (0, 0, 0, 150, 288000):
        raise ValueError("Fixture no longer contains the original complete six seconds")
    audio = crop.reference16k
    if audio is None or audio.shape != (1, 1, 96000) or audio.dtype != torch.float32:
        raise ValueError("Fixture is missing its exact original input tensor")
    if not bool(torch.isfinite(audio).all()):
        raise ValueError("Nonfinite fixture input")
    # The containing old cache is byte-pinned by DiagnosticContext. No fixture
    # is regenerated from an RNG, source label, latent zero or truncated crop.
    return audio.detach().contiguous().clone(), {
        "source_id": sid, "source_kind": "synthetic_fixture",
        "prepared_audio_sha256": tensor_sha(audio), "prepared_samples16k": 96000,
        "sample_rate_hz": 16000, "scope": "Exact full historical six-second fixture input",
        "origin_target_cache_sha256": ctx.receipt["target_cache_sha256"],
        "gain_policy": "unchanged", "reference_reconstructed": False}


def inventory(ctx):
    from diagnostic_common import crop_mask
    rows, pins = source_registry(ctx)
    ids = list(dict.fromkeys(c.source_id for c in ctx.heldout))
    if len(ctx.heldout) != 285 or len(ids) != 147 or not FIXTURES.issubset(ids):
        raise ValueError("Unexpected original heldout panel geometry")
    report = {"version": VERSION, "source_manifest_sha256": pins, "sources": [],
              "historical_target_cache_sha256": ctx.receipt["target_cache_sha256"],
              "crop_count": 285, "source_count": 147, "natural_sources": 144,
              "fixture_sources": 3, "source_errors": [], "parameter_updates": 0,
              "source_scope": "Complete manifest segment, not necessarily complete parent recording"}
    for sid in ids:
        try:
            audio, proof = source_input(ctx, sid, rows)
            crops = [c for c in ctx.heldout if c.source_id == sid]
            n = audio.shape[-1]
            for c in crops:
                end = c.start_frame * 1920 + c.valid_scored_samples
                if end > n * 3 or c.context_start_frame + c.context_frames != c.start_frame:
                    raise ValueError("Historical scored interval exceeds authentic source")
                if c.latents.shape[-1] != c.context_frames + c.scored_frames:
                    raise ValueError("Historical storage does not match frame geometry")
            proof["crops"] = [{**{k: getattr(c, k) for k in GEOMETRY},
                                "historical_mask_samples": int(crop_mask(c, c.teacher_audio).sum()),
                                "historical_reference_present": c.reference16k is not None}
                               for c in crops]
            report["sources"].append(proof)
        except (ValueError, OSError, KeyError) as error:
            report["source_errors"].append({"source_id": sid, "error": str(error)})
    report["ready"] = len(report["sources"]) == 147 and not report["source_errors"]
    report["identity_sha256"] = digest(report)
    return rows, report


def _slice(value, start, length):
    import torch.nn.functional as F
    part = value[..., start:start + length]
    return F.pad(part, (0, length - part.shape[-1])).contiguous().clone()


def changed(a, b):
    error = a.double() - b.double()
    return {"max_abs": float(error.abs().max()), "rms": float(error.square().mean().sqrt()),
            "bitwise_equal": bool(a.equal(b))}


def render(ctx, out, rows, source_inventory):
    import torch
    from audiovae_student.batching import _validate_crop
    from audiovae_student.objective_comparison import state_fingerprint
    from audiovae_student.quiet_audio import quiet_window_metrics, QuietAudioConfig
    from diagnostic_common import crop_mask, status
    from audiovae_student.preflight_distillation import _crop_identity
    if not source_inventory["ready"]:
        raise ValueError("Every source must verify before any model call")
    if (out / "heldout.pt").exists() or (out / "receipt.json").exists():
        raise ValueError("Refusing to overwrite a canonical target cache")
    teacher = ctx.teacher()
    model_sha = state_fingerprint(teacher.model.state_dict())
    modes_before = [m.training for m in teacher.model.modules()]
    if any(modes_before) or any(p.requires_grad for p in teacher.model.parameters()):
        raise ValueError("Teacher must be frozen and in evaluation mode")
    original_target_hashes = [(tensor_sha(c.latents), tensor_sha(c.teacher_audio),
                               tensor_sha(c.reference16k) if c.reference16k is not None else None)
                              for c in ctx.heldout]
    source_proofs = {p["source_id"]: p for p in source_inventory["sources"]}
    replacement = {}; receipts = []; quiet = {"all": 0, "natural": 0, "fixtures": 0}; steady_zero = None
    with torch.no_grad():
        for sid in dict.fromkeys(c.source_id for c in ctx.heldout):
            status("canonical_heldout_source", source=sid, completed=len(receipts), total=147)
            audio, proof = source_input(ctx, sid, rows)
            if proof["prepared_audio_sha256"] != source_proofs[sid]["prepared_audio_sha256"]:
                raise ValueError("Source input changed after inventory")
            n = audio.shape[-1]; frames = (n + 639) // 640
            device_audio = audio.to(teacher.device); input_sha = tensor_sha(device_audio)
            old_enabled = torch.backends.cudnn.enabled
            with torch.backends.cudnn.flags(enabled=False, benchmark=False,
                                             deterministic=True, allow_tf32=False):
                z = teacher.encode(device_audio)
            if torch.backends.cudnn.enabled != old_enabled:
                raise ValueError("Encoder scope failed to restore cuDNN backend state")
            if tensor_sha(device_audio) != input_sha:
                raise ValueError("Frozen encoder mutated original input")
            if z.shape != (1, 64, frames) or z.dtype != torch.float32 or not bool(torch.isfinite(z).all()):
                raise ValueError("Canonical encoder violated raw-mu FP32 contract")
            latent_sha = tensor_sha(z)
            # Decode the complete source once, never restart at a scored crop.
            y = teacher.decode(z)
            if tensor_sha(z) != latent_sha:
                raise ValueError("Teacher decoder mutated its input latents")
            if y.shape != (1, 1, frames * 1920) or y.dtype != torch.float32 or not bool(torch.isfinite(y).all()):
                raise ValueError("Canonical teacher decoder shape/dtype/finite mismatch")
            z = z.detach().cpu(); y = y.detach().cpu()
            source_receipt = {"source_id": sid, "input_samples16k": n,
                              "prepared_audio_sha256": input_sha, "latent_sha256": latent_sha,
                              "teacher_audio_sha256": tensor_sha(y), "latent_frames": frames,
                              "decode_cudnn_enabled": old_enabled, "crop_differences": []}
            cache_key = digest({"version": VERSION, "source_id": sid,
                                "prepared_audio_sha256": input_sha, "teacher_state_sha256": model_sha,
                                "latent_sha256": latent_sha, "teacher_audio_sha256": tensor_sha(y)})
            for index, old in enumerate(ctx.heldout):
                if old.source_id != sid:
                    continue
                length = old.context_frames + old.scored_frames
                c = replace(old, latents=_slice(z, old.context_start_frame, length),
                            teacher_audio=_slice(y, old.context_start_frame * 1920, length * 1920),
                            reference16k=(_slice(audio, old.context_start_frame * 640, length * 640)
                                          if old.reference16k is not None else None), cache_key=cache_key)
                _validate_crop(c)
                if any(getattr(c, k) != getattr(old, k) for k in GEOMETRY):
                    raise ValueError("Historical crop geometry changed")
                if not crop_mask(c, c.teacher_audio).equal(crop_mask(old, old.teacher_audio)):
                    raise ValueError("Historical scored mask changed")
                replacement[index] = c
                q = quiet_window_metrics(c.teacher_audio, c.teacher_audio, crop_mask(c, c.teacher_audio))
                count = sum(w["is_quiet"] for w in q["windows"])
                quiet["all"] += count; quiet["fixtures" if sid in FIXTURES else "natural"] += count
                if sid == "encoded_zero":
                    steady = [w for w in q["windows"] if w["is_quiet"] and w["start_sample"] >= 96000]
                    steady_zero = {"quiet_windows": len(steady),
                                   "quiet_samples": sum(w["valid_samples"] for w in steady)}
                source_receipt["crop_differences"].append({"start_frame": c.start_frame,
                    "latents": changed(c.latents, old.latents),
                    "latent_channel63": changed(c.latents[:, 63:64], old.latents[:, 63:64]),
                    "target": changed(c.teacher_audio, old.teacher_audio), "canonical_quiet_windows": count})
            receipts.append(source_receipt)
            write_json(out / "progress.json", {"completed_sources": len(receipts), "total_sources": 147,
                                               "version": VERSION, "source_receipts": receipts})
    crops = [replacement[i] for i in range(285)]
    if state_fingerprint(teacher.model.state_dict()) != model_sha or modes_before != [m.training for m in teacher.model.modules()]:
        raise ValueError("Frozen teacher state or modes changed")
    if original_target_hashes != [(tensor_sha(c.latents), tensor_sha(c.teacher_audio),
                                   tensor_sha(c.reference16k) if c.reference16k is not None else None)
                                  for c in ctx.heldout]:
        raise ValueError("Historical panel tensors changed")
    preservation = ctx.verify_files()
    contract = {"version": VERSION, "format_version": 1,
        "source_inventory_identity_sha256": source_inventory["identity_sha256"],
        "historical_target_cache_sha256": ctx.receipt["target_cache_sha256"],
        "teacher": teacher.provenance, "teacher_state_sha256": model_sha,
        "encoder_execution": "Full authentic source, singleton batch, FP32 raw_mu, scoped cudnn.enabled=False",
        "decoder_execution": "Normal frozen teacher decode, singleton full source, unchanged48k conditioning",
        "crop_policy": "Historical start/context/frame counts and valid samples preserved; full-source targets sliced afterwards",
        "quiet_mask_policy": "Recomputed from canonical teacher targets; historical4073windows is not asserted",
        "quiet_config": asdict(QuietAudioConfig()), "canonical_quiet_windows": quiet,
        "canonical_zero_steady": steady_zero,
        "mask_policy": "Original valid samples; exclude first6samples only when context_start_frame>0",
        "crop_count": 285, "source_count": 147, "natural_sources": 144, "fixture_sources": 3,
        "scored_samples": sum(int(crop_mask(c, c.teacher_audio).sum()) for c in crops),
        "old_panel_preserved": True, "parameter_updates": 0, "checkpoint_preservation": preservation,
        "runtime": {"torch": str(torch.__version__), "cudnn": torch.backends.cudnn.version(),
                    "cuda": torch.version.cuda, "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                    "deterministic": torch.are_deterministic_algorithms_enabled()},
        "renderer_sha256": sha(__file__), "source_receipts": receipts}
    contract["heldout_identity"] = _crop_identity(crops)
    contract["metadata_sha256"] = digest(ctx.metadata)
    contract["identity_sha256"] = digest(contract)
    temp = out / "heldout.pt.tmp"
    with temp.open("xb") as handle:
        torch.save({"format_version": 1, "contract": contract,
                    "heldout": [asdict(c) for c in crops], "metadata": ctx.metadata}, handle)
        handle.flush(); os.fsync(handle.fileno())
    os.link(temp, out / "heldout.pt"); temp.unlink()
    write_json(out / "receipt.json", {"path": str(out / "heldout.pt"),
               "sha256": sha(out / "heldout.pt"), "contract": contract})
    status("canonical_heldout_complete", path=str(out / "heldout.pt"), quiet_windows=quiet)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--render", action="store_true")
    a = p.parse_args()
    out = a.output.resolve()
    if not out.is_relative_to(Path("/tmp")):
        raise ValueError("Canonical output must be a new separate directory under /tmp")
    out.mkdir(parents=True, exist_ok=False)
    from diagnostic_common import load_context
    ctx = load_context()
    rows, report = inventory(ctx)
    write_json(out / "source-inventory.json", report)
    print(json.dumps({"ready": report["ready"], "sources": len(report["sources"]),
                      "errors": report["source_errors"], "path": str(out / "source-inventory.json")}))
    if a.render:
        render(ctx, out, rows, report)


if __name__ == "__main__":
    main()
