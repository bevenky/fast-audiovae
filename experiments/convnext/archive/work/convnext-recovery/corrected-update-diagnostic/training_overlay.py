"""Explicit sealed fresh-encoder training pairs, never an old-cache relabel."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path

VERSION = "corrected_training_pairs_v1"


def geometry(crops):
    # Order matters: pools are an indexed schedule, not a set of source IDs.
    return [(c.source_id, c.start_frame, c.context_start_frame, c.context_frames,
             c.scored_frames, c.valid_scored_samples, list(c.latents.shape), list(c.teacher_audio.shape)) for c in crops]


def _pool_contract(ctx, name, fresh):
    import torch
    from audiovae_student.batching import _validate_crop
    from audiovae_student.preflight_distillation import _crop_identity
    from audiovae_student.restart_data import digest
    if name not in ctx.pools or not fresh or len(fresh) > len(ctx.pools[name]):
        raise ValueError("Fresh pool must be a nonempty prefix of an existing indexed pool")
    baseline = ctx.pools[name][:len(fresh)]
    for crop in fresh:
        _validate_crop(crop)
        if crop.latents.dtype != torch.float32 or crop.teacher_audio.dtype != torch.float32:
            raise TypeError("Fresh decoder-training pairs must remain FP32")
    if geometry(fresh) != geometry(baseline):
        raise ValueError("Fresh pool changed original index order, source or crop geometry")
    for old, new in zip(baseline, fresh):
        if ((old.reference16k is None) != (new.reference16k is None)
                or (old.reference16k is not None and not torch.equal(old.reference16k, new.reference16k))):
            raise ValueError("Fresh pool changed the pinned reference waveform")
    return {"count": len(fresh), "baseline_pool_prefix_identity_sha256": digest(_crop_identity(baseline)),
            "fresh_pool_identity_sha256": digest(_crop_identity(fresh)),
            "ordered_geometry_sha256": digest(geometry(fresh))}


def seal_training_overlay(ctx, pools, path, *, provenance):
    """Root-generated fresh TrainingCrop lists -> .pt plus sibling receipt.json.

    pools may contain gradient_calibration[:32] for the small diagnostic and/or
    all12800 targeted_generator windows for the later two-arm continuation.
    This function does not run an encoder or generate any audio.
    """
    import torch
    from audiovae_student.restart_data import digest, file_sha
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Fresh-source generation and verified backend provenance required")
    json.dumps(provenance, allow_nan=False)
    if not pools or not set(pools).issubset({"gradient_calibration", "targeted_generator"}):
        raise ValueError("Only the declared diagnostic or continuation pool may be overlaid")
    path = Path(path).resolve()
    receipt_path = path.with_name("receipt.json")
    if path.exists() or receipt_path.exists():
        raise FileExistsError("Fresh versioned overlay paths are required")
    contract = {"version": VERSION, "base_target_cache_sha256": ctx.receipt["target_cache_sha256"],
        "base_data_plan_sha256": ctx.identity["data_plan_sha256"],
        "teacher_state_sha256": ctx.parent["identity"]["data"]["teacher_state_sha256"],
        "pools": {name: _pool_contract(ctx, name, list(crops)) for name, crops in pools.items()},
        "provenance": provenance,
        "interpretation": "Fresh whole-source encoder means and teacher decoder targets, sliced to the existing indexed crop geometry; original caches preserved"}
    contract["identity_sha256"] = digest(contract)
    payload = {"format_version": VERSION, "contract": contract,
               "pools": {name: [asdict(crop) for crop in crops] for name, crops in pools.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        torch.save(payload, handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)
    receipt = {"format_version": 1, "kind": VERSION, "path": str(path), "sha256": file_sha(path), "contract": contract}
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return receipt_path


def load_training_overlay(ctx, receipt_path, *, required_counts):
    import torch
    from audiovae_student.cache import TrainingCrop
    from audiovae_student.restart_data import digest, file_sha
    receipt_path = Path(receipt_path).resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("format_version") != 1 or receipt.get("kind") != VERSION:
        raise ValueError("Wrong training overlay format")
    path = Path(receipt["path"]).resolve(strict=True)
    if path.parent != receipt_path.parent or file_sha(path) != receipt["sha256"]:
        raise ValueError("Training overlay bytes/path changed")
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    contract = receipt["contract"]
    if saved.get("format_version") != VERSION or saved["contract"] != contract:
        raise ValueError("Training overlay receipt/payload disagree")
    if contract.get("version") != VERSION or digest({k: v for k, v in contract.items() if k != "identity_sha256"}) != contract["identity_sha256"]:
        raise ValueError("Training overlay contract checksum differs")
    expected = {"base_target_cache_sha256": ctx.receipt["target_cache_sha256"],
        "base_data_plan_sha256": ctx.identity["data_plan_sha256"],
        "teacher_state_sha256": ctx.parent["identity"]["data"]["teacher_state_sha256"]}
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError("Training overlay belongs to a different baseline: " + key)
    pools = {name: [TrainingCrop(**crop) for crop in crops] for name, crops in saved["pools"].items()}
    if set(pools) != set(contract["pools"]):
        raise ValueError("Training overlay pools changed")
    for name, count in required_counts.items():
        if name not in pools or len(pools[name]) != count:
            raise ValueError("Training overlay missing exact required indexed pool: " + name)
    for name, crops in pools.items():
        if _pool_contract(ctx, name, crops) != contract["pools"][name]:
            raise ValueError("Training overlay tensor identity differs: " + name)
    return {"pools": pools, "receipt": receipt, "identity": {"receipt_path": str(receipt_path),
        "receipt_sha256": file_sha(receipt_path), "cache_path": str(path), "cache_sha256": receipt["sha256"],
        "contract_sha256": contract["identity_sha256"], "pool_contracts": contract["pools"]}}


def verify_overlay_files(overlay):
    from audiovae_student.restart_data import file_sha
    identity = overlay["identity"]
    if (file_sha(identity["receipt_path"]) != identity["receipt_sha256"]
            or file_sha(identity["cache_path"]) != identity["cache_sha256"]):
        raise RuntimeError("Training overlay changed during the diagnostic/continuation")
    return {"fresh_training_overlay_files_unchanged": True}
