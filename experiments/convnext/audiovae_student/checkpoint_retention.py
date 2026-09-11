"""Crash-recoverable, bounded retention of immutable training snapshots."""
import hashlib
import json
import os
from pathlib import Path
import shutil


CONTINUATION_FORMAT = "recipe_v2_continuation_v1"


def identity_digest(value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def checkpoint_position(payload, *, expected_identity=None):
    """Validate global updates against this checkpoint's finite segment cursor."""
    identity, sampler = payload["identity"], payload["sampler"]
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Checkpoint run identity changed")
    step, batch, cursor = payload["engine"]["step"], identity["batch_size"], sampler["cursor"]
    if any(type(v) is not int for v in (step, batch, cursor)) or min(step, cursor) < 0 or batch < 1:
        raise ValueError("Invalid checkpoint step, batch size or cursor")
    kind = payload.get("format_version")
    start = 0
    if kind == CONTINUATION_FORMAT:
        start, total = identity["global_start_step"], identity["global_total_steps"]
        count = identity["segment_window_count"]
        if (identity.get("kind") != kind or any(type(v) is not int for v in (start, total, count))
                or start < 1 or count < 1 or identity["parent"]["step"] != start
                or identity["parent"]["batch_size"] != batch
                or total != start + (count + batch - 1) // batch
                or not start <= step <= total):
            raise ValueError("Continuation global budget or parent binding disagrees")
    elif kind == "recipe_v2_pilot":
        count = identity.get("actual_plan", {}).get("metadata", {}).get("windows")
        # Older pilot snapshots lack a finite-plan count and used full batches.
        total = identity.get("recipe", {}).get("total_steps")
        if count is not None and (type(count) is not int or count < 1
                or (total is not None and total != (count + batch - 1) // batch)):
            raise ValueError("Pilot finite window budget disagrees")
        if total is not None and (type(total) is not int or not 0 <= step <= total):
            raise ValueError("Checkpoint exceeds its global update budget")
    else:
        raise ValueError("Unexpected checkpoint format")
    expected_cursor = (step - start) * batch
    if count is not None:
        expected_cursor = min(expected_cursor, count)
        if step - start > (count + batch - 1) // batch:
            raise ValueError("Checkpoint exceeds its finite window plan")
    if cursor != expected_cursor:
        raise ValueError("Checkpoint step and exposure cursor disagree")
    if "window_identity" in identity and sampler.get("identity_sha256") != identity["window_identity"]:
        raise ValueError("Checkpoint sampler identity changed")
    return {"format_version": kind, "step": step, "cursor": cursor,
            "global_start_step": start, "segment_step": step - start,
            "segment_window_count": count, "run_identity_sha256": identity_digest(identity)}


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    tmp.replace(path)


def snapshot_latest(latest, destination, threshold, *, expected_identity=None):
    """Hard-link one atomic checkpoint generation, then inspect that exact inode."""
    import torch
    torch.set_num_threads(1)
    destination.mkdir(parents=True, exist_ok=True)
    pending = destination / "capture.pending.pt"
    if pending.exists():
        raise RuntimeError("Previous checkpoint capture needs inspection")
    # Reserve room for the next trainer atomic save as well as data preparation.
    if shutil.disk_usage(destination).free < 4 * 1024**3 + 2 * latest.stat().st_size:
        return {"state": "deferred_disk_reserve", "threshold": threshold}
    os.link(latest, pending)
    try:
        payload = torch.load(pending, map_location="cpu", weights_only=True)
        position = checkpoint_position(payload, expected_identity=expected_identity)
        step = position["step"]
        if step < threshold:
            pending.unlink()
            return {"state": "waiting_for_committed_checkpoint", "step": step, "threshold": threshold}
        final = destination / f"checkpoint-step{step:06d}.pt"
        with pending.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        receipt = {**position, "state": "retained", "threshold": threshold, "step": step,
                   "path": str(final), "sha256": digest, "bytes": pending.stat().st_size,
                   "cursor": payload["sampler"]["cursor"],
                   "journal_sha256": payload["journal_sha256"],
                   "scope": "Model/optimizer snapshot. Exact live-run resume also requires matching journals; do not replay later exposure."}
        if "metrics_sha256" in payload:
            receipt["metrics_sha256"] = payload["metrics_sha256"]
        if position["format_version"] == CONTINUATION_FORMAT:
            receipt["parent"] = payload["identity"]["parent"]
        if final.exists():
            old = json.loads(final.with_suffix(".json").read_text())
            with final.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != digest or old["sha256"] != digest:
                    raise ValueError("Retained checkpoint generation changed")
            pending.unlink()
            return old
        # Publish recovery metadata first. Startup can finish a rename interrupted here.
        atomic_json(final.with_suffix(".json"), receipt)
        pending.rename(final)
        return receipt
    except BaseException:
        # A capture failure never deletes the original latest checkpoint.
        if pending.exists():
            pending.unlink()
        raise


def reconcile_retention(destination, state, *, thresholds=(3000, 5000, 10000)):
    """Recover publication/pruning independently of the last observer state write."""
    destination.mkdir(parents=True, exist_ok=True)
    pending = destination / "capture.pending.pt"
    receipts = []
    for path in sorted(destination.glob("checkpoint-step*.json")):
        receipt = json.loads(path.read_text())
        expected = destination / f"checkpoint-step{receipt['step']:06d}.pt"
        if Path(receipt["path"]) != expected or expected.is_symlink():
            raise ValueError("Checkpoint receipt points outside owned retention")
        receipts.append(receipt)
    if pending.exists():
        with pending.open("rb") as handle:
            pending_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        matches = [r for r in receipts if not Path(r["path"]).exists() and r["sha256"] == pending_hash]
        if len(matches) > 1:
            raise ValueError("Ambiguous pending checkpoint receipt")
        if matches:
            pending.rename(matches[0]["path"])
        else:
            # Unpublished observer-owned hard link; latest remains untouched.
            pending.unlink()
    retained = []
    for receipt in receipts:
        path = Path(receipt["path"])
        if not path.exists():
            continue  # Receipts for previously pruned snapshots remain historical evidence.
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != receipt["sha256"]:
                raise ValueError("Retained checkpoint hash changed")
        retained.append(receipt)
    state["checkpoints"] = sorted(retained, key=lambda r: r["step"])
    completed = set(state.get("completed_thresholds", ()))
    for receipt in retained:
        completed.add(receipt["threshold"])
        completed.update(s for s in thresholds if s <= receipt["step"])
    state["completed_thresholds"] = sorted(completed)
    while len(state["checkpoints"]) > 2:
        old = state["checkpoints"].pop(0)
        Path(old["path"]).unlink(missing_ok=True)
