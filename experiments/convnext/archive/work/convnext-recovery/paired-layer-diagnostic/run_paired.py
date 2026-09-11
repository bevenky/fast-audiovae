"""Trace both frozen decoders on the identical canonical silence latents."""
import argparse
import fcntl
import importlib
from pathlib import Path
import time

import torch
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.restart_data import file_sha
from native_chain import QUARTER_SHA, restore_quarter
from paired_layers import probe_paired_layers
from run_update_experiment import load_canonical_panel


def run(args):
    from diagnostic_common import atomic_json, status, load_context
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    if file_sha(checkpoint) != QUARTER_SHA:
        raise ValueError("Wrong student checkpoint")
    ctx = load_context()
    panel = load_canonical_panel(ctx, {"requires_canonical_evaluation": True}, args.canonical_receipt)
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    engine = ctx.engine("targeted", device="cuda")
    original = restore_quarter(engine, saved["engine"])
    teacher = ctx.teacher()
    teacher_sha = ctx.parent["identity"]["data"]["teacher_state_sha256"]
    zero = next(c for c in panel["crops"] if c.source_id == "encoded_zero")
    files = {Path(__file__).resolve()}
    for name in ("paired_layers", "diagnose_architecture", "native_chain", "diagnostic_common",
                 "audiovae_student.model", "audiovae_student.fusion_architecture", "audiovae_student.teacher"):
        files.add(Path(importlib.import_module(name).__file__).resolve())
    identity = {"checkpoint_path": str(checkpoint), "checkpoint_sha256": QUARTER_SHA,
        "restored_engine_sha256": original, "canonical_panel": panel["identity"],
        "teacher_model_state_sha256": teacher_sha, "teacher_provenance": teacher.provenance,
        "torch": str(torch.__version__), "cudnn": torch.backends.cudnn.version(),
        "device": str(engine.device), "source_hashes": {str(p): file_sha(p) for p in files},
        "scope": "Paired full six-second digital-silence trace. No natural-audio or peak trace claimed."}
    atomic_json(out / "identity.json", identity)
    status("paired_layer_trace_start")
    started = time.monotonic()
    result = probe_paired_layers(teacher, engine.model, zero,
        expected_teacher_state_sha256=teacher_sha)
    if state_fingerprint(engine.state_dict()) != original or file_sha(checkpoint) != QUARTER_SHA:
        raise RuntimeError("Student state or checkpoint changed")
    if state_fingerprint(teacher.model.state_dict()) != teacher_sha:
        raise RuntimeError("Teacher changed")
    for key in ("receipt", "cache"):
        if file_sha(panel["identity"][key + "_path"]) != panel["identity"][key + "_sha256"]:
            raise RuntimeError("Canonical input changed")
    result.update(identity=identity, seconds=time.monotonic()-started, engine_state_unchanged=True)
    atomic_json(out / "paired-layers.json", result)
    atomic_json(out / "complete.json", {"complete": True, "retained_updates": 0,
        "checkpoint_sha256": QUARTER_SHA, "report_sha256": file_sha(out / "paired-layers.json")})
    status("paired_layer_trace_complete", report=str(out / "paired-layers.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--canonical-receipt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    lock = Path("/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock")
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
