"""Run the fixed-checkpoint architecture diagnosis without updating weights."""
import argparse
import fcntl
import importlib
from pathlib import Path
import time

import torch
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.quiet_audio import QuietAudioConfig
from audiovae_student.restart_data import file_sha
from diagnose_architecture import probe_silence_architecture, select_natural_quiet
from native_chain import QUARTER_SHA, restore_quarter
from run_update_experiment import load_canonical_panel


def run(args):
    from diagnostic_common import atomic_json, status, load_context
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    if file_sha(checkpoint) != QUARTER_SHA:
        raise ValueError("Wrong retained checkpoint")
    ctx = load_context()
    panel = load_canonical_panel(ctx, {"requires_canonical_evaluation": True}, args.canonical_receipt)
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    engine = ctx.engine("targeted", device="cuda")
    original = restore_quarter(engine, saved["engine"])
    q = engine.config.quiet_audio
    config = QuietAudioConfig(**q) if isinstance(q, dict) else q
    selected, selection = select_natural_quiet(panel["crops"], panel["metadata"],
        history_samples=engine.model.config.history_frames * 480 + 1920, limit=6, config=config)
    zero = next(c for c in panel["crops"] if c.source_id == "encoded_zero")
    identity = {"checkpoint_path": str(checkpoint), "checkpoint_sha256": QUARTER_SHA,
        "restored_engine_sha256": original, "canonical_panel": panel["identity"],
        "torch": str(torch.__version__), "cudnn": torch.backends.cudnn.version(),
        "device": str(engine.device), "model_config": engine.model.config.to_dict(),
        "natural_selection": selection,
        "source_hashes": {str(Path(importlib.import_module(name).__file__).resolve()):
                          file_sha(importlib.import_module(name).__file__)
                          for name in ("diagnose_architecture", "native_chain", "diagnostic_common",
                                       "audiovae_student.model", "audiovae_student.fusion_architecture")}}
    identity["source_hashes"][str(Path(__file__).resolve())] = file_sha(__file__)
    atomic_json(out / "identity.json", identity)
    status("architecture_probe_start", natural_sources=len(selected))
    started = time.monotonic()
    result = probe_silence_architecture(engine.model, zero, device=engine.device,
        quiet_crops=selected, quiet_config=config)
    if state_fingerprint(engine.state_dict()) != original or file_sha(checkpoint) != QUARTER_SHA:
        raise RuntimeError("Preserved checkpoint or loaded state changed")
    for key in ("receipt", "cache"):
        if file_sha(panel["identity"][key + "_path"]) != panel["identity"][key + "_sha256"]:
            raise RuntimeError("Canonical input changed")
    result.update(identity=identity, seconds=time.monotonic()-started, engine_state_unchanged=True)
    atomic_json(out / "architecture.json", result)
    atomic_json(out / "complete.json", {"complete": True, "retained_updates": 0,
        "checkpoint_sha256": QUARTER_SHA, "report_sha256": file_sha(out / "architecture.json")})
    status("architecture_probe_complete", report=str(out / "architecture.json"))


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
