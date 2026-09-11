"""Replay the failed pre-tanh fixture using its exact original source geometry."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import torch

from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.quiet_audio import QuietAudioConfig
from audiovae_student.restart_data import file_sha
from diagnostic_common import load_context, atomic_json, status
from full_source_teacher_head import capture_full_source_pre_tanh
from native_chain import QUARTER_SHA
from training_overlay import load_training_overlay, verify_overlay_files


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
        raise ValueError("Position 483 is not the previously failing crop: " + str((crop.source_id, crop.start_frame)))
    saved = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    config = saved["engine"]["config"]["quiet_audio"]
    if isinstance(config, dict):
        config = QuietAudioConfig(**config)
    teacher = ctx.teacher()
    teacher_sha = ctx.parent["identity"]["data"]["teacher_state_sha256"]
    identity = {"fit_position_one_based": args.fit_position, "pool_index": positions[args.fit_position - 1],
                "source_id": crop.source_id, "start_frame": crop.start_frame,
                "checkpoint_sha256": QUARTER_SHA, "selection_sha256": selection_sha,
                "overlay": overlay["identity"], "teacher_state_sha256": teacher_sha,
                "torch": str(torch.__version__), "cudnn": torch.backends.cudnn.version(),
                "script_sha256": file_sha(Path(__file__)),
                "helper_sha256": file_sha(Path(__file__).parent / "full_source_teacher_head.py")}
    atomic_json(out / "identity.json", identity)
    status("full_source_head_diagnostic", source_id=crop.source_id, fit_position=args.fit_position)
    started = time.monotonic()
    result = capture_full_source_pre_tanh(teacher, crop, ctx.data, config,
                                        expected_teacher_state_sha256=teacher_sha)
    if state_fingerprint(teacher.model.state_dict()) != teacher_sha:
        raise RuntimeError("Teacher state changed")
    if file_sha(selection_path) != selection_sha or file_sha(args.checkpoint) != QUARTER_SHA:
        raise RuntimeError("Selection or preserved checkpoint changed")
    atomic_json(out / "complete.json", {"complete": True, "identity": identity,
        "capture": result["receipt"], "elapsed_seconds": time.monotonic() - started,
        "fresh_files": verify_overlay_files(overlay), "original_files": ctx.verify_files(),
        "student_or_optimizer_updates": 0, "audio_or_weights_written": False,
        "original_cache_targets_changed": False, "exact_contract_preserved": True})
    status("full_source_head_diagnostic_complete", out=str(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "training-receipt", "selection", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--fit-position", type=int, default=483)
    arguments = parser.parse_args()
    lock = Path("/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock")
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(arguments)
