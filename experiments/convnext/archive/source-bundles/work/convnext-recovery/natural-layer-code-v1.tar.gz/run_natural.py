"""Four fixed natural-audio cases, frozen traces and local branch sensitivities."""
import argparse
import fcntl
import importlib
from pathlib import Path
import time

import torch

from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.quiet_audio import QuietAudioConfig
from audiovae_student.restart_data import file_sha
from diagnose_architecture import _eval_preserved
from native_chain import QUARTER_SHA, restore_quarter
from natural_layers import probe_natural_layers
from run_update_experiment import load_canonical_panel


CASES = (
    ("cmn_hans_cn:validation:10312747577451821721.wav", 178, "teacher_selected_quiet_1"),
    ("cmn_hans_cn:validation:10116334326946811387.wav", 396, "teacher_selected_quiet_2"),
    ("sindhi:5629499534317154_chunk_1.flac", 17, "previously_identified_sindhi_peak"),
    ("freesound:25794", 45, "previously_identified_laughter_peak"),
)


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
    lookup = {(c.source_id, c.start_frame): c for c in panel["crops"]}
    cases = [(lookup[(sid, start)], role) for sid, start, role in CASES]
    config = engine.config.quiet_audio
    if isinstance(config, dict):
        config = QuietAudioConfig(**config)
    files = {Path(__file__).resolve()}
    for name in ("natural_layers", "paired_layers", "diagnose_architecture", "native_chain", "diagnostic_common",
                 "audiovae_student.model", "audiovae_student.fusion_architecture", "audiovae_student.teacher"):
        files.add(Path(importlib.import_module(name).__file__).resolve())
    identity = {"checkpoint_path": str(checkpoint), "checkpoint_sha256": QUARTER_SHA,
        "restored_engine_sha256": original, "canonical_panel": panel["identity"],
        "teacher_model_state_sha256": teacher_sha, "teacher_provenance": teacher.provenance,
        "torch": str(torch.__version__), "cudnn": torch.backends.cudnn.version(),
        "device": str(engine.device), "source_hashes": {str(p): file_sha(p) for p in files},
        "selection": [{"source_id": sid, "start_frame": start, "role": role} for sid, start, role in CASES],
        "scope": "Four preselected diagnostic cases, not a representative quality benchmark. Native forward hooks plus local residual-branch gain derivatives at unchanged weights. No gain perturbations or layer replacements."}
    atomic_json(out / "identity.json", identity)
    status("natural_layer_trace_start")
    started = time.monotonic()
    zero = next(c for c in panel["crops"] if c.source_id == "encoded_zero")
    with torch.no_grad(), _eval_preserved(teacher.model), torch.autocast(device_type="cuda", enabled=False):
        values = [teacher.decode(zero.latents.to(teacher.device)) for _ in range(3)]
        target = zero.teacher_audio.to(teacher.device)
        warmup = {"calls": 3, "first_vs_canonical_max_abs": float((values[0]-target).abs().max()),
                  "second_equals_third": torch.equal(values[1], values[2]),
                  "third_equals_canonical": torch.equal(values[2], target)}
        if not warmup["second_equals_third"] or not warmup["third_equals_canonical"]:
            raise RuntimeError("Frozen teacher execution did not stabilize exactly on canonical silence")
    result = probe_natural_layers(teacher, engine.model, cases,
        quiet_config=config, expected_teacher_state_sha256=teacher_sha)
    if state_fingerprint(engine.state_dict()) != original or file_sha(checkpoint) != QUARTER_SHA:
        raise RuntimeError("Student state or checkpoint changed")
    if state_fingerprint(teacher.model.state_dict()) != teacher_sha:
        raise RuntimeError("Teacher changed")
    for key in ("receipt", "cache"):
        if file_sha(panel["identity"][key + "_path"]) != panel["identity"][key + "_sha256"]:
            raise RuntimeError("Canonical input changed")
    result.update(identity=identity, seconds=time.monotonic()-started,
                  engine_state_unchanged=True, teacher_execution_warmup=warmup)
    atomic_json(out / "natural-layers.json", result)
    atomic_json(out / "complete.json", {"complete": True, "retained_updates": 0,
        "checkpoint_sha256": QUARTER_SHA, "report_sha256": file_sha(out / "natural-layers.json")})
    status("natural_layer_trace_complete", report=str(out / "natural-layers.json"))


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
