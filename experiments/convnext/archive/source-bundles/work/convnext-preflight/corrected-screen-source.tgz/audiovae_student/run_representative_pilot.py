"""Launch the bounded fresh pilot from an audited plan and local Runpod audio.

No downloads occur here. Teacher source/checkpoint hashes and native prepared
audio hashes are checked by the existing frozen-teacher corpus implementation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .data import load_manifest
from .distillation_training import DistillationTrainingConfig
from .losses_distillation import DistillationLossConfig
from .model import StudentConfig
from .preflight_distillation import PreflightConfig, fixed_crops
from .prepare_targets import _target_teacher
from .representative_pilot import (RepresentativePilotConfig, _atomic_json, load_restart_plan,
                                    run_representative_pilot)
from .restart_data import file_sha
from .source_corpus import SourceCorpus, read_native_16k
from .teacher import CHECKPOINT_SHA256, FrozenAudioVAE2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("restart-plan", "heldout-manifest", "sample-counts", "teacher-source",
                 "teacher-checkpoint", "cache-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--config", type=Path, help="JSON with pilot, training, model, loss and heldout sections")
    parser.add_argument("--heldout-metadata", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--run-name", default="representative-reconstruction-pilot-v1")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--cache-mib", type=int, default=512)
    args = parser.parse_args(argv)
    if args.cache_mib < 1:
        parser.error("--cache-mib must be positive")
    spec = json.loads(args.config.read_text()) if args.config else {}
    if set(spec) - {"pilot", "training", "model", "loss", "heldout"}:
        raise ValueError("Unknown pilot configuration section")
    plan = load_restart_plan(args.restart_plan)
    heldout = load_manifest(args.heldout_manifest)
    counts = json.loads(args.sample_counts.read_text())
    if any(counts.get(r.source_id) != plan["counts"][r.source_id] for r in plan["rows"]):
        raise ValueError("Training sample counts differ from the immutable restart plan")
    all_rows = tuple(plan["rows"]) + tuple(heldout)
    needed_counts = {r.source_id: counts[r.source_id] for r in all_rows}
    pilot_values = spec.get("pilot", {})
    batch_size = pilot_values.get("batch_size", 32)
    config = RepresentativePilotConfig(**{"total_steps": math.ceil(len(plan["windows"]) / batch_size), **pilot_values})
    training = DistillationTrainingConfig(**{
        "total_steps": config.total_steps, "warmup_steps": 50, "freeze_normalization_step": 200,
        "reconstruction_waveform_share": 1.0, "quiet_gradient_share": config.quiet_gradient_share,
        "quiet_window_gate": True, "learning_rate_schedule": "constant_after_warmup",
        "parameter_update_metrics_interval": 100, **spec.get("training", {})})
    model = StudentConfig(**{"normalization_mode": "masked_batch_norm", **spec.get("model", {})})
    loss = DistillationLossConfig(**spec.get("loss", {}))
    panel_config = PreflightConfig(**{"scored_frames": 16, "context_frames": config.context_frames,
                                    **spec.get("heldout", {})})
    metadata = json.loads(args.heldout_metadata.read_text()) if args.heldout_metadata else None
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    teacher = FrozenAudioVAE2.from_files(args.teacher_source, args.teacher_checkpoint, device=args.device)
    teacher = _target_teacher(teacher, plan["rows"][0])
    warmup_source = plan["rows"][0]
    audio = Path(warmup_source.audio_path).read_bytes()
    if hashlib.sha256(audio).hexdigest() != warmup_source.audio_sha256:
        raise ValueError("Teacher warmup source checksum changed")
    audio = read_native_16k(audio, warmup_source).to(teacher.device)
    with torch.no_grad():
        for _ in range(2):
            teacher.decode(teacher.encode(audio))
    corpus = SourceCorpus(all_rows, teacher, cache_dir=args.cache_dir, input_sample_counts=needed_counts,
        max_disk_bytes=args.cache_mib * 1024**2, min_free_bytes=config.min_free_bytes,
        max_memory_utterances=4, allow_prepared_source=True)
    try:
        report = corpus.prefetch([r.source_id for r in heldout], max_batch_size=config.teacher_batch_size,
                                max_total_input_samples=config.teacher_batch_samples)
        prefill_path = args.cache_dir / "initial-pilot-prefill.json"
        if prefill_path.exists():
            if json.loads(prefill_path.read_text())["source_corpus_identity_sha256"] != corpus.identity_sha256:
                raise ValueError("Initial teacher qualification belongs to a different corpus")
        else:
            _atomic_json(prefill_path, {"source_corpus_identity_sha256": corpus.identity_sha256, "report": report})
        panel = tuple(c for row in heldout for c in fixed_crops(corpus.get(row.source_id), role="sentinel",
                                    config=panel_config, minimum_samples=max(loss.fft_sizes)))
        data_identity = {"teacher_checkpoint_sha256": CHECKPOINT_SHA256, "source_corpus": corpus.identity,
            "teacher_batch_qualification": {"initial_report_sha256": file_sha(prefill_path),
                                            "source_corpus_identity_sha256": corpus.identity_sha256},
            "restart_plan": plan["identity"], "heldout_manifest_sha256": file_sha(args.heldout_manifest),
            "sample_counts_sha256": file_sha(args.sample_counts), "panel_config": asdict(panel_config)}
        result = run_representative_pilot(corpus, plan["windows"], plan["rows"], counts, plan["ledger"],
            panel, heldout, args.output_dir, data_identity=data_identity, config=config, training_config=training,
            model_config=model, loss_config=loss, device=args.device, resume_from=args.resume,
            max_updates=args.max_updates, heldout_metadata=metadata, reserved_rows=plan["reserved"],
            excluded_sources=plan["diagnostic"], log_dir=args.log_dir, run_name=args.run_name)
    finally:
        corpus.close()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
