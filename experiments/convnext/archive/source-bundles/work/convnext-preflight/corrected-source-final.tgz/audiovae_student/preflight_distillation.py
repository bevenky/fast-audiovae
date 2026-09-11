"""Bounded real-teacher waveform diagnostic; never launches main training.

The fixed diagnostic clips are intentionally repeated and excluded from main
exposure accounting. Sentinels are evaluation-only. A passed gate is saved for
review, not used to automatically enable perceptual training or another run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .cache import DECODER_HOP, TrainingCrop, UtteranceCache
from .corpus_training import _atomic_json
from .data import load_manifest, validate_manifest
from .distillation_training import (DistillationEngine, DistillationTrainingConfig, evaluate_crops,
    reconstruction_gradient_report, scored_batch, waveform_gate)
from .model import StudentConfig, StudentDecoder
from .source_corpus import SourceCorpus, read_native_16k
from .teacher import CHECKPOINT_SHA256, FrozenAudioVAE2
from .training import _atomic_save, _restore_rng, _rng_state, _runtime_spec


def _plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class PreflightConfig:
    scored_frames: int = 16
    context_frames: int = 29
    evaluation_interval: int = 100
    checkpoint_interval: int = 100
    seed: int = 7
    save_audio: bool = True
    check_fold_streaming: bool = True

    def __post_init__(self):
        for name in ("scored_frames", "evaluation_interval", "checkpoint_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.context_frames) is not int or self.context_frames < 29:
            raise ValueError("At least 29 latent frames of original context are required")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")


def fixed_crops(record: UtteranceCache, *, role: str,
                config: PreflightConfig = PreflightConfig(), minimum_samples: int = 4096) -> tuple[TrainingCrop, ...]:
    """Create genuine beginning/interior views of continuous cached targets."""
    source = record.metadata["identity"]["source"]
    expected = {"diagnostic": "train", "sentinel": "dev"}.get(role)
    if expected is None or source["split"] != expected:
        raise ValueError("Diagnostic/sentinel role does not match the pinned source split")
    maximum_start = (record.valid_output_samples - minimum_samples) // DECODER_HOP
    if maximum_start < 1:
        raise ValueError("Each selected utterance needs valid beginning and interior scored regions")
    interior = min(maximum_start, max(1, (record.latent_frames - config.scored_frames) // 2))
    result = []
    for start in (0, interior):
        context_start = max(0, start - config.context_frames)
        context = start - context_start
        frames = context + config.scored_frames
        def window(value, offset, length):
            selected = value[..., offset:offset + length]
            return F.pad(selected, (0, length - selected.shape[-1])).contiguous().clone()
        valid = min(config.scored_frames * DECODER_HOP, record.valid_output_samples - start * DECODER_HOP)
        if valid < minimum_samples:
            raise ValueError("Fixed crop cannot accommodate all loss FFTs")
        result.append(TrainingCrop(window(record.latents, context_start, frames),
            window(record.teacher_audio, context_start * DECODER_HOP, frames * DECODER_HOP),
            None, record.cache_key, source["source_id"], start, context_start, context,
            config.scored_frames, valid))
    return tuple(result)


def load_selection(diagnostic_manifest: Path, sentinel_manifest: Path,
                   sample_counts: Path, selection_audit: Path):
    audit = json.loads(selection_audit.read_text())
    if audit.get("format_version") != 1 or audit.get("state") != "ready":
        raise ValueError("Selection audit must identify a ready version1 selection")
    files = {"diagnostic.jsonl": diagnostic_manifest, "sentinel.jsonl": sentinel_manifest,
             "input-sample-counts.json": sample_counts}
    for name, path in files.items():
        if audit.get("files", {}).get(name, {}).get("sha256") != _sha(path):
            raise ValueError(f"Selection audit hash mismatch for {name}")
    groups = []
    for path, split in ((diagnostic_manifest, "train"), (sentinel_manifest, "dev")):
        rows = load_manifest(path)
        if len(rows) != 16 or any(row.split != split for row in rows):
            raise ValueError("Exactly 16 diagnostic train and 16 sentinel dev utterances are required")
        rows = [replace(row, audio_path=str((path.parent / row.audio_path).resolve())
                        if not Path(row.audio_path).is_absolute() else row.audio_path) for row in rows]
        groups.append(tuple(rows))
    combined = (*groups[0], *groups[1])
    validate_manifest(combined)
    if len({r.source_id for r in combined}) != 32:
        raise ValueError("Diagnostic and sentinel source identities must be disjoint")
    counts = json.loads(sample_counts.read_text())
    selected_counts = {}
    for row in combined:
        count = counts.get(row.source_id)
        if type(count) is not int or count < 20480:
            raise ValueError("Every selected utterance needs an exact count of at least 1.28 seconds")
        if row.split == "train" and count > 12 * 16000:
            raise ValueError("Diagnostic whole utterances must not exceed 12 seconds")
        selected_counts[row.source_id] = count
    identity = {"selection_audit": audit, "selection_audit_sha256": _sha(selection_audit),
                "files": {name: _sha(path) for name, path in files.items()},
                "teacher_checkpoint_sha256": CHECKPOINT_SHA256}
    return groups[0], groups[1], selected_counts, identity


def _crop_identity(crops):
    return [{"source_id": c.source_id, "cache_key": c.cache_key, "start_frame": c.start_frame,
             "context_frames": c.context_frames, "scored_frames": c.scored_frames,
             "valid_scored_samples": c.valid_scored_samples, "latents_sha256": _tensor_sha(c.latents),
             "teacher_sha256": _tensor_sha(c.teacher_audio)} for c in crops]


def _implementation_identity():
    names = ("preflight_distillation.py", "distillation_training.py", "model.py", "optimizers.py",
             "losses_distillation.py", "discriminators.py", "gradient_balancer.py", "batching.py",
            "cache.py", "source_corpus.py", "teacher.py", "prepare_targets.py", "batched_teacher.py")
    return {name: _sha(Path(__file__).with_name(name)) for name in names}


@contextmanager
def _writer(log_dir, run_name, identity, step, resuming):
    if log_dir is None:
        yield None
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError("Run name must be a single safe directory name")
    path = Path(log_dir) / run_name
    path.mkdir(parents=True, exist_ok=True)
    record = path / "run.json"
    if record.exists() and json.loads(record.read_text()) != identity:
        raise ValueError("TensorBoard run identity changed")
    if any(path.glob("events.out.tfevents.*")) and (not resuming or not record.exists()):
        raise ValueError("Existing TensorBoard events require their exact checkpoint resume")
    saved_rng = _rng_state()
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        _atomic_json(record, identity)
        writer = SummaryWriter(str(path), max_queue=1, flush_secs=1,
                               purge_step=step + 1 if resuming else None)
        writer.add_text("run/qualification", "Real-teacher repeated fixed-set diagnostic. "
                        "Sentinels are evaluation-only. This is not unique main-corpus exposure.", step)
        writer.flush()
    finally:
        _restore_rng(saved_rng)
    try:
        yield writer
    finally:
        if writer is not None:
            writer.flush()
            writer.close()


def _evaluate(engine, diagnostic, sentinel):
    rows = {"diagnostic": evaluate_crops(engine, diagnostic), "sentinel": evaluate_crops(engine, sentinel)}
    gates = {}
    for role, entries in rows.items():
        gates[role] = {position: waveform_gate(engine, [r for r in entries if (r["start_frame"] == 0) == (position == "beginning")])
                       for position in ("beginning", "interior")}
        gates[role]["all"] = waveform_gate(engine, entries)
    return {"step": engine.step, "rows": rows, "gates": gates,
            "diagnostic_passed": gates["diagnostic"]["all"]["passed"],
            "sentinel_passed": gates["sentinel"]["all"]["passed"]}


def _log_evaluation(writer, report):
    if writer is None:
        return
    for role, rows in report["rows"].items():
        for position in ("beginning", "interior"):
            subset = [r for r in rows if (r["start_frame"] == 0) == (position == "beginning")]
            for key in ("teacher_waveform_raw", "teacher_waveform", "teacher_mel", "rms_db_error",
                        "waveform_to_silence_error_ratio", "waveform_cosine"):
                writer.add_scalar(f"evaluation/{role}/{position}/{key}", sum(r[key] for r in subset) / len(subset), report["step"])
        writer.add_scalar(f"evaluation/{role}/gate_passed", float(report["gates"][role]["all"]["passed"]), report["step"])
    writer.flush()


@torch.no_grad()
def fold_streaming_check(model, crop):
    """Eight small CPU chunks, with no timing or RTF measurement."""
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        reference = deepcopy(model).cpu().eval()
        if not all(bool(getattr(reference, name).statistics_frozen) for name in ("stem_norm", "affine")):
            raise ValueError("Acceptance requires explicitly frozen normalization statistics")
        latents = crop.latents[..., :min(16, crop.latents.shape[-1])].cpu()
        expected = reference(latents)
        folded = reference.fold_normalization()
        folded_audio = folded(latents)
        stream = folded.stream()
        parts = [stream.decode_chunk(chunk) for chunk in torch.tensor_split(latents, 8, dim=-1) if chunk.shape[-1]]
        streamed = torch.cat(parts, dim=-1)
        torch.testing.assert_close(folded_audio, expected, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(streamed, expected, atol=1e-5, rtol=1e-4)
        if expected.shape[-1] != latents.shape[-1] * DECODER_HOP or streamed.shape != expected.shape:
            raise AssertionError("Fold/stream sample accounting changed")
        return {"passed": True, "device": "cpu", "threads": 1, "chunks": len(parts),
                "samples": expected.shape[-1], "fold_max_abs": float((folded_audio-expected).abs().max()),
                "stream_max_abs": float((streamed-expected).abs().max()), "rtf_measured": False}
    finally:
        torch.set_num_threads(original_threads)


@torch.no_grad()
def _save_audio(engine, groups, output_dir):
    import soundfile as sf
    previous = engine.model.training
    results = []
    try:
        engine.model.eval()
        directory = output_dir / "audio"
        directory.mkdir(exist_ok=True)
        for role, crops in groups.items():
            for crop in crops:
                prefix = f"{role}-{hashlib.sha256(crop.source_id.encode()).hexdigest()[:16]}-{crop.start_frame}"
                batch = scored_batch(engine.model, [crop], engine.criterion, engine.device)
                names = {}
                for label, value in (("teacher", batch.targets[0]), ("student", batch.predictions[0])):
                    name = prefix + f"-{label}.wav"
                    sf.write(directory / name, value.detach().cpu()[0, 0].numpy(), 48000, subtype="FLOAT")
                    names[label] = f"audio/{name}"
                results.append({"source_id": crop.source_id, "start_frame": crop.start_frame,
                                "role": role, "samples": crop.valid_scored_samples, **names})
    finally:
        engine.model.train(previous)
    _atomic_json(output_dir / "audio.json", {"sample_rate": 48000, "subtype": "FLOAT", "gain_changed": False, "items": results})
    return results


def run_preflight(engine: DistillationEngine, diagnostic_crops: Sequence[TrainingCrop],
                  sentinel_crops: Sequence[TrainingCrop], output_dir: str | Path, *,
                  data_identity: dict, config: PreflightConfig = PreflightConfig(),
                  resume_from: str | Path | None = None, log_dir=None,
                  run_name="corrected-waveform-preflight-v1", max_updates_this_call: int | None = None):
    """Run at most the declared 500 updates, then stop even when the gate passes."""
    diagnostic, sentinel = tuple(diagnostic_crops), tuple(sentinel_crops)
    if not diagnostic or not sentinel or engine.config.total_steps > 500:
        raise ValueError("Preflight requires both sets and a budget at most500 updates")
    if engine.perceptual_start is not None:
        raise ValueError("This runner only performs the reconstruction trainability diagnostic")
    if data_identity.get("teacher_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("Data identity must bind the original teacher checkpoint")
    if {c.source_id for c in diagnostic} & {c.source_id for c in sentinel}:
        raise ValueError("Sentinel identities cannot enter the diagnostic training set")
    for role, crops in (("diagnostic", diagnostic), ("sentinel", sentinel)):
        if not any(c.start_frame == 0 for c in crops) or not any(c.start_frame > 0 for c in crops):
            raise ValueError(f"{role} needs beginning and interior crops")
    if max_updates_this_call is not None and (type(max_updates_this_call) is not int or max_updates_this_call < 1):
        raise ValueError("Partial invocation budget must be positive")
    runtime = {**_runtime_spec(), "device": str(engine.device), "dtype": str(next(engine.model.parameters()).dtype),
               "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    if engine.device.type == "cuda":
        runtime.update({"gpu_name": torch.cuda.get_device_name(engine.device), "cuda_version": torch.version.cuda})
    identity = _plain({"kind": "bounded_real_teacher_waveform_preflight", "format_version": 1,
        "config": asdict(config), "training": asdict(engine.config), "model": engine.model.config.to_dict(),
        "loss": asdict(engine.criterion.config), "balancer": asdict(engine.balancer.config),
        "discriminator": asdict(engine.discriminators.config), "data": data_identity,
        "diagnostic": _crop_identity(diagnostic), "sentinel": _crop_identity(sentinel),
        "runtime": runtime, "implementation": _implementation_identity(), "run_name": run_name,
        "main_exposure_hours": 0, "repeated_diagnostic": True})
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This diagnostic run already has an active writer") from error
        record = directory / "identity.json"
        if record.exists():
            if json.loads(record.read_text()) != identity or resume_from is None:
                raise ValueError("Existing output requires an exact identified resume; choose a new directory")
        elif any(p.name != ".run.lock" for p in directory.iterdir()):
            raise ValueError("Fresh diagnostic output directory must be empty")
        else:
            if resume_from is not None:
                raise ValueError("Resume must use its existing identified output directory")
            if engine.step != 0:
                raise ValueError("The corrected diagnostic requires a fresh student/optimizer")
            _atomic_json(record, identity)
        evaluated_step, exposure, latest_metrics = -1, 0, {}
        latest_checkpoint = None
        if resume_from is not None:
            path = Path(resume_from).resolve(strict=True)
            if path.parent != directory or path.is_symlink():
                raise ValueError("Resume checkpoint must belong to this run directory")
            pointer = json.loads((directory / "latest.json").read_text())
            if path.name != pointer["path"] or _sha(path) != pointer["sha256"]:
                raise ValueError("Resume must use the verified latest checkpoint")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved.get("format_version") != 1 or saved.get("identity") != identity:
                raise ValueError("Checkpoint data/model/runtime identity changed")
            engine.load_state_dict(saved["engine"])
            if engine.perceptual_start is not None:
                raise ValueError("Perceptual state cannot enter the diagnostic runner")
            _restore_rng(saved["rng"])
            evaluated_step, exposure = saved["evaluated_step"], saved["repeated_scored_samples"]
            latest_metrics, latest_checkpoint = saved["latest_metrics"], pointer
        started = time.monotonic()
        def status(state, **extra):
            _atomic_json(directory / "status.json", {"state": state, "step": engine.step,
                "total_steps": engine.config.total_steps, "phase": "reconstruction_diagnostic",
                "repeated_scored_hours": exposure / 48000 / 3600, "main_exposure_hours": 0,
                "latest_metrics": latest_metrics, "latest_checkpoint": latest_checkpoint,
                "elapsed_this_invocation_seconds": time.monotonic() - started, **extra})
        def checkpoint():
            nonlocal latest_checkpoint
            name = f"checkpoint-step{engine.step:06d}.pt"
            path = directory / name
            if path.exists():
                if latest_checkpoint is None or latest_checkpoint["path"] != name or _sha(path) != latest_checkpoint["sha256"]:
                    raise ValueError("An immutable checkpoint already exists for this step")
                return
            _atomic_save({"format_version": 1, "identity": identity, "engine": engine.state_dict(),
                "rng": _rng_state(), "evaluated_step": evaluated_step, "repeated_scored_samples": exposure,
                "latest_metrics": latest_metrics}, path)
            latest_checkpoint = {"path": name, "sha256": _sha(path), "step": engine.step}
            _atomic_json(directory / "latest.json", latest_checkpoint)
        try:
            with _writer(log_dir, run_name, identity, engine.step, resume_from is not None) as writer:
                if engine.step == 0 and not (directory / "gradient-initial.json").exists():
                    _atomic_json(directory / "gradient-initial.json", reconstruction_gradient_report(engine, diagnostic[:2]))
                if evaluated_step != engine.step and engine.step == 0:
                    report = _evaluate(engine, diagnostic, sentinel)
                    _atomic_json(directory / f"evaluation-step{engine.step:06d}.json", report)
                    _log_evaluation(writer, report)
                    evaluated_step = engine.step
                checkpoint()
                limit = engine.config.total_steps if max_updates_this_call is None else min(engine.config.total_steps, engine.step + max_updates_this_call)
                while engine.step < limit:
                    before = time.monotonic()
                    latest_metrics = engine.train_step(diagnostic)
                    latest_metrics["step_seconds"] = time.monotonic() - before
                    exposure += latest_metrics["scored_samples"]
                    if writer is not None:
                        for key, value in latest_metrics.items():
                            writer.add_scalar(f"train/{key}", value, engine.step)
                        writer.add_scalar("exposure/repeated_diagnostic_hours", exposure / 48000 / 3600, engine.step)
                        writer.flush()
                    if engine.step % config.evaluation_interval == 0 or engine.step == engine.config.total_steps:
                        report = _evaluate(engine, diagnostic, sentinel)
                        _atomic_json(directory / f"evaluation-step{engine.step:06d}.json", report)
                        _log_evaluation(writer, report)
                        evaluated_step = engine.step
                    if engine.step % config.checkpoint_interval == 0 or engine.step == limit:
                        checkpoint()
                    status("running")
                if engine.step < engine.config.total_steps:
                    status("paused")
                    return {"state": "paused", "step": engine.step, "latest_checkpoint": latest_checkpoint}
                summary_path = directory / "summary.json"
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text())
                    status(summary["state"], diagnostic_passed=summary["diagnostic_passed"])
                    return summary
                report = json.loads((directory / f"evaluation-step{engine.step:06d}.json").read_text())
                _atomic_json(directory / "gradient-final.json", reconstruction_gradient_report(engine, diagnostic[:2]))
                streaming = fold_streaming_check(engine.model, diagnostic[0]) if config.check_fold_streaming else {"performed": False}
                audio = _save_audio(engine, {"diagnostic": diagnostic, "sentinel": sentinel}, directory) if config.save_audio else []
                summary = {"state": "gate_passed_awaiting_review" if report["diagnostic_passed"] else "gate_failed",
                    "step": engine.step, "diagnostic_passed": report["diagnostic_passed"], "sentinel_passed": report["sentinel_passed"],
                    "gates": report["gates"], "fold_streaming": streaming, "audio_items": len(audio),
                    "latest_checkpoint": latest_checkpoint, "repeated_scored_hours": exposure / 48000 / 3600,
                    "main_exposure_hours": 0, "automatically_launched_training": False, "runtime": runtime}
                _atomic_json(summary_path, summary)
                status(summary["state"], diagnostic_passed=summary["diagnostic_passed"])
                return summary
        except BaseException as error:
            status("error", error_type=type(error).__name__, error=str(error))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic-manifest", "sentinel-manifest", "sample-counts", "selection-audit",
                 "teacher-source", "teacher-checkpoint", "cache-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--run-name", default="corrected-waveform-preflight-v1")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    diagnostic_rows, sentinel_rows, counts, data_identity = load_selection(args.diagnostic_manifest,
        args.sentinel_manifest, args.sample_counts, args.selection_audit)
    print("Selection verified: 16 diagnostic utterances and 16 held-out utterances.", flush=True)
    print("Loading the pinned frozen AudioVAE2 teacher.", flush=True)
    teacher = FrozenAudioVAE2.from_files(args.teacher_source, args.teacher_checkpoint, device=args.device)
    from .prepare_targets import _target_teacher
    teacher = _target_teacher(teacher, diagnostic_rows[0])
    payload = Path(diagnostic_rows[0].audio_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != diagnostic_rows[0].audio_sha256:
        raise ValueError("Teacher warmup source SHA mismatch")
    warmup = read_native_16k(payload, diagnostic_rows[0]).to(teacher.device)
    print("Qualifying the original teacher startup path.", flush=True)
    for _ in range(2):
        teacher.decode(teacher.encode(warmup))
    corpus = SourceCorpus((*diagnostic_rows, *sentinel_rows), teacher, cache_dir=args.cache_dir,
        input_sample_counts=counts, max_disk_bytes=1024**3, min_free_bytes=4 * 1024**3,
        max_memory_utterances=4, allow_prepared_source=True)
    try:
        print("Checking serial/batched teacher parity and preparing cached targets.", flush=True)
        prefill = corpus.prefetch([row.source_id for row in (*diagnostic_rows, *sentinel_rows)],
                                  max_batch_size=8, max_total_input_samples=1_920_000)
        print(json.dumps({"teacher_prefill": prefill}, allow_nan=False), flush=True)
        prefill_path = args.cache_dir / "initial-prefill.json"
        if prefill_path.exists():
            if json.loads(prefill_path.read_text()).get("source_corpus_identity_sha256") != corpus.identity_sha256:
                raise ValueError("Existing teacher-prefill report belongs to a different source corpus")
        else:
            _atomic_json(prefill_path, {"source_corpus_identity_sha256": corpus.identity_sha256,
                                       "report": prefill})
        # Timing/cache-hit counts differ on resume. Preserve and bind the
        # initial preparation evidence instead of changing the run identity.
        data_identity["initial_teacher_prefill_sha256"] = _sha(prefill_path)
        config = PreflightConfig()
        diagnostic = tuple(c for row in diagnostic_rows for c in fixed_crops(corpus.get(row.source_id), role="diagnostic", config=config))
        sentinel = tuple(c for row in sentinel_rows for c in fixed_crops(corpus.get(row.source_id), role="sentinel", config=config))
        data_identity["source_corpus"] = corpus.identity
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        model = StudentDecoder(StudentConfig(normalization_mode="masked_batch_norm")).to(args.device)
        engine = DistillationEngine(model, config=DistillationTrainingConfig())
        print("Starting the bounded 500-update reconstruction diagnostic; the main run remains stopped.", flush=True)
        result = run_preflight(engine, diagnostic, sentinel, args.output_dir, data_identity=data_identity,
            config=config, resume_from=args.resume, log_dir=args.log_dir, run_name=args.run_name)
    finally:
        corpus.close()
    print(json.dumps(result, indent=2))
    return 0 if result["diagnostic_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
