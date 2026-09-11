"""One-pass speech distillation with true batches and a bounded teacher cache.

The source manifest is fixed before a run begins. The original teacher remains
frozen, and the no-replacement sampler never wraps. A bootstrap checkpoint may
initialize this explicitly new data phase without discarding optimizer state.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from .batching import batched_scored_crop_loss
from .cache import sample_training_crop
from .corpus_training import _atomic_json, _fixed_dev_crop, scored_crop_loss
from .data import load_manifest
from .losses import WarmupLossConfig, WarmupReconstructionLoss
from .model import StudentConfig, StudentDecoder
from .optimizers import build_optimizer_bundle
from .sampling import NoRepeatSegmentSampler
from .source_corpus import SourceCorpus
from .teacher import CHECKPOINT_SHA256, FrozenAudioVAE2
from .training import _atomic_save, _restore_rng, _rng_state, _runtime_spec


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def implementation_identity():
    names = ("batching", "cache", "corpus_training", "data", "losses", "model", "optimizers",
             "sampling", "source_corpus", "source_training", "teacher", "training")
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / (name + ".py")).read_bytes()).hexdigest() for name in names}


@dataclass(frozen=True)
class SourceTrainingConfig:
    total_steps: int = 10000
    batch_size: int = 64
    scored_frames: int = 64
    context_frames: int = 29
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip_norm: float = 1.0
    optimizer: str = "muon_adamw"
    checkpoint_interval: int = 1000
    seed: int = 7

    def __post_init__(self):
        for name in ("total_steps", "batch_size", "scored_frames", "checkpoint_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("context_frames", "warmup_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("learning_rate", "grad_clip_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")


def finite_parameters(parameters):
    """One host synchronization instead of one per parameter tensor."""
    return bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all().item())


def batch_metrics(result, gradient_norm=None):
    values = dict(result.mean)
    if gradient_norm is not None:
        values["gradient_norm"] = gradient_norm
    numbers = torch.stack([x.detach() for x in values.values()]).cpu().tolist()
    if not all(math.isfinite(x) for x in numbers):
        raise FloatingPointError("Nonfinite loss or gradient norm")
    return {**dict(zip(values, numbers)), "examples": result.examples,
            "reference_examples": sum(len(g.indices) for g in result.groups if g.has_reference),
            "scored_samples": result.valid_scored_samples,
            "padded_latent_frames": result.padded_latent_frames}


def initialize_bootstrap(path, model, optimizer, config, loss_config):
    """Import only an identified student, retaining its optimizer/global step."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    identity = checkpoint.get("identity", {})
    if identity.get("kind") != "cached_corpus_reconstruction_warmup":
        raise ValueError("Initialization requires an identified corpus warmup checkpoint")
    if identity.get("teacher", {}).get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("Bootstrap checkpoint does not identify the pinned original teacher")
    if identity.get("model_config") != plain(model.config.to_dict()):
        raise ValueError("Bootstrap student architecture differs")
    if identity.get("loss_config") != plain(asdict(loss_config)):
        raise ValueError("Do not silently change the reconstruction objective")
    old = identity["training_config"]
    for key in ("learning_rate", "warmup_steps", "weight_decay", "betas", "grad_clip_norm", "optimizer"):
        if old[key] != plain(asdict(config))[key]:
            raise ValueError(f"Bootstrap optimizer/schedule setting differs: {key}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    _restore_rng(checkpoint["rng"])
    step = int(checkpoint["step"])
    if not 0 <= step < config.total_steps:
        raise ValueError("Bootstrap step must precede the target step")
    with Path(path).open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    lineage = {"checkpoint_sha256": checkpoint_sha256,
               "checkpoint_step": step, "previous_data_fingerprint": identity["data_fingerprint"],
               "previous_effective_batch": old["accumulation_steps"],
               "new_batch_size": config.batch_size,
               "change": "new fixed source corpus; no-replacement scored segments; true model batches"}
    return step, lineage


@torch.no_grad()
def evaluate(model, corpus, config, criterion, device):
    """One final reconstruction check; no updates or repeated training samples."""
    previous, rng = model.training, _rng_state()
    totals, examples, samples = {}, 0, 0
    try:
        model.eval()
        for row in corpus.rows:
            if row.split != "dev":
                continue
            crop = _fixed_dev_crop(corpus.get(row.source_id), config.scored_frames)
            values = scored_crop_loss(model, crop, criterion, device)
            numbers = torch.stack([v.detach() for v in values.values()]).cpu().tolist()
            for name, number in zip(values, numbers):
                totals[name] = totals.get(name, 0.0) + number
            examples += 1
            samples += crop.valid_scored_samples
        if not examples:
            raise ValueError("Source corpus has no held-out development utterances")
        return {**{k: v / examples for k, v in totals.items()}, "examples": examples, "scored_samples": samples}
    finally:
        model.train(previous)
        _restore_rng(rng)


def train_sources(model, corpus, output_dir, *, config=SourceTrainingConfig(),
                  loss_config=WarmupLossConfig(), device="cpu", initialize_from=None,
                  resume_from=None, log_dir=None, run_name="warmup-speech-500h-v1",
                  stop_after_updates=None, preparation_identity=None):
    if not isinstance(model, StudentDecoder):
        raise TypeError("Only the standalone StudentDecoder may be trained")
    if initialize_from is not None and resume_from is not None:
        raise ValueError("Choose initialization or exact resume")
    if config.context_frames * 4 < model.config.history_frames:
        raise ValueError("Student context is shorter than its receptive field")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Training requires explicit CPU or CUDA")
    model.to(device)
    optimizer = build_optimizer_bundle(model, optimizer=config.optimizer, lr=config.learning_rate,
                                        weight_decay=config.weight_decay, betas=config.betas)
    parameters = list(model.parameters())
    minimum = max(math.ceil(max(loss_config.teacher_fft_sizes) / 3), max(loss_config.reference_fft_sizes_16k))
    sampler = NoRepeatSegmentSampler([r for r in corpus.rows if r.split == "train"],
                                    {r.source_id: corpus.input_sample_counts[r.source_id]
                                     for r in corpus.rows if r.split == "train"},
                                    scored_frames=config.scored_frames, seed=config.seed,
                                    include_short_tail=True, min_input_samples=minimum)
    runtime = _runtime_spec()
    if device.type == "cuda":
        runtime.update(cuda_runtime=str(torch.version.cuda),
                       cuda_device=torch.cuda.get_device_name(device),
                       cudnn_version=torch.backends.cudnn.version())
    identity = plain({"kind": "source_corpus_one_pass", "format_version": 1,
                      "corpus": corpus.identity, "model_config": model.config.to_dict(),
                      "preparation": preparation_identity, "implementation": implementation_identity(),
                      "training_config": asdict(config), "loss_config": asdict(loss_config),
                      "optimizer_groups": optimizer.group_fingerprint, "runtime": runtime,
                      "device_type": device.type,
                      "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                      "cudnn_benchmark": torch.backends.cudnn.benchmark,
                      "cudnn_deterministic": torch.backends.cudnn.deterministic})
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "latest.pt"
    if latest.exists() and resume_from is None:
        raise ValueError("Resume the existing checkpoint or choose another output directory")
    step, lineage, last_train, validation, seconds = 0, None, {}, {}, 0.0
    if initialize_from is not None:
        step, lineage = initialize_bootstrap(initialize_from, model, optimizer, config, loss_config)
    if resume_from is not None:
        saved = torch.load(resume_from, map_location="cpu", weights_only=True)
        if saved.get("identity") != identity:
            raise ValueError("Resume source/model/loss/optimizer/runtime identity mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        sampler.load_state_dict(saved["sampler"])
        _restore_rng(saved["rng"])
        step, lineage = saved["step"], saved["lineage"]
        last_train, validation, seconds = saved["last_train"], saved["validation"], saved["seconds"]
    origin_step = lineage["checkpoint_step"] if lineage else 0
    if (type(step) is not int or type(origin_step) is not int or
            not 0 <= origin_step <= step <= config.total_steps or
            sampler.emitted_segments != (step - origin_step) * config.batch_size):
        raise ValueError("Checkpoint step, lineage and unique segment consumption disagree")
    if (config.total_steps - step) * config.batch_size > sampler.remaining_segments:
        raise ValueError("Insufficient unique scored segments: acquire more audio or reduce the requested batch/run length")
    if not any(r.split == "dev" for r in corpus.rows):
        raise ValueError("A held-out development split is required")
    if any(corpus.input_sample_counts[r.source_id] < minimum for r in corpus.rows if r.split == "dev"):
        raise ValueError("Development clips must contain enough valid samples for every loss FFT")
    if stop_after_updates is not None and (type(stop_after_updates) is not int or stop_after_updates < 1):
        raise ValueError("stop_after_updates must be a positive integer")
    # Inventory is an O(corpus size) report. Do not recompute it per update.
    available_scored_hours = sampler.inventory()["unique_scored_audio_hours"]
    writer = None
    if log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        log_path = Path(log_dir) / run_name
        if any(log_path.glob("events.out.tfevents.*")) and resume_from is None:
            raise ValueError("TensorBoard run exists; use exact resume or a new run name")
        writer = SummaryWriter(str(log_path), purge_step=step + 1 if resume_from else None, flush_secs=10)
        writer.add_text("run/stage", "One-pass real speech distillation. Original encoder/teacher frozen. "
                        "True batches; final-only validation; bounded rolling teacher cache.", step)
        writer.add_text("run/lineage", json.dumps(lineage), step)
    criterion = WarmupReconstructionLoss(loss_config)
    checkpoint_step = step if resume_from is not None else None
    initial_step = step
    end = config.total_steps if stop_after_updates is None else min(config.total_steps, step + stop_after_updates)
    status = {}

    def publish(state, error=None):
        nonlocal status
        status = {"stage": "source_corpus_one_pass", "state": state, "step": step,
                  "total_steps": config.total_steps, "batch_size": config.batch_size,
                  "run_name": run_name, "train": last_train, "validation": validation,
                  "checkpoint_step": checkpoint_step, "lineage": lineage,
                  "unique_scored_hours_seen": sampler.emitted_input_samples / 16000 / 3600,
                  "remaining_unique_segments": sampler.remaining_segments,
                  "corpus_scored_hours": available_scored_hours,
                  "seconds_in_training_loop": seconds, "cache": corpus.metrics(),
                  "updated_at_unix": time.time(), "error": error}
        _atomic_json(directory / "status.json", status)

    def checkpoint():
        nonlocal checkpoint_step
        corpus.flush()
        _atomic_save({"format_version": 1, "identity": identity, "step": step,
                      "lineage": lineage, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "sampler": sampler.state_dict(), "rng": _rng_state(), "last_train": last_train,
                      "validation": validation, "seconds": seconds}, latest)
        checkpoint_step = step

    try:
        model.train()
        publish("training")
        # Preserve an imported checkpoint before the first new-data update.
        if initialize_from is not None:
            checkpoint()
        while step < end:
            started = time.perf_counter()
            segments = sampler.take_batch(config.batch_size)
            crops = []
            for segment in segments:
                crop = sample_training_crop(corpus.get(segment.row_id), segment.start_frame,
                                            scored_frames=segment.scored_frames,
                                            context_frames=config.context_frames)
                if crop.valid_scored_samples != segment.valid_output_samples:
                    raise ValueError("Sampler/cache valid sample accounting differs")
                crops.append(crop)
            prepared = time.perf_counter()
            lr = config.learning_rate * (min(1.0, (step + 1) / config.warmup_steps)
                                         if config.warmup_steps else 1.0)
            for member in optimizer.optimizers.values():
                for group in member.param_groups:
                    group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            result = batched_scored_crop_loss(model, crops, criterion, device)
            result.mean["total"].backward()
            if any(p.grad is None for p in parameters):
                raise FloatingPointError("Student parameter has no gradient")
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            if not finite_parameters(parameters):
                raise FloatingPointError("Student parameter became nonfinite")
            last_train = batch_metrics(result, gradient_norm)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            last_train.update(learning_rate=lr, step_seconds=elapsed,
                              prepare_seconds=prepared - started, student_seconds=elapsed - (prepared - started))
            seconds += elapsed
            step += 1
            if writer is not None:
                for key, value in last_train.items():
                    writer.add_scalar("train/" + key, value, step)
                writer.add_scalar("exposure/unique_scored_hours", sampler.emitted_input_samples / 16000 / 3600, step)
                writer.add_scalar("progress/fraction", step / config.total_steps, step)
            del result, crops
            if step % config.checkpoint_interval == 0 or step == end:
                checkpoint()
            publish("training")
        if step == config.total_steps and not validation:
            validation = evaluate(model, corpus, config, criterion, device)
            if writer is not None:
                for key, value in validation.items():
                    writer.add_scalar("validation/" + key, value, step)
            checkpoint()
        publish("completed" if step == config.total_steps else "paused")
    except BaseException as error:
        publish("failed", f"{type(error).__name__}: {error}")
        raise
    finally:
        corpus.flush()
        if writer is not None:
            writer.close()
    return {**status, "updates_this_run": step - initial_step}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path,
                        help="Verified corpus readiness report, pinned in every checkpoint")
    parser.add_argument("--teacher-source", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-gib", type=float, default=12)
    parser.add_argument("--cache-memory-utterances", type=int, default=256)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--run-name", default="warmup-speech-500h-v1")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(7)
    rows = load_manifest(args.manifest)
    preparation_identity = None
    if args.corpus_audit is not None:
        audit_bytes = args.corpus_audit.read_bytes()
        audit = json.loads(audit_bytes)
        if audit.get("state") != "ready":
            raise ValueError("The supplied corpus audit is not ready for training")
        manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        if audit.get("source_manifest_sha256") != manifest_sha256:
            raise ValueError("The corpus audit does not identify this source manifest")
        preparation_identity = {"audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
                                "manifest_sha256": manifest_sha256,
                                "audit": audit}
    teacher = FrozenAudioVAE2.from_files(args.teacher_source, args.teacher_checkpoint, device=args.device)
    # Match the qualified original-teacher startup policy before caching targets.
    from .prepare_targets import _target_teacher
    from .source_corpus import read_native_16k
    first = next(row for row in rows if row.split == "train")
    teacher = _target_teacher(teacher, first)
    source = Path(first.audio_path)
    if not source.is_absolute():
        source = args.manifest.parent / source
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != first.audio_sha256:
        raise ValueError("Teacher warmup input SHA-256 mismatch")
    wave = read_native_16k(payload, first)
    for _ in range(2):
        teacher.decode(teacher.encode(wave.to(teacher.device)))
    corpus = SourceCorpus(args.manifest, teacher, cache_dir=args.cache_dir,
                          input_sample_counts={row.source_id: round(row.duration_seconds * 16000) for row in rows},
                          max_disk_bytes=int(args.cache_gib * 2**30), min_free_bytes=4 * 2**30,
                          max_memory_utterances=args.cache_memory_utterances, allow_prepared_source=True)
    try:
        result = train_sources(StudentDecoder(), corpus, args.output_dir,
                               config=SourceTrainingConfig(total_steps=args.steps, batch_size=args.batch_size),
                               device=args.device, initialize_from=args.initialize_from, resume_from=args.resume,
                               log_dir=args.log_dir, run_name=args.run_name,
                               preparation_identity=preparation_identity)
    finally:
        corpus.close()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
