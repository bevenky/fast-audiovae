"""Bounded reconstruction warmup on verified, continuously encoded speech caches.

This stage trains the student only. It has no adversarial discriminator and
does not claim to establish final perceptual quality. Each accumulated example
keeps its actual available history; only its valid scored waveform is lost.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .cache import DECODER_HOP, ENCODER_HOP, TrainingCrop, UtteranceCache, load_cache, sample_training_crop
from .data import ManifestRow, validate_manifest
from .losses import WarmupLossConfig, WarmupReconstructionLoss
from .model import StudentConfig, StudentDecoder
from .optimizers import build_optimizer_bundle
from .training import TrainingConfig, _atomic_save, _restore_rng, _rng_state, _runtime_spec


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp",
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class CorpusTrainingConfig:
    total_steps: int = 1000
    accumulation_steps: int = 8
    scored_frames: int = 64
    context_frames: int = 29
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip_norm: float = 1.0
    optimizer: str = "muon_adamw"
    validation_interval: int = 100
    checkpoint_interval: int = 100
    seed: int = 7
    max_cached_utterances: int = 8

    def __post_init__(self):
        for name in ("total_steps", "accumulation_steps", "scored_frames", "validation_interval",
                     "checkpoint_interval", "max_cached_utterances"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.context_frames) is not int or self.context_frames < 29:
            raise ValueError("context_frames must be at least 29")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        TrainingConfig(learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                       betas=self.betas, grad_clip_norm=self.grad_clip_norm,
                       optimizer=self.optimizer, seed=self.seed)


@dataclass(frozen=True)
class CacheEntry:
    path: Path
    cache_key: str
    split: str
    source_id: str
    language: str
    input_samples: int
    tensor_sha256: dict[str, str]
    has_reference: bool


class CachedCorpus:
    """Validated index with bounded CPU caching and immutable target identities.

    Schema: {"format_version":1,"train":[{"path":"cache/a.pt","cache_key":"..."}],
             "dev":[{"path":"cache/b.pt","cache_key":"..."}]}.
    Relative paths resolve against the index file. Source/speaker/session and
    prepared-audio checks reject known train/dev overlap; unknown identities
    cannot establish acoustic or speaker disjointness.
    """

    def __init__(self, index_path: str | Path, *, max_cached_utterances: int = 8):
        index_path = Path(index_path).resolve()
        payload = json.loads(index_path.read_text())
        if not isinstance(payload, dict) or set(payload) != {"format_version", "train", "dev"}:
            raise ValueError("Cache index must contain format_version, train and dev")
        if payload["format_version"] != 1:
            raise ValueError("Unsupported cache-index format_version")
        if type(max_cached_utterances) is not int or max_cached_utterances < 1:
            raise ValueError("max_cached_utterances must be positive")
        self.max_cached_utterances = max_cached_utterances
        self.entries: dict[str, list[CacheEntry]] = {"train": [], "dev": []}
        self._loaded: OrderedDict[str, UtteranceCache] = OrderedDict()
        rows, identities, keys = [], [], set()
        prepared_owners: dict[str, str] = {}
        teacher = None
        for split in ("train", "dev"):
            if not isinstance(payload[split], list) or not payload[split]:
                raise ValueError(f"Cache index requires a nonempty {split} partition")
            for entry in payload[split]:
                if not isinstance(entry, dict) or set(entry) != {"path", "cache_key"}:
                    raise ValueError("Each cache-index entry must contain path and cache_key")
                if not isinstance(entry["path"], str) or not entry["path"]:
                    raise ValueError("Cache path must be a nonempty string")
                path = Path(entry["path"])
                path = (index_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
                record = load_cache(path, expected_cache_key=entry["cache_key"])
                identity = record.metadata["identity"]
                source = ManifestRow.from_dict(identity["source"])
                if source.split != split:
                    raise ValueError(f"Cache source {source.source_id} is {source.split}, not indexed {split}")
                if record.cache_key in keys:
                    raise ValueError("Duplicate cache key in training index")
                keys.add(record.cache_key)
                prepared = identity["prepared_audio_sha256"]
                if prepared in prepared_owners and prepared_owners[prepared] != split:
                    raise ValueError("Prepared-audio leakage between train and dev")
                prepared_owners[prepared] = split
                current_teacher = _plain(identity["teacher"])
                if teacher is None:
                    teacher = current_teacher
                elif current_teacher != teacher:
                    raise ValueError("All caches must use one fixed teacher/configuration/latent contract")
                rows.append(source)
                hashes = _plain(record.metadata["tensor_sha256"])
                self.entries[split].append(CacheEntry(path, record.cache_key, split, source.source_id,
                                                      source.language, record.input_samples, hashes,
                                                      record.reference16k is not None))
                identities.append({"split": split, "cache_key": record.cache_key, "tensor_sha256": hashes})
        validate_manifest(rows)
        self.teacher = teacher
        self.data_fingerprint = _digest({"format_version": 1, "entries": identities})
        self.train_languages = sorted({entry.language for entry in self.entries["train"]})
        self.language_indices = {language: [i for i, entry in enumerate(self.entries["train"])
                                            if entry.language == language]
                                 for language in self.train_languages}
        self.by_key = {entry.cache_key: entry for split in ("train", "dev") for entry in self.entries[split]}

    def get(self, split: str, index: int) -> UtteranceCache:
        entry = self.entries[split][index]
        if entry.cache_key in self._loaded:
            self._loaded.move_to_end(entry.cache_key)
            return self._loaded[entry.cache_key]
        record = load_cache(entry.path, expected_cache_key=entry.cache_key)
        if record.metadata["tensor_sha256"] != entry.tensor_sha256:
            raise ValueError("Cached tensors changed after corpus identity was established")
        self._loaded[entry.cache_key] = record
        while len(self._loaded) > self.max_cached_utterances:
            self._loaded.popitem(last=False)
        return record


def minimum_scored_samples(loss_config: WarmupLossConfig, has_reference: bool) -> int:
    samples = max(loss_config.teacher_fft_sizes)
    if has_reference:
        samples = max(samples, 3 * max(loss_config.reference_fft_sizes_16k))
    return samples


def _validate_crop_lengths(corpus: CachedCorpus, config: CorpusTrainingConfig,
                           loss_config: WarmupLossConfig) -> None:
    for split in ("train", "dev"):
        for entry in corpus.entries[split]:
            required = minimum_scored_samples(loss_config, entry.has_reference)
            if min(config.scored_frames * DECODER_HOP, 3 * entry.input_samples) < required:
                raise ValueError(f"{entry.source_id} has insufficient valid scored audio for the loss FFTs")


def draw_training_crop(corpus: CachedCorpus, config: CorpusTrainingConfig,
                       loss_config: WarmupLossConfig, sampler: random.Random) -> TrainingCrop:
    language = corpus.train_languages[sampler.randrange(len(corpus.train_languages))]
    candidates = corpus.language_indices[language]
    index = candidates[sampler.randrange(len(candidates))]
    entry = corpus.entries["train"][index]
    minimum = minimum_scored_samples(loss_config, entry.has_reference)
    latest_start = (3 * entry.input_samples - minimum) // DECODER_HOP
    start = sampler.randrange(latest_start + 1)
    return sample_training_crop(corpus.get("train", index), start, config.scored_frames,
                                 config.context_frames)


def _fixed_dev_crop(record: UtteranceCache, scored_frames: int) -> TrainingCrop:
    if record.metadata["identity"]["source"]["split"] != "dev":
        raise ValueError("Held-out reconstruction requires a dev cache")
    def window(value, samples):
        selected = value[..., :samples]
        return F.pad(selected, (0, samples - selected.shape[-1]))
    frames = scored_frames
    reference = None if record.reference16k is None else window(record.reference16k, frames * ENCODER_HOP)
    return TrainingCrop(window(record.latents, frames), window(record.teacher_audio, frames * DECODER_HOP),
                         reference, record.cache_key, record.metadata["identity"]["source"]["source_id"],
                         0, 0, 0, frames, min(frames * DECODER_HOP, record.valid_output_samples))


def scored_crop_loss(model: StudentDecoder, crop: TrainingCrop,
                     criterion: WarmupReconstructionLoss, device: torch.device | str) -> dict[str, torch.Tensor]:
    """Exclude all left context and invalid tail samples before any loss FFT."""
    prediction = model(crop.latents.to(device))
    if prediction.shape != crop.teacher_audio.shape:
        raise ValueError("Student output does not match the cache crop's raw sample contract")
    prediction = prediction[..., crop.scored_slice]
    target = crop.teacher_audio[..., crop.scored_slice].to(device)
    reference = None
    if crop.reference16k is not None:
        reference = crop.reference16k[..., crop.reference_scored_slice].to(device)
    return criterion(prediction, target, reference)


class _Metrics:
    def __init__(self):
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.examples = self.samples = self.references = 0

    def add(self, values: dict[str, torch.Tensor], crop: TrainingCrop):
        self.examples += 1
        self.samples += crop.valid_scored_samples
        self.references += int(crop.reference16k is not None)
        for name, value in values.items():
            if name == "reference_spectral" and crop.reference16k is None:
                continue
            number = float(value.detach())
            if not math.isfinite(number):
                raise FloatingPointError(f"Nonfinite {name}")
            self.sums[name] = self.sums.get(name, 0.0) + number
            self.counts[name] = self.counts.get(name, 0) + 1

    def result(self) -> dict[str, float | int]:
        return {**{name: value / self.counts[name] for name, value in self.sums.items()},
                "examples": self.examples, "reference_examples": self.references,
                "scored_samples": self.samples}


@torch.no_grad()
def evaluate_reconstruction(model: StudentDecoder, corpus: CachedCorpus,
                             config: CorpusTrainingConfig, loss_config: WarmupLossConfig,
                             device: torch.device | str) -> dict[str, float | int]:
    """Evaluate one fixed start crop per dev utterance, without optimizer use."""
    was_training, rng = model.training, _rng_state()
    criterion, metrics = WarmupReconstructionLoss(loss_config), _Metrics()
    try:
        model.eval()
        for index in range(len(corpus.entries["dev"])):
            crop = _fixed_dev_crop(corpus.get("dev", index), config.scored_frames)
            metrics.add(scored_crop_loss(model, crop, criterion, device), crop)
        return metrics.result()
    finally:
        model.train(was_training)
        _restore_rng(rng)


_EXTENSION_FIELDS = ("total_steps", "validation_interval", "checkpoint_interval")


def _resume_is_extension(previous: dict[str, Any], current: dict[str, Any], *,
                         allow_run_extension: bool) -> bool:
    """Accept schedule growth only; never relax model, corpus, or optimizer identity."""
    if previous == current:
        return False
    mismatch = "Corpus checkpoint data/model/loss/optimizer/runtime identity mismatch"
    if not allow_run_extension:
        raise ValueError(mismatch)
    before, after = previous.get("training_config"), current.get("training_config")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError(mismatch)
    old_steps, new_steps = before.get("total_steps"), after.get("total_steps")
    if (type(old_steps) is not int or type(new_steps) is not int or
            old_steps < 1 or new_steps <= old_steps):
        raise ValueError("Run extension requires a strict increase in total_steps")
    if ({key: value for key, value in previous.items() if key != "training_config"} !=
            {key: value for key, value in current.items() if key != "training_config"}):
        raise ValueError(mismatch)
    before_fixed = {key: value for key, value in before.items() if key not in _EXTENSION_FIELDS}
    after_fixed = {key: value for key, value in after.items() if key not in _EXTENSION_FIELDS}
    if before_fixed != after_fixed:
        raise ValueError(mismatch)
    return True


@contextmanager
def _writer(log_dir: Path | None, run_name: str, identity: dict[str, Any], step: int, resuming: bool,
            extension_from: dict[str, Any] | None = None):
    if log_dir is None:
        yield None
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError("run_name must be a single simple directory name")
    path, writer = log_dir / run_name, None
    manifest = path / "run.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text())
        if recorded != identity and (extension_from is None or recorded != extension_from):
            raise ValueError("TensorBoard run identity differs; choose another run_name")
    events_exist = path.exists() and any(path.glob("events.out.tfevents.*"))
    if events_exist and (not resuming or not manifest.exists()):
        raise ValueError("TensorBoard run exists; resume its checkpoint or choose another run_name")
    try:
        rng = _rng_state()
        try:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise RuntimeError("Install tensorboard in the training environment for dashboard logging") from error
            _atomic_json(manifest, identity)
            writer = SummaryWriter(str(path), max_queue=1, flush_secs=1,
                                   purge_step=step + 1 if resuming else None)
            writer.add_text("run/stage", "Cached-speech reconstruction warmup. Frozen teacher targets; "
                            "fresh student. Held-out curves use fixed start crops. No GAN training yet.", step)
            writer.flush()
        finally:
            _restore_rng(rng)
        yield writer
    finally:
        if writer is not None:
            writer.close()


def _log(writer, prefix: str, values: dict[str, float | int], step: int):
    if writer is not None:
        for name, value in values.items():
            writer.add_scalar(f"{prefix}/{name}", value, step)
        writer.flush()


def train_cached_corpus(model: StudentDecoder, index_path: str | Path, *, output_dir: str | Path,
                         config: CorpusTrainingConfig = CorpusTrainingConfig(),
                         loss_config: WarmupLossConfig = WarmupLossConfig(),
                         device: str | torch.device = "cpu", resume_from: str | Path | None = None,
                         stop_after_updates: int | None = None, log_dir: str | Path | None = None,
                         run_name: str = "warmup-speech", allow_run_extension: bool = False) -> dict[str, Any]:
    """Train to total_steps, or pause after a bounded number of new updates.

    Per-example backward accumulation avoids fabricated history in padded
    batches. Sampling is uniform over manifest language labels, then utterances
    within that language, then eligible frame starts. This is not the later
    30/30/40 exposure recipe. This initial trainer has no GAN stage.
    Checkpoints are written only at completed optimizer-update boundaries.
    Resume is strict unless allow_run_extension explicitly permits increasing
    total_steps and changing validation/checkpoint intervals. All other settings
    and the corpus must remain identical; warmup uses the preserved global step.
    """
    if not isinstance(model, StudentDecoder):
        raise TypeError("Pass the standalone StudentDecoder, not a teacher or combined module")
    if config.context_frames * model.phases < model.config.history_frames:
        raise ValueError("Configured left context is shorter than the student's receptive history")
    if stop_after_updates is not None and (type(stop_after_updates) is not int or stop_after_updates < 1):
        raise ValueError("stop_after_updates must be a positive integer")
    if allow_run_extension and resume_from is None:
        raise ValueError("allow_run_extension requires resume_from")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only explicitly selected CPU or CUDA training is supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested, but no CUDA device is available")
    model.to(device)
    corpus = CachedCorpus(index_path, max_cached_utterances=config.max_cached_utterances)
    _validate_crop_lengths(corpus, config, loss_config)
    optimizer = build_optimizer_bundle(model, optimizer=config.optimizer, lr=config.learning_rate,
                                        weight_decay=config.weight_decay, betas=config.betas)
    parameters = [p for p in model.parameters() if p.requires_grad]
    runtime = _runtime_spec()
    if device.type == "cuda":
        runtime.update({"cuda_runtime": str(torch.version.cuda),
                        "cuda_device": torch.cuda.get_device_name(device),
                        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
                        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)})
    identity = _plain({"kind": "cached_corpus_reconstruction_warmup", "format_version": 1,
                       "data_fingerprint": corpus.data_fingerprint, "teacher": corpus.teacher,
                       "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
                       "model_config": model.config.to_dict(), "training_config": asdict(config),
                       "sampler": "uniform_manifest_language_then_utterance_then_eligible_frame_v1",
                       "loss_config": asdict(loss_config), "optimizer_groups": optimizer.group_fingerprint,
                       "device_type": device.type, "runtime": runtime})
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "latest.pt"
    if latest.exists() and resume_from is None:
        raise ValueError("output_dir already has a checkpoint; resume it or use a new directory")
    step, scored_samples_seen, training_seconds = 0, 0, 0.0
    last_train: dict[str, Any] = {}
    last_validation: dict[str, Any] = {}
    validation_step = -1
    language_exposure = {language: {"examples": 0, "scored_samples": 0} for language in corpus.train_languages}
    last_sources: list[dict[str, Any]] = []
    continuation_lineage: list[dict[str, Any]] = []
    extension_from: dict[str, Any] | None = None
    sampler = random.Random(config.seed)
    if resume_from is not None:
        saved = torch.load(resume_from, map_location="cpu", weights_only=True)
        if saved.get("format_version") != 1 or not isinstance(saved.get("identity"), dict):
            raise ValueError("Corpus checkpoint data/model/loss/optimizer/runtime identity mismatch")
        previous_identity = saved["identity"]
        is_extension = _resume_is_extension(previous_identity, identity, allow_run_extension=allow_run_extension)
        step = int(saved["step"])
        if not 0 <= step <= previous_identity["training_config"]["total_steps"]:
            raise ValueError("Corpus checkpoint step is outside its original training schedule")
        continuation_lineage = list(saved.get("continuation_lineage", []))
        if is_extension:
            extension_from = previous_identity
            continuation_lineage.append({
                "checkpoint_step": step, "source_checkpoint": str(Path(resume_from)),
                "from_schedule": {key: previous_identity["training_config"][key] for key in _EXTENSION_FIELDS},
                "to_schedule": {key: identity["training_config"][key] for key in _EXTENSION_FIELDS},
                "source_identity_sha256": _digest(previous_identity), "target_identity_sha256": _digest(identity),
            })
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"])
        sampler.setstate(saved["sampler_rng"])
        step, scored_samples_seen = int(saved["step"]), int(saved["scored_samples_seen"])
        training_seconds = float(saved["training_seconds"])
        last_train, last_validation = saved["last_train"], saved["last_validation"]
        validation_step = int(saved["validation_step"])
        language_exposure = saved["language_exposure"]
        last_sources = saved["last_sources"]
        if not 0 <= step <= config.total_steps:
            raise ValueError("Corpus checkpoint step is outside the configured training schedule")
    else:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
    end_step = config.total_steps if stop_after_updates is None else min(config.total_steps, step + stop_after_updates)
    criterion = WarmupReconstructionLoss(loss_config)
    checkpoint_step = step if resume_from is not None else None
    last_checkpoint = str(Path(resume_from)) if resume_from is not None else None
    start_step = step
    status: dict[str, Any] = {}

    def publish(state: str, error: str | None = None):
        nonlocal status
        status = {"stage": "reconstruction_warmup", "state": state,
                  "step": step, "total_steps": config.total_steps,
                  "progress": step / config.total_steps,
                  "device": str(device), "optimizer": config.optimizer,
                  "train_utterances": len(corpus.entries["train"]),
                  "dev_utterances": len(corpus.entries["dev"]),
                  "train_examples_seen": step * config.accumulation_steps,
                  "sampling_policy": identity["sampler"], "language_exposure": language_exposure,
                  "last_training_sources": last_sources,
                  "scored_samples_seen": scored_samples_seen,
                  "scored_audio_hours_seen": scored_samples_seen / 48000 / 3600,
                  "training_step_seconds_total": training_seconds,
                  "mean_training_step_seconds": training_seconds / step if step else None,
                  "train": last_train, "validation": last_validation,
                  "validation_step": validation_step,
                  "validation_policy": "one fixed start crop per dev utterance; no parameter updates",
                  "data_fingerprint": corpus.data_fingerprint,
                  "checkpoint": last_checkpoint, "checkpoint_step": checkpoint_step,
                  "continuation_lineage": continuation_lineage,
                  "run_name": run_name, "error": error, "updated_at_unix": time.time()}
        _atomic_json(directory / "status.json", status)

    def checkpoint():
        nonlocal checkpoint_step, last_checkpoint
        _atomic_save({"format_version": 1, "identity": identity, "step": step,
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "rng": _rng_state(), "sampler_rng": sampler.getstate(),
                      "scored_samples_seen": scored_samples_seen, "training_seconds": training_seconds,
                      "last_train": last_train, "last_validation": last_validation,
                      "language_exposure": language_exposure, "last_sources": last_sources,
                      "continuation_lineage": continuation_lineage,
                      "validation_step": validation_step}, latest)
        checkpoint_step, last_checkpoint = step, str(latest)

    model.train()
    try:
        with _writer(Path(log_dir) if log_dir is not None else None, run_name, identity,
                      step, resume_from is not None, extension_from=extension_from) as writer:
            if validation_step != step:
                publish("validating")
                last_validation = evaluate_reconstruction(model, corpus, config, loss_config, device)
                validation_step = step
                _log(writer, "validation", last_validation, step)
            if resume_from is None:
                checkpoint()
            publish("training")
            while step < end_step:
                started = time.perf_counter()
                next_step = step + 1
                warmup = min(1.0, next_step / config.warmup_steps) if config.warmup_steps else 1.0
                learning_rate = config.learning_rate * warmup
                for member in optimizer.optimizers.values():
                    for group in member.param_groups:
                        group["lr"] = learning_rate
                optimizer.zero_grad(set_to_none=True)
                metrics = _Metrics()
                sources = []
                pending_exposure = {language: {"examples": 0, "scored_samples": 0}
                                    for language in corpus.train_languages}
                for _ in range(config.accumulation_steps):
                    crop = draw_training_crop(corpus, config, loss_config, sampler)
                    values = scored_crop_loss(model, crop, criterion, device)
                    if not torch.isfinite(values["total"]):
                        raise FloatingPointError("Nonfinite student reconstruction loss")
                    (values["total"] / config.accumulation_steps).backward()
                    metrics.add(values, crop)
                    sources.append({"source_id": crop.source_id, "start_frame": crop.start_frame})
                    language = corpus.by_key[crop.cache_key].language
                    pending_exposure[language]["examples"] += 1
                    pending_exposure[language]["scored_samples"] += crop.valid_scored_samples
                gradients = [p.grad for p in parameters if p.grad is not None]
                if len(gradients) != len(parameters) or any(not torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("Student has missing or nonfinite gradients")
                norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip_norm,
                                                       error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in parameters):
                    raise FloatingPointError("Student has nonfinite parameters after update")
                norm_value = float(norm)
                # Tensor scalar/finite checks synchronize device work before
                # timing; elapsed includes cache I/O but excludes log/checkpoint I/O.
                elapsed = time.perf_counter() - started
                step = next_step
                last_train = {**metrics.result(), "gradient_norm": norm_value,
                              "learning_rate": learning_rate, "step_seconds": elapsed}
                scored_samples_seen += metrics.samples
                training_seconds += elapsed
                last_sources = sources
                for language, values in pending_exposure.items():
                    for key, value in values.items():
                        language_exposure[language][key] += value
                _log(writer, "train", last_train, step)
                _log(writer, "exposure", {"scored_audio_hours": scored_samples_seen / 48000 / 3600}, step)
                _log(writer, "progress", {"completed_steps": step, "remaining_steps": config.total_steps - step,
                                            "fraction": step / config.total_steps}, step)
                _log(writer, "exposure/language_examples", {language: values["examples"]
                                                             for language, values in language_exposure.items()}, step)
                if writer is not None:
                    for name in optimizer.optimizers:
                        writer.add_scalar(f"learning_rate/{name}", learning_rate, step)
                    writer.flush()
                if step % config.validation_interval == 0 or step == end_step:
                    publish("validating")
                    last_validation = evaluate_reconstruction(model, corpus, config, loss_config, device)
                    validation_step = step
                    _log(writer, "validation", last_validation, step)
                if step % config.checkpoint_interval == 0 or step == end_step:
                    checkpoint()
                publish("training")
            publish("completed" if step == config.total_steps else "paused")
    except BaseException as error:
        # Only previous complete checkpoints are resumable. Do not save a
        # partially accumulated or partially applied optimizer update on error.
        publish("failed", f"{type(error).__name__}: {error}")
        raise
    return {**status, "updates_this_run": step - start_step}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--steps", type=int, default=1000, help="Total schedule steps, including resumed updates")
    parser.add_argument("--stop-after-updates", type=int, help="Pause after this many new updates")
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--scored-frames", type=int, default=64)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--optimizer", choices=("adamw", "muon_adamw"), default="muon_adamw")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-run-extension", action="store_true",
                        help="With --resume, allow a longer total schedule and changed validation/checkpoint intervals only")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--run-name", default="warmup-speech")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.allow_run_extension and args.resume is None:
        parser.error("--allow-run-extension requires --resume")
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        # Initial student qualification uses explicit FP32 backend policy.
        # This does not claim bitwise CPU/CUDA training equivalence.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    config = CorpusTrainingConfig(total_steps=args.steps, accumulation_steps=args.accumulation_steps,
                                   learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
                                   scored_frames=args.scored_frames, validation_interval=args.validation_interval,
                                   checkpoint_interval=args.checkpoint_interval, optimizer=args.optimizer,
                                   seed=args.seed)
    # The CLI always trains the complete ten-block architecture. Tiny widths
    # are available only through the Python API for behavioral unit tests.
    result = train_cached_corpus(StudentDecoder(StudentConfig()), args.index, output_dir=args.output_dir,
                                  config=config, device=args.device, resume_from=args.resume,
                                  stop_after_updates=args.stop_after_updates, log_dir=args.log_dir,
                                  run_name=args.run_name, allow_run_extension=args.allow_run_extension)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
