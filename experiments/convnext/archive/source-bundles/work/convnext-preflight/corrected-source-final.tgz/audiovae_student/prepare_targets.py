"""Prepare bounded, resumable original-teacher caches from local mono 16 kHz files.

This first preparation path never downloads, resamples, mixes channels, changes
gain, or crops a recording. A manifest row must describe the complete file at
audio_path; parent_start_seconds is source provenance, not a file seek offset.
The resulting index is consumable by corpus_training. Use one writer per index.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import tempfile
from typing import Any, Callable, Sequence

import numpy as np
import soundfile as sf
import torch

from .cache import UtteranceCache, load_cache, prepare_utterance_cache, save_cache
from .data import ManifestRow, load_manifest, validate_manifest
from .teacher import FrozenAudioVAE2


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, value: Any) -> None:
    """Publish complete JSON, preserving the previous index on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".index-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_canonical(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_verified_audio(
    row: ManifestRow, *, manifest_directory: Path, max_utterance_seconds: float = 120,
) -> torch.Tensor:
    """Hash the exact bytes subsequently decoded, and preserve their amplitude."""
    if not math.isfinite(max_utterance_seconds) or max_utterance_seconds <= 0:
        raise ValueError("max_utterance_seconds must be positive and finite")
    if row.sample_rate_hz != 16000:
        raise ValueError(f"{row.source_id}: this preparation path requires original mono 16000 Hz files")
    if row.duration_seconds > max_utterance_seconds:
        raise ValueError(f"{row.source_id}: utterance exceeds explicit duration limit")
    for name in ("gain_policy", "resampler_policy"):
        # Fail closed if metadata requests preprocessing this path cannot apply.
        policy = getattr(row, name).casefold().split(":", 1)[0].strip()
        if policy not in {"none", "unchanged"}:
            raise ValueError(f"{row.source_id}: {name} requires unsupported preprocessing")
    path = Path(row.audio_path).expanduser()
    if not path.is_absolute():
        path = manifest_directory / path
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row.audio_sha256:
        raise ValueError(f"{row.source_id}: source audio SHA-256 mismatch")
    with sf.SoundFile(io.BytesIO(payload)) as handle:
        if handle.samplerate != 16000 or handle.channels != 1:
            raise ValueError(f"{row.source_id}: actual file must be mono 16000 Hz; no implicit conversion")
        if not 0 < handle.frames <= math.floor(max_utterance_seconds * 16000):
            raise ValueError(f"{row.source_id}: invalid or excessive decoded sample count")
        expected_duration = handle.frames / 16000
        if abs(expected_duration - row.duration_seconds) > 1 / 16000 + 1e-10:
            raise ValueError(f"{row.source_id}: manifest duration differs from the complete file")
        audio = handle.read(dtype="float32", always_2d=True)
    if audio.shape != (round(expected_duration * 16000), 1):
        raise ValueError(f"{row.source_id}: incomplete audio decode")
    if not np.isfinite(audio).all():
        raise ValueError(f"{row.source_id}: source audio contains nonfinite samples")
    return torch.from_numpy(np.ascontiguousarray(audio[:, 0])).reshape(1, 1, -1)


def select_rows(
    rows: Sequence[ManifestRow], *, max_records: int | None, max_hours: float | None,
) -> list[ManifestRow]:
    """Bound total selected records/hours, reserving the first train and dev rows.

    Remaining order is the manifest's order, which the data preparer should
    balance by source and language. Limits include already cached records, so
    resuming does not silently grow the selected corpus. No utterance is cropped.
    """
    if max_records is None and max_hours is None:
        raise ValueError("Supply max_records or max_hours to bound target preparation")
    if max_records is not None and (type(max_records) is not int or max_records < 2):
        raise ValueError("max_records must be an integer >= 2 to include train and dev")
    if max_hours is not None and (
        isinstance(max_hours, bool) or not isinstance(max_hours, (float, int))
        or not math.isfinite(max_hours) or max_hours <= 0
    ):
        raise ValueError("max_hours must be positive and finite")
    if any(row.split not in {"train", "dev"} for row in rows):
        raise ValueError("Target preparation index accepts train/dev only; keep test/regression separate")
    initial = []
    for split in ("train", "dev"):
        matches = [i for i, row in enumerate(rows) if row.split == split]
        if not matches:
            raise ValueError("Manifest must include both train and dev rows")
        initial.append(matches[0])
    order = initial + [i for i in range(len(rows)) if i not in initial]
    selected, seconds = [], 0.0
    for position, index in enumerate(order):
        row = rows[index]
        if max_records is not None and len(selected) >= max_records:
            break
        if max_hours is not None and seconds + row.duration_seconds > max_hours * 3600 + 1e-9:
            if position < 2:
                raise ValueError("Hour limit cannot include the initial complete train/dev utterances")
            continue
        selected.append(row)
        seconds += row.duration_seconds
    return selected


def _source_identity(row: ManifestRow) -> dict[str, Any]:
    source = row.to_dict()
    source["teacher_cache_key"] = None
    return source


def _slot(row: ManifestRow, teacher: FrozenAudioVAE2) -> str:
    # The file slot is discoverable without executing the teacher on resume.
    # The inner cache_key independently covers exact prepared PCM and targets.
    identity = {"source": _source_identity(row), "teacher": teacher.provenance,
                "reader": {"soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
                           "dtype": "float32", "mono_policy": "require_mono", "resample": False}}
    return hashlib.sha256(_canonical(identity).encode()).hexdigest()


def _target_teacher(teacher: FrozenAudioVAE2, first_row: ManifestRow) -> FrozenAudioVAE2:
    """Attach preparation identity while sharing the same frozen original model."""
    policy = {
        "jit_optimized_execution": True,
        "warmup_encode_decode_passes": 2,
        "warmup_input_policy": "first_selected_complete_utterance_including_resume",
        "warmup_source_sha256": first_row.audio_sha256,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if teacher.device.type == "cuda" and (
        not policy["deterministic_algorithms"] or not policy["cudnn_deterministic"]
        or policy["cudnn_benchmark"] or policy["cuda_matmul_allow_tf32"]
        or policy["cudnn_allow_tf32"] or policy["cublas_workspace_config"] != ":4096:8"
    ):
        raise ValueError("CUDA target preparation requires the qualified deterministic FP32 policy; use the CLI")
    provenance = teacher.provenance
    provenance["target_preparation_policy"] = policy
    return FrozenAudioVAE2(teacher.model, provenance)


def _check_record(
    record: UtteranceCache, row: ManifestRow, teacher: FrozenAudioVAE2,
    audio: torch.Tensor | None = None,
) -> None:
    identity = record.metadata["identity"]
    if identity["source"] != _source_identity(row) or identity["teacher"] != teacher.provenance:
        raise ValueError(f"{row.source_id}: existing cache has different source or teacher provenance")
    if row.teacher_cache_key is not None and row.teacher_cache_key != record.cache_key:
        raise ValueError(f"{row.source_id}: manifest teacher_cache_key mismatch")
    if record.reference16k is None:
        raise ValueError(f"{row.source_id}: cache is missing original 16 kHz reference")
    if audio is not None and not torch.equal(audio, record.reference16k):
        raise ValueError(f"{row.source_id}: existing cache does not match current decoded source PCM")


def _existing_entries(
    index_path: Path, destinations: dict[Path, ManifestRow], teacher: FrozenAudioVAE2,
) -> dict[Path, dict[str, str]]:
    if not index_path.exists():
        return {}
    value = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"format_version", "train", "dev"} or value["format_version"] != 1:
        raise ValueError("Existing cache index has an unsupported schema")
    entries = {}
    for split in ("train", "dev"):
        if not isinstance(value[split], list):
            raise ValueError("Existing cache index split must be a list")
        for entry in value[split]:
            if (not isinstance(entry, dict) or set(entry) != {"path", "cache_key"}
                    or not isinstance(entry["path"], str) or not isinstance(entry["cache_key"], str)):
                raise ValueError("Existing cache index row has an unsupported schema")
            path = (index_path.parent / entry["path"]).resolve()
            if path not in destinations or destinations[path].split != split or path in entries:
                raise ValueError("Existing index does not match this bounded manifest/teacher selection")
            record = load_cache(path, expected_cache_key=entry["cache_key"])
            _check_record(record, destinations[path], teacher)
            entries[path] = {"path": os.path.relpath(path, index_path.parent), "cache_key": record.cache_key}
    return entries


def prepare_targets(
    manifest_path: str | Path, teacher: FrozenAudioVAE2, *, output_dir: str | Path,
    index_path: str | Path, max_records: int | None = None, max_hours: float | None = None,
    reserved_manifests: Sequence[str | Path] = (), max_utterance_seconds: float = 120,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Prepare and index selected utterances, aborting on the first failed check.

    Completed caches are individually atomic. The index is atomically rewritten
    only after saved tensors have been reloaded and verified. Interrupted target
    generation leaves no index reference to an incomplete cache. A complete but
    unindexed cache is discovered and verified on resume without recomputation.
    """
    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    index_path = Path(index_path).expanduser().resolve()
    if manifest == index_path:
        raise ValueError("Index path must not overwrite the source manifest")
    if not isinstance(teacher, FrozenAudioVAE2):
        raise TypeError("teacher must be FrozenAudioVAE2")
    rows = load_manifest(manifest)
    reserved = [row for path in reserved_manifests for row in load_manifest(path)]
    validate_manifest(rows, reserved_rows=reserved)
    selected = select_rows(rows, max_records=max_records, max_hours=max_hours)
    teacher = _target_teacher(teacher, selected[0])
    destination.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    destinations = {destination / (_slot(row, teacher) + ".pt"): row for row in selected}
    if len(destinations) != len(selected):
        raise ValueError("Cache destination identity collision")
    entries = _existing_entries(index_path, destinations, teacher)
    generated = resumed = completed = 0
    selected_by_split, selected_by_language = defaultdict(float), defaultdict(float)
    for row in selected:
        selected_by_split[row.split] += row.duration_seconds / 3600
        selected_by_language[row.language] += row.duration_seconds / 3600
    summary = {
        "format_version": 1, "status": "preparing", "manifest_records": len(rows),
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "selected_records": len(selected), "selected_hours": sum(r.duration_seconds for r in selected) / 3600,
        "hours_by_split": dict(selected_by_split), "hours_by_language": dict(selected_by_language),
        "teacher": teacher.provenance, "soundfile_version": sf.__version__,
        "libsndfile_version": sf.__libsndfile_version__, "resampling": False, "gain_change": False,
    }
    report_path = index_path.with_name(index_path.name + ".preparation.json")
    _atomic_json(report_path, summary)
    # The scripted upstream Snake can change execution plan between its first
    # and second calls. Warm both encoder and decoder before retaining targets,
    # including on resumed runs where the first cache already exists. This is
    # compiler startup, not model training; all parameters remain frozen.
    warm_audio = read_verified_audio(selected[0], manifest_directory=manifest.parent,
                                     max_utterance_seconds=max_utterance_seconds).to(teacher.device)
    with torch.jit.optimized_execution(True):
        for _ in range(2):
            warm_latents = teacher.encode(warm_audio)
            warm_target = teacher.decode(warm_latents)
    del warm_audio, warm_latents, warm_target
    if any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
        raise RuntimeError("Teacher must remain frozen without gradients after startup warmup")
    for path, row in destinations.items():
        audio = read_verified_audio(row, manifest_directory=manifest.parent,
                                    max_utterance_seconds=max_utterance_seconds)
        if path.exists():
            expected = entries.get(path, {}).get("cache_key")
            record = load_cache(path, expected_cache_key=expected)
            _check_record(record, row, teacher, audio)
            resumed += 1
        else:
            with torch.jit.optimized_execution(True):
                record = prepare_utterance_cache(audio, row, teacher, reserved_rows=reserved, include_reference=True)
            save_cache(record, path)
            # Establish a verified on-disk artifact before publishing its index.
            record = load_cache(path, expected_cache_key=record.cache_key)
            _check_record(record, row, teacher, audio)
            generated += 1
        entries[path] = {"path": os.path.relpath(path, index_path.parent), "cache_key": record.cache_key}
        index = {"format_version": 1, "train": [], "dev": []}
        for target_path, source in destinations.items():
            if target_path in entries:
                index[source.split].append(entries[target_path])
        _atomic_json(index_path, index)
        completed += 1
        summary.update(completed_records=completed, generated_records=generated, resumed_records=resumed)
        _atomic_json(report_path, summary)
        if progress is not None:
            progress({"completed_records": completed, "selected_records": len(selected),
                      "generated_records": generated, "resumed_records": resumed,
                      "source_id": row.source_id, "split": row.split, "language": row.language})
    summary["status"] = "complete"
    _atomic_json(report_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--teacher-source", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--max-utterance-seconds", type=float, default=120)
    parser.add_argument("--reserved-manifest", type=Path, action="append", default=[])
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.max_records is None and args.max_hours is None:
        parser.error("Supply --max-records or --max-hours to bound preparation")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    # Establish the qualified cuBLAS setting before from_files initializes CUDA.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        teacher = FrozenAudioVAE2.from_files(args.teacher_source, args.teacher_checkpoint, device=args.device)
        summary = prepare_targets(args.manifest, teacher, output_dir=args.output_dir, index_path=args.index,
                                  max_records=args.max_records, max_hours=args.max_hours,
                                  reserved_manifests=args.reserved_manifest,
                                  max_utterance_seconds=args.max_utterance_seconds,
                                  progress=lambda event: print(json.dumps(event, sort_keys=True), flush=True))
        print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
