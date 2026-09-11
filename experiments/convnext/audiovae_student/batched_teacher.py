"""Qualified, bounded whole-utterance batching of the original FP32 teacher.

No source is cropped, normalized or resampled. Right padding extends each
utterance to the longest 640-sample-aligned input in its batch. Only its original
ceil(N/640) latent frames and corresponding raw waveform frames are retained.
Every actual batch compares those retained values against original serial
execution, including the valid 3N waveform region, before any batch is cached.
An initial qualification alone cannot authorize later shapes or input values.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .cache import (FORMAT_VERSION, ENCODER_HOP, DECODER_HOP, UtteranceCache,
                    _identity_hash, _json_copy, _tensor_hash, _validate_teacher_provenance)
from .data import ManifestRow, validate_manifest
from .teacher import FrozenAudioVAE2


@dataclass(frozen=True)
class TeacherBatchConfig:
    max_batch_size: int = 8
    max_total_input_samples: int = 1_920_000
    max_abs_error: float = 1e-3
    minimum_snr_db: float = 70.0

    def __post_init__(self):
        if type(self.max_batch_size) is not int or not 1 <= self.max_batch_size <= 8:
            raise ValueError("max_batch_size must be between 1 and 8")
        if type(self.max_total_input_samples) is not int or self.max_total_input_samples < ENCODER_HOP:
            raise ValueError("max_total_input_samples must accommodate one 640-sample frame")
        if not math.isfinite(self.max_abs_error) or self.max_abs_error <= 0:
            raise ValueError("max_abs_error must be positive and finite")
        if not math.isfinite(self.minimum_snr_db) or self.minimum_snr_db <= 0:
            raise ValueError("minimum_snr_db must be positive and finite")

    @property
    def max_relative_rms(self):
        return 10 ** (-self.minimum_snr_db / 20)


def padded_length(samples: int) -> int:
    return ((samples + ENCODER_HOP - 1) // ENCODER_HOP) * ENCODER_HOP


def length_buckets(lengths: Sequence[int], config: TeacherBatchConfig) -> list[tuple[int, ...]]:
    """Stable length ordering with a bound on *padded* samples, not true sums.

    An individually oversized utterance remains a singleton for the unchanged
    serial path. We never split its causal history merely to meet a batch cap.
    """
    if any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError("Every whole-utterance length must be positive")
    buckets, current = [], []
    for index in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        if current and (len(current) >= config.max_batch_size
                        or padded_length(lengths[index]) * (len(current) + 1) > config.max_total_input_samples):
            buckets.append(tuple(current))
            current = []
        current.append(index)
        if padded_length(lengths[index]) > config.max_total_input_samples:
            buckets.append(tuple(current))
            current = []
    if current:
        buckets.append(tuple(current))
    return buckets


def _validate_audio(audio: Tensor) -> None:
    if (not isinstance(audio, Tensor) or audio.ndim != 3 or audio.shape[:2] != (1, 1)
            or audio.shape[-1] < 1 or audio.device.type != "cpu" or audio.dtype != torch.float32
            or audio.requires_grad or not bool(torch.isfinite(audio).all())):
        raise ValueError("Prepared teacher inputs must be finite detached CPU FP32 [1,1,N]")


@torch.no_grad()
def _batch_outputs(teacher: FrozenAudioVAE2, audios: Sequence[Tensor], config: TeacherBatchConfig):
    if not isinstance(teacher, FrozenAudioVAE2):
        raise TypeError("Only the explicitly loaded original teacher is supported")
    if not audios or len(audios) > config.max_batch_size:
        raise ValueError("Teacher batch size exceeds its configured bound")
    for audio in audios:
        _validate_audio(audio)
    longest = max(padded_length(audio.shape[-1]) for audio in audios)
    if longest * len(audios) > config.max_total_input_samples:
        raise ValueError("Padded teacher input exceeds the configured sample budget")
    batch = torch.cat([F.pad(audio, (0, longest - audio.shape[-1])) for audio in audios]).to(teacher.device)
    latents = teacher.encode(batch)
    waveform = teacher.decode(latents)
    results = []
    for i, audio in enumerate(audios):
        frames = padded_length(audio.shape[-1]) // ENCODER_HOP
        results.append((latents[i:i+1, :, :frames].detach().cpu().contiguous().clone(),
                        waveform[i:i+1, :, :frames * DECODER_HOP].detach().cpu().contiguous().clone()))
    return results


def _errors(reference: Tensor, actual: Tensor, config: TeacherBatchConfig) -> dict:
    if reference.shape != actual.shape or not bool(torch.isfinite(actual).all()):
        return {"passed": False, "reason": "shape_or_nonfinite_output"}
    # FP64 diagnostic reduction avoids losing precision on near-identical FP32.
    ref, error = reference.double(), actual.double() - reference.double()
    rms = float(ref.square().mean().sqrt())
    error_rms = float(error.square().mean().sqrt())
    maximum = float(error.abs().max())
    relative = error_rms / rms if rms > 0 else (0.0 if error_rms == 0 else None)
    snr = None if error_rms == 0 or rms == 0 else 20 * math.log10(rms / error_rms)
    return {"passed": maximum <= config.max_abs_error and relative is not None
                      and relative <= config.max_relative_rms,
            "max_abs": maximum, "reference_rms": rms, "error_rms": error_rms,
            "relative_rms": relative, "snr_db": snr, "exact": error_rms == 0}


@torch.no_grad()
def _verify_actual_batch(teacher: FrozenAudioVAE2, audios: Sequence[Tensor], batched,
                         config: TeacherBatchConfig, *, source_ids: Sequence[str]):
    """Check these values and retain serial outputs for whole-batch fallback.

    A serial execution error propagates before publication. No partial set of
    serial references can authorize any member of the pending batch.
    """
    if not audios or len(source_ids) != len(audios) or len(batched) != len(audios):
        raise ValueError("Batch outputs and source IDs must match inputs")
    report = {"passed": False, "policy": "each_actual_batch_vs_serial_fp32_v2",
              "max_abs_error": config.max_abs_error, "minimum_snr_db": config.minimum_snr_db,
              "max_relative_rms": config.max_relative_rms, "batch_size": len(audios),
              "padded_input_samples": len(audios) * max(padded_length(a.shape[-1]) for a in audios),
              "teacher": _json_copy(teacher.provenance), "clips": []}
    serial = []
    for audio, source_id, (latent, wave) in zip(audios, source_ids, batched):
        original_latent = teacher.encode(audio.to(teacher.device))
        original_wave = teacher.decode(original_latent)
        original_latent = original_latent.detach().cpu().contiguous()
        original_wave = original_wave.detach().cpu().contiguous()
        serial.append((original_latent, original_wave))
        n = audio.shape[-1]
        measures = {"latents": _errors(original_latent, latent, config),
                    "raw_waveform": _errors(original_wave, wave, config),
                    "valid_waveform": _errors(original_wave[..., :3*n], wave[..., :3*n], config)}
        report["clips"].append({"source_id": source_id, "input_samples": n,
            "prepared_audio_sha256": _tensor_hash(audio),
            "latent_frames": latent.shape[-1], "raw_output_samples": wave.shape[-1],
            "valid_output_samples": 3*n, **measures,
            "passed": all(m["passed"] for m in measures.values())})
    report["passed"] = all(c["passed"] for c in report["clips"])
    if not report["passed"]:
        report["reason"] = "numerical_parity_threshold_exceeded"
    return report, serial


@torch.no_grad()
def qualify_batch(teacher: FrozenAudioVAE2, audios: Sequence[Tensor], config: TeacherBatchConfig,
                  *, source_ids: Sequence[str]) -> dict:
    """Preliminary mixed-length screen, followed by a gate on every real batch.

    Runtime exceptions become an explicit failed report so callers can fall
    back to serial. Source-byte validation happens before this function. A
    serial teacher error still propagates if fallback cannot prepare the source.
    """
    if len(audios) < 2 or len(set(a.shape[-1] for a in audios)) < 2:
        return {"passed": False, "reason": "mixed_length_qualification_unavailable", "clips": []}
    if len(source_ids) != len(audios):
        raise ValueError("Qualification source IDs must match inputs")
    report = {"passed": False, "policy": "each_actual_batch_vs_serial_fp32_v2",
              "scope": "preliminary_mixed_length_screen_only",
              "max_abs_error": config.max_abs_error, "minimum_snr_db": config.minimum_snr_db,
              "max_relative_rms": config.max_relative_rms, "batch_size": len(audios),
              "padded_input_samples": len(audios) * max(padded_length(a.shape[-1]) for a in audios),
              "teacher": _json_copy(teacher.provenance), "clips": []}
    try:
        # Two full calls settle the original TorchScript profiling executor for
        # the actual batch shape, just as existing serial preparation warms up.
        for _ in range(2):
            _batch_outputs(teacher, audios, config)
        batched = _batch_outputs(teacher, audios, config)
        report, _ = _verify_actual_batch(teacher, audios, batched, config, source_ids=source_ids)
        report["scope"] = "preliminary_mixed_length_screen_only"
    except (RuntimeError, ValueError, TypeError) as error:
        report["reason"] = "qualification_execution_failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
    return report


def record_from_outputs(audio: Tensor, source: ManifestRow, teacher: FrozenAudioVAE2,
                        latents: Tensor, waveform: Tensor, *, reserved_rows=()) -> UtteranceCache:
    """Build the existing cache schema using already-qualified actual outputs.

    This mirrors prepare_utterance_cache's identity exactly, including tensor
    hashes; tests compare the complete metadata against the original serial API.
    The source's optional expected cache key is enforced before publication.
    """
    _validate_audio(audio)
    validate_manifest([source], reserved_rows=reserved_rows)
    provenance = _json_copy(teacher.provenance)
    _validate_teacher_provenance(provenance)
    n = audio.shape[-1]
    if abs(source.duration_seconds - n / 16000) > 1 / 16000 + 1e-10:
        raise ValueError("Prepared audio duration differs from source metadata")
    source_values = source.to_dict()
    expected_key = source_values["teacher_cache_key"]
    source_values["teacher_cache_key"] = None
    reference = audio.detach().contiguous().clone()
    identity = {"format_version": FORMAT_VERSION, "source": source_values, "teacher": provenance,
                "input_samples": n, "sample_rate_hz": 16000, "posterior": "raw_mu",
                "prepared_audio_sha256": _tensor_hash(reference),
                "target_policy": "continuous_utterance_raw_decode_with_original_length_metadata"}
    key = _identity_hash(identity)
    if expected_key is not None and expected_key != key:
        raise ValueError("Source teacher_cache_key differs from current source/teacher identity")
    latents, waveform = (value.detach().cpu().contiguous().clone() for value in (latents, waveform))
    metadata = {"identity": identity, "tensor_sha256": {
        name: _tensor_hash(value) for name, value in
        (("latents", latents), ("teacher_audio", waveform), ("reference16k", reference))}}
    record = UtteranceCache(latents, waveform, reference, metadata, key)
    record.validate()
    return record


def prefetch_corpus(corpus, row_ids, config: TeacherBatchConfig) -> dict:
    """SourceCorpus adapter; publication retains its original ownership checks."""
    from dataclasses import asdict, replace
    import time

    requested = []
    for row_id in row_ids:
        if row_id not in requested:
            requested.append(row_id)
        if len(requested) > 256:
            raise ValueError("Prefetch accepts at most 256 upcoming unique rows, not the entire corpus")
    started = time.perf_counter()
    report = {"format_version": 2, "config": asdict(config), "requested_row_ids": requested,
              "cache_hits": 0, "cache_misses": 0, "batched_utterances": 0,
              "serial_utterances": 0, "batches": [], "qualification": None}
    with corpus._guard:
        if corpus._closed:
            raise RuntimeError("SourceCorpus is closed")
        if corpus._teacher.provenance != corpus.teacher_identity:
            raise ValueError("Teacher identity changed after SourceCorpus construction")
        # Validate all IDs before reading or writing the first source.
        for row_id in requested:
            if row_id not in corpus.rows_by_id:
                raise KeyError(row_id)
        missing = []
        for row_id in requested:
            key = corpus._lookup_key(corpus.rows_by_id[row_id])
            if row_id in corpus._memory or key in corpus._entries:
                corpus.get(row_id)  # Existing verified memory/disk path.
                report["cache_hits"] += 1
            else:
                missing.append(row_id)
        report["cache_misses"] = len(missing)
        lengths = [corpus.input_sample_counts[row_id] for row_id in missing]
        groups = length_buckets(lengths, config)
        eligible = sorted((i for i, n in enumerate(lengths)
                           if 2 * padded_length(n) <= config.max_total_input_samples), key=lambda i: (lengths[i], i))
        qualification_key = (config, corpus.identity_sha256)
        previous = getattr(corpus, "_teacher_batch_qualification", None)
        # A process-local qualification is reused only for the identical frozen
        # teacher/corpus/config. This preliminary screen never replaces the
        # numerical gate on every newly generated batch below.
        if previous is not None and previous[0] == qualification_key:
            report["qualification"] = previous[1]
        elif config.max_batch_size > 1 and len(eligible) >= 2:
            longest = padded_length(lengths[eligible[-1]])
            size = min(config.max_batch_size, len(eligible), config.max_total_input_samples // longest)
            indices = [eligible[(i * (len(eligible) - 1)) // (size - 1)] for i in range(size)]
            samples = [corpus._read_source(corpus.rows_by_id[missing[i]]) for i in indices]
            qualification_started = time.perf_counter()
            report["qualification"] = qualify_batch(corpus._teacher, samples, config,
                                                     source_ids=[missing[i] for i in indices])
            report["qualification"]["wall_seconds"] = time.perf_counter() - qualification_started
            corpus._teacher_batch_qualification = (qualification_key, report["qualification"])
            del samples
        else:
            report["qualification"] = {"passed": False, "reason": "no_qualifiable_mixed_batch", "clips": []}
        qualified = report["qualification"]["passed"] is True
        for indices in groups:
            ids = [missing[i] for i in indices]
            padded = len(ids) * max(padded_length(lengths[i]) for i in indices)
            use_batch = qualified and len(ids) > 1 and padded <= config.max_total_input_samples
            if not use_batch:
                for row_id in ids:
                    corpus.get(row_id)
                report["serial_utterances"] += len(ids)
                report["batches"].append({"mode": "serial", "row_ids": ids,
                                          "input_samples": [corpus.input_sample_counts[i] for i in ids]})
                continue
            rows = [corpus.rows_by_id[row_id] for row_id in ids]
            audios = [corpus._read_source(row) for row in rows]
            teacher_started = time.perf_counter()
            try:
                values = _batch_outputs(corpus._teacher, audios, config)
            except RuntimeError as error:
                # No output from a failed batch is ever published. Subsequent
                # groups also take the original serial path in this process.
                qualified = False
                failure = {"passed": False, "reason": "qualified_batch_execution_failed",
                           "error_type": type(error).__name__, "error": str(error), "clips": []}
                # Release failed-forward tensors before a serial OOM fallback.
                error.__traceback__ = None
                corpus._teacher_batch_qualification = (qualification_key, failure)
                report["runtime_fallback"] = failure
                for row_id in ids:
                    corpus.get(row_id)
                report["serial_utterances"] += len(ids)
                report["batches"].append({"mode": "serial_after_batch_failure", "row_ids": ids,
                                          "input_samples": [corpus.input_sample_counts[i] for i in ids]})
                del audios
                continue
            verification, serial_values = _verify_actual_batch(
                corpus._teacher, audios, values, config, source_ids=ids)
            mode = "batch"
            if not verification["passed"]:
                # One failed row rejects the entire batch. Reuse the complete
                # serial references instead of encoding these sources again.
                values = serial_values
                qualified = False
                mode = "serial_after_batch_parity_failure"
                corpus._teacher_batch_qualification = (qualification_key, verification)
                report["parity_fallback"] = verification
            del serial_values
            # Integrity-check every selected output before publishing any member.
            records = [record_from_outputs(audio, row, corpus._teacher, latent, waveform,
                                           reserved_rows=corpus._reserved)
                       for audio, row, (latent, waveform) in zip(audios, rows, values)]
            corpus._stats["teacher_seconds"] += time.perf_counter() - teacher_started
            for record, row in zip(records, rows):
                key = corpus._lookup_key(row)
                metadata = _json_copy(record.metadata)
                metadata["source_preparation"] = {"reader": corpus.reader_identity, "lookup_key": key,
                                                   "source_file_sha256": row.audio_sha256}
                record = replace(record, metadata=metadata)
                corpus._verify_record(record, row, key)
                write_started = time.perf_counter()
                corpus._store_record(record, row, key)
                corpus._stats["cache_write_seconds"] += time.perf_counter() - write_started
                corpus._stats["cache_misses"] += 1
                corpus._stats["prepared_utterances"] += 1
                corpus._stats["prepared_input_samples"] += record.input_samples
                if corpus.max_memory_utterances:
                    corpus._memory[row.source_id] = record
                    while len(corpus._memory) > corpus.max_memory_utterances:
                        corpus._memory.popitem(last=False)
                # Each complete file/index pair survives interruption independently.
                corpus.flush()
            report["batched_utterances" if mode == "batch" else "serial_utterances"] += len(ids)
            report["batches"].append({"mode": mode, "row_ids": ids,
                                      "input_samples": [corpus.input_sample_counts[i] for i in ids],
                                      "padded_input_samples": padded,
                                      "verification": verification})
            del audios, values, records
        corpus.flush()
        report["mode"] = "batched" if report["batched_utterances"] else "serial_or_cache_hits"
        report["wall_seconds"] = time.perf_counter() - started
        report["cache_metrics"] = corpus.metrics()
        return report
