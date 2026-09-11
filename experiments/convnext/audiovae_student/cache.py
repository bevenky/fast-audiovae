"""Continuous AudioVAE2 targets and aligned causal training crops.

This module accepts already prepared local 16 kHz FP32 mono utterances. It
does not download, resample, normalize, or identify benchmark recordings.
Supply the reserved regression/test manifest when preparing training caches,
and audit the complete corpus with data.build_training_manifest beforehand.
Original source-file checksums must have been computed by data preparation;
the independent prepared-tensor checksum here also pins the actual teacher input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from .data import ManifestRow, validate_manifest
from .teacher import FrozenAudioVAE2


FORMAT_VERSION = 1
ENCODER_HOP = 640
DECODER_HOP = 1920
LATENT_CHANNELS = 64


def _json_copy(value: Any) -> Any:
    # Enforce plain JSON metadata: no arbitrary Python classes enter .pt files.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _identity_hash(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tensor_hash(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _check_tensor(value: Tensor, shape: tuple[int, int, int], label: str) -> None:
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if value.dtype != torch.float32 or value.device.type != "cpu":
        raise ValueError(f"cached {label} must be CPU float32")
    if value.requires_grad or not torch.isfinite(value).all().item():
        raise ValueError(f"cached {label} must be finite and detached")


@dataclass(frozen=True)
class UtteranceCache:
    """One continuous utterance; retain 1,920T raw target samples and original N.

    Treat the tensors and metadata as immutable after construction. Complete
    checksum verification happens at preparation, load and save, not per crop.
    """

    latents: Tensor
    teacher_audio: Tensor
    reference16k: Tensor | None
    metadata: dict[str, Any]
    cache_key: str

    @property
    def input_samples(self) -> int:
        return self.metadata["identity"]["input_samples"]

    @property
    def valid_output_samples(self) -> int:
        return 3 * self.input_samples

    @property
    def latent_frames(self) -> int:
        return (self.input_samples + ENCODER_HOP - 1) // ENCODER_HOP

    def validate(self) -> None:
        metadata = _json_copy(self.metadata)
        identity = metadata.get("identity")
        if not isinstance(identity, dict) or identity.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported or missing cache format_version")
        length = identity.get("input_samples")
        if type(length) is not int or length < 1:
            raise ValueError("cache input_samples must be a positive integer")
        source = ManifestRow.from_dict(identity["source"])
        teacher = identity.get("teacher")
        _validate_teacher_provenance(teacher)
        if identity.get("sample_rate_hz") != 16000 or identity.get("posterior") != "raw_mu":
            raise ValueError("cache must contain raw posterior means from 16 kHz input")
        if abs(source.duration_seconds - length / 16000) > 1 / 16000 + 1e-10:
            raise ValueError("source duration does not match cached input sample count")
        if _identity_hash(identity) != self.cache_key:
            raise ValueError("cache key does not match source/teacher/preprocessing identity")
        frames = self.latent_frames
        _check_tensor(self.latents, (1, LATENT_CHANNELS, frames), "latents")
        _check_tensor(self.teacher_audio, (1, 1, frames * DECODER_HOP), "teacher_audio")
        tensors = {"latents": self.latents, "teacher_audio": self.teacher_audio}
        if self.reference16k is not None:
            _check_tensor(self.reference16k, (1, 1, length), "reference16k")
            tensors["reference16k"] = self.reference16k
            if _tensor_hash(self.reference16k) != identity.get("prepared_audio_sha256"):
                raise ValueError("reference16k no longer matches the prepared teacher input")
        hashes = metadata.get("tensor_sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(tensors):
            raise ValueError("cache tensor checksum inventory mismatch")
        for name, value in tensors.items():
            if _tensor_hash(value) != hashes[name]:
                raise ValueError(f"cache tensor SHA-256 mismatch: {name}")


def _validate_teacher_provenance(provenance: Any) -> None:
    if not isinstance(provenance, dict) or not isinstance(provenance.get("config"), dict):
        raise ValueError("teacher provenance must include serialized configuration")
    for name in ("source_sha256", "checkpoint_sha256"):
        value = provenance.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"teacher provenance must include {name}")
    expected = {"posterior": "raw_mu", "sample_rate_in": 16000, "sample_rate_out": 48000,
                "latent_channels": LATENT_CHANNELS, "encoder_hop": ENCODER_HOP,
                "decoder_hop": DECODER_HOP, "sr_cond": 48000, "dtype": "float32"}
    for name, value in expected.items():
        if provenance.get(name) != value:
            raise ValueError(f"teacher provenance violates {name}={value!r}")


@torch.no_grad()
def prepare_utterance_cache(
    audio16k: Tensor, source: ManifestRow, teacher: FrozenAudioVAE2, *,
    reserved_rows: Iterable[ManifestRow] = (), include_reference: bool = True,
) -> UtteranceCache:
    """Encode/decode once over the complete supplied utterance, before cropping.

    audio16k is [1,1,N] CPU float32. Its duration must match the manifest segment.
    Only the explicit teacher device is used; this function never picks a GPU.
    Caching dev/test material is allowed for evaluation, but sampling it for
    training is rejected separately. Existing teacher_cache_key is checked if set.
    """
    validate_manifest([source], reserved_rows=reserved_rows)
    if not isinstance(teacher, FrozenAudioVAE2):
        raise TypeError("teacher must be a FrozenAudioVAE2 instance")
    if not isinstance(audio16k, Tensor) or audio16k.ndim != 3 or audio16k.shape[:2] != (1, 1):
        raise ValueError("audio16k must have shape [1,1,N]")
    length = audio16k.shape[-1]
    if length < 1:
        raise ValueError("empty utterances cannot form a training cache")
    if audio16k.dtype != torch.float32 or audio16k.device.type != "cpu":
        raise ValueError("prepared audio16k must be CPU float32")
    if not torch.isfinite(audio16k).all().item():
        raise ValueError("prepared audio16k must be finite")
    if abs(source.duration_seconds - length / 16000) > 1 / 16000 + 1e-10:
        raise ValueError("prepared audio duration differs from the manifest source segment")
    provenance = _json_copy(teacher.provenance)
    _validate_teacher_provenance(provenance)
    # Copy before target generation so later caller mutation cannot invalidate inputs.
    audio = audio16k.detach().contiguous().clone()
    source_values = source.to_dict()
    expected_key = source_values["teacher_cache_key"]
    source_values["teacher_cache_key"] = None  # Avoid a circular cache identity.
    identity = {
        "format_version": FORMAT_VERSION, "source": source_values, "teacher": provenance,
        "input_samples": length, "sample_rate_hz": 16000, "posterior": "raw_mu",
        "prepared_audio_sha256": _tensor_hash(audio),
        "target_policy": "continuous_utterance_raw_decode_with_original_length_metadata",
    }
    cache_key = _identity_hash(identity)
    if expected_key is not None and expected_key != cache_key:
        raise ValueError("source teacher_cache_key does not match current input/teacher identity")
    latents = teacher.encode(audio.to(teacher.device))
    target = teacher.decode(latents)
    latents = latents.detach().cpu().contiguous().clone()
    target = target.detach().cpu().contiguous().clone()
    reference = audio if include_reference else None
    tensors = {"latents": latents, "teacher_audio": target}
    if reference is not None:
        tensors["reference16k"] = reference
    metadata = {"identity": identity, "tensor_sha256": {name: _tensor_hash(value) for name, value in tensors.items()}}
    record = UtteranceCache(latents, target, reference, metadata, cache_key)
    record.validate()
    return record


def save_cache(record: UtteranceCache, path: str | Path, *, overwrite: bool = False) -> None:
    """Publish a complete weights-only-safe .pt atomically in the target directory."""
    record.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": FORMAT_VERSION, "cache_key": record.cache_key,
               "metadata": _json_copy(record.metadata), "latents": record.latents,
               "teacher_audio": record.teacher_audio, "reference16k": record.reference16k}
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".utterance-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Atomic create-without-overwrite, including concurrent writers.
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_cache(path: str | Path, *, expected_cache_key: str | None = None) -> UtteranceCache:
    """Load tensors/plain metadata with weights_only=True and verify integrity."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    expected_fields = {"format_version", "cache_key", "metadata", "latents", "teacher_audio", "reference16k"}
    if not isinstance(payload, dict) or set(payload) != expected_fields or payload["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported utterance cache payload")
    if expected_cache_key is not None and payload["cache_key"] != expected_cache_key:
        raise ValueError("cache does not match the requested cache key")
    record = UtteranceCache(payload["latents"], payload["teacher_audio"], payload["reference16k"], payload["metadata"], payload["cache_key"])
    record.validate()
    return record


@dataclass(frozen=True)
class TrainingCrop:
    """A causal window with scored and unscored regions explicitly separated.

    Shapes retain a singleton batch: latents [1,64,C+S], waveform [1,1,1920(C+S)].
    C is actual available context, which may be below the requested history at a
    real utterance start. Never left-pad short history to simulate fake context.
    Right zero padding gives S requested score frames; scored_slice masks padding,
    including the partial last latent's invalid decoder samples.
    """

    latents: Tensor
    teacher_audio: Tensor
    reference16k: Tensor | None
    cache_key: str
    source_id: str
    start_frame: int
    context_start_frame: int
    context_frames: int
    scored_frames: int
    valid_scored_samples: int

    @property
    def scored_slice(self) -> slice:
        start = self.context_frames * DECODER_HOP
        return slice(start, start + self.valid_scored_samples)

    @property
    def reference_scored_slice(self) -> slice:
        start = self.context_frames * ENCODER_HOP
        return slice(start, start + self.valid_scored_samples // 3)

    @property
    def starts_at_utterance_start(self) -> bool:
        return self.context_start_frame == 0

    def loss_mask(self) -> Tensor:
        result = torch.zeros_like(self.teacher_audio, dtype=torch.bool)
        result[..., self.scored_slice] = True
        return result


def _window(value: Tensor, start: int, length: int) -> Tensor:
    output = value[..., start : start + length]
    return F.pad(output, (0, length - output.shape[-1])).contiguous().clone()


def sample_training_crop(
    record: UtteranceCache, start_frame: int, scored_frames: int = 64, context_frames: int = 29,
) -> TrainingCrop:
    """Slice previously continuous targets; do not run or reset the teacher.

    This function does no random sampling. A reproducible sampler chooses
    start_frame, permitting genuine starts/short utterances deliberately. A
    batch collator must respect varying available history and per-example masks;
    do not pass unmasked whole windows into reconstruction/adversarial losses.
    """
    for name, value, minimum in (("start_frame", start_frame, 0), ("scored_frames", scored_frames, 1), ("context_frames", context_frames, 29)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if record.metadata["identity"]["source"]["split"] != "train":
        raise ValueError("evaluation or regression cache cannot be sampled for training")
    if start_frame >= record.latent_frames:
        raise ValueError("start_frame must identify a real latent frame")
    context_start = max(0, start_frame - context_frames)
    context = start_frame - context_start
    frames = context + scored_frames
    valid_samples = min(scored_frames * DECODER_HOP, record.valid_output_samples - start_frame * DECODER_HOP)
    latents = _window(record.latents, context_start, frames)
    target = _window(record.teacher_audio, context_start * DECODER_HOP, frames * DECODER_HOP)
    reference = None
    if record.reference16k is not None:
        reference = _window(record.reference16k, context_start * ENCODER_HOP, frames * ENCODER_HOP)
    return TrainingCrop(latents, target, reference, record.cache_key,
                        record.metadata["identity"]["source"]["source_id"], start_frame,
                        context_start, context, scored_frames, valid_samples)
