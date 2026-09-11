"""True student batches with the original per-example scored-loss semantics.

Only the right ends of latent crops are padded for the shared causal forward.
Each example retains its real left context and its own scored start offset.
Before any FFT, predictions and targets are sliced to their exact valid scored
length, then grouped with equally long examples. Losses are weighted by example
count, matching serial per-example gradient accumulation, not by audio length.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .cache import DECODER_HOP, ENCODER_HOP, TrainingCrop
from .losses import WarmupReconstructionLoss
from .model import StudentDecoder


@dataclass(frozen=True)
class ScoredLossGroup:
    """Equal-length examples whose scalar loss reductions share one FFT batch."""

    indices: tuple[int, ...]
    values: dict[str, Tensor]
    has_reference: bool
    valid_scored_samples: int


@dataclass(frozen=True)
class BatchedCropLoss:
    """Differentiable example means plus correctly weighted logging groups.

    Backpropagate ``mean['total']`` once. The other ``mean`` values also include
    every example; absent-reference examples contribute zero reference loss.
    For the existing reference-only logging convention, aggregate that metric
    over groups with has_reference=True, weighted by len(group.indices).
    """

    mean: dict[str, Tensor]
    groups: tuple[ScoredLossGroup, ...]
    examples: int
    latent_batch_frames: int
    padded_latent_frames: int
    valid_scored_samples: int


def _validate_crop(crop: TrainingCrop) -> None:
    if not isinstance(crop, TrainingCrop):
        raise TypeError("Every batch example must be a TrainingCrop")
    for name, minimum in (("context_frames", 0), ("scored_frames", 1), ("valid_scored_samples", 1)):
        value = getattr(crop, name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    frames = crop.context_frames + crop.scored_frames
    if not isinstance(crop.latents, Tensor) or tuple(crop.latents.shape) != (1, 64, frames):
        raise ValueError("Crop latents must be [1,64,context_frames+scored_frames]")
    if not crop.latents.is_floating_point():
        raise ValueError("Crop latents must be floating-point")
    if (not isinstance(crop.teacher_audio, Tensor)
            or tuple(crop.teacher_audio.shape) != (1, 1, frames * DECODER_HOP)
            or not crop.teacher_audio.is_floating_point()):
        raise ValueError("Crop teacher audio must match its complete raw frame count")
    if not 0 < crop.valid_scored_samples <= crop.scored_frames * DECODER_HOP:
        raise ValueError("Valid scored samples exceed the scored crop region")
    if crop.valid_scored_samples % 3:
        raise ValueError("Valid scored samples must preserve the 16-to-48 kHz integer ratio")
    if crop.reference16k is not None and (
        not isinstance(crop.reference16k, Tensor)
        or tuple(crop.reference16k.shape) != (1, 1, frames * ENCODER_HOP)
        or not crop.reference16k.is_floating_point()
    ):
        raise ValueError("Crop original reference must match its complete raw frame count")


def batched_scored_crop_loss(
    model: StudentDecoder, crops: Sequence[TrainingCrop], criterion: WarmupReconstructionLoss,
    device: torch.device | str,
) -> BatchedCropLoss:
    """Run one genuine model batch, retaining each crop's independent loss mask.

    Caller controls grad mode, autocast, model mode and optimizer accumulation.
    A microbatch's mean total should be scaled by microbatch_size/effective_batch
    when several unequal microbatches contribute to a single optimizer update.
    No teacher call, cache mutation, sampling, trimming to a shared shorter
    duration, or extra left padding occurs here.
    """
    if not isinstance(model, StudentDecoder):
        raise TypeError("Batching requires the causal StudentDecoder")
    if not isinstance(criterion, WarmupReconstructionLoss):
        raise TypeError("Batching requires WarmupReconstructionLoss")
    crops = tuple(crops)
    if not crops:
        raise ValueError("A crop batch must contain at least one example")
    for crop in crops:
        _validate_crop(crop)
    first = crops[0].latents
    if any(crop.latents.dtype != first.dtype or crop.latents.device != first.device for crop in crops):
        raise ValueError("Crop latents must share source dtype and device before packing")
    device = torch.device(device)
    longest = max(crop.latents.shape[-1] for crop in crops)
    latent_batch = torch.cat([
        F.pad(crop.latents, (0, longest - crop.latents.shape[-1])) for crop in crops
    ], dim=0).to(device)
    prediction = model(latent_batch)
    expected_shape = (len(crops), 1, longest * DECODER_HOP)
    if tuple(prediction.shape) != expected_shape:
        raise ValueError("Student batch output violates the raw latent/sample contract")

    buckets: dict[tuple[int, bool], list[int]] = defaultdict(list)
    for index, crop in enumerate(crops):
        buckets[(crop.valid_scored_samples, crop.reference16k is not None)].append(index)
    groups = []
    for (samples, has_reference), indices in buckets.items():
        # Slice before concatenating or entering criterion; padded/context samples
        # never enter its window construction, spectral logs or normalization.
        predicted = torch.cat([prediction[i:i + 1, :, crops[i].scored_slice] for i in indices], dim=0)
        targets = torch.cat([crops[i].teacher_audio[..., crops[i].scored_slice].to(device) for i in indices], dim=0)
        references = None
        if has_reference:
            references = torch.cat([
                crops[i].reference16k[..., crops[i].reference_scored_slice].to(device) for i in indices
            ], dim=0)
        values = criterion(predicted, targets, references)
        groups.append(ScoredLossGroup(tuple(indices), values, has_reference, samples))
    keys = set(groups[0].values)
    if any(set(group.values) != keys or any(value.ndim != 0 for value in group.values.values()) for group in groups):
        raise ValueError("Loss criterion must return the same scalar components for every group")
    means = {
        name: torch.stack([group.values[name] * len(group.indices) for group in groups]).sum() / len(crops)
        for name in groups[0].values
    }
    return BatchedCropLoss(
        means, tuple(groups), len(crops), longest,
        sum(longest - crop.latents.shape[-1] for crop in crops),
        sum(crop.valid_scored_samples for crop in crops),
    )
