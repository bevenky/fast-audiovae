"""Existing-layer refinement and strictly aligned auxiliary teacher features.

No helper changes the deployed decoder architecture. The auxiliary affine map
is fitted on declared training calibration sources, frozen during student
adaptation, and discarded for inference. Its target is the first complete
AudioVAE2 decoder block, not a claimed layer-to-layer semantic correspondence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.teacher import SOURCE_SHA256, CHECKPOINT_SHA256
from full_source_teacher_head import (authenticated_source, crop_window,
    full_source_cache_key, preserved_teacher, tensor_bytes_sha, _exact)
from run_joint_heads import HEAD_NAMES


TEACHER_FEATURE_CHANNELS = 1024
TEACHER_FEATURE_RATE = 200
SAMPLES_PER_TEACHER_FEATURE = 240


def last_block_parameter_names(model):
    if len(model.blocks) != 10:
        raise ValueError("Expected the retained ten-block student")
    names = {name for name, _ in model.named_parameters()
             if name.startswith("blocks.9.") or name in HEAD_NAMES}
    if not HEAD_NAMES.issubset(names) or not any(n.startswith("blocks.9.") for n in names):
        raise ValueError("Missing retained final-block or head parameters")
    return names


def select_last_block_parameters(model):
    names = last_block_parameter_names(model)
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
        parameter.grad = None
        if name in names:
            selected.append((name, parameter))
    return selected


def last_block_forward(model, frozen_input, *, return_features=False):
    if model.training or len(model.blocks) != 10:
        raise ValueError("Last-block replay requires the retained eval decoder")
    value = model.blocks[9](frozen_input)
    features = value
    value = model.affine(value)
    audio = model._waveform(model.output(model.activation(model.head(value))))
    return (audio, features) if return_features else audio


@torch.no_grad()
def capture_last_block_input(model, latents):
    if model.training or len(model.blocks) != 10:
        raise ValueError("Frozen-prefix capture requires the retained eval decoder")
    observed = []
    handle = model.blocks[9].register_forward_pre_hook(
        lambda _module, arguments: observed.append(arguments[0].detach().clone()))
    try:
        baseline = model(latents)
    finally:
        handle.remove()
    if len(observed) != 1:
        raise RuntimeError("Expected one final-block input capture")
    _exact(last_block_forward(model, observed[0]), baseline, "Frozen-prefix replay")
    return observed[0], baseline.detach()


def complete_teacher_feature_mask(valid):
    """Use complete physical 240-sample cells, with no averaging or shifts."""
    if (valid.ndim != 3 or valid.shape[1] != 1 or valid.dtype != torch.bool
            or valid.shape[-1] % SAMPLES_PER_TEACHER_FEATURE):
        raise ValueError("Expected a boolean [B,1,T] mask on whole teacher cells")
    return valid.reshape(*valid.shape[:2], -1, SAMPLES_PER_TEACHER_FEATURE).all(-1)


def capture_full_source_teacher_features(teacher, crop, data, *,
        expected_teacher_state_sha256, verify_teacher_state_hash=True):
    """Authenticate complete z/y source execution before exposing hidden targets.

    One reference encoder call and two ordinary-geometry decoder calls retain
    the existing sealed-cache contract. At 200 Hz, teacher cells 2i and 2i+1
    share the latest latent available to student 100 Hz frame i: floor(i/4).
    Cropping happens only after the authenticated full-source execution.
    """
    if (teacher.provenance.get("source_sha256") != SOURCE_SHA256
            or teacher.provenance.get("checkpoint_sha256") != CHECKPOINT_SHA256):
        raise ValueError("Unexpected teacher provenance")
    if (type(verify_teacher_state_hash) is not bool
            or not isinstance(expected_teacher_state_sha256, str)
            or len(expected_teacher_state_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_teacher_state_sha256)):
        raise ValueError("Expected an explicit teacher state hash")
    config = teacher.provenance.get("config", {})
    if (config.get("decoder_rates") != [8, 6, 5, 2, 2, 2]
            or not config.get("depthwise") or config.get("use_noise_block")
            or config.get("cond_type") != "scale_bias" or config.get("cond_out_layer")):
        raise ValueError("Unexpected teacher feature architecture or conditioning")
    layers = teacher.model.decoder.model
    if len(layers) != 11 or type(layers[2]).__name__ != "CausalDecoderBlock":
        raise ValueError("Expected first complete decoder block at index 2")
    before = (state_fingerprint(teacher.model.state_dict()) if verify_teacher_state_hash
              else expected_teacher_state_sha256)
    if before != expected_teacher_state_sha256:
        raise ValueError("Teacher state hash differs")
    audio, inventory = authenticated_source(data, crop)
    frames = (audio.shape[-1] + 639) // 640
    observed = []
    with preserved_teacher(teacher.model), torch.no_grad(), torch.autocast(
            device_type=teacher.device.type, enabled=False):
        with torch.backends.cudnn.flags(enabled=False, benchmark=False,
                                        deterministic=True, allow_tf32=False):
            z = teacher.model.encode(audio.to(teacher.device), 16000)
        if z.shape != (1, 64, frames) or z.dtype != torch.float32 or not bool(torch.isfinite(z).all()):
            raise RuntimeError("Full-source encoder feature contract failed")
        n = crop.latents.shape[-1]
        _exact(crop_window(z, crop.context_start_frame, n), crop.latents.to(teacher.device),
               "Full-source cropped latents versus cache")
        ordinary = teacher.decode(z)
        handle = layers[2].register_forward_hook(
            lambda _module, _arguments, output: observed.append(output.detach().clone()))
        try:
            hooked = teacher.decode(z)
        finally:
            handle.remove()
        if len(observed) != 1 or ordinary.shape != (1, 1, frames * 1920):
            raise RuntimeError("Full-source decoder shape or hook count failed")
        _exact(hooked, ordinary, "Hooked versus ordinary full-source teacher")
        features = observed[0]
        if (features.shape != (1, TEACHER_FEATURE_CHANNELS, frames * 8)
                or features.dtype != torch.float32 or not bool(torch.isfinite(features).all())):
            raise RuntimeError("Teacher feature rate/channel contract failed")
        key = full_source_cache_key(inventory, before, z, ordinary)
        if key != crop.cache_key:
            raise RuntimeError("Original full-source cache key is not identical")
        target = crop_window(ordinary, crop.context_start_frame * 1920, n * 1920)
        _exact(target, crop.teacher_audio.to(teacher.device), "Entire context/padding post-tanh target")
        cropped = crop_window(features, crop.context_start_frame * 8, n * 8)
        valid = torch.zeros_like(target, dtype=torch.bool)
        start = crop.scored_slice.start + (6 if crop.context_start_frame > 0 else 0)
        stop = crop.scored_slice.stop
        if not 0 <= start < stop <= target.shape[-1]:
            raise ValueError("Invalid unchanged waveform score mask")
        valid[..., start:stop] = True
        mask = complete_teacher_feature_mask(valid)
        result = {"teacher_features": cropped, "feature_valid": mask, "valid": valid,
            "receipt": {"source_id": crop.source_id, "start_frame": crop.start_frame,
                "context_start_frame": crop.context_start_frame, "cache_key_recreated": key,
                "teacher_state_sha256": before, "feature_layer": "decoder.model.2",
                "feature_rate_hz": TEACHER_FEATURE_RATE, "feature_channels": TEACHER_FEATURE_CHANNELS,
                "full_source_feature_sha256": tensor_bytes_sha(features),
                "crop_feature_sha256": tensor_bytes_sha(cropped),
                "full_source_z_sha256": tensor_bytes_sha(z), "full_source_y_sha256": tensor_bytes_sha(ordinary),
                "entire_cached_latents_exact": True, "entire_cached_post_target_exact": True,
                "original_full_source_cache_key_exact": True, "native_hooked_exact": True,
                "encoder_forwards": 1, "singleton_decoder_forwards": 2,
                "feature_start_index": crop.context_start_frame * 8,
                "valid_teacher_cells": int(mask.sum()), "target_changed": False,
                "alignment": "200Hz teacher cells2i,2i+1 for100Hz student framei; no interpolation"}}
    if verify_teacher_state_hash and state_fingerprint(teacher.model.state_dict()) != before:
        raise RuntimeError("Teacher state changed during feature capture")
    return result


def _check_feature_row(row):
    target, valid = row["teacher_features"], row["feature_valid"]
    if (target.ndim != 3 or valid.shape != (target.shape[0], 1, target.shape[-1])
            or valid.dtype != torch.bool or target.shape[-1] % 2
            or not target.is_floating_point() or not bool(torch.isfinite(target).all())):
        raise ValueError("Invalid aligned teacher feature row")
    return target, valid


@dataclass(frozen=True)
class FeatureNormalization:
    mean: torch.Tensor
    scale: torch.Tensor
    valid_cells: int
    std_floor: float
    source_ids: tuple[str, ...]

    def normalize(self, target):
        if target.shape[1] != self.mean.numel():
            raise ValueError("Teacher normalization channel mismatch")
        return ((target - self.mean.to(target)[None, :, None])
                / self.scale.to(target)[None, :, None])


@torch.no_grad()
def fit_feature_normalization(calibration_rows, *, std_floor_ratio=.01, absolute_floor=1e-6):
    """Fixed pooled channel statistics from the caller's training-only rows."""
    rows = tuple(calibration_rows)
    if not rows or not 0 < std_floor_ratio <= 1 or absolute_floor <= 0:
        raise ValueError("Positive normalization floor and training calibration rows required")
    sums = squares = None
    count = 0
    ids = []
    for row in rows:
        target, valid = _check_feature_row(row)
        value, mask = target.detach().cpu().double(), valid.detach().cpu()
        if sums is None:
            sums = torch.zeros(value.shape[1], dtype=torch.float64)
            squares = sums.clone()
        if value.shape[1] != sums.numel():
            raise ValueError("Calibration teacher channels changed")
        selected = torch.where(mask, value, 0.)
        sums += selected.sum((0, 2)); squares += selected.square().sum((0, 2))
        count += int(mask.sum())
        ids.append(str(row["source_id"]))
    if count < 2 or len(set(ids)) != len(ids):
        raise ValueError("Distinct training sources and two valid cells required")
    mean = sums / count
    variance = (squares / count - mean.square()).clamp_min(0.)
    floor = max(float(variance.mean().sqrt()) * std_floor_ratio, absolute_floor)
    scale = variance.sqrt().clamp_min(floor)
    return FeatureNormalization(mean, scale, count, floor, tuple(ids))


class AuxiliaryPhaseReadout(nn.Module):
    """Training-only affine map with an exact chronological two-phase shuffle."""
    def __init__(self, student_channels=512, teacher_channels=1024):
        super().__init__()
        self.teacher_channels = teacher_channels
        self.projection = nn.Conv1d(student_channels, teacher_channels * 2, 1)
        with torch.no_grad():
            self.projection.weight.zero_(); self.projection.bias.zero_()
        self.register_buffer("initialized", torch.tensor(False))
        self.requires_grad_(False)

    def forward(self, features):
        if not bool(self.initialized):
            raise RuntimeError("Fit the auxiliary readout on frozen training features first")
        packed = self.projection(features)
        return packed.reshape(features.shape[0], self.teacher_channels, 2, features.shape[-1]).permute(
            0, 1, 3, 2).reshape(features.shape[0], self.teacher_channels, -1)


@torch.no_grad()
def fit_auxiliary_ridge(adapter, calibration_rows, normalization, *, ridge_ratio=1e-3,
                        device="cpu", input_std_floor_ratio=.01):
    """Fit once, freeze, then pass feature-loss gradients into the student.

    Rows require student_features at100Hz, teacher_features at200Hz and masks.
    Only pairs of complete teacher cells enter the affine fit. The positive
    ridge is relative to the centered, standardized design covariance trace;
    intercepts are unpenalized. No student/teacher parameter is changed.
    """
    rows = tuple(calibration_rows)
    if not rows or not math.isfinite(ridge_ratio) or ridge_ratio <= 0 or not 0 < input_std_floor_ratio <= 1:
        raise ValueError("Positive fixed ridge and calibration rows required")
    if tuple(str(r["source_id"]) for r in rows) != normalization.source_ids:
        raise ValueError("Ridge and normalization calibration source identities differ")
    inputs = adapter.projection.in_channels
    outputs = adapter.teacher_channels * 2
    sx = torch.zeros(inputs, device=device, dtype=torch.float64)
    sy = torch.zeros(outputs, device=device, dtype=torch.float64)
    xx = torch.zeros(inputs, inputs, device=device, dtype=torch.float64)
    xy = torch.zeros(inputs, outputs, device=device, dtype=torch.float64)
    count = 0
    for row in rows:
        target, valid = _check_feature_row(row)
        student = row["student_features"]
        if (student.shape != (target.shape[0], inputs, target.shape[-1] // 2)
                or target.shape[1] != adapter.teacher_channels
                or not bool(torch.isfinite(student).all())):
            raise ValueError("Student/teacher feature clocks or channels do not align")
        n = student.shape[-1]
        mask = valid.reshape(target.shape[0], 1, n, 2).all(-1)[:, 0].to(device)
        x = student.detach().to(device=device, dtype=torch.float64).transpose(1, 2)[mask]
        normalized = normalization.normalize(target.detach().to(device=device, dtype=torch.float64))
        y = normalized.reshape(target.shape[0], adapter.teacher_channels, n, 2).permute(
            0, 1, 3, 2).reshape(target.shape[0], outputs, n).transpose(1, 2)[mask]
        sx += x.sum(0); sy += y.sum(0)
        xx += x.T @ x; xy += x.T @ y
        count += x.shape[0]
    if count < 2:
        raise ValueError("Ridge needs two complete aligned training pairs")
    mx, my = sx / count, sy / count
    covariance = xx / count - mx[:, None] * mx[None, :]
    covariance = (covariance + covariance.T) / 2
    cross = xy / count - mx[:, None] * my[None, :]
    std = covariance.diagonal().clamp_min(0).sqrt()
    input_floor = max(float(std.square().mean().sqrt()) * input_std_floor_ratio, 1e-6)
    std = std.clamp_min(input_floor)
    normalized_cov = covariance / std[:, None] / std[None, :]
    effective_ridge = max(float(normalized_cov.diagonal().mean()), 1e-12) * ridge_ratio
    solution = torch.linalg.solve(normalized_cov + effective_ridge * torch.eye(
        inputs, device=device, dtype=torch.float64), cross / std[:, None]) / std[:, None]
    intercept = my - mx @ solution
    if not bool(torch.isfinite(solution).all() and torch.isfinite(intercept).all()):
        raise FloatingPointError("Nonfinite auxiliary ridge solution")
    adapter.projection.weight.copy_(solution.T[..., None].to(adapter.projection.weight))
    adapter.projection.bias.copy_(intercept.to(adapter.projection.bias))
    adapter.initialized.fill_(True)
    adapter.requires_grad_(False)
    return {"calibration_source_ids": normalization.source_ids, "complete_student_frames": count,
        "ridge_ratio": ridge_ratio, "effective_standardized_ridge": effective_ridge,
        "input_std_floor": input_floor, "target_std_floor": normalization.std_floor,
        "adapter_weight_norm": float(solution.norm()), "adapter_bias_norm": float(intercept.norm()),
        "adapter_frozen_during_student_fit": True, "student_updates": 0,
        "scope": "Predict first teacher upsampling-block features; not final-waveform cancellation"}


def auxiliary_feature_loss(adapter, student_features, teacher_features, feature_valid,
                           normalization, *, total_valid_cells=None):
    target, valid = _check_feature_row({"teacher_features": teacher_features, "feature_valid": feature_valid})
    prediction = adapter(student_features)
    if prediction.shape != target.shape or valid.device != prediction.device or target.device != prediction.device:
        raise ValueError("Auxiliary prediction and target geometry/device differ")
    count = int(valid.sum())
    total = count if total_valid_cells is None else total_valid_cells
    if type(total) is not int or total < count:
        raise ValueError("Pooled feature-cell denominator is invalid")
    error = prediction - normalization.normalize(target.detach())
    return torch.where(valid, error.square(), 0.).sum() / max(total * target.shape[1], 1)
