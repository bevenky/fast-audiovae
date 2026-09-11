"""Recreate authenticated original full-source geometry for teacher head targets.

The fresh training cache decoded complete sources before slicing crops. A
mathematically sufficient latent context does not require different CUDA shapes
to produce identical floating point values. This helper preserves the original
geometry and authenticates the full-source z/y hashes through the saved crop key.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

from audiovae_student.batching import _validate_crop
from audiovae_student.data import ManifestRow
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.quiet_audio import _window_values
from audiovae_student.restart_data import digest
from audiovae_student.source_corpus import read_native_16k
from audiovae_student.teacher import SOURCE_SHA256, CHECKPOINT_SHA256


VERSION = "full-source-corrected-training-pairs-v1"


def tensor_bytes_sha(value):
    """Exact byte-only digest used by prepare_fresh_pairs.py."""
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def crop_window(value, start, length):
    if type(start) is not int or type(length) is not int or start < 0 or length < 1:
        raise ValueError("Invalid original crop window")
    part = value[..., start:start + length]
    return F.pad(part, (0, length - part.shape[-1])).contiguous().clone()


def full_source_cache_key(inventory, teacher_state_sha256, latents, waveform):
    return digest({"version": VERSION, "source": inventory,
                   "teacher_state_sha256": teacher_state_sha256,
                   "z": tensor_bytes_sha(latents), "y": tensor_bytes_sha(waveform)})


def authenticated_source(data, crop):
    """Reproduce the original no-new-transform audio and geometry qualification."""
    _validate_crop(crop)
    sid = crop.source_id
    row = ManifestRow.from_dict(data["rows"][sid])
    if row.source_id != sid:
        raise ValueError("Source row identity differs")
    native = row.original_sample_rate_hz == 16000 and all(
        v.casefold().split(":", 1)[0].strip() in {"none", "unchanged"}
        for v in (row.resampler_policy, row.gain_policy))
    prepared = (row.resampler_policy in {"prepared-native-16000-float-v1", "prepared-soxr-vhq-to-16000-v1"}
                and row.gain_policy == "no-additional-gain-normalization")
    if prepared and ((row.resampler_policy == "prepared-native-16000-float-v1") != (row.original_sample_rate_hz == 16000)):
        raise ValueError("Prepared source rate policy mismatch")
    if not native and not prepared:
        raise ValueError("Source transformation is not the originally qualified policy")
    path = Path(row.audio_path)
    if not path.is_absolute():
        raise ValueError("Authenticated source path must be absolute")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row.audio_sha256:
        raise ValueError("Authenticated source file hash differs")
    audio = read_native_16k(payload, row)
    count = data["counts"][sid]
    if (type(count) is not int or count < 1 or audio.shape != (1, 1, count)
            or audio.dtype != torch.float32 or not bool(torch.isfinite(audio).all())):
        raise ValueError("Authentic source sample geometry differs")
    if abs(count / 16000 - row.duration_seconds) > 1 / 16000 + 1e-10:
        raise ValueError("Authentic source duration differs")
    if (crop.context_start_frame < 0 or crop.start_frame < 0
            or crop.context_start_frame + crop.context_frames != crop.start_frame
            or crop.start_frame * 1920 + crop.valid_scored_samples > count * 3):
        raise ValueError("Crop's absolute source coordinates are invalid")
    reference = crop_window(audio, crop.context_start_frame * 640, crop.latents.shape[-1] * 640)
    if crop.reference16k is None or not torch.equal(reference, crop.reference16k.cpu()):
        raise ValueError("Cached reference does not exactly match authenticated source PCM")
    inventory = {"samples16k": count, "source_audio_sha256": row.audio_sha256,
                 "prepared_audio_sha256": tensor_bytes_sha(audio)}
    return audio, inventory


@contextmanager
def preserved_teacher(model):
    modes = tuple((m, m.training) for m in model.modules())
    versions = tuple((k, v._version) for k, v in model.state_dict().items())
    flags = tuple(p.requires_grad for p in model.parameters())
    gradients = tuple((p.grad, p.grad._version if p.grad is not None else None) for p in model.parameters())
    try:
        model.eval()
        yield
    finally:
        for module, mode in modes:
            module.training = mode
        if versions != tuple((k, v._version) for k, v in model.state_dict().items()):
            raise RuntimeError("Frozen teacher parameter or buffer changed")
        if flags != tuple(p.requires_grad for p in model.parameters()):
            raise RuntimeError("Teacher requires-grad flags changed")
        if any(p.grad is not g or (g is not None and g._version != version)
               for p, (g, version) in zip(model.parameters(), gradients)):
            raise RuntimeError("Existing teacher gradients changed")


def _exact(actual, target, label):
    if actual.shape != target.shape or actual.dtype != target.dtype:
        raise RuntimeError(label + " shape/dtype mismatch")
    if not torch.equal(actual, target):
        delta = actual.double() - target.double()
        raise RuntimeError(label + " is not exact: " + str({"max_abs": float(delta.abs().max()),
            "rms": float(delta.square().mean().sqrt()), "nonzero": int((delta != 0).sum())}))


def capture_full_source_pre_tanh(teacher, crop, data, quiet_config, *, expected_teacher_state_sha256,
                                 verify_teacher_state_hash=True):
    """Return original-geometry pre-tanh crop only after every exact gate passes.

    Executes one authentic full-source reference encoder and two full-source
    decoder calls. The caller supplies the already pinned teacher state hash.
    Source bytes, reference PCM, full crop latents, full crop post-tanh target,
    and the original cache key all pass exact checks. No targets are replaced.
    A bulk runner may set verify_teacher_state_hash=False after checking that
    hash once, then checking it again at the end. Per-call immutable version,
    mode and gradient checks remain enabled in either mode.
    """
    if (teacher.provenance.get("source_sha256") != SOURCE_SHA256
            or teacher.provenance.get("checkpoint_sha256") != CHECKPOINT_SHA256):
        raise ValueError("Unexpected teacher source or checkpoint provenance")
    if (type(verify_teacher_state_hash) is not bool or not isinstance(expected_teacher_state_sha256, str)
            or len(expected_teacher_state_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_teacher_state_sha256)):
        raise ValueError("Expected an explicit teacher hash and Boolean verification policy")
    before = (state_fingerprint(teacher.model.state_dict()) if verify_teacher_state_hash
              else expected_teacher_state_sha256)
    if before != expected_teacher_state_sha256:
        raise ValueError("Frozen teacher state hash differs")
    layers = teacher.model.decoder.model
    if len(layers) < 2 or not isinstance(layers[-2], torch.nn.Conv1d) or not isinstance(layers[-1], torch.nn.Tanh):
        raise ValueError("Expected native final teacher convolution and tanh")
    audio, inventory = authenticated_source(data, crop)
    frames = (audio.shape[-1] + 639) // 640
    observed = []
    with preserved_teacher(teacher.model), torch.no_grad(), torch.autocast(device_type=teacher.device.type, enabled=False):
        with torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=True, allow_tf32=False):
            z = teacher.model.encode(audio.to(teacher.device), 16000)
        if z.shape != (1, 64, frames) or z.dtype != torch.float32 or not bool(torch.isfinite(z).all()):
            raise RuntimeError("Original-geometry reference encoder contract failed")
        n = crop.latents.shape[-1]
        z_crop = crop_window(z, crop.context_start_frame, n)
        _exact(z_crop, crop.latents.to(teacher.device), "Full-source cropped latents versus cache")
        ordinary = teacher.decode(z)
        handle = layers[-2].register_forward_hook(lambda _m, _a, v: observed.append(v.detach().clone()))
        try:
            hooked = teacher.decode(z)
        finally:
            handle.remove()
        if len(observed) != 1 or ordinary.shape != (1, 1, frames * 1920):
            raise RuntimeError("Original-geometry decoder shape or hook count failed")
        pre = observed[0]
        if pre.dtype != torch.float32 or not bool(torch.isfinite(pre).all()):
            raise RuntimeError("Nonfinite or non-FP32 teacher pre-activation")
        _exact(hooked, ordinary, "Hooked versus ordinary full-source teacher")
        _exact(torch.tanh(pre), ordinary, "Captured full-source pre-tanh versus native")
        key = full_source_cache_key(inventory, before, z, ordinary)
        if key != crop.cache_key:
            raise RuntimeError("Original full-source cache key is not identical: " + str({
                "expected": crop.cache_key, "actual": key,
                "source_inventory": inventory, "full_z_sha256": tensor_bytes_sha(z),
                "full_y_sha256": tensor_bytes_sha(ordinary)}))
        y_crop = crop_window(ordinary, crop.context_start_frame * 1920, n * 1920)
        pre_crop = crop_window(pre, crop.context_start_frame * 1920, n * 1920)
        target = crop.teacher_audio.to(teacher.device)
        _exact(y_crop, target, "Entire context/padding post-tanh crop versus cache")
        _exact(torch.tanh(pre_crop), target, "Entire pre-tanh crop through tanh versus cache")
        start = crop.scored_slice.start + (6 if crop.context_start_frame > 0 else 0)
        stop = crop.scored_slice.stop
        if not 0 <= start < stop <= target.shape[-1]:
            raise ValueError("Invalid unchanged score mask")
        valid = torch.zeros_like(target, dtype=torch.bool)
        valid[..., start:stop] = True
        _, _, _, _, quiet_windows = _window_values(target, target, valid, quiet_config)
        quiet = quiet_windows.repeat_interleave(quiet_config.window_samples, dim=-1).unsqueeze(1)
        quiet = quiet[..., :target.shape[-1]] & valid
        result = {"pre_tanh": pre_crop, "post_tanh": y_crop, "target": target,
            "valid": valid, "quiet": quiet, "receipt": {"source_id": crop.source_id,
                "start_frame": crop.start_frame, "context_start_frame": crop.context_start_frame,
                "source_inventory": inventory, "full_source_latent_shape": list(z.shape),
                "crop_latent_shape": list(z_crop.shape), "crop_waveform_shape": list(y_crop.shape),
                "full_source_z_sha256": tensor_bytes_sha(z), "full_source_y_sha256": tensor_bytes_sha(ordinary),
                "full_source_pre_tanh_sha256": tensor_bytes_sha(pre),
                "cache_key_recreated": key, "entire_cached_latents_exact": True,
                "entire_cached_post_target_exact": True, "original_full_source_cache_key_exact": True,
                "native_hooked_and_tanh_exact": True, "encoder_forwards": 1,
                "singleton_decoder_forwards": 2, "inverse_tanh_used": False,
                "teacher_hash_verified_in_this_call": verify_teacher_state_hash,
                "teacher_state_sha256": before,
                "target_changed": False, "tolerance_relaxed": False,
                "scored_samples": int(valid.sum()), "quiet_samples": int(quiet.sum())}}
    if verify_teacher_state_hash and state_fingerprint(teacher.model.state_dict()) != before:
        raise RuntimeError("Frozen teacher state hash changed")
    return result
