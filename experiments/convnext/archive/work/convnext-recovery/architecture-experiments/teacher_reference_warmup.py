"""Restore the previously qualified frozen-teacher startup protocol.

Three singleton reference encoder calls and three full-source decoder calls
must stabilize and reproduce the sealed cache exactly. No new targets,
tolerance changes, optimizer updates or model parameter changes are allowed.
"""
from __future__ import annotations

import torch

from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.teacher import SOURCE_SHA256, CHECKPOINT_SHA256
from full_source_teacher_head import (authenticated_source, crop_window,
    full_source_cache_key, preserved_teacher, tensor_bytes_sha, _exact)


def _startup_differences(values):
    reference = values[-1]
    rows = []
    for index, value in enumerate(values, 1):
        difference = value.double() - reference.double()
        rows.append({"call": index, "sha256": tensor_bytes_sha(value),
            "equal_to_final": torch.equal(value, reference),
            "max_abs_difference_to_final": float(difference.abs().max()),
            "rms_difference_to_final": float(difference.square().mean().sqrt())})
    return rows


def warmup_authenticated_teacher(teacher, crop, data, *, expected_teacher_state_sha256):
    """Warm only the first declared FIT source, then verify its original cache.

    The caller must supply a training crop. All source authentication, cached
    latent checks, and complete waveform checks use the same immutable
    contract as subsequent teacher-feature capture.
    """
    if (teacher.provenance.get("source_sha256") != SOURCE_SHA256
            or teacher.provenance.get("checkpoint_sha256") != CHECKPOINT_SHA256):
        raise ValueError("Unexpected warmup teacher provenance")
    if (not isinstance(expected_teacher_state_sha256, str)
            or len(expected_teacher_state_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_teacher_state_sha256)):
        raise ValueError("Expected the pinned teacher state hash")
    before = state_fingerprint(teacher.model.state_dict())
    if before != expected_teacher_state_sha256:
        raise ValueError("Teacher state differs before reference warmup")
    if teacher.device.type == "cuda" and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise ValueError("Reference warmup requires the original TF32-off policy")
    audio, inventory = authenticated_source(data, crop)
    frames = (audio.shape[-1] + 639) // 640
    n = crop.latents.shape[-1]
    encoded, decoded = [], []
    with preserved_teacher(teacher.model), torch.no_grad(), torch.autocast(
            device_type=teacher.device.type, enabled=False):
        with torch.backends.cudnn.flags(enabled=False, benchmark=False,
                                        deterministic=True, allow_tf32=False):
            for _ in range(3):
                z = teacher.model.encode(audio.to(teacher.device), 16000)
                if (z.shape != (1, 64, frames) or z.dtype != torch.float32
                        or not bool(torch.isfinite(z).all())):
                    raise RuntimeError("Warmup reference encoder shape/dtype/finite contract failed")
                encoded.append(z.detach().clone())
        _exact(encoded[1], encoded[2], "Reference encoder warmup last two calls")
        z = encoded[-1]
        _exact(crop_window(z, crop.context_start_frame, n), crop.latents.to(teacher.device),
               "Warmed full-source cropped latents versus cache")
        for _ in range(3):
            y = teacher.decode(z)
            if (y.shape != (1, 1, frames * 1920) or y.dtype != torch.float32
                    or not bool(torch.isfinite(y).all())):
                raise RuntimeError("Warmup reference decoder shape/dtype/finite contract failed")
            decoded.append(y.detach().clone())
        _exact(decoded[1], decoded[2], "Reference decoder warmup last two calls")
        target = decoded[-1]
        key = full_source_cache_key(inventory, before, z, target)
        if key != crop.cache_key:
            raise RuntimeError("Warmed full-source cache key is not identical")
        _exact(crop_window(target, crop.context_start_frame * 1920, n * 1920),
               crop.teacher_audio.to(teacher.device), "Warmed entire context/padding target versus cache")
        receipt = {"version": "authenticated_teacher_reference_warmup_v1",
            "source_id": crop.source_id, "start_frame": crop.start_frame,
            "context_start_frame": crop.context_start_frame,
            "source_inventory": inventory, "teacher_state_sha256": before,
            "encoder_calls": _startup_differences(encoded),
            "decoder_calls": _startup_differences(decoded),
            "encoder_last_two_exact": True, "decoder_last_two_exact": True,
            "entire_cached_latents_exact": True, "entire_cached_post_target_exact": True,
            "original_full_source_cache_key_exact": True, "cache_key_recreated": key,
            "full_source_latent_shape": list(z.shape), "full_source_waveform_shape": list(target.shape),
            "encoder_policy": "Singleton original FP32 raw mu; scoped cuDNN/TF32 disabled",
            "decoder_policy": "Singleton original full-source FP32 decoder; unchanged backend",
            "optimizer_updates": 0, "target_changed": False, "tolerance_relaxed": False,
            "interpretation": "Startup stabilization matching the earlier qualified capture protocol; not evidence that sealed targets or historical training were wrong."}
    if state_fingerprint(teacher.model.state_dict()) != before:
        raise RuntimeError("Teacher state changed during reference warmup")
    receipt["teacher_state_preserved"] = True
    return receipt
