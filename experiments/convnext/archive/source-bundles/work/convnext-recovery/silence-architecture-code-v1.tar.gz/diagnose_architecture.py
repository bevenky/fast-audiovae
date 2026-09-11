"""Frozen silence architecture diagnostics; no hooks, fitted weights or updates.

probe_silence_architecture(model, encoded_zero_crop, quiet_crops=...) returns
JSON-compatible evidence only. Natural crop selection uses teacher audio only.
The external readout calculation is a feasibility test for one stationary
signal, not a replacement decoder or evidence of generalization.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import math

import torch

from audiovae_student.batching import _validate_crop
from audiovae_student.quiet_audio import QuietAudioConfig, _window_values


def _rms(x):
    return float(x.double().square().mean().sqrt()) if x.numel() else None


def _ratio(a, b):
    return a / b if b else None


def _feature_components(features, start, period):
    """DC means per channel, phase differences, and cycle-to-cycle variation."""
    start = ((start + period - 1) // period) * period
    count = (features.shape[-1] - start) // period
    if count < 2:
        raise ValueError("Need at least two complete steady feature cycles")
    x = features[0, :, start:start + count * period].double().reshape(features.shape[1], count, period)
    phases = x.mean(1)
    channel_dc = phases.mean(-1, keepdim=True)
    phase_power = float((phases - channel_dc).square().mean())
    varying_power = float((x - phases[:, None]).square().mean())
    dc_power = float(channel_dc.square().mean())
    total = float(x.square().mean())
    return {"channels": features.shape[1], "complete_cycles": count, "period_frames": period,
        "steady_start_frame": start, "total_power": total, "channel_dc_power": dc_power,
        "phase_difference_power": phase_power, "cycle_varying_power": varying_power,
        "phase_difference_fraction": _ratio(phase_power, total),
        "cycle_varying_fraction": _ratio(varying_power, total),
        "cycle_varying_max_abs": float((x - phases[:, None]).abs().max()),
        "power_partition_error": total - dc_power - phase_power - varying_power}, phases.T.cpu()


def _wave_components(audio, start):
    start = ((start + 1919) // 1920) * 1920
    count = (audio.numel() - start) // 1920
    if count < 2:
        return {"defined": False, "reason": "fewer than two complete40ms cycles"}
    x = audio.reshape(-1)[start:start + count * 1920].double().reshape(count, 4, 480)
    phase = x.mean(0)
    mean480 = phase.mean(0, keepdim=True)
    dc = mean480.mean()
    parts = {"dc": float(dc.square()), "shared_480_pattern": float((mean480 - dc).square().mean()),
             "additional_1920_phase_pattern": float((phase - mean480).square().mean()),
             "cycle_varying": float((x - phase).square().mean())}
    total = float(x.square().mean())
    return {"defined": True, "complete_cycles": count, "steady_start_sample": start,
        "rms": math.sqrt(total), "power": total, "component_power": parts,
        "component_fraction": {k: _ratio(v, total) for k, v in parts.items()},
        "power_partition_error": total - sum(parts.values())}


@contextmanager
def _eval_preserved(model):
    modes = tuple((m, m.training) for m in model.modules())
    versions = tuple((k, v._version) for k, v in model.state_dict().items())
    flags = tuple(p.requires_grad for p in model.parameters())
    grads = tuple((p.grad, p.grad._version if p.grad is not None else None) for p in model.parameters())
    try:
        model.eval()
        yield
    finally:
        for module, mode in modes:
            module.training = mode
        if versions != tuple((k, v._version) for k, v in model.state_dict().items()):
            raise RuntimeError("Model parameter or buffer changed")
        if flags != tuple(p.requires_grad for p in model.parameters()):
            raise RuntimeError("Parameter requires_grad changed")
        if any(p.grad is not g or (g is not None and g._version != version)
               for p, (g, version) in zip(model.parameters(), grads)):
            raise RuntimeError("Existing parameter gradients changed")


def _trace(model, z, steady_seconds):
    if getattr(model, "output_filter", None) is not None or (
            hasattr(model, "fusion_config") and model.fusion_config.terminal_tanh):
        raise ValueError("This diagnostic requires the unchanged direct linear output head")
    rows = {}
    rows["encoder_latents"], _ = _feature_components(z, math.ceil(steady_seconds * 25), 1)
    first = math.ceil(steady_seconds * 100)
    def record(name, value):
        rows[name], phases = _feature_components(value, first, 4)
        return phases
    x = model._phase_frames(z); record("adapter_phase_frames", x)
    x = model.stem(x); record("stem", x)
    x = model.stem_norm(x); record("stem_norm", x)
    for i, block in enumerate(model.blocks):
        x = block(x); record("blocks." + str(i), x)
    x = model.affine(x); record("affine", x)
    x = model.head(x); record("head_before_prelu", x)
    x = model.activation(x); h = record("head_after_prelu", x)
    projected = model.output(x); projected_phases = record("output_480_channels", projected)
    waveform = model._waveform(projected)
    ordinary = model(z)
    if not torch.equal(waveform, ordinary):
        raise RuntimeError("Explicit stage replay differs from ordinary eval forward")
    return waveform, rows, h, projected_phases


def _readout_feasibility(h, weight, teacher_phases, measured_output_phases):
    """All algebra CPU FP64, four phase vectors; no new model weights made."""
    h, w, target, observed = (x.detach().cpu().double() for x in
                             (h, weight.squeeze(-1).T, teacher_phases, measured_output_phases))
    if h.ndim != 2 or h.shape[0] != 4 or target.shape != (4, 480) or w.shape != (h.shape[1], 480):
        raise ValueError("Expected4 phase vectors and480-output bias-free readout")
    u, s, _ = torch.linalg.svd(h, full_matrices=False)
    threshold = float(s.max()) * max(h.shape) * torch.finfo(torch.float64).eps
    active = s > threshold
    rank = int(active.sum())
    current = h @ w
    weight_norm = float(w.norm())
    candidates = {"teacher_phase_target": target,
                  "zero_waveform": torch.zeros_like(target),
                  "constant_waveform_1e-5": torch.full_like(target, 1e-5)}
    answers = {}
    for name, desired in candidates.items():
        residual = desired - current
        coordinates = u[:, active].T @ residual
        recoverable = u[:, active] @ coordinates
        error = residual - recoverable
        minimum_norm = float((coordinates / s[active, None]).norm()) if rank else 0.0
        answer = {"target_rms": _rms(desired), "current_target_error_rms": _rms(residual),
            "best_frozen_feature_error_rms": _rms(error),
            "minimum_delta_weight_frobenius_norm": minimum_norm,
            "delta_norm_over_current_weight_norm": _ratio(minimum_norm, weight_norm),
            "relative_target_residual": _ratio(float(error.norm()), float(desired.norm())),
            "max_abs_unrepresentable_residual": float(error.abs().max())}
        # A tiny singular value can make nominal representability ill-conditioned.
        for rtol in (1e-6, 1e-4):
            keep = s > float(s.max()) * rtol
            limited = residual - u[:, keep] @ (u[:, keep].T @ residual)
            answer["error_rms_at_relative_svd_cutoff_" + str(rtol)] = _rms(limited)
        answers[name] = answer
    return {"feature_shape": list(h.shape), "feature_singular_values": s.tolist(),
        "fp64_rank": rank, "rank_threshold": threshold,
        "retained_condition_number": float(s[active].max() / s[active].min()) if rank else None,
        "all_four_phase_targets_representable_in_exact_linear_algebra": rank == 4,
        "phase_features_nonzero": bool(h.norm() > 0), "current_weight_frobenius_norm": weight_norm,
        "same_features_fp64_vs_actual_fp32_projection_rms": _rms(current - observed),
        "same_features_fp64_projection_teacher_error_rms": _rms(current - target),
        "actual_fp32_phase_projection_teacher_error_rms": _rms(observed - target),
        "targets": answers,
        "scope": "One stationary signal and frozen phase features only. Feasibility/correction norm do not establish learning ease or preservation of other audio. No readout is installed or saved."}


def _mask(crop, reference, config, *, after_sample=0):
    valid = torch.zeros_like(reference, dtype=torch.bool)
    start = max(crop.scored_slice.start + (6 if crop.context_start_frame > 0 else 0), after_sample)
    valid[..., start:crop.scored_slice.stop] = True
    _, _, _, _, windows = _window_values(reference, reference, valid, config)
    quiet = windows.repeat_interleave(config.window_samples, -1).unsqueeze(1)[..., :reference.shape[-1]] & valid
    return quiet


def select_natural_quiet(crops, metadata, *, history_samples, limit=6, config=QuietAudioConfig()):
    """Select longest teacher-quiet exposure/source, never student-error ranking."""
    if not 0 <= limit <= 6:
        raise ValueError("At most six source-distinct natural cases")
    if limit == 0:
        return [], []
    candidates = []
    for c in crops:
        if c.source_id.startswith("encoded_"):
            continue
        count = int(_mask(c, c.teacher_audio, config, after_sample=history_samples).sum())
        if count:
            candidates.append((-count, c.source_id, c.start_frame, c))
    selected, seen = [], set()
    for negative_count, sid, start, crop in sorted(candidates, key=lambda x: x[:3]):
        if sid in seen:
            continue
        selected.append(crop); seen.add(sid)
        if len(selected) == limit:
            break
    return selected, [{"source_id": c.source_id, "start_frame": c.start_frame,
        "teacher_quiet_samples_after_history": int(_mask(c, c.teacher_audio, config, after_sample=history_samples).sum()),
        "metadata": metadata.get(c.source_id, {})} for c in selected]


@torch.inference_mode(False)
@torch.no_grad()
def probe_silence_architecture(model, encoded_zero_crop, *, device="cuda", quiet_crops=(),
                              quiet_config=QuietAudioConfig(), steady_seconds=2.0, expected_prediction=None):
    _validate_crop(encoded_zero_crop)
    if encoded_zero_crop.source_id != "encoded_zero" or encoded_zero_crop.context_start_frame != 0:
        raise ValueError("Supply the complete canonical encoded-zero fixture")
    if len(quiet_crops) > 6 or len({c.source_id for c in quiet_crops}) != len(quiet_crops):
        raise ValueError("At most six predefined source-distinct natural crops")
    device = torch.device(device)
    model_device = next(model.parameters()).device
    if device.type == "cuda" and device.index is None:
        device = model_device if model_device.type == "cuda" else device
    if device != model_device:
        raise ValueError("Supply the existing model on the requested diagnostic device")
    history_samples = model.config.history_frames * 480
    history_latents = (history_samples + 1919) // 1920
    prefix_frames = history_latents + 2
    steady_start = math.ceil(steady_seconds * 48000 / 1920) * 1920
    z = encoded_zero_crop.latents.detach().to(device).clone()
    teacher = encoded_zero_crop.teacher_audio.detach().to(device).clone()
    with _eval_preserved(model), torch.autocast(device_type=device.type, enabled=False):
        p, nodes, h, projected_phases = _trace(model, z, steady_seconds)
        if expected_prediction is not None and not torch.equal(p, expected_prediction.to(p)):
            raise RuntimeError("Encoded-zero prediction differs from supplied canonical output")
        stationary_z = z[..., steady_start // 1920:steady_start // 1920 + 1]
        difference = (z - stationary_z).abs().amax((0, 1))
        nonmatching = torch.nonzero(difference > 0).reshape(-1)
        exact_stationary_suffix_frame = int(nonmatching[-1]) + 1 if len(nonmatching) else 0
        constant_output = model(stationary_z.expand_as(z).contiguous())
        prefixed_output = model(torch.cat((stationary_z.expand(-1, -1, prefix_frames), z), -1))
        prefixed_output = prefixed_output[..., prefix_frames * 1920:]
        after = max(steady_start, exact_stationary_suffix_frame * 1920 + history_samples)
        cycles = (teacher.shape[-1] - steady_start) // 1920
        if cycles < 2:
            raise ValueError("Encoded-zero fixture needs at least two steady cycles")
        target_phases = teacher[0, 0, steady_start:steady_start + cycles * 1920].double().reshape(cycles, 4, 480).mean(0)
        feasibility = _readout_feasibility(h, model.output.weight, target_phases, projected_phases)
        natural = []
        for crop in quiet_crops:
            _validate_crop(crop)
            if crop.source_id.startswith("encoded_"):
                raise ValueError("Natural counterfactual list contains a fixture")
            t = crop.teacher_audio.to(device)
            mask = _mask(crop, t, quiet_config, after_sample=history_samples + 1920)
            if not bool(mask.any()):
                natural.append({"source_id": crop.source_id, "start_frame": crop.start_frame,
                                "defined": False, "reason": "no teacher-quiet samples after history"}); continue
            sample = int(torch.nonzero(mask[0, 0]).reshape(-1)[0])
            latent_index = sample // 1920
            original_z = crop.latents.to(device)
            fixed = original_z[..., latent_index:latent_index + 1].expand_as(original_z).contiguous()
            original, stationary = model(original_z), model(fixed)
            natural.append({"source_id": crop.source_id, "start_frame": crop.start_frame, "defined": True,
                "selected_by": "predefined teacher-only quiet coverage; reference is first eligible quiet sample",
                "quiet_samples": int(mask.sum()), "reference_latent_index": latent_index,
                "teacher_rms": _rms(t[mask]), "original_teacher_residual_rms": _rms((original - t)[mask]),
                "stationary_counterfactual_teacher_residual_rms": _rms((stationary - t)[mask]),
                "input_variation_output_difference_rms": _rms((original - stationary)[mask]),
                "constant_latent_waveform_components": _wave_components(stationary, history_samples + 1920),
                "warning": "Counterfactual latents no longer encode the original teacher waveform; a smaller residual is not quality improvement or a valid replacement."})
        result = {"format_version": 1, "source_id": encoded_zero_crop.source_id,
            "manual_stage_replay_matches_model_exactly": True,
            "expected_prediction_reproduced": True if expected_prediction is not None else None,
            "history_internal_frames": model.config.history_frames, "history_samples": history_samples,
            "steady_start_sample": steady_start, "node_components": nodes,
            "waveform_components": {"student": _wave_components(p, steady_start),
                "teacher": _wave_components(teacher, steady_start), "residual": _wave_components(p - teacher, steady_start)},
            "encoded_latents": {"raw_zero_is_encoded_silence": bool((z == 0).all()),
                "steady_reference_rms": _rms(stationary_z),
                "max_difference_from_reference": float(difference.max()),
                "exact_stationary_suffix_frame": exact_stationary_suffix_frame},
            "boundary_test": {"prefix_frames": prefix_frames,
                "startup_difference_rms": _rms((prefixed_output - p)[..., :history_samples]),
                "after_receptive_field_difference_rms": _rms((prefixed_output - p)[..., history_samples:]),
                "steady_difference_max_abs": float((prefixed_output - p)[..., steady_start:].abs().max())},
            "stationary_input_test": {"comparison_start_sample": after,
                "comparison_samples": max(0, p.shape[-1] - after),
                "output_difference_rms_after_input_and_decoder_settle": _rms((constant_output - p)[..., after:])},
            "projection_feasibility": feasibility, "natural_quiet_counterfactuals": natural,
            "quiet_config": asdict(quiet_config), "updates": 0, "weights_saved": False,
            "interpretation": "A phase-invariant internal feature vector may still map to480 different output-channel values, producing a10ms repeating waveform. Four distinct feature phases permit additional40ms structure. Neither observation proves architecture impossibility; feasibility on silence alone does not establish global quality."}
    result["model_modes_parameters_buffers_and_grad_fields_preserved"] = True
    return result


def self_test():
    # Known phase-invariant features can produce non-flat480-periodic audio.
    h = torch.ones(4, 1, dtype=torch.float64)
    w = torch.arange(480, dtype=torch.float64).reshape(480, 1, 1) / 480
    pattern = (h @ w.squeeze(-1).T).reshape(1, 1, 1920).repeat(1, 1, 3)
    c = _wave_components(pattern, 0)
    assert c["component_power"]["shared_480_pattern"] > 0
    assert c["component_power"]["additional_1920_phase_pattern"] == 0
    f = _readout_feasibility(h, w, torch.zeros(4, 480), pattern.reshape(3, 4, 480).mean(0))
    assert f["fp64_rank"] == 1
    assert f["targets"]["constant_waveform_1e-5"]["best_frozen_feature_error_rms"] < 1e-12
    independent = _readout_feasibility(torch.eye(4), torch.zeros(480, 4, 1),
        torch.arange(1920).reshape(4, 480), torch.zeros(4, 480))
    assert independent["fp64_rank"] == 4
    assert independent["targets"]["teacher_phase_target"]["best_frozen_feature_error_rms"] == 0
    impossible = _readout_feasibility(h, w, torch.arange(4)[:, None].expand(4, 480),
                                    pattern.reshape(3, 4, 480).mean(0))
    assert impossible["targets"]["teacher_phase_target"]["best_frozen_feature_error_rms"] > 1
    x = torch.tensor([[[0., 1., 2., 3., 0., 1., 2., 3.]]])
    stats, _ = _feature_components(x, 0, 4)
    assert stats["phase_difference_power"] > 0 and stats["cycle_varying_power"] == 0
    print("PASS: CPU phase/channel separation, rank1 constant-target feasibility, rank4 arbitrary-target feasibility, rank-deficient incompatible target, stationary phase decomposition")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args(); self_test()
