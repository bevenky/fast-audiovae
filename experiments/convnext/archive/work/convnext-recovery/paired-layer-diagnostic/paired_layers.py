"""Paired frozen decoder instrumentation on the canonical six-second silence.

No encoder inference, hooks, parameter changes, fitted layers or audio files.
Teacher and student hidden values are not aligned feature targets. Only their
physical-rate temporal/phase behavior and final waveform projections are compared.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.teacher import SOURCE_SHA256, CHECKPOINT_SHA256
from diagnose_architecture import _eval_preserved, _trace, _wave_components, _rms


def phase_statistics(value, rate, steady_seconds=2.):
    """Per-channel phase decomposition using40ms cycles at the true feature rate."""
    if rate % 25 or value.shape[0] != 1 or value.ndim != 3:
        raise ValueError("Expected a singleton temporal feature tensor at a multiple of25Hz")
    period = rate // 25
    start = math.ceil(steady_seconds * rate / period) * period
    count = (value.shape[-1] - start) // period
    if count < 2:
        raise ValueError("Too little steady context for two complete40ms cycles")
    x = value[0, :, start:start + count * period].double().reshape(value.shape[1], count, period)
    phase = x.mean(1)
    dc = phase.mean(-1, keepdim=True)
    powers = {"channel_dc": float(dc.square().mean()),
              "all_40ms_phase_structure": float((phase - dc).square().mean()),
              "cycle_varying": float((x - phase[:, None]).square().mean())}
    if rate % 100 == 0:
        p = phase.reshape(value.shape[1], 4, rate // 100)
        ten = p.mean(1, keepdim=True)
        powers.update(shared_10ms_phase_pattern=float((ten - dc[..., None]).square().mean()),
                      additional_40ms_phase_pattern=float((p - ten).square().mean()))
    total = float(x.square().mean())
    result = {"feature_rate_hz": rate, "channels": value.shape[1], "feature_frames": value.shape[-1],
        "period_40ms_frames": period, "period_10ms_frames": rate // 100 if rate % 100 == 0 else None,
        "steady_start_frame": start, "complete_cycles": count, "total_power": total,
        "component_power": powers, "phase_power_fraction": powers["all_40ms_phase_structure"] / total if total else None,
        "cycle_varying_fraction": powers["cycle_varying"] / total if total else None,
        "cycle_varying_max_abs": float((x - phase[:, None]).abs().max()),
        "hidden_scale_warning": "Feature power is descriptive, not a teacher/student quality error or an invariant measure across differently scaled layers."}
    return result, phase.detach().cpu()


def _difference(left, right):
    delta = left.detach().double() - right.detach().double()
    return {"bitwise_equal": torch.equal(left, right), "max_abs": float(delta.abs().max()),
            "rms": float(delta.square().mean().sqrt()), "different_elements": int((left != right).sum())}


def native_startup_check(wrapper, z, target):
    """Three native forwards, recording cold-start differences without tolerance."""
    first = wrapper.decode(z)
    second = wrapper.decode(z)
    third = wrapper.decode(z)
    report = {"ordinary_calls_before_manual": 3,
        "first_vs_second": _difference(first, second),
        "second_vs_third": _difference(second, third),
        "first_vs_canonical": _difference(first, target),
        "second_vs_canonical": _difference(second, target),
        "third_vs_canonical": _difference(third, target),
        "interpretation": "Cold-start output differences are recorded, not ignored. A difference that disappears with native forwards demonstrates execution-startup dependence; it does not by itself isolate TorchScript, cuDNN or another backend component."}
    if not torch.equal(third, target):
        raise RuntimeError("Third native teacher decode differs from canonical target: " + str(report))
    return third, report


def teacher_stage_trace(wrapper, z, steady_seconds=2., *, stable_ordinary=None):
    """Manually follow the verified source, calling its untouched leaf modules."""
    decoder = wrapper.model.decoder
    cfg = wrapper.provenance["config"]
    if (cfg["decoder_rates"] != [8, 6, 5, 2, 2, 2] or not cfg["depthwise"]
            or cfg["use_noise_block"] or cfg["cond_type"] != "scale_bias" or cfg["cond_out_layer"]):
        raise ValueError("Unexpected teacher architecture/conditioning")
    rows = []
    def record(name, op, value, rate):
        info, phase = phase_statistics(value, rate, steady_seconds)
        rows.append({"name": name, "operation": op, **info})
        return phase
    def visit(layer, x, rate, name):
        kind = type(layer).__name__
        if kind == "CausalDecoderBlock":
            for j, child in enumerate(layer.block):
                x, rate = visit(child, x, rate, name + ".block." + str(j))
            record(name, "decoder_block_output", x, rate)
        elif kind == "CausalResidualUnit":
            original = x
            for j, child in enumerate(layer.block):
                x, rate = visit(child, x, rate, name + ".residual_branch." + str(j))
            if original.shape != x.shape:
                raise ValueError("Teacher residual unexpectedly requires cropping")
            x = original + x
            record(name + ".add", "residual_add", x, rate)
        else:
            if kind == "NoiseBlock":
                raise ValueError("Stochastic decoder is outside this frozen diagnostic")
            x = layer(x)
            if isinstance(layer, torch.nn.ConvTranspose1d):
                rate *= layer.stride[0]
            elif isinstance(layer, torch.nn.Conv1d) and layer.stride[0] != 1:
                raise ValueError("Unexpected decoder downsampling")
            record(name, kind, x, rate)
        return x, rate
    rate, x = 25, z
    record("decoder_input", "raw_encoder_mu", x, rate)
    conditions = torch.full((1,), wrapper.sr_cond, device=z.device, dtype=torch.int32)
    indices = decoder.get_sr_idx(conditions)
    final_features = pre_tanh = None
    layers = tuple(decoder.model)
    for i, (layer, condition) in enumerate(zip(layers, decoder.sr_cond_model, strict=True)):
        if condition is not None:
            x = condition(x, indices)
            record("decoder.sr_cond_model." + str(i), "sample_rate_scale_bias_48000", x, rate)
        x, rate = visit(layer, x, rate, "decoder.model." + str(i))
        if i == len(layers) - 3:
            _, final_features = phase_statistics(x, rate, steady_seconds)
        elif i == len(layers) - 2:
            _, pre_tanh = phase_statistics(x, rate, steady_seconds)
    if rate != 48000 or final_features is None or pre_tanh is None:
        raise ValueError("Missing teacher final feature/projection stages")
    ordinary = wrapper.decode(z)
    replay = {"manual_vs_final_ordinary": _difference(x, ordinary)}
    if stable_ordinary is not None:
        replay.update(third_vs_fourth_ordinary=_difference(stable_ordinary, ordinary),
                      manual_vs_third_ordinary=_difference(x, stable_ordinary))
        if not torch.equal(stable_ordinary, ordinary):
            raise RuntimeError("Native teacher output remains unstable after three forwards: " + str(replay))
    if not torch.equal(x, ordinary):
        raise RuntimeError("Manual teacher replay differs from ordinary frozen decode: " + str(replay))
    return {"waveform": x, "stages": rows, "final_features": final_features,
            "pre_tanh_phase": pre_tanh, "projection": layers[-2], "sr_bucket": int(indices[0]),
            "replay": replay}


def student_stage_trace(model, z, steady_seconds=2.):
    """Trace each primitive in the existing ConvNeXt blocks without hooks."""
    rows = []
    def record(name, operation, value, rate=100):
        info, phases = phase_statistics(value, rate, steady_seconds)
        rows.append({"name": name, "operation": operation, **info})
        return phases.T
    record("encoder_latents", "raw_encoder_mu", z, 25)
    x = model.adapter(z)
    record("adapter_before_phase_shuffle", type(model.adapter).__name__, x, 25)
    x = x.reshape(x.shape[0], 64, 4, x.shape[-1]).permute(0, 1, 3, 2).reshape(x.shape[0], 64, -1)
    record("adapter_phase_frames", "chronological_phase_shuffle", x)
    x = model.stem(x); record("stem", "causal_convolution", x)
    x = model.stem_norm(x); record("stem_norm", type(model.stem_norm).__name__, x)
    for i, block in enumerate(model.blocks):
        prefix = "blocks." + str(i)
        residual = x
        x = block.depthwise(x); record(prefix + ".depthwise", "causal_depthwise_convolution", x)
        value = block.norm(x.transpose(1, 2))
        record(prefix + ".norm", "channel_LayerNorm", value.transpose(1, 2))
        value = block.expand(value); record(prefix + ".expand", "linear_expand", value.transpose(1, 2))
        value = F.gelu(value, approximate="none"); record(prefix + ".gelu", "GELU_exact", value.transpose(1, 2))
        value = block.project(value); record(prefix + ".project", "linear_project", value.transpose(1, 2))
        value = value * block.scale; record(prefix + ".scale", "learned_channel_scale", value.transpose(1, 2))
        x = residual + value.transpose(1, 2); record(prefix + ".add", "residual_add", x)
    x = model.affine(x); record("affine", type(model.affine).__name__, x)
    x = model.head(x); record("head_before_prelu", "causal_convolution", x)
    x = model.activation(x); h = record("head_after_prelu", "PReLU", x)
    projected = model.output(x); phases = record("output_480_channels", "bias_free_linear_projection", projected)
    waveform = model._waveform(projected)
    record("waveform", "chronological480_sample_shuffle", waveform, 48000)
    old_waveform, _, old_h, old_phases = _trace(model, z, steady_seconds)
    if not (torch.equal(waveform, old_waveform) and torch.equal(h, old_h) and torch.equal(phases, old_phases)):
        raise RuntimeError("Primitive student replay differs from the original verified stage trace")
    return waveform, rows, h, phases


def _wave_from_phase(phase):
    return phase.reshape(1, 1, -1).repeat(1, 1, 2)


def _wave_summary(phase):
    return _wave_components(_wave_from_phase(phase), 0)


def _fp32_reconstruction_check(predicted, reference, absolute_sum, terms):
    predicted, reference, absolute_sum = (x.detach().cpu().double() for x in (predicted, reference, absolute_sum))
    error = predicted - reference
    eps = torch.finfo(torch.float32).eps
    gamma = (2 * terms + 1) * eps / (1 - (2 * terms + 1) * eps)
    bound = gamma * absolute_sum + torch.finfo(torch.float32).tiny
    passed = bool((error.abs() <= bound).all())
    report = {"error_rms": _rms(error), "error_max_abs": float(error.abs().max()),
        "reference_rms": _rms(reference), "relative_rms_error": float(error.norm() / reference.norm()) if bool(reference.norm()) else None,
        "conservative_fp32_forward_error_bound_max": float(bound.max()), "passed": passed,
        "policy": "Report actual FP64 algebra versus FP32 reference error; bound uses gamma_(2*dot_terms+1) times absolute product sum, not bitwise equality."}
    if not passed:
        raise RuntimeError("Projection decomposition exceeds its FP32 rounding bound: " + str(report))
    return report


def teacher_projection_decomposition(features, conv, observed_pre_tanh):
    """Actual multichannel7-tap convolution, with DC, phase and tap cancellation."""
    h = features.detach().cpu().double()
    w = conv.weight.detach().cpu().double()[0]
    bias = conv.bias.detach().cpu().double()[0] if conv.bias is not None else torch.tensor(0., dtype=torch.float64)
    if h.shape[1] != 1920 or w.shape != (h.shape[0], 7) or conv.groups != 1 or conv.dilation != (1,):
        raise ValueError("Expected the native multichannel seven-tap final convolution")
    dc_features = h.mean(-1, keepdim=True)
    non_dc = h - dc_features
    dc = (w * dc_features).sum() + bias
    taps, absolute_products = [], torch.full((1920,), float(bias.abs()), dtype=torch.float64)
    for k in range(7):
        shifted = torch.roll(h, 6 - k, -1)
        shifted_non_dc = torch.roll(non_dc, 6 - k, -1)
        taps.append((w[:, k, None] * shifted_non_dc).sum(0))
        absolute_products += (w[:, k, None].abs() * shifted.abs()).sum(0)
    taps = torch.stack(taps)
    periodic = taps.sum(0)
    reconstructed = dc + periodic
    check = _fp32_reconstruction_check(reconstructed, observed_pre_tanh[0], absolute_products, w.numel())
    tap_power = taps.square().mean(-1)
    output_power = float(periodic.square().mean())
    sum_power = float(tap_power.sum())
    return {"input_channels": h.shape[0], "temporal_taps": 7, "bias": float(bias),
        "constant_feature_component_output_value": float(dc),
        "constant_feature_component_output_time_variation": 0.,
        "constant_component_waveform": _wave_summary(dc.expand(1920)),
        "non_dc_feature_component_waveform": _wave_summary(periodic),
        "full_pre_tanh_waveform": _wave_summary(reconstructed),
        "tap_non_dc_power": tap_power.tolist(), "sum_separate_tap_non_dc_power": sum_power,
        "coherent_sum_non_dc_power": output_power,
        "signed_cross_tap_power": output_power - sum_power,
        "cross_tap_cancellation_fraction": 1 - output_power / sum_power if sum_power else None,
        "fp32_reference_reconstruction": check,
        "causal_scope": "Algebraic decomposition of the existing32-channel learned7-tap projection. A constant channel vector produces a constant waveform away from boundaries. Tap cross-terms describe cancellation or reinforcement, not a tested layer ablation and not equivalence to a single-channel output filter."}


def student_projection_decomposition(h, conv, observed_phases, teacher_phase):
    h, w = h.detach().cpu().double(), conv.weight.detach().cpu().double().squeeze(-1).T
    if h.shape[0] != 4 or w.shape != (h.shape[1], 480) or conv.bias is not None:
        raise ValueError("Expected student four-phase bias-free480-channel readout")
    phase_mean = h.mean(0, keepdim=True)
    shared = (phase_mean @ w).expand(4, 480)
    phase_difference = (h - phase_mean) @ w
    reconstructed = shared + phase_difference
    absolute_products = h.abs() @ w.abs()
    check = _fp32_reconstruction_check(reconstructed, observed_phases, absolute_products, h.shape[1])
    teacher_phase = teacher_phase.detach().cpu().double().reshape(4, 480)
    teacher_shared = teacher_phase.mean(0, keepdim=True).expand(4, 480)
    teacher_additional = teacher_phase - teacher_shared
    total_residual = reconstructed - teacher_phase
    shared_residual = shared - teacher_shared
    additional_residual = phase_difference - teacher_additional
    return {"feature_phase_mean_norm": float(phase_mean.norm()),
        "feature_phase_difference_norm": float((h - phase_mean).norm()),
        "constant_across_frames_feature_component_waveform": _wave_summary(shared),
        "four_phase_feature_variation_component_waveform": _wave_summary(phase_difference),
        "shared_480_teacher_residual_rms": _rms(shared_residual),
        "additional_1920_teacher_residual_rms": _rms(additional_residual),
        "shared_teacher_residual_power_fraction": float(shared_residual.square().mean() / total_residual.square().mean()) if bool(total_residual.norm()) else None,
        "additional_teacher_residual_power_fraction": float(additional_residual.square().mean() / total_residual.square().mean()) if bool(total_residual.norm()) else None,
        "residual_partition_rms_error": _rms(shared_residual + additional_residual - total_residual),
        "fp32_reference_reconstruction": check,
        "causal_scope": "The phase-MEAN feature vector alone maps to480 different waveform positions every10ms. Additional four-phase feature differences create40ms structure. This decomposition does not make adapter phase differences the cause of the shared480-periodic residual."}


@torch.inference_mode(False)
@torch.no_grad()
def probe_paired_layers(teacher, student, encoded_zero_crop, *, expected_teacher_state_sha256, steady_seconds=2.):
    provenance = teacher.provenance
    if provenance["source_sha256"] != SOURCE_SHA256 or provenance["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("The paired probe requires the pinned official teacher")
    if encoded_zero_crop.source_id != "encoded_zero" or encoded_zero_crop.context_start_frame != 0:
        raise ValueError("Start with the complete canonical encoded-zero fixture only")
    z = encoded_zero_crop.latents.detach().to(teacher.device).clone()
    target = encoded_zero_crop.teacher_audio.detach().to(teacher.device).clone()
    if z.shape != (1, 64, 150) or target.shape != (1, 1, 288000):
        raise ValueError("Expected the full six-second25Hz/48kHz fixture")
    if next(student.parameters()).device != teacher.device:
        raise ValueError("Both models must share the explicit diagnostic device")
    before = state_fingerprint(teacher.model.state_dict())
    if before != expected_teacher_state_sha256:
        raise ValueError("Frozen teacher state differs from the training teacher")
    with _eval_preserved(teacher.model), _eval_preserved(student), torch.autocast(device_type=teacher.device.type, enabled=False):
        stable_ordinary, startup = native_startup_check(teacher, z, target)
        try:
            teacher_trace = teacher_stage_trace(teacher, z, steady_seconds, stable_ordinary=stable_ordinary)
        except RuntimeError as exc:
            raise RuntimeError(str(exc) + "; preceding native startup evidence=" + str(startup)) from exc
        startup.update(teacher_trace["replay"])
        if not torch.equal(teacher_trace["waveform"], target):
            raise RuntimeError("Ordinary teacher decode does not bitwise match canonical target: max="
                               + str(float((teacher_trace["waveform"] - target).abs().max())))
        student_audio, student_nodes, h, projected = student_stage_trace(student, z, steady_seconds)
        start = math.ceil(steady_seconds * 25) * 1920
        _, teacher_phases = phase_statistics(teacher_trace["waveform"], 48000, steady_seconds)
        teacher_projection = teacher_projection_decomposition(teacher_trace["final_features"],
            teacher_trace["projection"], teacher_trace["pre_tanh_phase"])
        student_projection = student_projection_decomposition(h, student.output, projected, teacher_phases)
        pre = teacher_trace["pre_tanh_phase"]
        tanh_error = torch.tanh(pre) - teacher_phases
        tanh_change = teacher_phases - pre
        result = {"format_version": 1, "fixture": "canonical_encoded_zero_6s", "steady_start_sample": start,
            "teacher_source_sha256": SOURCE_SHA256, "teacher_checkpoint_sha256": CHECKPOINT_SHA256,
            "teacher_model_state_sha256": before, "teacher_manual_replay_equals_ordinary_decode": True,
            "teacher_execution_startup": startup,
            "teacher_ordinary_decode_equals_canonical_target": True,
            "student_manual_replay_equals_ordinary_decode": True,
            "shared_latents": {"shape": list(z.shape), "posterior": "raw_mu", "resampling_or_encoder_calls": 0},
            "teacher_sample_rate_bucket": teacher_trace["sr_bucket"], "teacher_stages": teacher_trace["stages"],
            "student_stages": student_nodes,
            "teacher_final_projection": teacher_projection, "student_final_projection": student_projection,
            "teacher_tanh": {"pre_tanh_max_abs": float(pre.abs().max()),
                "post_minus_pre_rms": _rms(tanh_change), "post_minus_pre_max_abs": float(tanh_change.abs().max()),
                "fp64_tanh_of_phase_mean_vs_actual_phase_mean_rms": _rms(tanh_error),
                "scope": "Stationary phase templates only; interpret after confirming zero cycle variation. Tanh may be effectively identity on these low-amplitude values."},
            "waveforms": {"teacher": _wave_components(target, start), "student": _wave_components(student_audio, start),
                          "residual": _wave_components(student_audio - target, start)},
            "limits": ["Hidden feature amplitudes/channels are not cross-architecture reconstruction errors.",
                       "Projection cancellation is descriptive linear algebra, not a trained replacement or causally isolated layer ablation.",
                       "Digital silence alone cannot establish speech/quiet natural-audio quality or an architecture-wide optimum."],
            "updates": 0, "waveform_or_checkpoint_files_written": 0}
    if state_fingerprint(teacher.model.state_dict()) != before:
        raise RuntimeError("Frozen teacher state changed")
    result["teacher_and_student_state_preserved"] = True
    return result


def self_test():
    # Physical-rate decomposition: constant,10ms shared,40ms extra are separable.
    p = torch.tensor([0., 1., 0., 1., 0., 1., 0., 1.]).reshape(1, 1, 8).repeat(1, 1, 3)
    stats, _ = phase_statistics(p, 200, 0)
    assert stats["component_power"]["shared_10ms_phase_pattern"] > 0
    assert stats["component_power"]["additional_40ms_phase_pattern"] == 0
    assert stats["component_power"]["cycle_varying"] == 0
    conv = torch.nn.Conv1d(2, 1, 7, bias=True)
    with torch.no_grad(): conv.weight.fill_(.01); conv.bias.fill_(.002)
    phases = torch.stack((torch.sin(torch.arange(1920).double()), torch.ones(1920)))
    # Independent direct circular convolution reference, including bias.
    reference = conv.bias.detach().double().expand(1920).clone()
    for k in range(7): reference += (conv.weight.detach().double()[0, :, k, None] * torch.roll(phases, 6-k, -1)).sum(0)
    report = teacher_projection_decomposition(phases, conv, reference[None])
    assert report["fp32_reference_reconstruction"]["error_max_abs"] < 1e-12
    assert report["constant_feature_component_output_time_variation"] == 0
    head = torch.nn.Conv1d(1, 480, 1, bias=False)
    with torch.no_grad(): head.weight.copy_(torch.arange(480).reshape(480,1,1)/480)
    h = torch.ones(4, 1); observed = h @ head.weight.detach().squeeze(-1).T
    student = student_projection_decomposition(h, head, observed, torch.zeros(1,1920))
    assert student["constant_across_frames_feature_component_waveform"]["component_power"]["shared_480_pattern"] > 0
    assert student["four_phase_feature_variation_component_waveform"]["power"] == 0
    print("PASS: CPU physical-rate phase separation, native multichannel7-tap decomposition, constant teacher output component, phase-invariant student480-pattern mechanism")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true", required=True)
    p.parse_args(); self_test()
