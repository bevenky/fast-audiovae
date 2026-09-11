"""Four externally selected natural cases: untouched forwards and local VJPs.

No hidden-feature matching, stationary substitution, circular phase model,
optimizer step, gain change, model update or saved audio/checkpoint.
"""
from contextlib import contextmanager

import torch

from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.teacher import SOURCE_SHA256, CHECKPOINT_SHA256
from diagnose_architecture import _eval_preserved
from paired_layers import _difference


def masked_feature_stats(value, rate, valid, layout="BCT"):
    if layout == "BTC":
        value = value.transpose(1, 2)
    if value.ndim != 3 or value.shape[0] != 1 or 48000 % rate:
        raise ValueError("Unsupported primitive feature shape/rate")
    hop = 48000 // rate
    if value.shape[-1] * hop != valid.shape[-1]:
        raise ValueError("Feature time grid does not cover the same waveform context")
    weights = valid.reshape(1, 1, value.shape[-1], hop).sum(-1).double()
    selected = weights[0, 0] > 0
    x = value.detach().double()
    denominator = weights.sum() * value.shape[1]
    return {"shape": list(value.shape), "feature_rate_hz": rate,
        "finite_entire_tensor": bool(torch.isfinite(x).all()),
        "scored_sample_coverage": int(weights.sum()), "scored_feature_cells": int(selected.sum()),
        "scored_rms": float(((x.square() * weights).sum() / denominator).sqrt()) if denominator else None,
        "scored_max_abs": float(x[0, :, selected].abs().max()) if bool(selected.any()) else None,
        "mask_interpretation": "Each feature cell covers its physical waveform-time interval. RMS weights partial cells by valid sample coverage; hidden feature values are descriptive and are not waveform-quality errors."}


def teacher_specs(teacher):
    decoder = teacher.model.decoder
    specs = []
    def add(module, name, rate):
        specs.append((module, name, rate, "BCT", False))
    def visit(module, name, rate):
        kind = type(module).__name__
        if kind in ("CausalDecoderBlock", "CausalResidualUnit"):
            for i, child in enumerate(module.block):
                rate = visit(child, name + ".block." + str(i), rate)
            add(module, name + (".add" if kind == "CausalResidualUnit" else ".output"), rate)
        else:
            if kind == "NoiseBlock":
                raise ValueError("Stochastic teacher unsupported")
            if isinstance(module, torch.nn.ConvTranspose1d):
                rate *= module.stride[0]
            add(module, name, rate)
        return rate
    rate = 25
    for i, (module, condition) in enumerate(zip(decoder.model, decoder.sr_cond_model, strict=True)):
        if condition is not None:
            add(condition, "decoder.sr_cond_model." + str(i), rate)
        rate = visit(module, "decoder.model." + str(i), rate)
    if rate != 48000:
        raise ValueError("Unexpected teacher output rate")
    return specs


def student_specs(model):
    specs = [(model.adapter, "adapter", 25, "BCT", False),
             (model.stem, "stem", 100, "BCT", False),
             (model.stem_norm, "stem_norm", 100, "BCT", False)]
    for i, block in enumerate(model.blocks):
        prefix = "blocks." + str(i)
        specs += [(block.depthwise, prefix + ".depthwise", 100, "BCT", False),
                  (block.norm, prefix + ".norm", 100, "BTC", False),
                  (block.expand, prefix + ".expand", 100, "BTC", False),
                  (block.project, prefix + ".gelu", 100, "BTC", True),
                  (block.project, prefix + ".project", 100, "BTC", False),
                  (block, prefix + ".add", 100, "BCT", False)]
    specs += [(model.affine, "affine", 100, "BCT", False),
              (model.head, "head", 100, "BCT", False),
              (model.activation, "head_prelu", 100, "BCT", False),
              (model.output, "output_480_channels", 100, "BCT", False)]
    return specs


@contextmanager
def capture_forward(model, specs, valid, *, blocks=(), retain_modules=()):
    """Hooks observe tensors only; every handle is removed even after failure."""
    handles, rows, inputs, outputs, retained = [], [], {}, {}, {}
    watched = {id(module): name for name, module in retain_modules}
    def observe(name, rate, layout, value):
        info = masked_feature_stats(value, rate, valid, layout)
        if not info["finite_entire_tensor"]:
            raise RuntimeError("Nonfinite feature at " + name)
        rows.append({"name": name, **info})
    try:
        for module, name, rate, layout, pre in specs:
            if pre:
                def pre_hook(module, args, name=name, rate=rate, layout=layout):
                    observe(name, rate, layout, args[0])
                handles.append(module.register_forward_pre_hook(pre_hook))
            else:
                def output_hook(module, args, value, name=name, rate=rate, layout=layout):
                    observe(name, rate, layout, value)
                    if id(module) in watched:
                        retained[watched[id(module)]] = value
                handles.append(module.register_forward_hook(output_hook))
        for index, block in enumerate(blocks):
            def block_pre(module, args, index=index):
                inputs[index] = args[0]
            def block_post(module, args, value, index=index):
                outputs[index] = value
            handles.append(block.register_forward_pre_hook(block_pre))
            handles.append(block.register_forward_hook(block_post))
        yield {"rows": rows, "inputs": inputs, "outputs": outputs, "retained": retained}
    finally:
        for handle in reversed(handles):
            handle.remove()


def gain_sensitivities(prediction, target, valid, quiet, block_inputs, block_outputs):
    """All gains stay at1. Derivatives do not establish finite-step improvement."""
    if len(block_inputs) != len(block_outputs) or not block_outputs:
        raise ValueError("Missing actual block activations")
    fixed_overshoot = (prediction.detach().abs() > 1) & valid
    active = valid & ~quiet & ~fixed_overshoot
    definitions = (("teacher_quiet_residual_mse", quiet, "mse"),
                   ("full_scale_excess_mse", valid, "excess"),
                   ("active_teacher_residual_mse", active, "mse"))
    outputs = tuple(block_outputs[i] for i in sorted(block_outputs))
    report = {}
    for name, mask, kind in definitions:
        count = int(mask.sum())
        if not count:
            report[name] = {"defined": False, "samples": 0, "reason": "No selected samples"}
            continue
        values = prediction[mask].double()
        loss = ((values - target[mask].double()).square().mean() if kind == "mse"
                else (values.abs() - 1).clamp_min(0).square().mean())
        gradients = torch.autograd.grad(loss, outputs, retain_graph=True, create_graph=False)
        rows = []
        for i, gradient in zip(sorted(block_outputs), gradients, strict=True):
            branch = block_outputs[i].detach().double() - block_inputs[i].detach().double()
            dot = float((gradient.detach().double() * branch).sum())
            rows.append({"block": i, "d_metric_d_residual_gain_at_1": dot,
                         "locally_lower_gain_decreases_metric": dot > 0,
                         "observed_branch_rms_over_full_context": float(branch.square().mean().sqrt())})
        report[name] = {"defined": True, "samples": count, "value": float(loss.detach()), "blocks": rows,
            "fixed_baseline_overshoot_samples": int(fixed_overshoot.sum()),
            "scope": "VJP at the untouched forward. A positive derivative predicts improvement only for an infinitesimal reduction of that block residual gain; no gain was changed. The branch is observed output-minus-input, including FP32 addition rounding, rather than a re-executed exact branch tensor."}
    return report


def scored_wave_stats(value, valid):
    selected = value.detach()[valid].double()
    excess = (selected.abs() - 1).clamp_min(0)
    return {"samples": selected.numel(), "rms": float(selected.square().mean().sqrt()),
            "max_abs": float(selected.abs().max()), "overshoot_samples": int((excess > 0).sum()),
            "full_scale_excess_mse": float(excess.square().mean())}


@torch.inference_mode(False)
def probe_natural_layers(teacher, student, cases, *, quiet_config, expected_teacher_state_sha256):
    from native_chain import _masks
    if len(cases) != 4 or len({c.source_id for c, _ in cases}) != 4:
        raise ValueError("Exactly four externally selected source-distinct natural cases")
    if teacher.provenance["source_sha256"] != SOURCE_SHA256 or teacher.provenance["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Unexpected teacher provenance")
    if state_fingerprint(teacher.model.state_dict()) != expected_teacher_state_sha256:
        raise ValueError("Unexpected frozen teacher state")
    if len(student.blocks) != 10 or next(student.parameters()).device != teacher.device:
        raise ValueError("Expected ten-block student on the teacher device")
    if getattr(student, "output_filter", None) is not None or (hasattr(student, "fusion_config") and student.fusion_config.terminal_tanh):
        raise ValueError("Expected unchanged direct student head")
    results = []
    with _eval_preserved(teacher.model), _eval_preserved(student), torch.autocast(device_type=teacher.device.type, enabled=False):
        for crop, role in cases:
            target, valid, quiet = _masks(crop, teacher.device, quiet_config)
            z = crop.latents.detach().to(teacher.device).clone()
            with torch.no_grad():
                ordinary_teacher = teacher.decode(z)
                with capture_forward(teacher.model, teacher_specs(teacher), valid,
                        retain_modules=(("pre_tanh", teacher.model.decoder.model[-2]),)) as trace:
                    teacher_prediction = teacher.decode(z)
                teacher_replay = _difference(teacher_prediction, ordinary_teacher)
                canonical = _difference(teacher_prediction[valid], target[valid])
                if not teacher_replay["bitwise_equal"] or not canonical["bitwise_equal"]:
                    raise RuntimeError("Teacher replay/canonical scored mismatch for " + crop.source_id
                                       + ": " + str({"replay": teacher_replay, "canonical": canonical}))
                pre_tanh = trace["retained"]["pre_tanh"]
                teacher_head = {"pre_tanh": scored_wave_stats(pre_tanh, valid),
                    "post_tanh": scored_wave_stats(teacher_prediction, valid),
                    "tanh_change_scored_rms": float((teacher_prediction[valid].double() - pre_tanh[valid].double()).square().mean().sqrt())}
                teacher_rows = trace["rows"]
                ordinary_student = student(z)
            with torch.enable_grad(), capture_forward(student, student_specs(student), valid,
                    blocks=student.blocks, retain_modules=(("projection", student.output),)) as trace:
                prediction = student(z.clone().requires_grad_(True))
                replay = _difference(prediction, ordinary_student)
                if not replay["bitwise_equal"]:
                    raise RuntimeError("Student hook/gradient-mode replay mismatch: " + str(replay))
                sensitivities = gain_sensitivities(prediction, target, valid, quiet, trace["inputs"], trace["outputs"])
                projection = trace["retained"]["projection"]
                if not torch.equal(student._waveform(projection), prediction):
                    raise RuntimeError("Actual student head does not reproduce output exactly")
                with torch.no_grad():
                    # Actual waveform sample positions, not a stationary/circular model.
                    excess_mask = (prediction.abs() > 1) & valid
                    counts = excess_mask.reshape(-1, 480).sum(0)
                    student_head = {"actual_projection_shuffle_equals_waveform": True,
                        "scored_waveform": scored_wave_stats(prediction, valid),
                        "overshoot_counts_by_output_position_within_480": counts.cpu().tolist(),
                        "scope": "Counts use actual valid waveform samples; they do not infer repeating phase errors or independent events."}
                results.append({"source_id": crop.source_id, "start_frame": crop.start_frame,
                    "context_start_frame": crop.context_start_frame, "role": role,
                    "scored_samples": int(valid.sum()), "teacher_quiet_samples": int(quiet.sum()),
                    "active_control_samples": int((valid & ~quiet & ~(prediction.detach().abs() > 1)).sum()),
                    "fixed_baseline_overshoot_samples": int(((prediction.detach().abs() > 1) & valid).sum()),
                    "teacher_hooked_vs_native": teacher_replay, "teacher_scored_vs_canonical": canonical,
                    "student_hooked_grad_mode_vs_native": replay,
                    "teacher_stages": teacher_rows, "student_stages": trace["rows"],
                    "teacher_head": teacher_head, "student_head": student_head,
                    "local_block_gain_derivatives": sensitivities})
    if state_fingerprint(teacher.model.state_dict()) != expected_teacher_state_sha256:
        raise RuntimeError("Teacher state changed")
    return {"version": 1, "cases": results, "teacher_and_student_state_preserved": True,
        "parameter_or_gain_updates": 0, "optimizer_steps": 0, "waveform_or_checkpoint_files_written": 0,
        "scope": "Four fixed source cases. Primitive hooks observe actual forwards. Local gain derivatives do not identify a historically bad layer, establish a finite edit, validate quality equivalence or compare hidden features between architectures."}
