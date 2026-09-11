"""CPU contracts for auxiliary readouts that never enter the decoder path."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "convnext"))
import projected_hints_v1 as hints
import compare_projected_hints_v1 as experiment
from test_group_model import TinyDecoder, assert_snapshot, gm, selections, snapshot


@pytest.fixture(autouse=True)
def cpu_state():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(381)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


@pytest.mark.parametrize("cell,channels", [(8, 256), (4, 128)])
def test_real_hint_dimensions_and_partial_context_tail_weights(cell, channels):
    valid = torch.zeros(1, 1, 32, dtype=torch.bool)
    valid[..., 9:29] = True
    weights = hints.feature_weights(valid, 32 // cell, cell)
    expected = [0, 7, 8, 5] if cell == 8 else [0, 0, 3, 4, 4, 4, 4, 1]
    assert weights.flatten().tolist() == expected
    assert int(weights.sum()) == 20
    student = torch.randn(1, 128, 32 // cell, requires_grad=True)
    teacher = torch.randn(1, channels, 32 // cell, requires_grad=True)
    projector = hints.LinearHint(torch.randn(channels, 128) * .02,
                                 torch.zeros(channels), trainable=True)
    loss = hints.weighted_hint_loss(projector, student, teacher, valid,
                                   total_valid_samples=20, samples_per_cell=cell)
    # Independent waveform-rate oracle counts each feature cell once per valid sample.
    expanded = (projector(student) - teacher.detach()).square().repeat_interleave(cell, -1)
    expected_loss = expanded.masked_select(valid.expand_as(expanded)).mean()
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    assert teacher.grad is None
    assert student.grad is not None and projector.weight.grad is not None
    assert projector.bias.grad is not None
    assert torch.count_nonzero(student.grad[..., weights.flatten() == 0]) == 0


@pytest.mark.parametrize("cell", [4, 8])
def test_pooled_hint_loss_and_gradient_equal_singletons_without_extra_division(cell):
    s = torch.randn(2, 3, 8, requires_grad=True)
    t = torch.randn(2, 5, 8, requires_grad=True)
    valid = torch.zeros(2, 1, 8 * cell, dtype=torch.bool)
    valid[0, :, cell + 1:6 * cell - 1] = True
    valid[1, :, 2 * cell:8 * cell - 3] = True
    total = int(valid.sum())
    p = hints.LinearHint(torch.randn(5, 3), torch.randn(5), trainable=True)
    joint = hints.weighted_hint_loss(p, s, t, valid, total_valid_samples=total,
                                    samples_per_cell=cell)
    separate = sum(hints.weighted_hint_loss(p, s[i:i + 1], t[i:i + 1], valid[i:i + 1],
                   total_valid_samples=total, samples_per_cell=cell) for i in range(2))
    variables = (s, p.weight, p.bias)
    a = torch.autograd.grad(joint, variables, retain_graph=True)
    b = torch.autograd.grad(separate, variables)
    torch.testing.assert_close(joint, separate)
    for actual, expected in zip(a, b):
        torch.testing.assert_close(actual, expected)
    assert t.grad is None


def test_time_misalignment_and_non_boolean_masks_are_rejected():
    with pytest.raises(ValueError):
        hints.feature_weights(torch.ones(1, 1, 32, dtype=torch.bool), 7, 4)
    with pytest.raises(ValueError):
        hints.feature_weights(torch.ones(1, 1, 32), 8, 4)


def fixture(monkeypatch, count=2):
    teacher_decoder = TinyDecoder().eval().requires_grad_(False)
    holder = nn.Module()
    holder.decoder = teacher_decoder
    teacher = nn.Module()
    teacher.model = holder
    student = gm.build_student(teacher_decoder, *selections())
    real_batch = hints.base.batch
    monkeypatch.setattr(hints.base, "batch", lambda rows: real_batch(rows, device="cpu"))
    monkeypatch.setattr(hints.base, "teacher_forward",
                        lambda teacher, z: gm.teacher_trace(teacher.model.decoder, z[:, :8]))
    crops = []
    for i in range(count):
        frames, context = 2 + i % 2, i % 2
        z = torch.randn(1, 64, frames) * .1
        with torch.no_grad():
            target = teacher_decoder(z[:, :8])
        crops.append({"source_id": f"fit-{i}", "latents": z, "teacher_audio": target,
                      "context_frames": context, "context_start_frame": 0,
                      "start_frame": context,
                      "valid_scored_samples": (frames - context) * 1920 - 13 - i})
    spectral = hints.base.ReconstructionV2(hints.base.ReconstructionV2Config(
        fft_sizes=(32, 64), mel_bands=(4, 8)))
    projectors = {
        "stage3_up": hints.LinearHint(torch.randn(16, 8) * .1, torch.zeros(16), trainable=True),
        "stage4_up": hints.LinearHint(torch.eye(8), torch.zeros(8), trainable=True),
    }
    return student, teacher, crops, spectral, projectors


def test_captures_are_before_residuals_and_readouts_do_not_change_inference(monkeypatch):
    model, teacher, crops, _, projectors = fixture(monkeypatch)
    z = crops[0]["latents"][:, :8]
    before = snapshot(model)
    with torch.no_grad():
        waveform_before = model(z)
        with hints.capture_hints(model.decoder) as captured:
            trace = model.forward_from_latents(z)
        assert captured["stage3_up"].shape == (1, 8, 2 * 240)
        assert captured["stage4_up"].shape == (1, 8, 2 * 480)
        assert not torch.equal(captured["stage3_up"], trace["stage3_output"])
        assert not torch.equal(captured["stage4_up"], trace["stage4_output"])
        _ = {key: projectors[key](value) for key, value in captured.items()}
        waveform_after = model(z)
    torch.testing.assert_close(waveform_before, waveform_after, rtol=0, atol=0)
    assert_snapshot(model, before)
    assert not any("projector" in key or "hint" in key for key in model.state_dict())
    with pytest.raises(RuntimeError, match="sentinel"):
        with hints.capture_hints(model.decoder):
            raise RuntimeError("sentinel")
    assert all(not model.decoder.get_submodule(path)._forward_hooks.values()
               for path in hints.HINT_PATHS.values())


def test_teacher_detached_student_and_readout_gradients_and_frozen_suffix(monkeypatch):
    model, teacher, crops, spectral, projectors = fixture(monkeypatch)
    model.train()
    before_teacher, before_student = snapshot(teacher.model), snapshot(model)
    coefficients = {"waveform": 1., "mel": .0007, "feature": .009}
    denominators = hints.base.reconstruction_denominators(crops, spectral)
    existing, auxiliary = hints.forward_branches(model, teacher, crops[0], spectral,
                                                coefficients, projectors, denominators)
    parameters = list(model.trainable_group_parameters())
    # Frozen suffix operations must preserve a path from waveform error to the group.
    wave_grads = torch.autograd.grad(existing["waveform"], parameters, retain_graph=True)
    assert sum(float(g.abs().sum()) for g in wave_grads) > 0
    total = sum(coefficients[k] * v for k, v in existing.items()) + sum(auxiliary.values()) * .01
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    assert all(p.grad is None for p in teacher.model.parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for projector in projectors.values() for p in projector.parameters())
    assert_snapshot(teacher.model, before_teacher)
    assert_snapshot(model, before_student)
    names = hints.upstream_parameter_names(model, "stage3_up")
    assert any(n.startswith("model.3.") for n in names)
    assert any(n.startswith("model.4.block.1.") for n in names)
    assert not any(n.startswith(("model.4.block.2.", "model.5.")) for n in names)


def test_ridge_initialization_preserves_student_teacher_rng_grads_and_inference(monkeypatch):
    model, teacher, crops, _, _ = fixture(monkeypatch)
    model.train()
    for p in model.trainable_group_parameters():
        p.grad = torch.ones_like(p) * .17
    old_grads = [p.grad.clone() for p in model.trainable_group_parameters()]
    before_model, before_teacher = snapshot(model), snapshot(teacher.model)
    modes = {n: module.training for n, module in model.named_modules()}
    rng = torch.get_rng_state()
    readouts, report = hints.fit_projectors(model, teacher, crops, trainable=True)
    assert report["source_ids"] == [c["source_id"] for c in crops]
    assert report["trainable_after_fit"] is True
    assert all(p.requires_grad for projector in readouts.values() for p in projector.parameters())
    assert readouts["stage3_up"].weight.shape == (16, 8)
    assert readouts["stage4_up"].weight.shape == (8, 8)
    assert_snapshot(model, before_model)
    assert_snapshot(teacher.model, before_teacher)
    assert {n: module.training for n, module in model.named_modules()} == modes
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for old, p in zip(old_grads, model.trainable_group_parameters()):
        torch.testing.assert_close(p.grad, old, rtol=0, atol=0)
    frozen = copy.deepcopy(readouts)
    with torch.no_grad():
        readouts["stage3_up"].weight.add_(.2)
    assert not torch.equal(frozen["stage3_up"].weight, readouts["stage3_up"].weight)
    with pytest.raises(ValueError, match="unique"):
        hints.fit_projectors(model, teacher, crops + crops[:1])


@pytest.mark.parametrize("record_components", [False, True])
def test_zero_hint_control_has_identical_student_update_and_active_arm_has_separate_projector_optimizer(monkeypatch, record_components):
    model, teacher, crops, spectral, projectors = fixture(monkeypatch)
    original = copy.deepcopy(model)
    coefficients = {"waveform": 1., "mel": .0007, "feature": .009}
    options = dict(lr=3e-5, betas=(.9, .99), eps=1e-8, weight_decay=0.)
    original_opt = torch.optim.AdamW(original.trainable_group_parameters(), **options)
    control_opt = torch.optim.AdamW(model.trainable_group_parameters(), **options)
    expected = hints.base.training_update(original, teacher, crops, spectral, coefficients, original_opt)
    actual = hints.training_update(model, teacher, crops, spectral, coefficients, control_opt,
                                   projectors, dict.fromkeys(hints.HINT_PATHS, 0.),
                                   record_components=record_components)
    for k in ("total", "waveform", "mel", "feature"):
        assert actual[k] == expected[k]
    assert_snapshot(model, snapshot(original))
    assert experiment.replay.compare_tree(control_opt.state_dict(), original_opt.state_dict())["equal"]
    assert all(p.grad is None for projector in projectors.values() for p in projector.parameters())
    decoder_keys = set(model.state_dict())
    old = snapshot(model)
    projected_old = {key: snapshot(p) for key, p in projectors.items()}
    projector_opt = torch.optim.AdamW([p for q in projectors.values() for p in q.parameters()], **options)
    values = hints.training_update(model, teacher, crops, spectral, coefficients, control_opt,
                 projectors, dict.fromkeys(hints.HINT_PATHS, .01), projector_optimizer=projector_opt)
    assert values["total"] > 0
    assert any(not torch.equal(model.state_dict()[k], v) for k, v in old.items())
    assert all(any(not torch.equal(p.state_dict()[k], v) for k, v in projected_old[key].items())
               for key, p in projectors.items())
    assert set(model.state_dict()) == decoder_keys
    assert all(float(s["step"]) == 2 for s in control_opt.state.values())
    assert all(float(s["step"]) == 1 for s in projector_opt.state.values())


def test_calibration_uses_pooled_connected_student_gradients_and_restores_state(monkeypatch):
    model, teacher, crops, spectral, projectors = fixture(monkeypatch)
    coefficients = {"waveform": 1., "mel": .0007, "feature": .009}
    denominator = hints.base.reconstruction_denominators(crops, spectral)
    terms = {"existing": 0., "stage3_up": 0., "stage4_up": 0.}
    for crop in crops:
        branches, auxiliary = hints.forward_branches(model, teacher, crop, spectral,
                                                      coefficients, projectors, denominator)
        terms["existing"] += sum(coefficients[k] * v for k, v in branches.items())
        for k, v in auxiliary.items():
            terms[k] += v
    named = model.group_named_parameters()
    params = [p for _, p in named]
    gradients = {k: torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
                 for k, v in terms.items()}
    expected = {}
    for key in hints.HINT_PATHS:
        connected = set(hints.upstream_parameter_names(model, key))
        energy = {}
        for branch in ("existing", key):
            energy[branch] = sum(float(g.double().square().sum())
                                 for (name, _), g in zip(named, gradients[branch])
                                 if name in connected and g is not None)
        expected[key] = .05 * (energy["existing"] / energy[key]) ** .5
    for p in params:
        p.grad = torch.ones_like(p) * .29
    saved_grads = [p.grad.clone() for p in params]
    saved_model, saved_teacher = snapshot(model), snapshot(teacher)
    saved_maps = {k: snapshot(p) for k, p in projectors.items()}
    rng = experiment.screen.rng_state()
    calculated, report = hints.calibrate_hint_coefficients(model, teacher, crops, spectral,
                                                           coefficients, projectors)
    for key in hints.HINT_PATHS:
        assert calculated[key] == pytest.approx(expected[key], rel=3e-6)
    assert report["source_ids"] == [c["source_id"] for c in crops]
    assert report["projector_gradients_excluded"] is True
    assert_snapshot(model, saved_model)
    assert_snapshot(teacher, saved_teacher)
    for key, p in projectors.items():
        assert_snapshot(p, saved_maps[key])
        assert all(v.grad is None for v in p.parameters())
    for p, old in zip(params, saved_grads):
        torch.testing.assert_close(p.grad, old, rtol=0, atol=0)
    assert experiment.replay.compare_tree(experiment.screen.rng_state(), rng)["equal"]


def test_restart_keeps_saved4625_student_moments_rng_and_independent_fresh_readouts(monkeypatch):
    model, teacher, crops, spectral, projectors = fixture(monkeypatch)
    options = dict(lr=3e-5, betas=(.9, .99), eps=1e-8, weight_decay=0.)
    opt = torch.optim.AdamW(model.trainable_group_parameters(), **options)
    hints.base.training_update(model, teacher, crops, spectral,
                               {"waveform": 1., "mel": .0007, "feature": .009}, opt)
    for state in opt.state.values():
        state["step"].fill_(4625)
    candidate = {"format": experiment.previous.VERSION, "optimizer_step": 4625,
                 "accumulation": 12, "group": {k: v.clone() for k, v in model.group_state_dict().items()},
                 "optimizer": copy.deepcopy(opt.state_dict()), "rng": experiment.screen.rng_state(),
                 "original_training_identity": {"learning_rate": 3e-5}}
    original_candidate = copy.deepcopy(candidate)
    first, checks = experiment.restart_candidate(model, candidate)
    first_random = torch.rand(9)
    with torch.no_grad():
        next(iter(model.trainable_group_parameters())).add_(.15)
    second, second_checks = experiment.restart_candidate(model, candidate)
    second_random = torch.rand(9)
    assert all(row["equal"] for row in checks.values())
    assert all(row["equal"] for row in second_checks.values())
    torch.testing.assert_close(first_random, second_random, rtol=0, atol=0)
    assert experiment.replay.compare_tree(candidate, original_candidate)["equal"]
    assert experiment.replay.compare_tree(first.state_dict(), second.state_dict())["equal"]
    copied = experiment.copy_projectors(projectors, trainable=True)
    popt = experiment.projector_optimizer(copied)
    assert not popt.state
    assert all(popt.param_groups[0][key] == value for key, value in options.items())
    assert not ({id(p) for p in model.parameters()} & {id(p) for q in copied.values() for p in q.parameters()})
    for key in projectors:
        assert copied[key].weight.data_ptr() != projectors[key].weight.data_ptr()
        torch.testing.assert_close(copied[key].weight, projectors[key].weight, rtol=0, atol=0)
    assert experiment.SOURCES == 1500 and experiment.ACCUMULATION == 12
    assert experiment.FRESH_START == 10500 and experiment.START + experiment.SOURCES // experiment.ACCUMULATION == 4750
    assert list(range(0, experiment.SOURCES, experiment.ACCUMULATION))[-1] == 1488
