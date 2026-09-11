"""Real tiny causal-decoder tests for nested recovered-weight migrations."""
import copy

import pytest
import torch
from torch import nn

import group_model as gm
import progressive_model as pm
from test_group_model import TinyDecoder, assert_snapshot, snapshot


SMALL_WIDTHS = ((32, 16), (24, 16), (24, 12), (16, 12), (16, 8))


@pytest.fixture(autouse=True)
def deterministic_cpu():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(3171)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


@pytest.fixture
def teacher():
    return TinyDecoder().eval().requires_grad_(False)


def schedule():
    return pm.make_schedule(list(range(31, 0, -2)), list(range(15, 0, -2)),
                            stage2_order=list(range(31, 0, -2)) + list(range(0, 32, 2)),
                            stage3_order=list(range(15, 0, -2)) + list(range(0, 16, 2)),
                            widths=SMALL_WIDTHS)


def relative(old, new):
    return {key: [old[key].index(channel) for channel in new[key]] for key in pm.KEYS}


def assert_effective_slice(source, result, inputs, outputs):
    w = gm.effective_weight(source).detach()
    if isinstance(source, nn.ConvTranspose1d):
        expected = w[inputs][:, outputs]
    elif source.groups == source.in_channels == source.out_channels:
        expected = w[outputs]
    else:
        expected = w[outputs][:, inputs]
    torch.testing.assert_close(gm.effective_weight(result), expected, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(result.bias, source.bias.detach()[outputs], rtol=0, atol=0)


def assert_all_coupled_weights(source, result, current_ids, next_ids):
    ids = relative(current_ids, next_ids)
    outer_in = list(range(source.decoder.model[3].input_channels))
    outer_out = list(range(source.decoder.model[5].block[1].out_channels))
    for stage, ins, outs in ((3, outer_in, ids[pm.KEYS[0]]),
                             (4, ids[pm.KEYS[0]], ids[pm.KEYS[1]]),
                             (5, ids[pm.KEYS[1]], outer_out)):
        a, b = source.decoder.model[stage], result.decoder.model[stage]
        torch.testing.assert_close(b.block[0].alpha, a.block[0].alpha[:, ins], rtol=0, atol=0)
        assert_effective_slice(a.block[1], b.block[1], ins, outs)
        for ar, br in zip(a.block[2:], b.block[2:]):
            for snake in (0, 2):
                torch.testing.assert_close(br.block[snake].alpha, ar.block[snake].alpha[:, outs], rtol=0, atol=0)
            assert_effective_slice(ar.block[1], br.block[1], outs, outs)
            assert_effective_slice(ar.block[3], br.block[3], outs, outs)
        for name in ("scale_embed", "bias_embed"):
            a_cond, b_cond = source.decoder.sr_cond_model[stage], result.decoder.sr_cond_model[stage]
            torch.testing.assert_close(getattr(b_cond, name).weight,
                                       getattr(a_cond, name).weight[:, ins], rtol=0, atol=0)


def test_production_schedule_keeps_exact_endpoint_and_single_boundary_cuts():
    final2, final3 = list(range(511, 0, -2)), list(range(255, 0, -2))
    steps = pm.make_schedule(final2, final3,
        stage2_order=final2 + list(range(0, 512, 2)),
        stage3_order=final3 + list(range(0, 256, 2)))
    assert [tuple(len(x[k]) for k in pm.KEYS) for x in steps] == list(pm.WIDTH_SCHEDULE)
    assert steps[-1] == dict(zip(pm.KEYS, (final2, final3)))
    copied = pm.validate_schedule(steps, steps[-1])
    copied[1][pm.KEYS[0]][0] = -1
    assert steps[1][pm.KEYS[0]][0] >= 0


@pytest.mark.parametrize("kind", ["wrong_endpoint", "nonnested", "duplicate", "two_boundaries", "reordered_unchanged"])
def test_invalid_schedule_is_rejected(kind):
    steps = schedule()
    widths = SMALL_WIDTHS
    final = copy.deepcopy(steps[-1])
    if kind == "wrong_endpoint":
        final[pm.KEYS[1]] = final[pm.KEYS[1]][::-1]
    elif kind == "nonnested":
        missing = next(x for x in range(32) if x not in steps[2][pm.KEYS[0]])
        steps[3][pm.KEYS[0]][0] = missing
    elif kind == "duplicate":
        steps[1][pm.KEYS[0]][0] = steps[1][pm.KEYS[0]][1]
    elif kind == "two_boundaries":
        steps[1][pm.KEYS[1]] = steps[1][pm.KEYS[1]][:12]
        widths = ((32, 16), (24, 12), *SMALL_WIDTHS[2:])
    else:
        steps[1][pm.KEYS[1]].reverse()
    with pytest.raises(ValueError):
        pm.validate_schedule(steps, final, original_widths=(32, 16), widths=widths)


def test_identity_control_keeps_teacher_math_and_independent_storage(teacher):
    z = torch.randn(2, 8, 2) * .15
    sr = torch.tensor([16000, 48000])
    before = snapshot(teacher)
    rng = torch.get_rng_state().clone()
    model = pm.initialize_from_teacher(teacher, schedule()[0])
    assert torch.equal(torch.get_rng_state(), rng)
    with torch.no_grad():
        actual = model.forward_from_latents(z, sr)
        expected = gm.teacher_trace(teacher, z, sr)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert_snapshot(teacher, before)
    teacher_state = teacher.state_dict()
    for name, value in model.decoder.state_dict().items():
        assert value.data_ptr() != teacher_state[name].data_ptr()
    assert not any("progressive" in name for name in model.state_dict())


def test_first_cut_keeps_nine_units_full_interface_and_finite_gradient_through_suffix(teacher):
    model = pm.initialize_from_teacher(teacher, schedule()[1])
    assert [(model.decoder.model[i].block[1].in_channels, model.decoder.model[i].block[1].out_channels)
            for i in (3, 4, 5)] == [(64, 24), (24, 16), (16, 8)]
    x = torch.randn(1, 64, 5) * .1
    h = model.group_from_input(x)
    assert h.shape == (1, 8, 5 * 60)
    wave = model.suffix_from_group(h)
    assert wave.shape == (1, 1, 5 * 240)
    wave.square().mean().backward()
    for name, p in model.decoder.named_parameters():
        if name.startswith(gm.GROUP_PREFIXES):
            assert p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all(), name
        else:
            assert not p.requires_grad and p.grad is None, name
    assert any(p.grad.abs().max() > 0 for p in model.trainable_group_parameters())
    for i in (3, 4, 5):
        stage = model.decoder.model[i]
        assert len(stage.block[2:]) == 3
        assert [r.block[1].dilation for r in stage.block[2:]] == [(1,), (3,), (9,)]
        assert stage.block[1].kernel_size == (2 * stage.block[1].stride[0],)


def test_each_cut_inherits_recovered_effective_weights_and_original_id_mapping(teacher):
    steps = schedule()
    model = pm.initialize_from_teacher(teacher, steps[1])
    teacher_before = snapshot(teacher)
    for step in steps[2:]:
        # Recovered g/v/alpha/conditioning/bias differ from the original teacher;
        # no forward refreshes cached WN weights before migration.
        with torch.no_grad():
            for i, p in enumerate(model.trainable_group_parameters()):
                p.add_(0.002 * (i + 1))
        before = snapshot(model)
        next_model = pm.migrate(model, step)
        assert_all_coupled_weights(model, next_model, model.progressive_selection, step)
        assert_snapshot(model, before)
        assert_snapshot(teacher, teacher_before)
        assert next_model.selections == next_model.progressive_selection == step
        assert next_model.progressive_provenance["outer_widths"] == [64, 8]
        model = next_model
    assert model.progressive_selection == steps[-1]
    assert len(model.progressive_provenance["selections"]) == 4


def test_migration_preserves_source_modes_grad_slots_rng_and_frozen_state(teacher):
    steps = schedule()
    model = pm.initialize_from_teacher(teacher, steps[1]).train()
    for p in model.trainable_group_parameters():
        p.grad = torch.full_like(p, .25)
    state = snapshot(model)
    modes = {n: m.training for n, m in model.named_modules()}
    grads = {n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    out = pm.migrate(model, steps[2])
    assert_snapshot(model, state)
    assert modes == {n: m.training for n, m in model.named_modules()}
    assert torch.equal(rng, torch.get_rng_state())
    for name, p in model.named_parameters():
        if grads[name] is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, grads[name], rtol=0, atol=0)
    assert not out.training
    for name, p in out.named_parameters():
        assert p.grad is None
        assert p.data_ptr() != dict(model.named_parameters())[name].data_ptr()
    for name, value in out.decoder.state_dict().items():
        if not name.startswith(gm.GROUP_PREFIXES):
            torch.testing.assert_close(value, teacher.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["no_cut", "widen", "missing", "invalid_type", "metadata"])
def test_invalid_migration_never_changes_previous_model(teacher, kind):
    steps = schedule()
    model = pm.initialize_from_teacher(teacher, steps[1])
    chosen = copy.deepcopy(steps[2])
    if kind == "no_cut":
        chosen = steps[1]
    elif kind == "widen":
        chosen = steps[0]
    elif kind == "missing":
        chosen[pm.KEYS[0]][0] = next(x for x in range(32) if x not in steps[1][pm.KEYS[0]])
    elif kind == "invalid_type":
        chosen[pm.KEYS[1]][0] = True
    else:
        model.selections[pm.KEYS[0]].reverse()
    before = snapshot(model)
    with pytest.raises(ValueError):
        pm.migrate(model, chosen)
    assert_snapshot(model, before)


def test_zero_effective_rows_survive_cut_and_have_finite_gradients(teacher):
    steps = schedule()
    model = pm.initialize_from_teacher(teacher, steps[1])
    up = model.decoder.model[4].block[1]
    zero = gm.effective_weight(up).detach().clone()
    zero[0].zero_()
    gm.assign_effective_weight(up, zero)
    out = pm.migrate(model, steps[2])
    assert torch.count_nonzero(gm.effective_weight(out.decoder.model[4].block[1])[0]) == 0
    out.group_from_input(torch.randn(1, 64, 3) * .1).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in out.trainable_group_parameters())


def test_fresh_adam_after_cut_has_empty_moments_and_no_old_parameter_alias(teacher):
    steps = schedule()
    model = pm.initialize_from_teacher(teacher, steps[1])
    optimizer = pm.fresh_optimizer(model)
    for p in model.trainable_group_parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    moments = copy.deepcopy(optimizer.state_dict())
    out = pm.migrate(model, steps[2])
    fresh = pm.fresh_optimizer(out)
    assert not fresh.state
    assert (fresh.param_groups[0]["lr"], fresh.param_groups[0]["betas"],
            fresh.param_groups[0]["eps"], fresh.param_groups[0]["weight_decay"]) == (3e-5, (.9, .99), 1e-8, 0)
    assert not {id(p) for p in optimizer.param_groups[0]["params"]} & {id(p) for p in fresh.param_groups[0]["params"]}
    for key, state in optimizer.state_dict()["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(value, moments["state"][key][field], rtol=0, atol=0)


def test_restored_previous_selection_and_checkpoint_give_same_next_cut(teacher):
    steps = schedule()
    original = pm.initialize_from_teacher(teacher, steps[1])
    with torch.no_grad():
        for p in original.trainable_group_parameters():
            p.add_(.05)
    checkpoint = {name: value.clone() for name, value in original.group_state_dict().items()}
    restored = pm.initialize_from_teacher(teacher, steps[1])
    restored.load_group_state_dict(checkpoint)
    a, b = pm.migrate(original, steps[2]), pm.migrate(restored, steps[2])
    assert_snapshot(a, snapshot(b))


def test_first_cut_preserves_causal_prefix_of_group_output(teacher):
    model = pm.initialize_from_teacher(teacher, schedule()[1])
    x = torch.randn(1, 64, 6) * .1
    changed = x.clone()
    changed[..., 4:] += 3
    with torch.no_grad():
        before, after = model.group_from_input(x), model.group_from_input(changed)
    torch.testing.assert_close(before[..., :4 * 60], after[..., :4 * 60], rtol=0, atol=0)
