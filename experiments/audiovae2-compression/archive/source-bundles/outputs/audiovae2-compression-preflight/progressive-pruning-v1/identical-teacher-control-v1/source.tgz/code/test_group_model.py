"""CPU contracts for the actual stage-group compression helpers.

The fixture runs the upstream operation pattern at small channel widths. It
uses nontrivial legacy weight normalization and sample-rate conditioning so a
constructor-only copy or slicing normalization factors cannot pass by accident.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("compression_group_model_under_test", HERE / "group_model.py")
gm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gm
SPEC.loader.exec_module(gm)


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, padding=0, output_padding=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding

    def forward(self, x):
        return super().forward(F.pad(x, (2 * self.__padding - self.__output_padding, 0)))


class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(self, *args, padding=0, output_padding=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding

    def forward(self, x):
        return super().forward(x)[..., :-(2 * self.__padding - self.__output_padding)]


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).square()


class CausalResidualUnit(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(channels),
            weight_norm(CausalConv1d(channels, channels, 7, dilation=dilation,
                                    padding=3 * dilation, groups=channels)),
            Snake1d(channels),
            weight_norm(CausalConv1d(channels, channels, 1)),
        )

    def forward(self, x):
        return x + self.block(x)


class CausalDecoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride, groups=None, use_noise_block=False):
        super().__init__()
        assert not use_noise_block
        self.input_channels = input_dim
        self.block = nn.Sequential(
            Snake1d(input_dim),
            weight_norm(CausalTransposeConv1d(input_dim, output_dim, 2 * stride,
                                             stride=stride, padding=(stride + 1) // 2,
                                             output_padding=stride % 2)),
            *(CausalResidualUnit(output_dim, dilation) for dilation in (1, 3, 9)),
        )

    def forward(self, x):
        return self.block(x)


class SampleRateConditionLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cond_type = "scale_bias"
        self.scale_embed = nn.Embedding(4, channels)
        self.bias_embed = nn.Embedding(4, channels)
        self.out_layer = nn.Identity()

    def forward(self, x, index):
        return self.out_layer(x * self.scale_embed(index).unsqueeze(-1)
                              + self.bias_embed(index).unsqueeze(-1))


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [weight_norm(CausalConv1d(8, 8, 7, padding=3, groups=8)),
                  weight_norm(CausalConv1d(8, 128, 1))]
        for i, stride in enumerate((8, 6, 5, 2, 2, 2)):
            layers.append(CausalDecoderBlock(128 // 2**i, 128 // 2**(i + 1), stride))
        layers += [Snake1d(2), weight_norm(CausalConv1d(2, 1, 7, padding=3)), nn.Tanh()]
        self.model = nn.ModuleList(layers)
        self.register_buffer("sr_bin_boundaries", torch.tensor([20000, 30000, 40000], dtype=torch.int32))
        self.sr_bin_buckets = 4
        self.sr_cond_model = nn.ModuleList([
            SampleRateConditionLayer(layer.input_channels) if isinstance(layer, CausalDecoderBlock) else None
            for layer in layers
        ])
        with torch.no_grad():
            for module in self.modules():
                if hasattr(module, "weight_g"):
                    module.weight_g.mul_(torch.linspace(.4, 1.1, module.weight_g.shape[0]).view(-1, 1, 1))
                if isinstance(module, Snake1d):
                    module.alpha.copy_(torch.linspace(.35, 1.25, module.alpha.shape[1]).view(1, -1, 1))
                if isinstance(module, SampleRateConditionLayer):
                    module.scale_embed.weight.uniform_(.85, 1.15)
                    module.bias_embed.weight.uniform_(-.02, .02)

    def get_sr_idx(self, value):
        return torch.bucketize(value, self.sr_bin_boundaries)

    def forward(self, x, sr_cond=None):
        if sr_cond is None:
            sr_cond = torch.tensor([48000], dtype=torch.int32, device=x.device)
        index = self.get_sr_idx(sr_cond)
        for layer, conditioning in zip(self.model, self.sr_cond_model):
            if conditioning is not None:
                x = conditioning(x, index)
            x = layer(x)
        return x


@pytest.fixture(autouse=True)
def cpu_determinism():
    state = torch.random.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(9137)
    yield
    torch.random.set_rng_state(state)
    torch.set_num_threads(threads)


@pytest.fixture
def teacher():
    return TinyDecoder().eval().requires_grad_(False)


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_snapshot(model, expected):
    actual = model.state_dict()
    assert set(actual) == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0, msg=name)


def wn_oracle(module):
    if hasattr(module, "weight_v"):
        v = module.weight_v.detach()
        return v * (module.weight_g.detach() / torch.linalg.vector_norm(v, dim=(1, 2), keepdim=True))
    return module.weight.detach()


def selections():
    # Deliberately not the first contiguous coordinates, nor sorted order.
    return list(range(31, 0, -2)), list(range(15, 0, -2))


@pytest.mark.parametrize("transpose", [False, True])
def test_effective_weight_reads_loaded_normalization_not_stale_cached_tensor(transpose):
    module = weight_norm((nn.ConvTranspose1d if transpose else nn.Conv1d)(5, 7, 4))
    stale = module.weight.detach().clone()
    with torch.no_grad():
        module.weight_g.mul_(2.7)
        module.weight_v.add_(.11)
    actual = gm.effective_weight(module)
    torch.testing.assert_close(actual, wn_oracle(module), rtol=1e-6, atol=1e-7)
    assert not torch.allclose(actual, stale)


@pytest.mark.parametrize("transpose", [False, True])
def test_assign_effective_weight_renormalizes_sliced_axes_and_preserves_bias(transpose):
    cls = nn.ConvTranspose1d if transpose else nn.Conv1d
    old = weight_norm(cls(5, 7, 4))
    with torch.no_grad():
        old.weight_g.mul_(1.9)
    inputs, outputs = torch.tensor([4, 1, 3]), torch.tensor([6, 2])
    value = wn_oracle(old)
    desired = value[inputs][:, outputs] if transpose else value[outputs][:, inputs]
    module = weight_norm(cls(3, 2, 4))
    bias = old.bias.detach()[outputs]
    gm.assign_effective_weight(module, desired.clone(), bias.clone())
    torch.testing.assert_close(gm.effective_weight(module), desired, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(module.bias, bias, rtol=0, atol=0)
    x = torch.randn(2, 3, 9)
    oracle = F.conv_transpose1d(x, desired, bias) if transpose else F.conv1d(x, desired, bias)
    torch.testing.assert_close(module(x), oracle, rtol=1e-5, atol=1e-6)
    assert {"weight_g", "weight_v"}.issubset(dict(module.named_parameters()))


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_zero_effective_weight_rows_stay_zero_and_have_finite_backward(dim):
    module = weight_norm(nn.ConvTranspose1d(4, 3, 2), dim=dim)
    desired = torch.randn_like(module.weight) * .1
    if dim == -1:
        desired.zero_()
    else:
        desired.select(dim, 1).zero_()
    gm.assign_effective_weight(module, desired)
    actual = gm.effective_weight(module)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, desired, rtol=1e-6, atol=1e-7)
    module(torch.randn(1, 4, 7)).square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


@pytest.mark.parametrize("frames", [1, 3])
def test_full_copy_is_independent_and_matches_full_original(teacher, frames):
    before = snapshot(teacher)
    copied = gm.clone_teacher(teacher)
    z = torch.randn(1, 8, frames) * .15
    with torch.no_grad():
        torch.testing.assert_close(copied(z), teacher(z), rtol=1e-5, atol=1e-6)
    assert_snapshot(teacher, before)
    original_params = dict(teacher.named_parameters())
    for name, param in copied.named_parameters():
        assert param.data_ptr() != original_params[name].data_ptr()
    original_buffers = dict(teacher.named_buffers())
    for name, value in copied.named_buffers():
        assert value.data_ptr() != original_buffers[name].data_ptr()
    for name, module in copied.named_modules():
        if hasattr(module, "weight_v"):
            torch.testing.assert_close(gm.effective_weight(module), wn_oracle(dict(teacher.named_modules())[name]),
                                       rtol=1e-6, atol=1e-7)


def test_identity_selection_matches_teacher_at_group_and_waveform(teacher):
    student = gm.build_student(teacher, list(range(32)), list(range(16)))
    z = torch.randn(1, 8, 2) * .15
    with torch.no_grad():
        ref = gm.teacher_trace(teacher, z)
        waveform, h = student.forward_latents(z)
        from_group = student.group_from_input(ref["group_input"])
        suffix = student.suffix_from_group(ref["group_output"])
    torch.testing.assert_close(h, ref["group_output"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(from_group, ref["group_output"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(waveform, ref["waveform"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(suffix, ref["waveform"], rtol=1e-5, atol=1e-6)
    assert h.shape == (1, 8, 2 * 480)
    assert waveform.shape == (1, 1, 2 * 1920)


def test_full_width_permutation_preserves_the_complete_function(teacher):
    # Coupled reordering is an exact reparameterization, unlike dropping rows.
    student = gm.build_student(teacher, list(reversed(range(32))), list(reversed(range(16))))
    z = torch.randn(2, 8, 2) * .1
    rates = torch.tensor([16000, 48000], dtype=torch.int32)
    with torch.no_grad():
        expected = teacher(z, rates)
        traced = gm.teacher_trace(teacher, z, rates)
        actual = student.forward_from_latents(z, rates)
    torch.testing.assert_close(traced["waveform"], expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual["group_output"], traced["group_output"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual["waveform"], expected, rtol=1e-5, atol=1e-6)


def test_narrowing_couples_all_weight_axes_snakes_and_conditioning(teacher):
    i2, i3 = selections()
    original = snapshot(teacher)
    student = gm.build_student(teacher, i2, i3)
    decoder = student.decoder
    all1, all4 = list(range(64)), list(range(8))
    for index, ins, outs in [(3, all1, i2), (4, i2, i3), (5, i3, all4)]:
        old, new = teacher.model[index], decoder.model[index]
        torch.testing.assert_close(new.block[0].alpha, old.block[0].alpha[:, ins], rtol=0, atol=0)
        expected = wn_oracle(old.block[1])[ins][:, outs]
        torch.testing.assert_close(gm.effective_weight(new.block[1]), expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(new.block[1].bias, old.block[1].bias[outs], rtol=0, atol=0)
        assert new.input_channels == len(ins)
        for n in (2, 3, 4):
            ro, rn = old.block[n], new.block[n]
            assert rn.block[1].groups == len(outs)
            assert rn.block[1].dilation == ro.block[1].dilation
            for a in (0, 2):
                torch.testing.assert_close(rn.block[a].alpha, ro.block[a].alpha[:, outs], rtol=0, atol=0)
            torch.testing.assert_close(gm.effective_weight(rn.block[1]), wn_oracle(ro.block[1])[outs], rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(gm.effective_weight(rn.block[3]), wn_oracle(ro.block[3])[outs][:, outs], rtol=1e-6, atol=1e-7)
            for conv in (1, 3):
                torch.testing.assert_close(rn.block[conv].bias, ro.block[conv].bias[outs], rtol=0, atol=0)
        for embed in ("scale_embed", "bias_embed"):
            torch.testing.assert_close(getattr(decoder.sr_cond_model[index], embed).weight,
                                       getattr(teacher.sr_cond_model[index], embed).weight[:, ins], rtol=0, atol=0)
    assert_snapshot(teacher, original)


def test_all_nine_residual_units_and_dilations_survive(teacher):
    student = gm.build_student(teacher, *selections())
    units = [u for i in (3, 4, 5) for u in student.decoder.model[i].block[2:]]
    assert len(units) == 9
    assert [u.block[1].dilation[0] for u in units] == [1, 3, 9] * 3
    assert isinstance(student.decoder.model[-1], nn.Tanh)


def test_only_group_is_trainable_but_frozen_suffix_transmits_gradient(teacher):
    source_before = snapshot(teacher)
    student = gm.build_student(teacher, *selections())
    student.train()
    for i, layer in enumerate(student.decoder.model):
        assert layer.training is (i in (3, 4, 5))
    for i, layer in enumerate(student.decoder.sr_cond_model):
        if layer is not None:
            assert layer.training is (i in (3, 4, 5))
    before = snapshot(student.decoder)
    expected = {name for name, _ in student.decoder.named_parameters()
                if any(name.startswith(prefix + str(i) + ".")
                       for prefix in ("model.", "sr_cond_model.") for i in (3, 4, 5))}
    actual = {name for name, p in student.decoder.named_parameters() if p.requires_grad}
    assert actual == expected
    selected = list(student.group_named_parameters())
    assert {id(p) for _, p in selected} == {id(p) for _, p in student.decoder.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for _, p in selected], lr=1e-3, weight_decay=0)
    z = torch.randn(1, 8, 2) * .15
    y, h = student.forward_latents(z)
    h.retain_grad()
    y.square().mean().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all() and h.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in selected)
    assert all(p.grad is None for name, p in student.decoder.named_parameters() if name not in expected)
    optimizer.step()
    after = student.decoder.state_dict()
    outside = [n for n in before if not any(n.startswith(prefix + str(i) + ".")
                                          for prefix in ("model.", "sr_cond_model.") for i in (3, 4, 5))]
    for name in outside:
        torch.testing.assert_close(after[name], before[name], rtol=0, atol=0, msg=name)
    assert any(not torch.equal(after[name], before[name]) for name in expected)
    assert_snapshot(teacher, source_before)


@pytest.mark.parametrize("frames", [1, 2, 5])
def test_group_and_decoder_lengths_are_exact(teacher, frames):
    student = gm.build_student(teacher, *selections())
    with torch.no_grad():
        h = student.group_from_input(torch.randn(1, 64, frames) * .1)
        y, from_z = student.forward_latents(torch.randn(1, 8, frames) * .1)
    assert h.shape == (1, 8, 60 * frames)
    assert from_z.shape == (1, 8, 480 * frames)
    assert y.shape == (1, 1, 1920 * frames)
    assert torch.isfinite(y).all() and y.abs().max() <= 1


def test_group_future_cannot_change_prior_outputs(teacher):
    student = gm.build_student(teacher, *selections())
    x = torch.randn(1, 64, 7) * .1
    changed = x.clone()
    changed[..., 4:] += torch.randn_like(changed[..., 4:]) * 5
    with torch.no_grad():
        left, right = student.group_from_input(x), student.group_from_input(changed)
    torch.testing.assert_close(left[..., :240], right[..., :240], rtol=1e-6, atol=1e-7)
    assert not torch.allclose(left[..., 240:], right[..., 240:])


def test_group_state_restore_recovers_all_trainable_parameters(teacher):
    student = gm.build_student(teacher, *selections())
    saved = {k: v.detach().clone() for k, v in student.group_state_dict().items()}
    all_before = snapshot(student.decoder)
    with torch.no_grad():
        for _, p in student.group_named_parameters():
            p.add_(.03)
    student.load_group_state_dict(saved)
    assert_snapshot(student.decoder, all_before)


def test_checkpoint_rejection_is_atomic_and_construction_preserves_rng(teacher):
    random_before = torch.random.get_rng_state().clone()
    student = gm.build_student(teacher, *selections())
    assert torch.equal(torch.random.get_rng_state(), random_before)
    before = snapshot(student.decoder)
    bad = {k: v.clone() + .01 for k, v in student.group_state_dict().items()}
    key = list(bad)[-1]
    bad[key].view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        student.load_group_state_dict(bad)
    assert_snapshot(student.decoder, before)


@pytest.mark.parametrize("first,second", [([0, 0], [0, 1]), ([32], [0]), ([-1], [0]), ([], [0]), ([0], [16])])
def test_invalid_coordinate_selections_fail_without_mutating_teacher(teacher, first, second):
    before = snapshot(teacher)
    with pytest.raises((ValueError, IndexError, TypeError)):
        gm.build_student(teacher, first, second)
    assert_snapshot(teacher, before)
