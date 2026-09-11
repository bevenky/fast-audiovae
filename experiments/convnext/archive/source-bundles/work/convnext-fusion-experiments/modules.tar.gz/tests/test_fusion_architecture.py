"""CPU correctness for independent, non-production architecture candidates."""
from copy import deepcopy
from itertools import product

import pytest
import torch

from audiovae_student.fusion_architecture import FusionArchitectureConfig, FusionStudentDecoder
from audiovae_student.model import MaskedBatchNorm, StudentConfig, StudentDecoder


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(1)
    torch.manual_seed(281)


def base_decoder():
    model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16,
        head_channels=16, layer_scale_init=1., normalization_mode="masked_batch_norm",
        adapter_mode="raw_repeat_phase_bias"))
    with torch.no_grad():
        model(torch.randn(2, 64, 11))
        for norm in (model.stem_norm, model.affine):
            norm.weight.copy_(torch.linspace(.7, 1.3, 8))
            norm.bias.copy_(torch.linspace(-.1, .2, 8))
        model.adapter.phase_bias.copy_(torch.linspace(-.2, .3, 256).reshape(64, 4))
    return model.freeze_normalization_statistics().eval()


CONFIGS = [FusionArchitectureConfig(*values) for values in product((False, True), repeat=3)]


@pytest.mark.parametrize("config", CONFIGS)
def test_variants_copy_parameters_and_calibration_without_mutating_source_or_rng(config):
    base = base_decoder()
    before = deepcopy(base.state_dict())
    rng = torch.get_rng_state().clone()
    model = FusionStudentDecoder.from_decoder(base, config)
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.config == base.config and model.fusion_config == config
    assert model.required_context_latent_frames == (30 if config.causal_output_filter else 29)
    old_parameters, new_parameters = dict(base.named_parameters()), dict(model.named_parameters())
    assert set(new_parameters) - set(old_parameters) == (
        {"output_filter.conv.weight"} if config.causal_output_filter else set())
    for name, parameter in old_parameters.items():
        torch.testing.assert_close(new_parameters[name], parameter, atol=0, rtol=0)
        assert new_parameters[name].data_ptr() != parameter.data_ptr()
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
        torch.testing.assert_close(base.state_dict()[name], value, atol=0, rtol=0)
    for norm in (model.stem_norm, model.affine):
        assert isinstance(norm, MaskedBatchNorm) and bool(norm.statistics_frozen)
    # Actual candidate training must not mutate the original source model.
    model.train()
    model(torch.randn(2, 64, 4)).square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for name, value in before.items():
        torch.testing.assert_close(base.state_dict()[name], value, atol=0, rtol=0)
    for name, value in base.named_buffers():
        torch.testing.assert_close(model.get_buffer(name), value, atol=0, rtol=0)


@pytest.mark.parametrize("config", CONFIGS)
def test_batch_streaming_sample_counts_empty_reset_and_causality(config):
    model = FusionStudentDecoder.from_decoder(base_decoder(), config)
    # Nonidentity taps ensure filter state errors cannot hide behind the initial identity.
    if model.output_filter is not None:
        with torch.no_grad():
            model.output_filter.conv.weight.copy_(torch.tensor([[[.03, -.02, .04, .01, -.05, .1, .9]]]))
    z = torch.randn(2, 64, 43)
    with torch.no_grad():
        expected = model(z)
        assert expected.shape == (2, 1, 43 * 1920)
        state = model.initial_state(2)
        original_state = state
        empty, state = model.forward_stream(z[..., :0], state)
        assert empty.shape == (2, 1, 0) and state is original_state
        chunks = []
        for start, stop in ((0, 1), (1, 5), (5, 6), (6, 40), (40, 43)):
            piece, state = model.forward_stream(z[..., start:stop], state)
            assert piece.shape[-1] == (stop - start) * 1920
            chunks.append(piece)
        torch.testing.assert_close(torch.cat(chunks, -1), expected, atol=2e-6, rtol=1e-4)
        changed = z.clone()
        changed[..., 8:] = torch.randn_like(changed[..., 8:]) * 100
        torch.testing.assert_close(model(changed)[..., :8 * 1920], expected[..., :8 * 1920], atol=0, rtol=0)
        if config.terminal_tanh:
            assert expected.abs().max() <= 1
    with model.stream() as stream:
        pieces = [stream.decode_chunk(z[:1, :, :1]), stream.decode_chunk(z[:1, :, 1:])]
        torch.testing.assert_close(torch.cat(pieces, -1), expected[:1], atol=2e-6, rtol=1e-4)
        assert stream.frames_decoded == 43 and stream.flush().shape == (1, 1, 0)
        stream.reset()
        torch.testing.assert_close(stream.decode_chunk(z[:1]), expected[:1], atol=2e-6, rtol=1e-4)
        for history in stream.state.histories:
            assert history.untyped_storage().nbytes() == history.numel() * history.element_size()


@pytest.mark.parametrize("tanh", [False, True])
def test_identity_filter_and_tanh_have_exact_declared_initial_function(tanh):
    base = base_decoder()
    z = torch.randn(2, 64, 6)
    expected = base(z)
    if tanh:
        expected = expected.tanh()
    for filtering in (False, True):
        model = FusionStudentDecoder.from_decoder(base,
            FusionArchitectureConfig(terminal_tanh=tanh, causal_output_filter=filtering))
        torch.testing.assert_close(model(z), expected, atol=0, rtol=0)
        for phase in range(4):
            torch.testing.assert_close(model._phase_frames(z)[..., phase::4],
                z + base.adapter.phase_bias[:, phase][None, :, None], atol=0, rtol=0)


def test_zero_padding_changes_only_startup_and_preserves_interior_after_full_history():
    base = base_decoder()
    model = FusionStudentDecoder.from_decoder(base, FusionArchitectureConfig(zero_startup_padding=True))
    z = torch.randn(1, 64, 39)
    with torch.no_grad():
        expected, actual = base(z), model(z)
    assert not torch.equal(expected[..., :1920], actual[..., :1920])
    # 116 internal frames = 29 original latent frames. No earlier boundary can
    # enter the output once that complete causal history has passed.
    torch.testing.assert_close(actual[..., 29 * 1920:], expected[..., 29 * 1920:], atol=0, rtol=0)


def test_filter_applies_to_waveform_before_tanh_and_uses_past_not_future_samples():
    base = base_decoder()
    model = FusionStudentDecoder.from_decoder(base,
        FusionArchitectureConfig(terminal_tanh=True, causal_output_filter=True))
    with torch.no_grad():
        model.output_filter.conv.weight.zero_()
        model.output_filter.conv.weight[..., -2] = 1  # One-sample causal delay.
    z = torch.randn(1, 64, 4)
    raw = base(z)
    expected = torch.cat((torch.zeros_like(raw[..., :1]), raw[..., :-1]), -1).tanh()
    torch.testing.assert_close(model(z), expected, atol=0, rtol=0)


def test_variant_identity_state_and_export_contracts_are_explicit():
    base = base_decoder()
    plain = FusionStudentDecoder.from_decoder(base)
    zero = FusionStudentDecoder.from_decoder(base, FusionArchitectureConfig(zero_startup_padding=True))
    with pytest.raises(ValueError, match="Normalization layout"):
        zero.forward_stream(torch.randn(1, 64, 1), plain.initial_state())
    with pytest.raises(ValueError, match="export-qualified"):
        plain.fold_normalization()
    with pytest.raises(ValueError, match="unfolded"):
        FusionStudentDecoder.from_decoder(base.fold_normalization(), FusionArchitectureConfig(zero_startup_padding=True))
    with pytest.raises(ValueError, match="already wrapped"):
        FusionStudentDecoder.from_decoder(plain)
    for values in ({"terminal_tanh": 1}, {"causal_output_filter": None}, {"zero_startup_padding": "yes"}):
        with pytest.raises(ValueError, match="boolean"):
            FusionArchitectureConfig(**values)
    assert FusionArchitectureConfig(**zero.fusion_config.to_dict()) == zero.fusion_config
    assert "terminal_tanh" not in zero.config.to_dict()


def test_saved_variant_requires_its_architecture_and_strictly_restores_new_filter_weights():
    base = base_decoder()
    config = FusionArchitectureConfig(True, True, True)
    first = FusionStudentDecoder.from_decoder(base, config)
    with torch.no_grad():
        first.output_filter.conv.weight[..., -2] = .125
    restored = FusionStudentDecoder.from_decoder(base, FusionArchitectureConfig(**config.to_dict()))
    restored.load_state_dict(deepcopy(first.state_dict()), strict=True)
    z = torch.randn(1, 64, 8)
    torch.testing.assert_close(restored(z), first(z), atol=0, rtol=0)
    with pytest.raises(RuntimeError):
        base.load_state_dict(first.state_dict(), strict=True)


def test_filter_training_crops_use_extended_real_context_without_changing_scored_audio():
    model = FusionStudentDecoder.from_decoder(base_decoder(), FusionArchitectureConfig(causal_output_filter=True))
    with torch.no_grad():
        model.output_filter.conv.weight.fill_(1 / 7)
        z = torch.randn(1, 64, 47)
        full = model(z)
        scored_start, scored_stop = 35, 41
        context_start = scored_start - model.required_context_latent_frames
        crop = model(z[..., context_start:scored_stop])
    offset = model.required_context_latent_frames * model.samples_per_latent
    expected = full[..., scored_start * 1920:scored_stop * 1920]
    torch.testing.assert_close(crop[..., offset:], expected, atol=2e-6, rtol=1e-4)
