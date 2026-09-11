"""Masked statistics and exact fixed-statistics deployment obligations."""
from copy import deepcopy
import io

import pytest
import torch

from audiovae_student.model import MaskedBatchNorm, StudentConfig, StudentDecoder


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(1)
    torch.manual_seed(912)


def normalized_model():
    return StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16,
                                        head_channels=16, layer_scale_init=1.0,
                                        normalization_mode="masked_batch_norm"))


def updated_model():
    model = normalized_model().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for index in range(3):
        z = torch.randn(2, 64, 10) + index
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[0, 2:8], mask[1, 4:9] = True, True
        optimizer.zero_grad()
        model(z, mask).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    # Nonzero affine offsets exercise the otherwise invisible boundary terms.
    with torch.no_grad():
        for norm in (model.stem_norm, model.affine):
            norm.weight.copy_(torch.linspace(0.7, 1.3, 8))
            norm.bias.copy_(torch.linspace(-0.3, 0.4, 8))
    return model.freeze_normalization_statistics().eval()


def test_masked_statistics_match_unpadded_batchnorm_population_and_running_variance():
    norm = MaskedBatchNorm(2, momentum=0.25)
    reference = torch.nn.BatchNorm1d(2, eps=norm.eps, momentum=norm.momentum)
    x = torch.tensor([[[99., 1., 3., 999.], [77., 2., 6., 777.]],
                      [[88., 5., 88., 888.], [66., 10., 66., 666.]]], requires_grad=True)
    mask = torch.tensor([[False, True, True, False], [False, True, False, False]])
    selected = x.transpose(1, 2)[mask].T.unsqueeze(0)
    expected = reference(selected)
    result = norm(x, mask)
    actual = result.transpose(1, 2)[mask].T.unsqueeze(0)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(norm.running_mean, reference.running_mean)
    torch.testing.assert_close(norm.running_var, reference.running_var)
    assert norm.num_batches_tracked.item() == 1
    # Excluded padding and context do not enter statistics, but still receive
    # the affine transform so causal convolutions have normalized real context.
    mean, variance = selected.mean((0, 2)), selected.var((0, 2), unbiased=False)
    full_expected = (x - mean[None, :, None]) / (variance[None, :, None] + norm.eps).sqrt()
    torch.testing.assert_close(result, full_expected)
    actual.square().sum().backward()
    assert torch.count_nonzero(x.grad.transpose(1, 2)[~mask]) == 0
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("count", [0, 1])
def test_insufficient_scored_frames_use_fixed_stats_without_updates(count):
    norm = MaskedBatchNorm(3)
    with torch.no_grad():
        norm.running_mean.copy_(torch.tensor([1., 2., 3.]))
        norm.running_var.copy_(torch.tensor([2., 3., 4.]))
    x = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    if count:
        mask[0, 2] = True
    before = deepcopy(norm.state_dict())
    expected = norm.fixed(x)
    result = norm(x, mask)
    torch.testing.assert_close(result, expected)
    result.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert norm.weight.grad is not None and norm.bias.grad is not None
    for key, value in before.items():
        torch.testing.assert_close(norm.state_dict()[key], value, atol=0, rtol=0)


def test_decoder_expands_scored_mask_to_exact_chronological_phases():
    model = normalized_model()
    z = torch.randn(2, 64, 5)
    mask = torch.tensor([[False, True, True, False, False], [False, False, True, True, False]])
    before_blocks = []
    hook = model.blocks[-1].register_forward_hook(lambda _m, _a, out: before_blocks.append(out.detach()))
    stem = model.stem(model._phase_frames(z)).detach()
    model(z, mask)
    hook.remove()
    expanded = mask.repeat_interleave(4, -1)
    for norm, features in [(model.stem_norm, stem), (model.affine, before_blocks[0])]:
        selected = features.transpose(1, 2)[expanded]
        torch.testing.assert_close(norm.running_mean, selected.mean(0) * norm.momentum)
        torch.testing.assert_close(norm.running_var,
                                   1 - norm.momentum + selected.var(0, unbiased=True) * norm.momentum)


def test_extra_right_padding_cannot_change_scored_output_or_either_running_statistic():
    original = normalized_model().train()
    padded = deepcopy(original)
    z = torch.randn(2, 64, 6)
    mask = torch.tensor([[False, False, True, True, True, True], [False, True, True, True, True, True]])
    expected = original(z, mask)
    extended = torch.cat((z, torch.randn(2, 64, 9) * 100), -1)
    extended_mask = torch.cat((mask, torch.zeros(2, 9, dtype=torch.bool)), -1)
    actual = padded(extended, extended_mask)[..., :6 * 1920]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
    for name, buffer in original.named_buffers():
        torch.testing.assert_close(dict(padded.named_buffers())[name], buffer, atol=1e-7, rtol=1e-6)


def test_mixed_precision_keeps_statistics_fp32_and_affine_gradients_finite():
    model = normalized_model().train()
    z = torch.randn(2, 64, 5)
    mask = torch.tensor([[False, True, True, True, False], [False, False, True, True, False]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = model(z, mask)
    assert y.dtype == torch.bfloat16 and torch.isfinite(y).all()
    y.float().square().mean().backward()
    for norm in (model.stem_norm, model.affine):
        assert norm.running_mean.dtype == norm.running_var.dtype == torch.float32
        assert torch.isfinite(norm.running_mean).all() and torch.isfinite(norm.running_var).all()
        assert torch.isfinite(norm.weight.grad).all() and torch.isfinite(norm.bias.grad).all()


def test_fixed_statistics_ignore_other_batch_members_and_stay_frozen_in_train_mode():
    model = updated_model().train()
    z = torch.randn(1, 64, 7)
    before = deepcopy(model.state_dict())
    alone = model(z)
    together = model(torch.cat((z, torch.randn_like(z) * 100), 0))[:1]
    torch.testing.assert_close(alone, together, atol=1e-6, rtol=1e-4)
    alone.square().mean().backward()
    for norm in (model.stem_norm, model.affine):
        assert norm.weight.requires_grad and norm.bias.requires_grad
        assert norm.weight.grad is not None and norm.bias.grad is not None
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(buffer, before[name], atol=0, rtol=0)


def test_functional_streaming_never_updates_or_uses_dynamic_statistics():
    model = normalized_model().train()
    model(torch.randn(2, 64, 6))
    z = torch.randn(1, 64, 8)
    before = deepcopy(model.state_dict())
    state = model.initial_state()
    pieces = []
    for offset in range(0, 8, 2):
        output, state = model.forward_stream(z[..., offset:offset+2], state)
        pieces.append(output)
    reference = deepcopy(model).eval()(z)
    torch.testing.assert_close(torch.cat(pieces, -1), reference, atol=1e-6, rtol=1e-4)
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(buffer, before[name], atol=0, rtol=0)


@pytest.mark.parametrize("fold", [False, True])
def test_after_training_updates_prefix_and_streaming_remain_causal(fold):
    model = updated_model()
    if fold:
        model = model.fold_normalization()
    z = torch.randn(1, 64, 37, requires_grad=True)
    original = model(z)
    changed = z.detach().clone()
    changed[..., 5:] = torch.randn_like(changed[..., 5:]) * 50
    torch.testing.assert_close(model(changed)[..., :9600], original[..., :9600], atol=0, rtol=0)
    original[..., :9600].square().mean().backward()
    assert torch.count_nonzero(z.grad[..., 5:]) == 0
    with model.stream() as stream:
        pieces = [stream.decode_chunk(z.detach()[..., start:start+3]) for start in range(0, 37, 3)]
        assert stream.frames_decoded == 37
    torch.testing.assert_close(torch.cat(pieces, -1), original.detach(), atol=1e-6, rtol=1e-4)


def test_two_site_fold_preserves_start_boundary_and_does_not_mutate_source_or_reuse_state():
    model = updated_model()
    snapshot = deepcopy(model.state_dict())
    old_state = model.initial_state()
    folded = model.fold_normalization()
    assert folded.config.normalization_mode == "folded"
    assert not any(isinstance(module, MaskedBatchNorm) for module in folded.modules())
    assert isinstance(folded.stem_norm, torch.nn.Identity) and isinstance(folded.affine, torch.nn.Identity)
    for frames in (1, 2, 33):
        z = torch.randn(1, 64, frames)
        torch.testing.assert_close(folded(z), model(z), atol=1e-6, rtol=1e-4)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, snapshot[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="new streaming state"):
        folded.forward_stream(torch.randn(1, 64, 1), old_state)
    # A source stream opened before conversion keeps using the untouched model.
    with model.stream() as live:
        z = torch.randn(1, 64, 4)
        first = live.decode_chunk(z[..., :1])
        model.fold_normalization()
        torch.testing.assert_close(torch.cat((first, live.decode_chunk(z[..., 1:])), -1),
                                   model(z), atol=1e-6, rtol=1e-4)


def test_fold_requires_explicit_freeze_and_evaluation():
    model = normalized_model()
    with pytest.raises(ValueError, match="eval"):
        model.fold_normalization()
    model.eval()
    with pytest.raises(ValueError, match="Freeze"):
        model.fold_normalization()
    model.freeze_normalization_statistics().fold_normalization()


@pytest.mark.parametrize("mode", ["affine", "masked_batch_norm", "folded"])
def test_checkpoint_roundtrip_and_legacy_configuration_loading(mode):
    model = updated_model() if mode != "affine" else StudentDecoder(StudentConfig(
        hidden_channels=8, expansion_channels=16, head_channels=16, layer_scale_init=1.0)).eval()
    if mode == "folded":
        model = model.fold_normalization()
    config = model.config.to_dict()
    if mode == "affine":
        # Exactly the old config and state_dict contract, with no new buffers.
        for key in ("normalization_mode", "batch_norm_eps", "batch_norm_momentum"):
            config.pop(key)
        assert not any(name.startswith("stem_norm.") for name in model.state_dict())
        assert "affine.scale" in model.state_dict() and "affine.bias" in model.state_dict()
    memory = io.BytesIO()
    torch.save({"config": config, "model": model.state_dict()}, memory)
    memory.seek(0)
    payload = torch.load(memory, weights_only=True)
    restored = StudentDecoder(StudentConfig(**payload["config"])).eval()
    restored.load_state_dict(payload["model"], strict=True)
    if mode == "masked_batch_norm":
        assert restored.stem_norm.statistics_frozen and restored.affine.statistics_frozen
    z = torch.randn(1, 64, 5)
    torch.testing.assert_close(restored(z), model(z), atol=0, rtol=0)


def test_mask_contract_rejects_shape_dtype_and_configuration_errors():
    model = normalized_model()
    z = torch.randn(2, 64, 4)
    for mask in (torch.ones(2, 4), torch.ones(1, 4, dtype=torch.bool), [True] * 4):
        with pytest.raises(ValueError, match="scored_latent_mask"):
            model(z, mask)
    assert model(z[..., :0], torch.empty(2, 0, dtype=torch.bool)).shape == (2, 1, 0)
    for settings in ({"normalization_mode": "bad"}, {"batch_norm_eps": 0}, {"batch_norm_momentum": 0}):
        with pytest.raises(ValueError):
            StudentConfig(**settings)
