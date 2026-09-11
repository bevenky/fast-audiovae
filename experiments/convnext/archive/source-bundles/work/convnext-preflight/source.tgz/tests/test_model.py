"""Causality, ordering and state checks independent of any trained checkpoint."""
import pytest
import torch

from audiovae_student.model import CausalConv, StudentConfig, StudentDecoder, architecture_summary


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(1)
    torch.manual_seed(312)


def small_model():
    # Strong residuals ensure a broken deep-layer history cannot hide behind
    # the intentionally small residual initialization used for fresh training.
    return StudentDecoder(StudentConfig(hidden_channels=16, expansion_channels=32,
                                        head_channels=32, layer_scale_init=1.0)).eval()


@pytest.mark.parametrize("frames,pattern", [(1, [1]), (9, [2]), (37, [1]), (67, [2, 4, 1, 9])])
def test_stream_matches_whole_through_all_histories(frames, pattern):
    model = small_model()
    z = torch.randn(1, 64, frames)
    with torch.inference_mode():
        expected = model(z)
    pieces, position, index = [], 0, 0
    with model.stream() as stream:
        while position < frames:
            stop = min(frames, position + pattern[index % len(pattern)])
            chunk = stream.decode_chunk(z[..., position:stop])
            assert chunk.shape == (1, 1, (stop - position) * 1920)
            pieces.append(chunk)
            position, index = stop, index + 1
        assert stream.frames_decoded == frames
        assert stream.flush().shape[-1] == 0
        for history in stream.state.histories:
            assert history.untyped_storage().nbytes() == history.numel() * history.element_size()
    torch.testing.assert_close(torch.cat(pieces, -1), expected, atol=1e-6, rtol=1e-4)


def test_temporal_shuffle_has_channel_time_phase_order():
    model = small_model()
    with torch.no_grad():
        model.adapter.weight.zero_()
        model.adapter.bias.copy_(torch.arange(256, dtype=torch.float32))
    result = model._phase_frames(torch.zeros(1, 64, 3))
    for channel in range(64):
        expected = torch.arange(4 * channel, 4 * channel + 4, dtype=torch.float32).repeat(3)
        torch.testing.assert_close(result[0, channel], expected, atol=0, rtol=0)
    frames = torch.arange(2 * 480, dtype=torch.float32).reshape(1, 2, 480).transpose(1, 2)
    torch.testing.assert_close(model._waveform(frames).flatten(), torch.arange(960, dtype=torch.float32))


def test_dilated_layer_uses_only_current_and_past_inputs():
    conv = CausalConv(1, 1, 3, dilation=2, bias=False)
    with torch.no_grad():
        conv.conv.weight.copy_(torch.tensor([[[1.0, 10.0, 100.0]]]))
    x = torch.tensor([[[1., 2., 3., 4., 5., 6.]]])
    expected = torch.tensor([[[111., 211., 311., 421., 531., 642.]]])
    torch.testing.assert_close(conv(x), expected, atol=0, rtol=0)


def test_future_latents_cannot_change_output_prefix_or_gradients():
    model = small_model()
    z = torch.randn(1, 64, 11, requires_grad=True)
    original = model(z)
    changed = z.detach().clone()
    changed[..., 5:] = torch.randn_like(changed[..., 5:]) * 100
    torch.testing.assert_close(model(changed)[..., :5 * 1920], original[..., :5 * 1920], atol=0, rtol=0)
    original[..., :5 * 1920].square().mean().backward()
    assert torch.count_nonzero(z.grad[..., 5:]) == 0
    assert torch.count_nonzero(z.grad[..., :5]) > 0


def test_stream_reset_isolation_empty_and_invalid_input_are_transactional():
    model = small_model()
    a, b = torch.randn(1, 64, 8), torch.randn(1, 64, 3)
    with model.stream() as first, model.stream() as second:
        saved = first.decode_chunk(a[..., :2])
        snapshot = saved.clone()
        second_result = second.decode_chunk(b)
        first_state = first.state
        with pytest.raises(ValueError):
            first.decode_chunk(torch.full((1, 64, 1), float("nan")))
        assert first.state is first_state and first.frames_decoded == 2
        assert first.decode_chunk(a[..., :0]).shape[-1] == 0
        rest = first.decode_chunk(a[..., 2:])
        torch.testing.assert_close(saved, snapshot, atol=0, rtol=0)
        torch.testing.assert_close(torch.cat((saved, rest), -1), model(a), atol=1e-6, rtol=1e-4)
        torch.testing.assert_close(second_result, model(b), atol=1e-6, rtol=1e-4)
        first.reset()
        torch.testing.assert_close(first.decode_chunk(a), model(a), atol=1e-6, rtol=1e-4)
    with pytest.raises(RuntimeError, match="closed"):
        first.flush()


def test_declared_default_capacity_and_state_budget():
    model = StudentDecoder()
    summary = architecture_summary()
    assert len(model.blocks) == 10
    assert model.blocks[0].expand.weight.shape == (2048, 512)
    assert model.head.conv.weight.shape == (2048, 512, 3)
    assert model.activation.weight.numel() == 1
    assert model.output.bias is None
    assert summary["history_internal_frames"] == 116
    assert summary["history_latent_frames"] == 29
    assert summary["state_fp32_bytes_per_stream"] == 56704 * 4 + 1
    assert sum(h.numel() for h in model.initial_state().histories) == 56704
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    assert all(p.requires_grad for p in model.parameters())


def test_full_capacity_decoder_streams_and_backpropagates():
    model = StudentDecoder().eval()
    z = torch.randn(1, 64, 3, requires_grad=True)
    y = model(z)
    assert y.shape == (1, 1, 5760)
    y.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    with model.stream() as stream:
        result = torch.cat([stream.decode_chunk(z.detach()[..., i:i+1]) for i in range(3)], -1)
    torch.testing.assert_close(result, y.detach(), atol=1e-5, rtol=1e-4)


def test_input_and_configuration_contracts():
    for config in [{"hidden_channels": 0}, {"dilations": [0]}, {"layer_norm_eps": 0}]:
        with pytest.raises(ValueError):
            StudentConfig(**config)
    model = small_model()
    for bad in [torch.zeros(1, 63, 1), torch.zeros(1, 64), torch.ones(1, 64, 1, dtype=torch.int64)]:
        with pytest.raises(ValueError):
            model(bad)
    assert model(torch.empty(2, 64, 0)).shape == (2, 1, 0)
    state = model.initial_state()
    with pytest.raises(ValueError):
        model.forward_stream(torch.zeros(2, 64, 1), state)
