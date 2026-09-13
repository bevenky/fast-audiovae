"""Small CPU reference fixtures; no MPS device or original model is loaded."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fast_audiovae import mps_decoder as decoder


def array(shape, seed=7):
    return np.random.default_rng(seed).normal(0, 0.1, shape).astype(np.float32)


@pytest.mark.parametrize("dilation", [1, 3, 9])
def test_depthwise_odd_short_and_empty_partitions(dilation):
    weight, bias, x = array((3, 1, 7)), array((3,), 8), array((1, 3, 15), 9)
    initial = array((1, 3, 6*dilation), 10)
    module = decoder._CausalConv1d(weight, bias, dilation=dilation, groups=3)
    joined = np.concatenate((initial, x), axis=-1)
    expected = np.empty_like(x)
    for c in range(3):
        for t in range(15):
            expected[0, c, t] = bias[c] + sum(float(weight[c, 0, k])*float(joined[0, c, t+k*dilation])
                                               for k in range(7))
    state = torch.from_numpy(initial.copy())
    chunks, pos = [], 0
    for length in (1, 0, 2, 3, 9):
        before = state.clone()
        y, updated = module.decode(torch.from_numpy(x[..., pos:pos+length].copy()), state)
        assert torch.equal(state, before)
        assert updated.data_ptr() != state.data_ptr()
        assert updated.is_contiguous()
        chunks.append(y)
        state = updated
        pos += length
    np.testing.assert_allclose(torch.cat(chunks, dim=-1).numpy(), expected, atol=2e-7, rtol=2e-6)
    np.testing.assert_array_equal(state.numpy(), joined[..., -6*dilation:])


def transpose_oracle(x, history, weight, bias, stride):
    joined = np.concatenate((history, x), axis=-1)
    result = np.zeros((1, weight.shape[1], (joined.shape[-1]+1)*stride), dtype=np.float64)
    for t in range(joined.shape[-1]):
        for i in range(weight.shape[0]):
            for o in range(weight.shape[1]):
                for k in range(2*stride):
                    result[0, o, t*stride+k] += float(joined[0, i, t])*float(weight[i, o, k])
    result += bias[None, :, None]
    return result[..., stride:stride+x.shape[-1]*stride].astype(np.float32)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("stride", [2, 5, 6, 8])
@pytest.mark.parametrize("pattern", ["signed", "quiet", "zero"])
def test_transpose_matrix_forms_preserve_prior_input_reference(stride, paired, pattern):
    weight, bias = array((3, 2, 2*stride)), array((2,), 11)
    x, initial = array((1, 3, 7), 12), array((1, 3, 1), 13)
    if pattern == "quiet":
        x *= np.float32(1e-5); initial *= np.float32(1e-5)
    elif pattern == "zero":
        x.fill(0); initial.fill(0)
    module = decoder._CausalTranspose1d(weight, bias, stride=stride, paired=paired)
    expected = transpose_oracle(x, initial, weight, bias, stride)
    state, chunks, pos = torch.from_numpy(initial.copy()), [], 0
    original_x = x.copy()
    packed, packed_pointer = module.packed_weight.clone(), module.packed_weight.data_ptr()
    for length in (1, 0, 3, 1, 2):
        before = state.clone()
        packet = torch.from_numpy(x[..., pos:pos+length].copy())
        y, updated = module.decode(packet, state)
        assert torch.equal(state, before)
        assert updated.data_ptr() != state.data_ptr()
        assert updated.is_contiguous()
        assert tuple(y.shape) == (1, 2, length*stride)
        if length:
            assert torch.equal(updated, packet[..., -1:])
            assert updated.data_ptr() != packet[..., -1:].data_ptr()
        else:
            assert torch.equal(updated, before)
        chunks.append(y)
        state = updated
        pos += length
    actual = torch.cat(chunks, dim=-1).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
    np.testing.assert_array_equal(x, original_x)
    assert torch.equal(module.packed_weight, packed) and module.packed_weight.data_ptr() == packed_pointer
    joined = torch.from_numpy(np.concatenate((initial, x), axis=-1))
    old_full = torch.nn.functional.conv_transpose1d(
        joined, torch.from_numpy(weight), torch.from_numpy(bias), stride=stride)
    torch.testing.assert_close(torch.from_numpy(actual), old_full[..., stride:8*stride],
                               atol=2e-7, rtol=2e-6)
    changed = x.copy(); changed[..., 4:] += np.float32(0.7)
    prefix, _ = module.decode(torch.from_numpy(changed), torch.from_numpy(initial.copy()))
    np.testing.assert_allclose(prefix.numpy()[..., :4*stride], actual[..., :4*stride], atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("stride", [2, 5, 6, 8])
def test_transpose_bias_added_once_and_activated_input_retained(stride, paired):
    # Distinct current/previous tap values catch phase swaps and bias retention.
    weight = np.concatenate((np.ones((1, 1, stride), np.float32),
                             np.full((1, 1, stride), 2, np.float32)), axis=-1)
    module = decoder._CausalTranspose1d(weight, np.array([8], np.float32), stride=stride, paired=paired)
    x = torch.ones(1, 1, 1)
    first, state = module.decode(x, torch.zeros(1, 1, 1))
    assert torch.equal(first, torch.full((1, 1, stride), 9.0))
    assert torch.equal(state, x) and state.data_ptr() != x.data_ptr()
    saved = state.clone()
    second, next_state = module.decode(x, state)
    assert torch.equal(second, torch.full((1, 1, stride), 11.0))
    assert torch.equal(state, saved) and torch.equal(next_state, saved)
    second.fill_(99)
    assert torch.equal(next_state, saved)


@pytest.mark.parametrize("paired", [False, True])
def test_two_interleaved_streams_have_independent_transpose_histories(paired):
    module = decoder._CausalTranspose1d(array((3, 2, 4)), array((2,), 14), stride=2, paired=paired)
    a, b = array((1, 3, 5), 15), array((1, 3, 5), 16)
    ah = torch.zeros(1, 3, 1); bh = ah.clone()
    a1, ah = module.decode(torch.from_numpy(a[..., :2].copy()), ah)
    b1, bh = module.decode(torch.from_numpy(b[..., :3].copy()), bh)
    a2, ah = module.decode(torch.from_numpy(a[..., 2:].copy()), ah)
    b2, bh = module.decode(torch.from_numpy(b[..., 3:].copy()), bh)
    afull, _ = module.decode(torch.from_numpy(a), torch.zeros(1, 3, 1))
    bfull, _ = module.decode(torch.from_numpy(b), torch.zeros(1, 3, 1))
    torch.testing.assert_close(torch.cat((a1, a2), dim=-1), afull, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(torch.cat((b1, b2), dim=-1), bfull, atol=2e-7, rtol=2e-6)
    assert torch.equal(ah, torch.from_numpy(a[..., -1:]))
    assert torch.equal(bh, torch.from_numpy(b[..., -1:]))


def test_model_initializes_original_history_and_rejects_projected_history():
    stage = decoder._Stage(np.ones(3, np.float32), np.zeros(3, np.float32),
                           decoder._Snake(np.ones(3, np.float32), np.ones(3, np.float32)),
                           decoder._CausalTranspose1d(array((3, 2, 16)), array((2,)), stride=8, paired=True), [])
    model = decoder.MPSModel(
        decoder._CausalConv1d(array((64, 1, 7)), array((64,)), groups=64),
        torch.nn.Identity(), [stage], decoder._Snake(np.ones(1, np.float32), np.ones(1, np.float32)),
        decoder._CausalConv1d(array((1, 1, 7)), array((1,))))
    empty = torch.empty(1, 64, 0)
    audio, states = model.decode(empty, {})
    assert tuple(audio.shape) == (1, 1, 0)
    assert model.state_shapes["stage0.transpose.history"] == (1, 3, 1)
    assert torch.equal(states["stage0.transpose.history"], torch.zeros(1, 3, 1))
    assert model.state_bytes == sum(value.numel()*4 for value in states.values())
    _, preserved = model.decode(empty, states)
    assert all(torch.equal(value, preserved[key]) and value.data_ptr() != preserved[key].data_ptr()
               for key, value in states.items())
    states["stage0.transpose.history"] = torch.zeros(1, 2, 8)
    with pytest.raises(ValueError, match="State shape"):
        model.decode(empty, states)


@pytest.mark.parametrize("channels", [(3, 5), (5, 3)])
@pytest.mark.parametrize("length", [1, 2, 13])
def test_pointwise_matrix_matches_original_convolution(channels, length):
    cin, cout = channels
    weight, bias = array((cout, cin, 1), 20), array((cout,), 21)
    layer = decoder._Pointwise(weight, bias)
    x = torch.from_numpy(array((1, cin, length), 22))
    reference = torch.nn.functional.conv1d(x, torch.from_numpy(weight), torch.from_numpy(bias))
    weight.fill(99); bias.fill(99)
    torch.testing.assert_close(layer(x), reference, atol=2e-7, rtol=2e-6)


def test_snake_preserves_literal_reciprocal_and_freezes_coefficients():
    alpha = np.array([2, 3], dtype=np.float32)
    reciprocal = np.array([0.3, 0.1], dtype=np.float32)
    snake = decoder._Snake(alpha, reciprocal)
    alpha[:] = 99; reciprocal[:] = 99
    x = torch.tensor([[[0., 0.2, -0.4], [0.1, -0.3, 0.5]]], dtype=torch.float32)
    sine = torch.sin(torch.tensor([2., 3.]).reshape(1, 2, 1)*x)
    expected = x + torch.tensor([0.3, 0.1]).reshape(1, 2, 1)*(sine*sine)
    assert torch.equal(snake(x), expected)
    assert not list(snake.parameters())
    assert all(not value.requires_grad for value in snake.buffers())


def test_failed_decode_preserves_supplied_state_and_input():
    class Fail(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("injected failure after first convolution")
    model = decoder.MPSModel(
        decoder._CausalConv1d(array((64, 1, 7)), array((64,)), groups=64),
        Fail(), [], decoder._Snake(np.ones(1, np.float32), np.ones(1, np.float32)),
        decoder._CausalConv1d(array((1, 1, 7)), array((1,))))
    z = torch.from_numpy(array((1, 64, 2)))
    states = {key: torch.from_numpy(array(shape)) for key, shape in model.state_shapes.items()}
    snapshots = {key: value.clone() for key, value in states.items()}
    before = z.clone()
    with pytest.raises(RuntimeError, match="injected failure"):
        model.decode(z, states)
    assert torch.equal(z, before)
    assert all(torch.equal(states[key], value) for key, value in snapshots.items())
    with pytest.raises(ValueError, match="Partial or unknown"):
        model.decode(z, {"stem.history": states["stem.history"]})


def test_unverified_source_rejected_before_onnx_load(monkeypatch, tmp_path):
    import onnx
    def reject(path):
        raise ValueError("unverified original export")
    def forbidden(*args, **kwargs):
        pytest.fail("Weights or graph were loaded before source verification")
    monkeypatch.setattr(decoder.assets, "verify_model", reject)
    monkeypatch.setattr(onnx, "load", forbidden)
    with pytest.raises(ValueError, match="unverified original"):
        decoder.MPSModel.from_onnx(tmp_path / "audio_vae_decoder.onnx", device="cpu")
