"""Raw-coordinate preservation and deployment parity for the explicit new mode."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from audiovae_student.export import export_decoder, OnnxStudentStream
from audiovae_student.model import RawRepeatPhaseAdapter, StudentConfig, StudentDecoder, architecture_summary


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(1)
    torch.manual_seed(231)


def small_model(normalization="masked_batch_norm"):
    model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16,
        head_channels=16, layer_scale_init=1., normalization_mode=normalization,
        adapter_mode="raw_repeat_phase_bias"))
    with torch.no_grad():
        model.adapter.phase_bias.copy_(torch.linspace(-.2, .3, 256).reshape(64, 4))
        if normalization == "masked_batch_norm":
            model.train()
            model(torch.randn(2, 64, 9))
            for norm in (model.stem_norm, model.affine):
                norm.weight.copy_(torch.linspace(.7, 1.3, 8))
                norm.bias.copy_(torch.linspace(-.1, .2, 8))
    return model.freeze_normalization_statistics().eval()


def test_every_phase_retains_each_raw_coordinate_and_chronological_order():
    adapter = RawRepeatPhaseAdapter()
    assert set(dict(adapter.named_parameters())) == {"phase_bias"}
    assert adapter.phase_bias.shape == (64, 4)
    assert torch.count_nonzero(adapter.phase_bias) == 0
    model = small_model()
    z = torch.arange(2 * 64 * 3, dtype=torch.float64).reshape(2, 64, 3)
    model.double()
    expanded = model._phase_frames(z)
    assert expanded.shape == (2, 64, 12)
    for phase in range(4):
        torch.testing.assert_close(expanded[..., phase::4],
            z + model.adapter.phase_bias[:, phase][None, :, None], atol=0, rtol=0)


@pytest.mark.parametrize("phase", range(4))
def test_exact_identity_jacobian_including_after_bias_changes(phase):
    model = small_model().double()
    z = torch.randn(1, 64, 2, dtype=torch.float64, requires_grad=True)
    for offset in (0., .75):
        with torch.no_grad():
            model.adapter.phase_bias.add_(offset)
        jacobian = torch.autograd.functional.jacobian(
            lambda value: model._phase_frames(value)[..., phase::4], z).reshape(128, 128)
        torch.testing.assert_close(jacobian, torch.eye(128, dtype=torch.float64), atol=0, rtol=0)


def test_phase_bias_receives_independent_gradients_without_a_learned_matrix():
    model = small_model().double()
    z = torch.randn(2, 64, 3, dtype=torch.float64, requires_grad=True)
    coefficients = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    phases = model._phase_frames(z).reshape(2, 64, 3, 4)
    (phases * coefficients).sum().backward()
    torch.testing.assert_close(model.adapter.phase_bias.grad,
        coefficients[None].expand(64, -1) * 6, atol=0, rtol=0)
    torch.testing.assert_close(z.grad, torch.full_like(z, 10.), atol=0, rtol=0)
    assert model.adapter_parameter_name == "adapter.phase_bias"
    assert model.get_parameter(model.adapter_parameter_name) is model.adapter.phase_bias
    assert sum(p.numel() for p in model.adapter.parameters()) == 256


@pytest.mark.parametrize("normalization", ["affine", "masked_batch_norm"])
def test_batch_stream_fold_and_future_prefix_parity(normalization):
    model = small_model(normalization)
    z = torch.randn(2, 64, 41)
    with torch.no_grad():
        expected = model(z)
        independent = torch.cat([model(item[None]) for item in z])
        folded = model.fold_normalization()
        torch.testing.assert_close(independent, expected, atol=2e-6, rtol=1e-4)
        torch.testing.assert_close(folded(z), expected, atol=2e-6, rtol=1e-4)
        state = model.initial_state(batch_size=2)
        chunks = []
        for start, stop in ((0, 1), (1, 7), (7, 8), (8, 39), (39, 41)):
            audio, state = model.forward_stream(z[..., start:stop], state)
            assert audio.shape[-1] == (stop - start) * 1920
            chunks.append(audio)
        torch.testing.assert_close(torch.cat(chunks, -1), expected, atol=2e-6, rtol=1e-4)
        changed = z.clone()
        changed[..., 9:] = 100 * torch.randn_like(changed[..., 9:])
        torch.testing.assert_close(model(changed)[..., :9 * 1920], expected[..., :9 * 1920],
                                   atol=0, rtol=0)
        with folded.stream() as stream:
            empty = stream.decode_chunk(z[:1, :, :0])
            assert empty.shape == (1, 1, 0) and stream.frames_decoded == 0
            pieces = [stream.decode_chunk(z[:1, :, start:stop])
                      for start, stop in ((0, 1), (1, 10), (10, 41))]
            assert stream.frames_decoded == 41 and stream.flush().shape[-1] == 0
            torch.testing.assert_close(torch.cat(pieces, -1), expected[:1], atol=2e-6, rtol=1e-4)


def test_complete_decoder_backpropagates_to_phase_bias_and_preserves_normalization():
    model = small_model().train()
    before = {name: value.clone() for name, value in model.named_buffers()}
    z = torch.randn(2, 64, 5, requires_grad=True)
    model(z).square().mean().backward()
    assert model.adapter.phase_bias.grad is not None
    assert torch.isfinite(model.adapter.phase_bias.grad).all()
    assert torch.count_nonzero(model.adapter.phase_bias.grad) > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_legacy_configuration_and_state_remain_exact_and_new_mode_is_explicit():
    legacy = StudentConfig(hidden_channels=8, expansion_channels=16, head_channels=16)
    historical = legacy.to_dict()
    assert "adapter_mode" not in historical
    restored = StudentConfig(**historical)
    assert restored == legacy and restored.adapter_mode == "learned_phase"
    old_model = StudentDecoder(restored)
    old_copy = StudentDecoder(StudentConfig(**historical))
    old_copy.load_state_dict(deepcopy(old_model.state_dict()), strict=True)
    z = torch.randn(1, 64, 4)
    torch.testing.assert_close(old_model(z), old_copy(z), atol=0, rtol=0)
    assert old_model.adapter_parameter_name == "adapter.weight"
    new = small_model()
    explicit = new.config.to_dict()
    assert explicit["adapter_mode"] == "raw_repeat_phase_bias"
    new_copy = StudentDecoder(StudentConfig(**explicit)).eval()
    new_copy.load_state_dict(deepcopy(new.state_dict()), strict=True)
    torch.testing.assert_close(new(z), new_copy(z), atol=0, rtol=0)
    with pytest.raises(RuntimeError):
        new_copy.load_state_dict(old_model.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        old_copy.load_state_dict(new.state_dict(), strict=True)
    with pytest.raises(ValueError, match="adapter_mode"):
        StudentConfig(adapter_mode="unknown")


def test_summary_removes_the_adapter_matrix_work_only():
    legacy = architecture_summary()
    new = architecture_summary(StudentConfig(adapter_mode="raw_repeat_phase_bias"))
    assert legacy["conv_linear_macs_per_audio_second"] - new["conv_linear_macs_per_audio_second"] == 25 * 64 * 256
    assert new["adapter"]["matrix_macs_per_audio_second"] == 0
    assert new["adapter"]["trainable_parameters"] == 256
    assert new["history_internal_frames"] == legacy["history_internal_frames"] == 116
    assert new["state_fp32_bytes_per_stream"] == legacy["state_fp32_bytes_per_stream"]


def test_export_preserves_new_mode_and_adds_no_adapter_matrix(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = small_model()
    before = deepcopy(model.state_dict())
    manifest = export_decoder(model, tmp_path, provenance={"qualification": "synthetic_adapter_parity_only"})
    assert manifest["architecture"]["config"]["adapter_mode"] == "raw_repeat_phase_bias"
    graph = onnx.load(str(tmp_path / "decoder.onnx"))
    assert not any("adapter" in node.name and node.op_type in {"Conv", "MatMul", "Gemm"}
                   for node in graph.graph.node)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    full = ort.InferenceSession(str(tmp_path / "decoder.onnx"), options, providers=["CPUExecutionProvider"])
    z = np.random.default_rng(2).standard_normal((1, 64, 9)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(z)).numpy()
    np.testing.assert_allclose(full.run(None, {"latents": z})[0], expected, atol=2e-6, rtol=1e-4)
    if ort.__version__ == "1.29.0":
        stream = OnnxStudentStream(tmp_path)
        pieces = [stream.decode_chunk(z[..., start:stop]) for start, stop in ((0, 1), (1, 4), (4, 9))]
        np.testing.assert_allclose(np.concatenate(pieces, -1), expected, atol=2e-6, rtol=1e-4)
        assert stream.frames_decoded == 9
    else:
        # Exercise the exported state graph with the installed CPU runtime;
        # the production wrapper's explicit version requirement stays intact.
        session = ort.InferenceSession(str(tmp_path / "decoder-stream.onnx"), options,
                                       providers=["CPUExecutionProvider"])
        histories = {item["input"]: np.zeros(item["shape"], np.float32) for item in manifest["states"]}
        started = np.zeros(1, np.bool_)
        pieces = []
        for start, stop in ((0, 1), (1, 4), (4, 9)):
            audio, started, *buffers = session.run(None,
                {"latents": z[..., start:stop].copy(), "started": started, **histories})
            histories = {item["input"]: value for item, value in zip(manifest["states"], buffers)}
            pieces.append(audio)
        np.testing.assert_allclose(np.concatenate(pieces, -1), expected, atol=2e-6, rtol=1e-4)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
