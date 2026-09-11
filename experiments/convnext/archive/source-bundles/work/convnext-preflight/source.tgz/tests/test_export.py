import json

import numpy as np
import onnxruntime as ort
import pytest
import torch

from audiovae_student.export import OnnxStudentStream, export_decoder
from audiovae_student.model import StudentConfig, StudentDecoder


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.set_num_threads(1)
    torch.manual_seed(81)
    model = StudentDecoder(StudentConfig(hidden_channels=16, expansion_channels=32,
                                        head_channels=32, layer_scale_init=1.0)).eval()
    path = tmp_path_factory.mktemp("exported-student")
    manifest = export_decoder(model, path, provenance={"qualification": "untrained_test_only"})
    assert not model.training
    assert manifest["architecture"]["initialization"] == "fresh"
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path / "decoder.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    return model, path, session


@pytest.mark.parametrize("frames,chunk", [(1, 1), (7, 2), (37, 4)])
def test_export_dynamic_lengths_and_streaming_match_pytorch(exported, frames, chunk):
    model, path, full = exported
    z = np.random.default_rng(frames).normal(size=(1, 64, frames)).astype(np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(z)).numpy()
    np.testing.assert_allclose(full.run(["audio"], {"latents": z})[0], expected, atol=1e-6, rtol=1e-4)
    stream = OnnxStudentStream(path)
    parts = [stream.decode_chunk(z[..., offset:offset + chunk]) for offset in range(0, frames, chunk)]
    np.testing.assert_allclose(np.concatenate(parts, -1), expected, atol=1e-6, rtol=1e-4)
    assert stream.frames_decoded == frames
    assert stream.flush().shape == (1, 1, 0)
    assert stream.session.get_providers() == ["CPUExecutionProvider"]


def test_export_stream_lifecycle_and_output_ownership(exported):
    _, path, _ = exported
    stream = OnnxStudentStream(path)
    z = np.random.default_rng(10).normal(size=(1, 64, 3)).astype(np.float32)
    first = stream.decode_chunk(z)
    original = first.copy()
    stream.decode_chunk(z)
    np.testing.assert_array_equal(first, original)
    history = stream.history
    with pytest.raises(ValueError):
        stream.decode_chunk(z.astype(np.float64))
    assert stream.history is history and stream.frames_decoded == 6
    stream.reset()
    assert stream.decode_chunk(z[..., :0]).size == 0
    assert not stream.started.any()
    np.testing.assert_array_equal(stream.decode_chunk(z), first)
    stream.close()
    with pytest.raises(RuntimeError):
        stream.decode_chunk(z)


def test_export_preserves_existing_artifacts(exported):
    model, path, _ = exported
    manifest = (path / "bundle.json").read_bytes()
    with pytest.raises(FileExistsError):
        export_decoder(model, path, provenance={"qualification": "test"})
    assert (path / "bundle.json").read_bytes() == manifest


def test_graph_checksum_is_checked_before_execution(exported, tmp_path):
    _, path, _ = exported
    spec = json.loads((path / "bundle.json").read_text())
    (tmp_path / "bundle.json").write_text(json.dumps(spec))
    (tmp_path / "decoder-stream.onnx").write_bytes(b"invalid graph")
    with pytest.raises(ValueError, match="checksum"):
        OnnxStudentStream(tmp_path)
