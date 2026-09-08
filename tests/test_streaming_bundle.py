"""Real ORT integration for preparation, selection and persistent history."""
import json

import numpy as np
import onnx
from onnx import TensorProto as TP, helper, numpy_helper
import pytest

from fast_audiovae import load_decoder, load_streaming_decoder
from fast_audiovae.assets import sha256
from fast_audiovae.prepare_streaming import prepare_streaming


def bundle(root):
    weights = [numpy_helper.from_array(np.full((1, 64, 3), 1/192, np.float32), "w")]
    nodes = [helper.make_node("Conv", ["z", "w"], ["x0"], pads=[2, 0])]
    for i, stride in enumerate((8, 6, 5, 2, 2, 2)):
        weights.append(numpy_helper.from_array(np.full((1, 1, 2*stride), .5, np.float32), f"w{i}"))
        for name, values in (("start", [0]), ("end", [-stride]), ("axis", [2]), ("step", [1])):
            weights.append(numpy_helper.from_array(np.array(values, np.int64), f"{name}{i}"))
        nodes += [helper.make_node("ConvTranspose", [f"x{i}", f"w{i}"], [f"t{i}"],
                                  strides=[stride], pads=[0, 0], dilations=[1], group=1),
                  helper.make_node("Slice", [f"t{i}"] + [f"{n}{i}" for n in ("start", "end", "axis", "step")],
                                   [f"x{i+1}"])]
    model = helper.make_model(helper.make_graph(nodes, "test", [helper.make_tensor_value_info("z", TP.FLOAT, [1, 64, "L"])],
        [helper.make_tensor_value_info("x6", TP.FLOAT, [1, 1, "samples"])], weights),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.save(model, root / "decoder.onnx")
    (root / "bundle.json").write_text(json.dumps({"onnxruntime": "1.29.0", "fallback": "decoder.onnx", "native": {}}))


def test_prepare_and_load_stream_without_altering_original(tmp_path):
    bundle(tmp_path)
    original_hash = sha256(tmp_path / "decoder.onnx")
    with pytest.raises(RuntimeError, match="prepare-streaming"):
        load_streaming_decoder(tmp_path)
    record = prepare_streaming(tmp_path)
    assert sha256(tmp_path / "decoder.onnx") == original_hash
    assert record["version"] == 1
    stream_decoder, info = load_streaming_decoder(tmp_path, threads=1)
    full, _ = load_decoder(tmp_path, threads=1)
    latent = np.random.default_rng(13).normal(size=(1, 64, 7)).astype(np.float32)
    expected = full.run(None, {"z": latent})[0]
    with stream_decoder.streaming_decode() as stream:
        actual = np.concatenate([stream.decode_chunk(latent[..., :1]), stream.decode_chunk(latent[..., 1:3]),
                                 stream.decode_chunk(latent[..., 3:])], axis=2)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert not info["fresh_call_only"] and info["streaming"]
    assert info["state_bytes_per_stream"] > 0
    assert info["providers"] == ["CPUExecutionProvider"]
    with pytest.raises(ValueError, match="already contains"):
        prepare_streaming(tmp_path)


def test_failed_prepare_leaves_original_bundle_usable(tmp_path):
    bundle(tmp_path)
    model = onnx.load(tmp_path / "decoder.onnx")
    model.graph.node[0].op_type = "UnsupportedMixing"
    onnx.save(model, tmp_path / "decoder.onnx")
    before = (tmp_path / "bundle.json").read_bytes()
    with pytest.raises(ValueError, match="Unsupported"):
        prepare_streaming(tmp_path)
    assert (tmp_path / "bundle.json").read_bytes() == before
    assert not (tmp_path / "streaming").exists()
    assert not list(tmp_path.glob(".streaming-*"))


def test_modified_streaming_model_fails_before_load(tmp_path):
    bundle(tmp_path)
    record = prepare_streaming(tmp_path)
    model = tmp_path / record["models"]["decoder.onnx"]["model"]
    model.write_bytes(model.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="differs"):
        load_streaming_decoder(tmp_path)


def test_modified_external_source_weights_are_detected(tmp_path):
    bundle(tmp_path)
    source = tmp_path / "decoder.onnx"
    model = onnx.load(source)
    onnx.save_model(model, source, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="decoder.data", size_threshold=0)
    record = prepare_streaming(tmp_path)
    assert "decoder.data" in record["models"]["decoder.onnx"]["source_external_sha256"]
    weights = tmp_path / "decoder.data"
    weights.write_bytes(weights.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="external weights"):
        load_streaming_decoder(tmp_path)


def test_manifest_hash_mismatch_is_rejected_during_prepare(tmp_path):
    bundle(tmp_path)
    manifest_path = tmp_path / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fallback_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from its manifest"):
        prepare_streaming(tmp_path)
    assert not (tmp_path / "streaming").exists()


def test_additional_library_hash_is_checked_before_registration(tmp_path):
    bundle(tmp_path)
    prepare_streaming(tmp_path)
    manifest_path = tmp_path / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    (tmp_path / "extra.so").write_bytes(b"not a real library")
    manifest["streaming"]["models"]["decoder.onnx"]["additional_libraries"] = [
        {"library": "extra.so", "sha256": "0" * 64}]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="differs from its manifest"):
        load_streaming_decoder(tmp_path)
