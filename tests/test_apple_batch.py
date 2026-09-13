"""Stateless boundary conversion only; no native library or inference calls."""
import copy
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from fast_audiovae.recipes import apple_batch as batch


def graph(shapes=((1, 3, 6), (1, 2, 18))):
    inputs = [helper.make_tensor_value_info("latent", TensorProto.FLOAT, [1, 64, "L"])]
    states, nodes, history_outputs = [], [], []
    for i, shape in enumerate(shapes):
        before, after = "history_" + str(i), "next_" + str(i)
        states.append({"input": before, "output": after, "shape": list(shape), "dtype": "float32"})
        inputs.append(helper.make_tensor_value_info(before, TensorProto.FLOAT, shape))
        history_outputs.append(helper.make_tensor_value_info(after, TensorProto.FLOAT, shape))
        nodes.append(helper.make_node("StatefulFixture", ["latent", before, "trained"],
                     ["audio" if i == len(shapes) - 1 else "intermediate_" + str(i), after],
                     name="state_node_" + str(i), domain="fixture.state", threads=1))
    outputs = [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, "samples"]), *history_outputs]
    weights = numpy_helper.from_array(np.array([[1.125, -0.0], [3.0, -4.0]], np.float32), "trained")
    model = helper.make_model(helper.make_graph(nodes, "fixture", inputs, outputs, [weights]),
                              opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("fixture.state", 1)])
    return model, {"latent_input": "latent", "audio_output": "audio", "states": states}


def test_specialize_preserves_every_node_output_slot_and_trained_byte():
    model, spec = graph()
    original, original_spec = model.SerializeToString(), copy.deepcopy(spec)
    converted, audit = batch.specialize(model, spec)
    assert model.SerializeToString() == original and spec == original_spec
    assert [x.name for x in converted.graph.input] == ["latent"]
    assert [x.name for x in converted.graph.output] == ["audio"]
    assert [n.SerializeToString() for n in converted.graph.node] == [n.SerializeToString() for n in model.graph.node]
    assert [list(n.output) for n in converted.graph.node] == [["intermediate_0", "next_0"], ["audio", "next_1"]]
    assert converted.graph.initializer[0].SerializeToString() == model.graph.initializer[0].SerializeToString()
    assert set(i.name for i in converted.graph.initializer).isdisjoint(i.name for i in converted.graph.input)
    for state, value in zip(spec["states"], converted.graph.initializer[1:]):
        array = numpy_helper.to_array(value)
        assert value.name == state["input"] and list(array.shape) == state["shape"]
        assert array.dtype == np.float32 and not array.view(np.uint32).any()
    assert audit["state_tensors"] == 2 and audit["state_bytes"] == (3 * 6 + 2 * 18) * 4
    assert audit["kernel_output_arity_preserved"] and audit["independent_calls"]
    assert audit["prepended_latent_frames"] == 0
    onnx.checker.check_model(converted)


@pytest.mark.parametrize("mutation,pattern", [
    (lambda m, s: s.update(states=[]), "Explicit initial history"),
    (lambda m, s: s["states"].append(copy.deepcopy(s["states"][0])), "Duplicate history"),
    (lambda m, s: s["states"][0].update(dtype="float16"), "float32 shape"),
    (lambda m, s: s["states"][0].update(shape=[1, 3, 7]), "shape differs"),
    (lambda m, s: s["states"][0].update(shape=[True, 3, 6]), "float32 shape"),
    (lambda m, s: s["states"][0].update(input="latent"), "overlaps latent"),
    (lambda m, s: s["states"][0].update(output="missing"), "endpoint missing"),
    (lambda m, s: m.graph.input.append(helper.make_tensor_value_info("extra", TensorProto.FLOAT, [1])), "undeclared"),
    (lambda m, s: m.graph.initializer.append(numpy_helper.from_array(np.zeros((1, 3, 6), np.float32), "history_0")), "override initializers"),
    (lambda m, s: setattr(m.graph.input[1].type.tensor_type, "elem_type", TensorProto.DOUBLE), "Float32 graph interface"),
])
def test_malformed_history_contract_rejected_without_source_mutation(mutation, pattern):
    model, spec = graph()
    mutation(model, spec)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match=pattern):
        batch.specialize(model, spec)
    assert model.SerializeToString() == before


def bundle(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "runtime").mkdir(parents=True)
    (source / "licenses").mkdir()
    (source / "licenses/dependencies.txt").write_text("Retained license fixture")
    # Tiny trained graph; the native contract's exact history size is retained.
    shapes = [(1, 1, 170816 - 25)] + [(1, 1, 1)] * 25
    model, spec = graph(shapes)
    onnx.save(model, source / "stream.onnx")
    stream_sha = batch.sha(source / "stream.onnx")
    monkeypatch.setattr(batch, "SELECTED_GRAPH", stream_sha)
    (source / "full.onnx").write_bytes(b"unchanged full reference fixture")
    (source / "fallback.onnx").write_bytes(b"unchanged portable fallback fixture")
    (source / "runtime/base.dylib").write_bytes(b"inert base; never loaded")
    additional = []
    for i in range(11):
        name = "runtime/operator_" + str(i) + ".dylib"
        (source / name).write_bytes(("inert operator " + str(i)).encode())
        additional.append({"library": name, "domain": "fixture." + str(i), "sha256": batch.sha(source / name)})
    (source / "runtime/dependency.dylib").write_bytes(b"inert transitive dependency")
    dependencies = [{"library": "runtime/dependency.dylib", "sha256": batch.sha(source / "runtime/dependency.dylib")}]
    spec.update(model="stream.onnx", model_sha256=stream_sha, source_sha256=batch.sha(source / "full.onnx"),
                source_external_sha256={}, additional_libraries=additional, dependencies=dependencies,
                apple_stream_selected={"selected_reference_graph_sha256": stream_sha})
    manifest = {"onnxruntime": "1.30.0", "fallback": "fallback.onnx",
        "fallback_sha256": batch.sha(source / "fallback.onnx"),
        "native": {"Darwin/arm64": {"model": "full.onnx", "library": "runtime/base.dylib",
             "library_sha256": batch.sha(source / "runtime/base.dylib"),
             "model_sha256": batch.sha(source / "full.onnx"),
             "required_cpu_features": ["sme", "sme2"], "dependencies": dependencies, "math": "vforce"}},
        "streaming": {"version": 1, "models": {"full.onnx": spec}}}
    (source / "bundle.json").write_text(json.dumps(manifest))
    (source / ".recipe-ready.json").write_text("source completion marker")
    return source, manifest


def test_prepare_creates_independent_batch_entry_and_preserves_entire_source(tmp_path, monkeypatch):
    source, manifest = bundle(tmp_path, monkeypatch)
    before = {str(p.relative_to(source)): batch.sha(p) for p in source.rglob("*") if p.is_file()}
    destination = tmp_path / "batch"
    audit = batch.prepare(source, destination)
    after = {str(p.relative_to(source)): batch.sha(p) for p in source.rglob("*") if p.is_file()}
    assert before == after
    converted = onnx.load(destination / batch.MODEL)
    assert [v.name for v in converted.graph.input] == ["latent"]
    assert [v.name for v in converted.graph.output] == ["audio"]
    assert len(converted.graph.initializer) == 27
    assert all(len(n.output) == 2 for n in converted.graph.node)
    result = json.loads((destination / "bundle.json").read_text())
    native = result["native"]["Darwin/arm64"]
    assert native["model"] == batch.MODEL and native["apple_batch_selected"] == audit
    assert audit["model_sha256"] == batch.sha(destination / batch.MODEL)
    assert native["model_sha256"] == audit["model_sha256"]
    assert native["model_sha256"] != manifest["native"]["Darwin/arm64"]["model_sha256"]
    assert audit["source_bundle_manifest_sha256"] == before["bundle.json"]
    assert (audit["state_tensors"], audit["state_bytes"]) == (26, 683264)
    assert result["fallback"] == manifest["fallback"]
    assert result["streaming"] == manifest["streaming"]
    assert batch.MODEL not in result["streaming"]["models"]
    original_spec = manifest["streaming"]["models"]["full.onnx"]
    assert native["additional_libraries"] == original_spec["additional_libraries"]
    assert native["dependencies"] == original_spec["dependencies"]
    assert native["required_cpu_features"] == ["sme", "sme2"]
    for name, digest in before.items():
        if name not in ("bundle.json", ".recipe-ready.json"):
            assert batch.sha(destination / name) == digest
    assert not (destination / ".recipe-ready.json").exists()
    with pytest.raises(ValueError, match="New separate"):
        batch.prepare(source, destination)


@pytest.mark.parametrize("corrupt", ["stream", "library", "base_library", "fallback", "closure", "state_count"])
def test_prepare_rejects_changed_identity_before_destination_created(tmp_path, monkeypatch, corrupt):
    source, manifest = bundle(tmp_path, monkeypatch)
    if corrupt == "stream":
        monkeypatch.setattr(batch, "SELECTED_GRAPH", "0" * 64)
    elif corrupt == "library":
        (source / "runtime/operator_3.dylib").write_bytes(b"changed")
    elif corrupt == "base_library":
        (source / "runtime/base.dylib").write_bytes(b"changed")
    elif corrupt == "fallback":
        (source / "fallback.onnx").write_bytes(b"changed")
    elif corrupt == "closure":
        spec = manifest["streaming"]["models"]["full.onnx"]
        spec["additional_libraries"].append(copy.deepcopy(spec["additional_libraries"][0]))
        (source / "bundle.json").write_text(json.dumps(manifest))
    else:
        model, spec = graph()
        onnx.save(model, source / "stream.onnx")
        old = manifest["streaming"]["models"]["full.onnx"]
        old.update(spec, model_sha256=batch.sha(source / "stream.onnx"))
        monkeypatch.setattr(batch, "SELECTED_GRAPH", old["model_sha256"])
        (source / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        batch.prepare(source, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
