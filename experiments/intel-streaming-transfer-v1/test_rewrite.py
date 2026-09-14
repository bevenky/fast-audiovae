"""Offline graph contracts; no audio, native loading, or model inference."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as TP
import pytest


ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


rewrite = load("intel_transfer_rewrite", Path(__file__).with_name("rewrite.py"))
fixtures = load("amd_transfer_fixtures", ROOT / "tests/test_amd_streaming_recipe.py")


def intel_graph():
    graph = fixtures._graph()
    rewrite.rules._pair(graph)
    pair = next(n for n in graph.graph.node if n.op_type == "PackedProjectionPairF32")
    pair.domain = rewrite.PAIR_DOMAIN; pair.name = "stream_first_projection_pair"
    next(x for x in graph.opset_import if x.domain == rewrite.rules.PAIR_DOMAIN).domain = rewrite.PAIR_DOMAIN
    return graph


@pytest.mark.parametrize("variant,removed,added", [("history", 24, 6), ("phase", 5, 1), ("combined", 29, 7)])
def test_transfer_preserves_intel_projection_and_unselected_payload(variant, removed, added):
    graph = intel_graph(); before = graph.SerializeToString()
    candidate, audit = rewrite.rewrite(graph, variant)
    assert graph.SerializeToString() == before
    assert all(audit["checks"].values())
    assert len(audit["removed_nodes"]) == removed and len(audit["inserted_nodes"]) == added
    assert next(n for n in candidate.graph.node if n.op_type == "PackedProjectionPairF32").domain == rewrite.PAIR_DOMAIN


@pytest.mark.parametrize("mutation", ["amd_pair", "wrong_pair_shape", "wrong_pair_attr", "extra_history_consumer", "changed_phase_order", "negative_zero_padding", "duplicate_initializer", "repeat_transfer"])
def test_transfer_rejects_different_contract(mutation):
    graph = intel_graph()
    pair = next(n for n in graph.graph.node if n.op_type == "PackedProjectionPairF32")
    if mutation == "amd_pair": pair.domain = rewrite.rules.PAIR_DOMAIN
    elif mutation == "wrong_pair_shape": graph.graph.initializer[0].dims[0] = 4096
    elif mutation == "wrong_pair_attr": next(a for a in pair.attribute if a.name == "K").i = 1024
    elif mutation == "extra_history_consumer":
        graph.graph.node.append(H.make_node("Identity", [rewrite.rules.HISTORY_NODES[0]+"_joined"], ["extra"], name="extra"))
    elif mutation == "changed_phase_order":
        phase = next(n for n in graph.graph.node if n.op_type == "PhaseSumBiasInterleaveF32")
        phase.input[0], phase.input[1] = phase.input[1], phase.input[0]
    elif mutation == "negative_zero_padding":
        next(w for w in graph.graph.initializer if w.name == "pad").CopyFrom(N.from_array(np.full((1,8192,1), -0.0, np.float32), "pad"))
    elif mutation == "duplicate_initializer": graph.graph.initializer.append(graph.graph.initializer[0])
    elif mutation == "repeat_transfer": graph, _ = rewrite.rewrite(graph, "history")
    with pytest.raises(ValueError): rewrite.rewrite(graph, "combined")


def test_shared_external_weights_roundtrip_exact(tmp_path):
    value = H.make_tensor_value_info("x", TP.FLOAT, [2])
    output = H.make_tensor_value_info("y", TP.FLOAT, [2])
    original = H.make_model(H.make_graph([H.make_node("Add", ["x", "w"], ["y"], name="add")],
        "tiny", [value], [output], [N.from_array(np.array([1.25, -2.5], np.float32), "w")]),
        opset_imports=[H.make_opsetid("", 18)])
    baseline = copy.deepcopy(original)
    onnx.save_model(baseline, tmp_path / "baseline.onnx", save_as_external_data=True,
                    all_tensors_to_one_file=True, location="weights.bin", size_threshold=0)
    stored = onnx.load(tmp_path / "baseline.onnx", load_external_data=False)
    weights_hash = rewrite.sha(tmp_path / "weights.bin")
    rewrite.save_shared(copy.deepcopy(original), tmp_path / "candidate.onnx", stored)
    loaded = onnx.load(tmp_path / "candidate.onnx")
    # ONNX materializes an explicit DEFAULT field when loading external data.
    # It changes protobuf presence, not the coefficient bytes.
    loaded.graph.initializer[0].ClearField("data_location")
    assert loaded.graph.initializer[0].SerializeToString() == original.graph.initializer[0].SerializeToString()
    assert rewrite.sha(tmp_path / "weights.bin") == weights_hash
    assert len(list(tmp_path.glob("*.bin"))) == 1


def test_prepare_rejects_unpinned_source_without_outputs(tmp_path):
    source = tmp_path / "other.onnx"; onnx.save(intel_graph(), source)
    with pytest.raises(ValueError, match="SHA256"):
        rewrite.prepare(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_prepare_emits_one_immutable_weight_file_for_all_variants(tmp_path, monkeypatch):
    source = tmp_path / "source.onnx"; original = intel_graph(); onnx.save(original, source)
    # The fixture's opaque large-weight sentinels are a graph-only contract,
    # so opt in to its digest only within this test, never the CLI.
    monkeypatch.setattr(rewrite, "SOURCE_SHA256", rewrite.sha(source))
    output = tmp_path / "screen"
    report = rewrite.prepare(source, output)
    assert set(report["variants"]) == set(rewrite.VARIANTS)
    assert len(list(output.glob("*.bin"))) == 1
    assert rewrite.sha(output / "weights.bin") == report["shared_weights"]["sha256"]
    for variant in rewrite.VARIANTS:
        stored = onnx.load(output / (variant+".onnx"), load_external_data=False)
        assert all(weight.data_location != TP.EXTERNAL for weight in stored.graph.initializer
                   if weight.data_type != TP.FLOAT)
        assert all(weight.data_type == TP.FLOAT for weight in stored.graph.initializer
                   if weight.data_location == TP.EXTERNAL)
        loaded = onnx.load(output / (variant+".onnx"))
        for weight in loaded.graph.initializer: weight.ClearField("data_location")
        expected, _ = rewrite.rewrite(original, variant)
        assert rewrite.rules._sequence(loaded.graph.initializer) == rewrite.rules._sequence(expected.graph.initializer)
        assert rewrite.rules._sequence(loaded.graph.input) == rewrite.rules._sequence(original.graph.input)
        assert rewrite.rules._sequence(loaded.graph.output) == rewrite.rules._sequence(original.graph.output)
    with pytest.raises(ValueError, match="already exists"):
        rewrite.prepare(source, output)
