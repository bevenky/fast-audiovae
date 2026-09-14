"""Production Intel graph, fallback and build provenance contracts; no inference."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import onnx
from onnx import helper as H, TensorProto as TP
import pytest
from fast_audiovae.recipes import intel_streaming as intel, amd_streaming as shared

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def graph():
    fixture = load("amd_contract_fixture", ROOT / "tests/test_amd_streaming_recipe.py")
    g = fixture._graph()
    shared._pair(g)
    p = next(n for n in g.graph.node if n.op_type == "PackedProjectionPairF32")
    p.domain = intel.PAIR_DOMAIN; p.name = "stream_first_projection_pair"
    next(x for x in g.opset_import if x.domain == shared.PAIR_DOMAIN).domain = intel.PAIR_DOMAIN
    g.graph.input.append(H.make_tensor_value_info("view_15", TP.FLOAT, [1, 1024, "T2"]))
    for weight, output in (("ncc_up_2_split_matmul_current_w", "ncc_up_2_split_matmul_current"),
                           ("ncc_up_2_split_matmul_previous_w", "ncc_up_2_split_matmul_previous_unshifted")):
        g.graph.initializer.append(TP(name=weight, data_type=TP.FLOAT, dims=[3072, 1024], float_data=[.5]))
        g.graph.node.append(H.make_node("PrecisionMatMulF32", [weight, "view_15"], [output], name=output+"_node",
            domain="fast.audiovae.precision.matrix.experimental", M=3072, K=1024, precision_mode=8, backend=1, native_abi=1, shards=1))
        g.graph.output.append(H.make_tensor_value_info(output, TP.FLOAT, [1, 3072, "T2"]))
    return g


def test_selected_graph_preserves_first_pair_weights_and_states():
    original = graph(); before = original.SerializeToString()
    selected, audit = intel.rewrite_graph(original)
    assert before == original.SerializeToString()
    assert all(audit["checks"].values())
    assert len(audit["inserted_nodes"]) == 8
    assert sum(n.domain == shared.HISTORY_DOMAIN for n in selected.graph.node) == 6
    assert sum(n.domain == shared.PHASE_DOMAIN for n in selected.graph.node) == 1
    node = next(n for n in selected.graph.node if n.domain == intel.MATRIX_DOMAIN)
    assert shared._attrs(node) == dict(M=3072, K=1024, mode=4, native_abi=1)
    assert list(node.output) == ["ncc_up_2_split_matmul_current", "ncc_up_2_split_matmul_previous_unshifted"]


def test_recipe_is_the_qualified_rewrite_except_production_operator_names():
    transfer = load("qualified_transfer", ROOT / "experiments/intel-streaming-transfer-v1/rewrite.py")
    bridge = load("qualified_bridge", ROOT / "experiments/intel-library-screen-v1/matrix/bridge.py")
    original = graph()
    reference, _ = transfer.rewrite(original, "combined")
    reference, _ = bridge.rewrite(reference, 4)
    selected, _ = intel.rewrite_graph(original)
    node = next(n for n in selected.graph.node if n.domain == intel.MATRIX_DOMAIN)
    node.domain = bridge.DOMAIN; node.name = "intel_second_projection_pair_screen"
    next(x for x in selected.opset_import if x.domain == intel.MATRIX_DOMAIN).domain = bridge.DOMAIN
    assert reference.SerializeToString() == selected.SerializeToString()


@pytest.mark.parametrize("kind", ["second_shards", "second_domain", "second_duplicate_attr", "first_pair", "state_shape", "repeated", "overridable"])
def test_rejects_changed_arithmetic_or_topology(kind):
    source = graph()
    if kind == "second_shards":
        next(a for a in source.graph.node[-1].attribute if a.name == "shards").i = 2
    elif kind == "second_domain":
        source.graph.node[-1].domain = "other"
    elif kind == "second_duplicate_attr":
        source.graph.node[-1].attribute.append(H.make_attribute("mode", 4))
        source.graph.node[-1].attribute.append(H.make_attribute("mode", 4))
    elif kind == "first_pair":
        next(n for n in source.graph.node if n.op_type == "PackedProjectionPairF32").domain = shared.PAIR_DOMAIN
    elif kind == "state_shape":
        next(v for v in source.graph.input if v.name == shared.HISTORY_NODES[0]+"_in").type.tensor_type.shape.dim[2].dim_value = 7
    elif kind == "repeated":
        source.opset_import.append(H.make_opsetid(intel.MATRIX_DOMAIN, 1))
    else:
        source.graph.input.append(H.make_tensor_value_info("ncc_up_2_split_matmul_current_w", TP.FLOAT, [3072, 1024]))
    with pytest.raises(ValueError):
        intel.rewrite_graph(source)


def test_native_arithmetic_is_qualified_source_with_only_selection_and_symbol_changes():
    screen = (ROOT / "experiments/intel-library-screen-v1/matrix/screen.cpp").read_text()
    production = (ROOT / "native/intel/streaming/matrix.cpp").read_text()
    screen = screen[screen.index('#include <oneapi'):]
    production = production[production.index('#include <oneapi'):]
    production = production.replace('iis_', 'ims_').replace('require(mode == 4, "Only accepted BRGeMM64 mode is enabled");',
                                                          'require(mode >= 0 && mode <= 4, "Invalid implementation");')
    assert production == screen
    bridge = (ROOT / "native/intel/streaming/custom_ops.cpp").read_text()
    assert 'mode==4' in bridge and 'mode>=2' not in bridge
    assert 'shape[2]==8 || shape[2]==16' in bridge
    assert 'ip_prepare(fallback0.get()' in bridge and 'ip_run_rows(fallback1.get()' in bridge
    assert 'iis_create("",3072,1024,8,mode' in bridge
    assert 'iis_create("",3072,1024,16,mode' in bridge


def test_source_closure_and_dependency_configuration_are_pinned():
    closure = json.loads((ROOT / "native/intel/streaming/sources.json").read_text())
    for relative, expected in closure["files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    pin = json.loads((ROOT / "native/intel/streaming/onednn.json").read_text())
    assert pin["version"] == "3.13.2"
    assert pin["configuration"]["ONEDNN_CPU_RUNTIME"] == "SEQ"
    assert pin["configuration"]["ONEDNN_GPU_RUNTIME"] == "NONE"
    assert pin["configuration"]["ONEDNN_EXPERIMENTAL_UKERNEL"] == "ON"
    assert pin["configuration"]["CMAKE_INSTALL_RPATH"] == "$ORIGIN"


def test_existing_build_requires_successful_real_provenance(tmp_path):
    builder = load("intel_builder", ROOT / "tools/build_intel_streaming.py")
    p = tmp_path / "provenance.json"
    p.write_text(json.dumps({"version": "3.13.2", "source_sha256": "wrong", "configure_exit": 0, "build_exit": 0}))
    with pytest.raises(ValueError, match="provenance"):
        builder.dependency(ROOT, existing=tmp_path, source=tmp_path, provenance=p)


def test_elf_policy_rejects_external_search_or_gpu(monkeypatch):
    builder = load("intel_builder_elf", ROOT / "tools/build_intel_streaming.py")
    for data in ('(RUNPATH) [/tmp/runtime]', '(NEEDED) [/some/lib.so]', '(NEEDED) [libcuda.so]', '(NEEDED) [libgomp.so.1]'):
        monkeypatch.setattr(builder.subprocess, "check_output", lambda *args, **kwargs: data)
        with pytest.raises(ValueError):
            builder.elf_metadata("unused")
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *args, **kwargs: '(RUNPATH) [$ORIGIN]\n(NEEDED) [libdnnl.so.3]')
    assert '$ORIGIN' in builder.elf_metadata("unused")


def bundle_fixture(tmp_path, monkeypatch):
    fixture = load("amd_payload_fixture", ROOT / "tests/test_amd_streaming_recipe.py")
    root, manifest, _ = fixture._bundle(tmp_path, monkeypatch)
    for key in ("SOURCE_FULL", "SOURCE_STREAM", "SELECTED_STREAM", "rewrite_graph"):
        monkeypatch.setattr(intel, key, getattr(shared, key))
    stream = manifest["streaming"]["models"]["streaming/full.onnx"]
    pair = fixture._file(root, "libs/libpaired_projection.so", b"original-pair")
    stream["additional_libraries"] = [{"library": pair["path"], "sha256": pair["sha256"]}]
    (root / "bundle.json").write_text(json.dumps(manifest))
    payload = tmp_path / "intel-payload"; payload.mkdir()
    roles = {role: fixture._file(payload, "libs/"+name, role.encode()) for role, name in intel.LIBRARIES.items()}
    for name in ("LICENSE", "THIRD-PARTY-PROGRAMS"):
        fixture._file(root, "licenses/onednn/"+name, b"notice")
    return root, manifest, (payload, {"vendor": "intel", "libraries": roles})


@pytest.mark.parametrize("runtime", ["1.29.0"])
def test_payload_augment_keeps_first_pair_batch_and_state(tmp_path, monkeypatch, runtime):
    root, old, payload = bundle_fixture(tmp_path, monkeypatch)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file() and p.name != "bundle.json"}
    intel.augment(tmp_path / "work", root, [], payload=payload, onnxruntime_version=runtime)
    new = json.loads((root / "bundle.json").read_text())
    assert old["native"] == new["native"] and old["fallback"] == new["fallback"]
    previous = old["streaming"]["models"]["streaming/full.onnx"]
    selected = new["streaming"]["models"]["streaming/full.onnx"]
    assert previous["states"] == selected["states"]
    assert selected["model"] == "streaming/decoder_intel_selected.onnx"
    assert selected["additional_libraries"][0] == previous["additional_libraries"][0]
    assert [Path(x["library"]).name for x in selected["additional_libraries"][1:]] == [
        "libintel_rawhistory.so", "libintel_phase.so", "libintel_stream_matrix_ops.so"]
    assert all((root / p).read_bytes() == data for p, data in before.items())
    assert (root / "libs/libintel_stream_matrix.so").is_file() and (root / "libs/libdnnl.so.3").is_file()


@pytest.mark.parametrize("problem", ["missing_role", "path_escape", "wrong_hash", "wrong_name", "runtime", "published", "source_changed", "license"])
def test_payload_preparation_fails_closed(tmp_path, monkeypatch, problem):
    root, _, payload = bundle_fixture(tmp_path, monkeypatch); runtime = "1.29.0"
    record = payload[1]["libraries"]["intel_stream_matrix"]
    if problem == "missing_role": del payload[1]["libraries"]["intel_stream_phase"]
    elif problem == "path_escape": record["path"] = "../outside.so"
    elif problem == "wrong_hash": record["sha256"] = "0"*64
    elif problem == "wrong_name":
        source = payload[0] / record["path"]; source.rename(source.with_name("wrong.so"))
        record["path"] = "libs/wrong.so"
    elif problem == "runtime": runtime = "1.30.0"
    elif problem == "published": (root / ".recipe-ready.json").write_text("{}")
    elif problem == "license": (root / "licenses/onednn/LICENSE").unlink()
    else: (root / "streaming/source.onnx").write_bytes(b"changed")
    before = (root / "bundle.json").read_bytes()
    with pytest.raises(ValueError):
        intel.augment(tmp_path / "work", root, [], payload=payload, onnxruntime_version=runtime)
    assert (root / "bundle.json").read_bytes() == before


def test_final_graph_mismatch_leaves_manifest_and_original_intact(tmp_path, monkeypatch):
    root, _, payload = bundle_fixture(tmp_path, monkeypatch)
    before = (root / "bundle.json").read_bytes()
    monkeypatch.setattr(intel, "SELECTED_STREAM", "0"*64)
    with pytest.raises(ValueError, match="accepted artifact"):
        intel.augment(tmp_path / "work", root, [], payload=payload, onnxruntime_version="1.29.0")
    assert (root / "bundle.json").read_bytes() == before
    assert not (root / "streaming/decoder_intel_selected.onnx").exists()
    assert not list((root / "streaming").glob("*.tmp-*"))
