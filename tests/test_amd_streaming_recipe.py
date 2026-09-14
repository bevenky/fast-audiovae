"""Offline graph/payload contracts; no native loading, model inference or build."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as TP
import pytest

from fast_audiovae.recipes import amd_streaming as amd


def _value(name, shape):
    return H.make_tensor_value_info(name, TP.FLOAT, shape)


def _graph():
    # These two opaque protobuf fixtures carry declared weight geometry and one
    # sentinel value, avoiding134MiB per weight in a graph-only contract test.
    # They are deliberately never materialized as NumPy or used for inference.
    constants = [TP(name=n, data_type=TP.FLOAT, dims=[8192, 2048], float_data=[v])
                 for n, v in (("wc", 1.0), ("wp", -1.0))]
    def const(name, array):
        constants.append(N.from_array(np.asarray(array), name)); return name
    maxend = const("end", np.array([9223372036854775807], np.int64))
    axis = const("axis", np.array([2], np.int64)); step = const("step", np.array([1], np.int64))
    def slice_node(name, src, dst, start):
        begin = const(name+"_start", np.array([start], np.int64))
        return H.make_node("Slice", [src, begin, maxend, axis, step], [dst], name=name)
    inputs = [_value("x", [1, 2048, "T"]), _value("x512", [1, 512, "T512"]), _value("phase_in", [1, 8192, 1])]
    outputs = [_value("y", [1, 512, "T512"]), _value("phase_out", [1, 8192, 1])]
    attrs = dict(K=2048, M=8192, backend=1, native_abi=1, precision_mode=8, shards=1)
    nodes = [H.make_node("PrecisionMatMulF32", [w, "x"], [branch], name="first_"+branch,
                        domain="fast.audiovae.precision.matrix.experimental", **attrs)
             for w, branch in (("wc", "current"), ("wp", "previous"))]
    zero = const("pad", np.zeros((1, 8192, 1), np.float32)); bias = const("bias", np.zeros(1024, np.float32))
    nodes += [H.make_node("Concat", ["phase_in", "previous"], ["previous_pad"], name="prev_join", axis=2),
              slice_node("phase_retain", "previous_pad", "phase_out", -1),
              H.make_node("Concat", [zero, "current"], ["current_pad"], name="cur_join", axis=2),
              H.make_node("PhaseSumBiasInterleaveF32", ["current_pad", "previous_pad", bias], ["padded_audio"],
                          name="ncc_phase_finish_17_op", domain="venky.audio.cpu.portable",
                          channels=1024, stride=8, native_abi=1, previous_shift=1, row_batches=0),
              slice_node("phase_crop", "padded_audio", "phase_audio", 8)]
    x = "phase_audio"
    for index, name in enumerate(amd.HISTORY_NODES):
        c = 1024 if index < 3 else 512; d = (1, 3, 9)[index % 3]; halo = 6*d
        if index == 3: x = "x512"
        hist, next_hist = name+"_in", name+"_out"
        inputs.append(_value(hist, [1, c, halo])); outputs.append(_value(next_hist, [1, c, halo]))
        coeff = [const(name+"_w", np.zeros((c, 1, 7), np.float32))]
        coeff += [const(name+"_c"+str(i), np.full(c, .25 if i else 0, np.float32)) for i in range(5)]
        nodes += [H.make_node("Concat", [hist, x], [name+"_joined"], name=name+"_join", axis=2),
                  slice_node(name+"_retain", name+"_joined", next_hist, -halo),
                  H.make_node("SnakeDW7SnakeF32", [name+"_joined", *coeff], [name+"_raw"], name=name,
                              domain="venky.audio.cpu.portable", native_abi=1, channels=c, dilation=d,
                              backend=5, row_batches=0, require_vector_sine=1),
                  slice_node(name+"_crop", name+"_raw", name+"_audio", halo)]
        x = name+"_audio"
    nodes.append(H.make_node("Identity", [x], ["y"], name="output"))
    return H.make_model(H.make_graph(nodes, "contracts_only", inputs, outputs, constants),
                        opset_imports=[H.make_opsetid("", 18), H.make_opsetid("venky.audio.cpu.portable", 1),
                                       H.make_opsetid("fast.audiovae.precision.matrix.experimental", 1)])


def test_exact_selected_topology_preserves_weights_io_and_source():
    source = _graph(); before = source.SerializeToString()
    result, audit = amd.rewrite_graph(source)
    assert source.SerializeToString() == before
    for field in ("initializer", "input", "output"):
        assert amd._sequence(getattr(source.graph, field)) == amd._sequence(getattr(result.graph, field))
    assert all(audit["checks"].values())
    assert len(audit["inserted_nodes"]) == 8
    pair = next(n for n in result.graph.node if n.domain == amd.PAIR_DOMAIN)
    assert list(pair.input) == ["wc", "wp", "x"] and list(pair.output) == ["current", "previous"]
    histories = [n for n in result.graph.node if n.domain == amd.HISTORY_DOMAIN]
    assert len(histories) == 6 and all(len(n.input) == 8 and len(n.output) == 2 for n in histories)
    phase = next(n for n in result.graph.node if n.domain == amd.PHASE_DOMAIN)
    assert list(phase.input) == ["current", "previous", "phase_in", "bias"]
    assert list(phase.output) == ["phase_audio", "phase_out"]
    assert amd._attrs(phase) == {"channels": 1024, "stride": 8, "phase_state_abi": 1, "threads": 1}
    assert [n.name for n in result.graph.node if n.op_type == "Identity"] == ["output"]


def test_history_and_phase_match_the_frozen_accepted_rewrites():
    repo = Path(__file__).resolve().parents[1]
    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, repo / relative)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module
    a2 = load("accepted_a2_graph", "experiments/amd-fp32-streaming-v1/a2/rewrite.py")
    a4 = load("accepted_a4_graph", "experiments/amd-fp32-streaming-v1/a4/prepare.py")
    source = _graph(); legacy = copy.deepcopy(source); amd._pair(legacy)
    regions = [a2.inspect(legacy, name) for name in amd.HISTORY_NODES]
    removed = {name for r in regions for name in r["remove"]}
    replacement = {r["remove"][-1]: a2.new_node(r) for r in regions}
    nodes = [replacement[n.name] if n.name in replacement else n for n in legacy.graph.node
             if n.name in replacement or n.name not in removed]
    del legacy.graph.node[:]; legacy.graph.node.extend(nodes)
    live = {x for n in nodes for x in (*n.input, *n.output)} | {v.name for v in (*legacy.graph.input, *legacy.graph.output)}
    infos = [v for v in legacy.graph.value_info if v.name in live]
    del legacy.graph.value_info[:]; legacy.graph.value_info.extend(infos)
    legacy.opset_import.append(H.make_opsetid(a2.DOMAIN, 1)); legacy, _ = a4.rewrite(legacy)
    actual, _ = amd.rewrite_graph(source)
    assert actual.SerializeToString() == legacy.SerializeToString()


@pytest.mark.parametrize("mutation", ["matrix_shards", "different_activation", "history_shape", "extra_consumer", "negative_zero_pad", "repeat_domain", "overridable"])
def test_rejects_malformed_or_different_contract(mutation):
    graph = _graph()
    if mutation == "matrix_shards":
        next(a for a in graph.graph.node[0].attribute if a.name == "shards").i = 4
    elif mutation == "different_activation":
        graph.graph.node[1].input[1] = "x512"
    elif mutation == "history_shape":
        next(v for v in graph.graph.input if v.name == amd.HISTORY_NODES[0]+"_in").type.tensor_type.shape.dim[2].dim_value = 7
    elif mutation == "extra_consumer":
        graph.graph.node.append(H.make_node("Identity", [amd.HISTORY_NODES[0]+"_joined"], ["extra"], name="extra"))
    elif mutation == "negative_zero_pad":
        next(w for w in graph.graph.initializer if w.name == "pad").CopyFrom(N.from_array(np.full((1, 8192, 1), -0.0, np.float32), "pad"))
    elif mutation == "repeat_domain":
        graph.opset_import.append(H.make_opsetid(amd.HISTORY_DOMAIN, 1))
    else:
        graph.graph.input.append(_value("wc", [8192, 2048]))
    with pytest.raises(ValueError):
        amd.rewrite_graph(graph)


def _file(root, name, data):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}


def _bundle(tmp_path, monkeypatch):
    root = tmp_path / "bundle"; root.mkdir()
    states = [{"input": "state"+str(i), "output": "state"+str(i)+"_out", "shape": [1, 1, 1 if i < 17 else 170799],
               "dtype": "float32", "time_axis": 2} for i in range(18)]
    simple = H.make_model(H.make_graph([H.make_node("Identity", ["x"], ["y"], name="identity")]+
        [H.make_node("Identity", [s["input"]], [s["output"]], name=s["input"]+"_copy") for s in states], "tiny",
        [_value("x", [1, 1, "T"])]+[_value(s["input"], s["shape"]) for s in states],
        [_value("y", [1, 1, "T"])]+[_value(s["output"], s["shape"]) for s in states]))
    source = simple.SerializeToString(); final = copy.deepcopy(simple); final.graph.name = "selected"
    _file(root, "streaming/full.onnx", b"full"); _file(root, "streaming/source.onnx", source)
    _file(root, "portable.onnx", b"fallback"); base = _file(root, "libs/base.so", b"base")
    monkeypatch.setattr(amd, "SOURCE_FULL", amd.sha256(root / "streaming/full.onnx"))
    monkeypatch.setattr(amd, "SOURCE_STREAM", amd.sha256(root / "streaming/source.onnx"))
    monkeypatch.setattr(amd, "SELECTED_STREAM", hashlib.sha256(final.SerializeToString()).hexdigest())
    monkeypatch.setattr(amd, "rewrite_graph", lambda original: (copy.deepcopy(final), {"checks": {"fixture": True}}))
    stream = {"model": "streaming/source.onnx", "model_sha256": amd.SOURCE_STREAM,
              "state_bytes": 683264, "states": states, "latent_input": "x", "audio_output": "y",
              "required_streaming_operators": [{"domain": "old", "operator": "Old"}]}
    native = {"model": "streaming/full.onnx", "model_sha256": amd.SOURCE_FULL, "library": base["path"],
              "library_sha256": base["sha256"], "additional_libraries": []}
    manifest = {"onnxruntime": "1.29.0", "native": {"Linux/x86_64": native},
                "fallback": "portable.onnx", "fallback_sha256": amd.sha256(root / "portable.onnx"),
                "streaming": {"models": {native["model"]: stream}}}
    (root / "bundle.json").write_text(json.dumps(manifest))
    payload = tmp_path / "payload"; payload.mkdir()
    roles = {role: _file(payload, "libs/"+name, role.encode()) for role, name in amd.LIBRARIES.items()}
    return root, manifest, (payload, {"vendor": "amd", "libraries": roles})


@pytest.mark.parametrize("runtime", ["1.29.0", "1.30.0"])
def test_prebuilt_augment_keeps_full_fallback_state_and_source(tmp_path, monkeypatch, runtime):
    root, old, payload = _bundle(tmp_path, monkeypatch)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file() and p.name != "bundle.json"}
    amd.augment(tmp_path / "work", root, [], payload=payload, onnxruntime_version=runtime)
    new = json.loads((root / "bundle.json").read_text())
    assert new["native"] == old["native"] and new["fallback"] == old["fallback"]
    stream = new["streaming"]["models"]["streaming/full.onnx"]
    assert new["onnxruntime"] == runtime and stream["states"] == old["streaming"]["models"]["streaming/full.onnx"]["states"]
    assert stream["model"] == "streaming/decoder_amd_selected.onnx" and stream["model_sha256"] == amd.SELECTED_STREAM
    assert stream["amd_stream_selected"]["onnxruntime"] == runtime
    assert [Path(x["library"]).name for x in stream["additional_libraries"]] == list(amd.LIBRARIES.values())
    assert all((root / p).read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("problem", ["missing_role", "path_escape", "wrong_hash", "wrong_name", "runtime", "published", "source_changed"])
def test_preparation_rejects_incomplete_or_untrusted_payload(tmp_path, monkeypatch, problem):
    root, _, payload = _bundle(tmp_path, monkeypatch); runtime = "1.30.0"
    if problem == "missing_role": del payload[1]["libraries"]["amd_stream_phase"]
    elif problem == "path_escape": payload[1]["libraries"]["amd_stream_phase"]["path"] = "../outside.so"
    elif problem == "wrong_hash": payload[1]["libraries"]["amd_stream_phase"]["sha256"] = "0"*64
    elif problem == "wrong_name":
        payload[1]["libraries"]["amd_stream_phase"] = _file(payload[0], "libs/other.so", b"other")
    elif problem == "runtime": runtime = "1.31.0"
    elif problem == "published": (root / ".recipe-ready.json").write_text("{}")
    else: (root / "streaming/source.onnx").write_bytes(b"changed")
    before = (root / "bundle.json").read_bytes()
    with pytest.raises(ValueError):
        amd.augment(tmp_path / "work", root, [], payload=payload, onnxruntime_version=runtime)
    assert (root / "bundle.json").read_bytes() == before


def test_final_hash_failure_keeps_manifest_and_source(tmp_path, monkeypatch):
    root, _, payload = _bundle(tmp_path, monkeypatch); before = (root / "bundle.json").read_bytes()
    monkeypatch.setattr(amd, "SELECTED_STREAM", "0"*64)
    with pytest.raises(ValueError, match="accepted artifact"):
        amd.augment(tmp_path / "work", root, [], payload=payload, onnxruntime_version="1.30.0")
    assert (root / "bundle.json").read_bytes() == before
    assert not (root / "streaming/decoder_amd_selected.onnx").exists()
    assert not list((root / "streaming").glob("*.tmp-*"))


def test_native_closure_is_exact_accepted_sources():
    repo = Path(__file__).resolve().parents[1]; closure = repo / "native/amd/streaming"
    pins = json.loads((closure / "sources.json").read_text())
    for name, row in pins["files"].items():
        assert amd.sha256(closure / name) == row["sha256"] == amd.sha256(repo / row["origin"])
    for name, digest in pins["base_sources"].items():
        assert amd.sha256(repo / name) == digest
