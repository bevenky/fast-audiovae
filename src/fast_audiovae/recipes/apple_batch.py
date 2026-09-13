"""Use the selected causal kernels for independent, zero-history full calls.

Only the graph boundary changes. Custom nodes still produce their history
outputs, including output slot 1; these outputs are no longer caller-visible.
No zero latent frames are prepended and no trained tensor is transformed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from ..assets import sha256 as sha

SELECTED_GRAPH = "5918e523939aba3a6f72e32b88b27a0a1f39828a376b0ed748a7cb43e70ce841"
VERSION = "apple_batch_selected_v1"
MODEL = "audio_vae_decoder_batch_selected.onnx"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _shape(value):
    require(value.type.HasField("tensor_type"), "Tensor input/output required")
    tensor = value.type.tensor_type
    require(tensor.elem_type == TensorProto.FLOAT, "Float32 graph interface required")
    require(tensor.HasField("shape"), "Graph interface shape required")
    return [d.dim_value if d.HasField("dim_value") else None for d in tensor.shape.dim]


def specialize(model, specification):
    """Return a copied model with constant initial history and only audio output.

The pure graph helper permits small fixtures. ``prepare`` additionally binds
the exact selected graph and its 26-state native contract.
"""
    require(isinstance(specification, dict), "Streaming specification required")
    latent, audio = specification.get("latent_input"), specification.get("audio_output")
    require(isinstance(latent, str) and latent and isinstance(audio, str) and audio and latent != audio,
            "Distinct latent input and audio output names required")
    states = specification.get("states")
    require(isinstance(states, list) and states, "Explicit initial history required")
    inputs, outputs = list(model.graph.input), list(model.graph.output)
    require(len({v.name for v in inputs}) == len(inputs) and len({v.name for v in outputs}) == len(outputs),
            "Duplicate graph interface names")
    input_map, output_map = {v.name: v for v in inputs}, {v.name: v for v in outputs}
    require(latent in input_map and audio in output_map, "Latent/audio graph endpoints missing")
    require(_shape(input_map[latent])[:2] == [1, 64] and len(_shape(input_map[latent])) == 3,
            "Expected float32 latent [1,64,L]")
    require(_shape(output_map[audio])[:2] == [1, 1] and len(_shape(output_map[audio])) == 3,
            "Expected float32 audio [1,1,T]")
    original_initializers = [t.SerializeToString() for t in model.graph.initializer]
    original_nodes = [n.SerializeToString() for n in model.graph.node]
    initializer_names = {t.name for t in model.graph.initializer}
    require(len(initializer_names) == len(original_initializers), "Duplicate initializer names")
    require(not initializer_names.intersection(input_map), "Graph inputs must not override initializers")
    state_inputs, state_outputs, zeros = [], [], []
    elements = 0
    for state in states:
        require(isinstance(state, dict), "History record must be a dictionary")
        before, after, shape = state.get("input"), state.get("output"), state.get("shape")
        require(isinstance(before, str) and before and isinstance(after, str) and after,
                "History names required")
        require(before not in (latent, audio) and after not in (latent, audio), "History overlaps latent/audio")
        require(before not in state_inputs and after not in state_outputs, "Duplicate history name")
        require(before in input_map and after in output_map, "History graph endpoint missing")
        require(state.get("dtype") == "float32" and isinstance(shape, list) and len(shape) == 3
                and all(type(d) is int and d > 0 for d in shape) and shape[0] == 1,
                "History must have a positive [1,C,H] float32 shape")
        require(_shape(input_map[before]) == shape and _shape(output_map[after]) == shape,
                "History shape differs from graph")
        state_inputs.append(before); state_outputs.append(after)
        value = np.zeros(shape, dtype=np.float32)
        elements += value.size
        zeros.append(numpy_helper.from_array(value, name=before))
    require(not set(state_inputs).intersection(state_outputs), "History input/output names overlap")
    require(set(input_map) == {latent, *state_inputs} and set(output_map) == {audio, *state_outputs},
            "Streaming graph has undeclared inputs or outputs")
    result = copy.deepcopy(model)
    del result.graph.input[:]
    result.graph.input.append(input_map[latent])
    del result.graph.output[:]
    result.graph.output.append(output_map[audio])
    result.graph.initializer.extend(zeros)
    require([n.SerializeToString() for n in result.graph.node] == original_nodes, "Custom nodes changed")
    require([t.SerializeToString() for t in result.graph.initializer[:len(original_initializers)]]
            == original_initializers, "Trained initializer bytes changed")
    # A path-based check in prepare resolves any external tensor files. The
    # helper checks embedded models directly without materializing weights.
    if not any(t.external_data for t in result.graph.initializer):
        onnx.checker.check_model(result)
    return result, {"version": VERSION, "method": "constant_zero_initial_history",
        "latent_input": latent, "audio_output": audio, "state_tensors": len(states),
        "state_bytes": int(elements * 4), "zero_initializer_names": state_inputs,
        "removed_external_history_outputs": state_outputs, "original_nodes_unchanged": True,
        "kernel_output_arity_preserved": True, "original_initializers_unchanged": True,
        "prepended_latent_frames": 0, "independent_calls": True}


def _file(root, name):
    require(isinstance(name, str) and name and "\\" not in name, "Invalid bundle path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "Bundle path escapes source")
    path = root / name
    require(not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(root),
            "Missing or unsafe bundle file")
    return path


def _libraries(source, records, *, registered):
    require(isinstance(records, list), "Explicit native closure required")
    seen = set()
    for record in records:
        require(isinstance(record, dict), "Invalid native library record")
        name = record.get("library")
        path = _file(source, name)
        require(name not in seen and sha(path) == record.get("sha256"), "Native closure changed or duplicated")
        if registered:
            require(isinstance(record.get("domain"), str) and record["domain"], "Native domain required")
        seen.add(name)
    return seen


def prepare(source, destination):
    """Create a separate batch-only native entry from the frozen selected bundle."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    require(not destination.exists() and not destination.is_relative_to(source), "New separate bundle destination required")
    manifest_path = _file(source, "bundle.json")
    source_manifest_sha = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("onnxruntime") == "1.30.0", "Selected Apple ORT1.30 bundle required")
    native = manifest["native"]["Darwin/arm64"]
    require(native.get("required_cpu_features") == ["sme", "sme2"], "Selected Apple CPU feature policy required")
    require(not native.get("apple_batch_selected"), "Source is already batch-specialized")
    original_full = native["model"]
    spec = manifest["streaming"]["models"][original_full]
    require(manifest["streaming"].get("version") == 1 and spec.get("apple_stream_selected"),
            "Verified selected streaming specification required")
    stream = _file(source, spec["model"])
    require(sha(stream) == spec.get("model_sha256") == SELECTED_GRAPH, "Selected stream graph identity changed")
    require(sha(_file(source, original_full)) == spec.get("source_sha256"), "Full reference identity changed")
    require(sha(_file(source, manifest["fallback"])) == manifest.get("fallback_sha256"),
            "Portable fallback identity changed")
    for name, digest in spec.get("source_external_sha256", {}).items():
        require(sha(_file(source, name)) == digest, "External trained weights changed")
    additional = spec.get("additional_libraries")
    dependencies = spec.get("dependencies", native.get("dependencies", []))
    registered = _libraries(source, additional, registered=True)
    transitive = _libraries(source, dependencies, registered=False)
    require(len(registered) == 11 and len(transitive) == 1 and not registered.intersection(transitive),
            "Selected native closure inventory changed")
    base_library = _file(source, native["library"])
    require(sha(base_library) == native.get("library_sha256"), "Base native library identity changed")
    require(native["library"] not in registered | transitive, "Base library duplicated in selected closure")
    result, audit = specialize(onnx.load(stream, load_external_data=False), spec)
    require((audit["state_tensors"], audit["state_bytes"]) == (26, 683264), "Selected history contract changed")
    require(not (source / MODEL).exists() and MODEL not in manifest["streaming"]["models"],
            "Batch model filename collision")
    # Copy the entire bundle, including licenses, fallback and original graphs.
    # The caller's completed_bundle helper can publish this new directory atomically.
    shutil.copytree(source, destination)
    model_path = destination / MODEL
    onnx.save(result, model_path)
    onnx.checker.check_model(str(model_path))
    audit.update(model_sha256=sha(model_path), source_stream_model=spec["model"],
        source_stream_sha256=SELECTED_GRAPH, source_full_model=original_full,
        source_full_sha256=spec["source_sha256"], source_bundle_manifest_sha256=source_manifest_sha,
        source_state_spec_sha256=hashlib.sha256(json.dumps(spec["states"], sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest(),
        base_library_sha256=sha(base_library), streaming_mapping_for_batch=False)
    native["model"] = MODEL
    native["model_sha256"] = audit["model_sha256"]
    native["additional_libraries"] = copy.deepcopy(additional)
    native["dependencies"] = copy.deepcopy(dependencies)
    native["apple_batch_selected"] = audit
    (destination / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / ".recipe-ready.json").unlink(missing_ok=True)
    require(sha(manifest_path) == source_manifest_sha and sha(stream) == SELECTED_GRAPH,
            "Source bundle changed during specialization")
    require(sha(destination / original_full) == spec["source_sha256"], "Copied full reference changed")
    return audit
