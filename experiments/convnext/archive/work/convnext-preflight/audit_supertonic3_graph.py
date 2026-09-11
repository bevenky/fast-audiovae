"""Static architecture inventory only. No ONNX session or model execution."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper

root = Path(__file__).resolve().parents[2]
graph_path = root / "work/supertonic3_assets/vocoder.onnx"
out = root / "outputs/convnext-restart-plan"
out.mkdir(parents=True, exist_ok=True)
raw = graph_path.read_bytes()
model = onnx.load_model_from_string(raw)
initializers = {value.name: value for value in model.graph.initializer}
constant_values = {}


def attrs(node):
    return {value.name: helper.get_attribute_value(value) for value in node.attribute}


def serial(value):
    if isinstance(value, onnx.TensorProto):
        array = numpy_helper.to_array(value)
        return {"shape": list(array.shape), "dtype": str(array.dtype), "constant_value": array.tolist()}
    if isinstance(value, bytes):
        return value.decode()
    return value


def fold_constant_only(node):
    """Evaluate only small exported shape/padding constants, never model tensors."""
    a = attrs(node)
    if node.op_type == "Constant":
        result = numpy_helper.to_array(a["value"])
    else:
        if any(name not in constant_values for name in node.input):
            return
        x = [constant_values[name] for name in node.input]
        if node.op_type == "ConstantOfShape":
            result = np.full(tuple(x[0]), numpy_helper.to_array(a["value"]).item(), dtype=np.int64)
        elif node.op_type == "Concat":
            result = np.concatenate(x, axis=a["axis"])
        elif node.op_type == "Reshape":
            result = x[0].reshape(tuple(x[1]))
        elif node.op_type == "Transpose":
            result = x[0].transpose(tuple(a["perm"]))
        elif node.op_type == "Slice":
            index = [slice(None)] * x[0].ndim
            axes = x[3] if len(x) > 3 else range(len(x[1]))
            steps = x[4] if len(x) > 4 else np.ones_like(x[1])
            for start, end, axis, step in zip(x[1], x[2], axes, steps):
                index[int(axis)] = slice(int(start), int(end), int(step))
            result = x[0][tuple(index)]
        elif node.op_type == "Cast":
            result = x[0].astype(helper.tensor_dtype_to_np_dtype(a["to"]))
        elif node.op_type == "Unsqueeze":
            result = np.expand_dims(x[0], axis=tuple(x[1]))
        elif node.op_type == "Squeeze":
            result = np.squeeze(x[0], axis=tuple(x[1]))
        else:
            return
    if result.size > 64:
        raise ValueError("Static shape constant unexpectedly large")
    constant_values[node.output[0]] = result


def node_info(index, node):
    value = {"index": index, "name": node.name, "op": node.op_type,
             "inputs": list(node.input), "outputs": list(node.output),
             "attributes": {key: serial(value) for key, value in attrs(node).items()},
             "parameter_shapes": {name: list(initializers[name].dims) for name in node.input if name in initializers}}
    if node.op_type == "Pad":
        value["resolved_pads"] = constant_values[node.input[1]].tolist()
    return value


for node in model.graph.node:
    fold_constant_only(node)
nodes = [node_info(i, node) for i, node in enumerate(model.graph.node)]
counts = dict(Counter(node.op_type for node in model.graph.node))
api = json.loads((out / "hf-model-metadata.json").read_text())
tree = json.loads((out / "hf-onnx-tree.json").read_text())
remote = next(value for value in tree if value["path"] == "onnx/vocoder.onnx")
digest = hashlib.sha256(raw).hexdigest()
assert digest == remote["lfs"]["oid"] and len(raw) == remote["size"]
config = json.loads((out / "tts.json").read_text())
blocks = []
for index in range(10):
    prefix = f"/decoder/convnext.{index}/"
    block = [value for value in nodes if value["name"].startswith(prefix)]
    blocks.append({"block": index, "nodes": block,
                   "gamma_shape": list(initializers[f"tts.ae.decoder.convnext.{index}.gamma"].dims)})
student_path = root / "work/fast-audiovae/experiments/convnext/audiovae_student/model.py"
report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "method": "ONNX protobuf inventory and constant-only shape/padding folding; no inference, benchmark, weight conversion or parameter import",
    "graph": {"path": str(graph_path), "sha256": digest, "bytes": len(raw),
              "official_current_revision": api["sha"], "official_last_modified": api["lastModified"],
              "verified_against_current_official_lfs_metadata": True,
              "producer": model.producer_name, "producer_version": model.producer_version,
              "ir_version": model.ir_version, "opsets": [{"domain": v.domain, "version": v.version} for v in model.opset_import],
              "nodes": len(nodes), "initializers": len(initializers), "op_counts": counts,
              "inputs": [helper.printable_value_info(value) for value in model.graph.input],
              "outputs": [helper.printable_value_info(value) for value in model.graph.output]},
    "student_model_sha256": hashlib.sha256(student_path.read_bytes()).hexdigest(),
    "official_decoder_config": config["ae"]["decoder"],
    "official_encoder_config": config["ae"]["encoder"],
    "official_audio_contract": {key: config["ae"][key] for key in ("sample_rate", "base_chunk_size", "ldim", "n_delay")},
    "input_wrapper": {"node_indices": list(range(35)), "packed_input": "[B,144,L] = [B,24*6,L]",
                      "operations": "divide TTS normalizer scale; reshape [B,24,6,L]; transpose [0,1,3,2]; reshape [B,24,6L]; multiply per-channel latent_std and add latent_mean",
                      "normalizer_scale": float(numpy_helper.to_array(initializers["tts.ttl.normalizer.scale"])),
                      "learned_normalization_values_exported_in_audit": False},
    "stem": [value for value in nodes if value["name"].startswith("/decoder/embed/")],
    "blocks": blocks,
    "final_norm": nodes[372],
    "head": nodes[373:],
    "resolved_padding": [{"node": value["name"], "mode": value["attributes"]["mode"], "pads": value["resolved_pads"]}
                         for value in nodes if value["op"] == "Pad"],
    "absent_in_inference_graph": {"Snake_or_sine_activation": counts.get("Sin", 0) == 0,
                                  "iSTFT_or_DFT_synthesis": all(counts.get(name, 0) == 0 for name in ("STFT", "DFT", "ISTFT")),
                                  "progressive_transposed_convolutions": counts.get("ConvTranspose", 0) == 0,
                                  "GRN_dynamic_response_normalization": all(counts.get(name, 0) == 0 for name in ("ReduceMean", "ReduceL2", "ReduceSum", "Sqrt")),
                                  "terminal_tanh_or_clip": all(counts.get(name, 0) == 0 for name in ("Tanh", "Clip"))},
    "training_unknowns": ["Whether additional input normalization was folded into the anonymous stem Conv",
                          "BatchNorm versus SyncBatchNorm training source, exact batch/time axes and schedule",
                          "Initialization, optimizer, loss weighting, discriminators and training curriculum",
                          "Whether dropout/stochastic depth existed during training but was removed for export",
                          "Encoder checkpoint behavior: tts.json describes architecture but vocoder graph contains no encoder"],
    "all_nodes": nodes,
    "initializers_shape_only": [{"name": v.name, "shape": list(v.dims), "dtype": onnx.TensorProto.DataType.Name(v.data_type)} for v in model.graph.initializer],
    "sources": ["https://huggingface.co/Supertone/supertonic-3/blob/main/onnx/vocoder.onnx",
                "https://huggingface.co/Supertone/supertonic-3/blob/main/onnx/tts.json",
                "https://huggingface.co/api/models/Supertone/supertonic-3",
                "https://huggingface.co/api/models/Supertone/supertonic-3/tree/main/onnx"],
}
(out / "supertonic3-static-graph-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"graph": report["graph"], "pads": report["resolved_padding"]}, indent=2))
