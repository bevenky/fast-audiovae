"""Transpose only the first streaming projection pair for fixed-right weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((source / "bundle.json").read_text())
    native = manifest["native"]["Darwin/arm64"]
    spec = manifest["streaming"]["models"][native["model"]]
    full_path, stream_path = source / native["model"], source / spec["model"]
    original_hashes = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    model = onnx.load(stream_path)
    weights = {x.name: x for x in model.graph.initializer}
    names = {"ncc_up_1_split_matmul_current_node", "ncc_up_1_split_matmul_previous_unshifted_node"}
    selected = [n for n in model.graph.node if n.name in names]
    assert len(selected) == 2 and selected[0].input[1] == selected[1].input[1]
    assert all(n.op_type == "MatMul" and not n.domain for n in selected)
    changes, output_nodes, removed_weights, new_weights = [], [], set(), []
    transposed_input = "stream_up1_shared_input_transposed"
    inserted_transpose = False
    for node in model.graph.node:
        if node.name not in names:
            output_nodes.append(node)
            continue
        weight = weights[node.input[0]]
        assert list(weight.dims) == [8192, 2048]
        array = numpy_helper.to_array(weight)
        assert array.dtype == np.float32 and len(node.output) == 1
        if not inserted_transpose:
            output_nodes.append(helper.make_node("Transpose", [node.input[1]], [transposed_input],
                                                  name="stream_up1_transpose_input", perm=[0, 2, 1]))
            inserted_transpose = True
        new_name = weight.name + "_transposed"
        transposed_weight = np.ascontiguousarray(array.T)
        assert np.array_equal(transposed_weight.T.view(np.uint32), array.view(np.uint32))
        new_weights.append(numpy_helper.from_array(transposed_weight, new_name))
        removed_weights.add(weight.name)
        intermediate = node.output[0] + "_transposed"
        output_nodes.extend([
            helper.make_node("MatMul", [transposed_input, new_name], [intermediate],
                             name=node.name + "_fixed_right"),
            helper.make_node("Transpose", [intermediate], list(node.output),
                             name=node.name + "_restore_layout", perm=[0, 2, 1]),
        ])
        changes.append({"node": node.name, "weight": weight.name, "weight_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                        "new_weight": new_name, "new_shape": list(transposed_weight.shape)})
    del model.graph.node[:]
    model.graph.node.extend(output_nodes)
    assert not any(x in removed_weights for n in model.graph.node for x in n.input)
    kept = [x for x in model.graph.initializer if x.name not in removed_weights]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept + new_weights)
    onnx.checker.check_model(model)
    shutil.copytree(source, destination)
    target = destination / spec["model"]
    onnx.save(model, target)
    spec["model_sha256"] = sha(target)
    spec["experiment"] = "streaming_first_projection_fixed_right"
    (destination / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assert sha(destination / native["model"]) == sha(full_path)
    for relative, digest in original_hashes.items():
        assert sha(source / relative) == digest, relative
        if relative not in ("bundle.json", spec["model"]):
            assert sha(destination / relative) == digest, relative
    receipt = {"source": str(source), "destination": str(destination), "source_sha256": original_hashes,
               "candidate_stream_sha256": sha(target), "changes": changes,
               "unchanged_full_graph": True, "unchanged_state_specification": True,
               "precision": "same FP32 values; only exact weight transpose", "runtime_validation": "pending"}
    (destination.parent / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"bundle": str(destination), "changes": len(changes), "stream_sha256": sha(target)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
