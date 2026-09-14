"""Isolated Intel streaming transfer screen, without running a model.

Keep the accepted Intel paired VNNI projection and oneMKL arithmetic. Transfer
only the six raw-history regions and/or the first phase assembly from the
qualified AMD experiment. The native custom domains deliberately stay identical
so those two native sources can be compiled without arithmetic changes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import onnx

from fast_audiovae.recipes import amd_streaming as rules

SOURCE_SHA256 = "59a937ee76d494896561b994c111b708603fb2f3c8e0fdf53df1380f2df41516"
PAIR_DOMAIN = "fast.audiovae.streaming.matrix.experimental"
VARIANTS = ("history", "phase", "combined")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rewrite(original, variant):
    """Pure rewrite; prepare() additionally enforces the actual source digest."""
    require(variant in VARIANTS, "Unknown transfer variant")
    require(not original.functions and not original.graph.sparse_initializer,
            "Flat dense graph required")
    require(not any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                    for n in original.graph.node for a in n.attribute), "Nested graph unsupported")
    require(not any(w.data_location == onnx.TensorProto.EXTERNAL
                    for w in original.graph.initializer), "Load embedded source tensors first")
    require(len({n.name for n in original.graph.node}) == len(original.graph.node),
            "Unique node names required")
    require(len({w.name for w in original.graph.initializer}) == len(original.graph.initializer),
            "Unique initializer names required")
    require(not {w.name for w in original.graph.initializer}.intersection(v.name for v in original.graph.input),
            "Overridable initializers unsupported")
    require(not {rules.HISTORY_DOMAIN, rules.PHASE_DOMAIN}.intersection(x.domain for x in original.opset_import),
            "Transfer already present")
    pairs = [n for n in original.graph.node if n.op_type == "PackedProjectionPairF32"]
    require(len(pairs) == 1, "One existing Intel first projection pair required")
    pair = pairs[0]
    require(pair.domain == PAIR_DOMAIN and pair.name == "stream_first_projection_pair"
            and rules._attrs(pair) == {"K": 2048, "M": 8192, "native_abi": 1}
            and len(pair.input) == 3 and len(pair.output) == 2,
            "Existing Intel projection contract changed")
    constants = {w.name: w for w in original.graph.initializer}
    require(all(w in constants and constants[w].data_type == onnx.TensorProto.FLOAT
                and list(constants[w].dims) == [8192, 2048] for w in pair.input[:2]),
            "Expected fixed first projection weights")
    graph = copy.deepcopy(original)
    removed, added = set(), set()
    operations = {"history": (rules._history,), "phase": (rules._phase,),
                  "combined": (rules._history, rules._phase)}[variant]
    for operation in operations:
        rm, add = operation(graph)
        require(not removed.intersection(rm), "Overlapping transfer regions")
        removed.update(rm); added.update(add)
    checks = {field + "_byte_identical": rules._sequence(getattr(original.graph, field)) ==
              rules._sequence(getattr(graph.graph, field)) for field in ("initializer", "input", "output")}
    checks["remaining_nodes_byte_identical"] = (
        rules._sequence(n for n in original.graph.node if n.name not in removed) ==
        rules._sequence(n for n in graph.graph.node if n.name not in added))
    checks["intel_pair_byte_identical"] = pair.SerializeToString() == next(
        n for n in graph.graph.node if n.name == pair.name).SerializeToString()
    require(all(checks.values()), "Unselected payload changed")
    onnx.checker.check_model(graph, check_custom_domain=False)
    return graph, {"checks": checks, "removed_nodes": sorted(removed), "inserted_nodes": sorted(added),
                   "required_transfer_domains": [op.domain for op in graph.opset_import
                                                 if op.domain in (rules.HISTORY_DOMAIN, rules.PHASE_DOMAIN)]}


def save_shared(graph, destination, external_template):
    """Write a graph referring to the baseline's exact immutable weight slices."""
    template = {w.name: w for w in external_template.graph.initializer}
    require(set(template) == {w.name for w in graph.graph.initializer}, "Initializer inventory changed")
    for weight in graph.graph.initializer:
        stored = template[weight.name]
        require(weight.data_type == stored.data_type and list(weight.dims) == list(stored.dims),
                "Initializer geometry changed")
        if stored.data_location == onnx.TensorProto.EXTERNAL:
            require(stored.data_type == onnx.TensorProto.FLOAT,
                    "Only FP32 coefficients may use the shared external file")
            weight.ClearField("raw_data")
            del weight.external_data[:]
            weight.external_data.extend(stored.external_data)
            weight.data_location = onnx.TensorProto.EXTERNAL
        else:
            require(weight.SerializeToString() == stored.SerializeToString(), "Embedded initializer changed")
    # save_model would try to rewrite external bytes. Serialize only the small
    # graph, whose relative references were copied from the checked baseline.
    Path(destination).write_bytes(graph.SerializeToString())
    onnx.checker.check_model(str(destination), check_custom_domain=False)


def prepare(source, output, variants=VARIANTS):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(tuple(variants) and len(set(variants)) == len(variants)
            and set(variants) <= set(VARIANTS), "Unique supported variants required")
    require(source.is_file() and sha(source) == SOURCE_SHA256, "Expected accepted Intel paired stream SHA256")
    require(not output.exists(), "Output directory already exists")
    original = onnx.load(source, load_external_data=False)
    require(not any(w.data_location == onnx.TensorProto.EXTERNAL for w in original.graph.initializer),
            "Pinned embedded source required")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".staging-", dir=output.parent))
    try:
        baseline = copy.deepcopy(original)
        # ORT needs shape/Slice/Reshape constants embedded during graph loading.
        # Mark only sizable FP32 coefficient payloads for shared external data;
        # never externalize integer, Boolean or small structural constants.
        for weight in baseline.graph.initializer:
            if (weight.data_type == onnx.TensorProto.FLOAT and weight.HasField("raw_data")
                    and len(weight.raw_data) >= 1024):
                onnx.external_data_helper.set_external_data(weight, location="weights.bin")
        onnx.save_model(baseline, staging / "baseline.onnx")
        stored = onnx.load(staging / "baseline.onnx", load_external_data=False)
        onnx.checker.check_model(str(staging / "baseline.onnx"), check_custom_domain=False)
        report = {"version": "intel_streaming_transfer_v1", "source_sha256": SOURCE_SHA256,
                  "cpu_only": True, "inference_executed": False, "intel_projection_unchanged": True,
                  "externalization": "Only FP32 raw initializer payloads >=1024 bytes; all integer and smaller constants stay embedded",
                  "baseline": {"model": "baseline.onnx", "sha256": sha(staging / "baseline.onnx")},
                  "shared_weights": {"file": "weights.bin", "sha256": sha(staging / "weights.bin"),
                                     "bytes": (staging / "weights.bin").stat().st_size}, "variants": {}}
        del baseline
        for variant in variants:
            candidate, audit = rewrite(original, variant)
            name = variant + ".onnx"
            save_shared(candidate, staging / name, stored)
            report["variants"][variant] = {"model": name, "sha256": sha(staging / name), **audit}
            del candidate
        require(sha(source) == SOURCE_SHA256, "Source changed during preparation")
        require(sha(staging / "weights.bin") == report["shared_weights"]["sha256"], "Shared weights changed")
        (staging / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output)
        return report
    except BaseException:
        # Staging contains only this call's generated graph copies.
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output_dir, tuple(args.variants)), indent=2))


if __name__ == "__main__":
    main()
