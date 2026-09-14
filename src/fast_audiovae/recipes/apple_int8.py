"""Bind the qualified first-upsampling INT8 projections to Apple streaming."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import onnx
from onnx import helper

from ..assets import sha256
from .apple_selected import SELECTED_GRAPH

DOMAIN = "fast.audiovae.apple.firstpair.int8.v1"
MODEL = "streaming/decoder_int8.onnx"
TARGETS = {
    "ncc_up_1_split_matmul_current_node_fixed_right":
        ("fast.audiovae.apple.libxsmm.panel.v1", "LibxsmmPanelSmeWeightLeftF32"),
    "ncc_up_1_split_matmul_previous_unshifted_node_fixed_right":
        ("fast.audiovae.apple.multitile.v1", "MultitileWeightLeftF32"),
}
ATTRIBUTES = {"k": 2048, "n": 8192, "max_m": 2, "matrix_abi": 1, "threads": 1}


def require(value, message):
    if not value:
        raise ValueError(message)


def rewrite(model):
    """Change two operators only; keep trained constants and all stream states."""
    result = copy.deepcopy(model)
    found = set()
    for node in result.graph.node:
        if node.name not in TARGETS:
            continue
        require(node.name not in found, "Duplicate first-upsampling projection")
        require((node.domain, node.op_type) == TARGETS[node.name], "Unexpected projection implementation")
        require({a.name: helper.get_attribute_value(a) for a in node.attribute} == ATTRIBUTES,
                "First-upsampling projection dimensions or thread policy changed")
        require(len(node.input) == 2 and len(node.output) == 1, "Projection interface changed")
        node.domain, node.op_type = DOMAIN, "FirstPairInt8"
        found.add(node.name)
    require(found == set(TARGETS), "Both qualified projections are required")
    require(all(o.domain != DOMAIN for o in result.opset_import), "Graph already contains INT8 operators")
    result.opset_import.append(helper.make_opsetid(DOMAIN, 1))
    return result


def prepare(source, destination, build_manifest):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    require(not destination.exists() and not destination.is_relative_to(source), "New bundle destination required")
    manifest = json.loads((source / "bundle.json").read_text())
    native = manifest["native"]["Darwin/arm64"]
    spec = manifest["streaming"]["models"][native["model"]]
    require(spec.get("apple_stream_selected") and sha256(source / spec["model"]) == spec["model_sha256"] == SELECTED_GRAPH,
            "INT8 preparation requires the qualified Apple FP32 streaming graph")
    build = json.loads(Path(build_manifest).read_text())
    require(build.get("version") == "apple_firstpair_int8_build_v1" and build.get("complete") is True,
            "Complete Apple INT8 build required")
    records = build["runtime_files"]
    require(len(records) == 1 and records[0].get("register") is True and records[0].get("domain") == DOMAIN,
            "A self-contained INT8 operator library is required")
    record = records[0]
    require(sha256(record["path"]) == record["sha256"], "INT8 native library changed")
    require(build["additional_libraries"] == [{"library": record["path"], "sha256": record["sha256"], "domain": DOMAIN}],
            "INT8 registration inventory differs")
    result = rewrite(onnx.load(source / spec["model"]))
    onnx.checker.check_model(result)
    shutil.copytree(source, destination)
    onnx.save(result, destination / MODEL)
    library = "runtime/" + Path(record["path"]).name
    require(not (destination / library).exists(), "INT8 library filename collision")
    shutil.copy2(record["path"], destination / library)
    require(sha256(destination / library) == record["sha256"], "INT8 library copy changed")
    # Preserve the original graph for explicit multithreaded loads of this bundle.
    fallback = copy.deepcopy(spec)
    fallback["precision"] = "FP32"
    spec["fp32_fallback"] = fallback
    spec["model"] = MODEL
    spec["model_sha256"] = sha256(destination / MODEL)
    spec["additional_libraries"].append({"library": library, "sha256": record["sha256"], "domain": DOMAIN})
    spec["precision"] = "mixed_fp32_int8"
    spec["apple_firstpair_int8"] = {"version": 1, "threads": 1, "nodes": sorted(TARGETS),
        "reference_graph_sha256": SELECTED_GRAPH, "internal_tile_frames": 2,
        "arbitrary_packet_lengths": True, "quantization": "symmetric per-output-channel weights and per-frame activations",
        "remainder": "unchanged FP32", "native_build_sha256": sha256(build_manifest)}
    (destination / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / ".recipe-ready.json").unlink(missing_ok=True)
    return spec["apple_firstpair_int8"]
