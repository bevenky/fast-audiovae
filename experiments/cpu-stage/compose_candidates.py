"""Compose audited CPU graph candidates proven from one immutable native model.

This does graph checks only. It never creates an inference session. The base is
reproved with the repository block-fusion pass. Every selected derivative region
must contain whole base regions and have identical activation boundaries.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from collections import Counter

KINDS = {
    "matrix": ("audio.cpu.pointwise.experimental", "PointwiseBiasResidualF32", ".fusion.json"),
    "stage": ("fast.audiovae.stage.experimental", "StageStackF32", ".stage.json"),
    "mkl": ("venky.audio.intel.decoder.experimental", "IntelPlainMatMulF32", ".json"),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def blob(value):
    return value.SerializeToString()


def unique(items, name, label):
    out = {}
    for item in items:
        key = name(item)
        if not key or key in out:
            raise ValueError("Missing or duplicate " + label + ": " + str(key))
        out[key] = item
    return out


def graph_index(model):
    import onnx
    if model.functions or model.graph.sparse_initializer or any(
        a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for n in model.graph.node for a in n.attribute
    ):
        raise ValueError("Only flat dense graphs without local functions are supported")
    nodes = unique(model.graph.node, lambda n: n.name, "node name")
    initializers = unique(model.graph.initializer, lambda t: t.name, "initializer")
    producers = {}
    users = {}
    for n in model.graph.node:
        for output in n.output:
            if not output or output in producers or output in initializers:
                raise ValueError("Duplicate or empty produced tensor: " + output)
            producers[output] = n.name
        for i, value in enumerate(n.input):
            if value:
                users.setdefault(value, []).append((n.name, i))
    if any(v.name in initializers for v in model.graph.input):
        raise ValueError("Overridable initializer inputs are unsupported")
    return {"nodes": nodes, "initializers": initializers, "producers": producers,
            "users": users, "outputs": {v.name for v in model.graph.output}}


def activation_inputs(node, initializers):
    # Shape programs of view-only Reshapes are established by the original proof.
    # They are metadata dependencies, not activation inputs to a fused CPU kernel.
    return {v for i, v in enumerate(node.input) if v and v not in initializers
            and not (node.domain == "" and node.op_type == "Reshape" and i == 1)}


def boundary(index, region):
    produced = {v for name in region for v in index["nodes"][name].output}
    inputs = set().union(*(activation_inputs(index["nodes"][name], index["initializers"])
                           for name in region)) - produced
    outputs = {v for v in produced if v in index["outputs"] or any(
        user not in region for user, _ in index["users"].get(v, []))}
    return inputs, outputs


def check_boundary(index, region, replacement, replacement_initializers):
    inputs, outputs = boundary(index, region)
    wanted_inputs = activation_inputs(replacement, replacement_initializers)
    if inputs != wanted_inputs or outputs != set(replacement.output):
        raise ValueError("Activation boundary/fanout mismatch for " + replacement.name
                         + ": " + repr((sorted(inputs), sorted(outputs))) + " versus "
                         + repr((sorted(wanted_inputs), list(replacement.output))))
    return inputs, outputs


def external_manifest(model, directory):
    import onnx
    result = {}
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        fields = {v.key: v.value for v in tensor.external_data}
        location = fields.get("location", "")
        if not location or Path(location).is_absolute() or ".." in Path(location).parts:
            raise ValueError("Unsafe external data location")
        path = Path(directory) / location
        length = path.stat().st_size
        offset = int(fields.get("offset", 0))
        size = int(fields.get("length", length - offset))
        if offset < 0 or size < 0 or offset + size > length:
            raise ValueError("Invalid external data bounds")
        if location not in result:
            result[location] = {"sha256": sha(path), "bytes": length}
    return result


def audit_binding(audit_path, model_path, source_hash):
    path = Path(audit_path)
    audit = json.loads(path.read_text())
    if audit.get("source_sha256") != source_hash:
        raise ValueError("Audit is not bound to the same original: " + str(path))
    if audit.get("output_sha256") != sha(model_path):
        raise ValueError("Derivative hash does not match its audit: " + str(path))
    return audit


def check_initializers(original, derivative):
    old, new = original["initializers"], derivative["initializers"]
    if list(new)[:len(old)] != list(old):
        raise ValueError("Original initializer sequence changed")
    for name, tensor in old.items():
        if name not in new or blob(tensor) != blob(new[name]):
            raise ValueError("Original initializer protobuf changed: " + name)
    aliases = {}
    # Only byte-identical rank-one channel-bias aliases are allowed additions.
    for name in list(new)[len(old):]:
        tensor = new[name]
        if len(tensor.dims) != 1 or tensor.data_type != 1:
            raise ValueError("Unexpected initializer addition: " + name)
        matches = []
        for old_name, source in old.items():
            if source.data_type != tensor.data_type or list(source.dims) not in (
                [1, tensor.dims[0], 1], [tensor.dims[0], 1], [tensor.dims[0]]
            ):
                continue
            alias = copy.deepcopy(source)
            alias.name = name
            del alias.dims[:]
            alias.dims.extend(tensor.dims)
            if blob(alias) == blob(tensor):
                matches.append(old_name)
        if not matches:
            raise ValueError("Alias coefficient bytes do not match an original bias: " + name)
        aliases[name] = matches
    return aliases


def check_interface(original, derivative):
    for field in ("input", "output"):
        if [blob(v) for v in getattr(original.graph, field)] != [blob(v) for v in getattr(derivative.graph, field)]:
            raise ValueError("Graph " + field + " interface changed")
    if original.ir_version != derivative.ir_version:
        raise ValueError("IR version changed")


def base_regions(original, base, audit, directory):
    from fast_audiovae.graph.block_fusion import rewrite_model
    variant = audit.get("variant", audit.get("mode"))
    if variant not in ("none", "adds", "chain", "both"):
        raise ValueError("Only a checked block-fusion base is supported")
    if audit.get("backend_override") not in (None, 0, 2, 4, 5):
        raise ValueError("Invalid base backend")
    counts = audit.get("counts")
    if not isinstance(counts, dict) or set(counts) != {"chain", "adds"}:
        raise ValueError("Missing base proof counts")
    proven, proof = rewrite_model(original, directory, variant=variant,
                                  backend=audit.get("backend_override"),
                                  expected_chains=counts["chain"], expected_adds=counts["adds"])
    if blob(proven) != blob(base):
        raise ValueError("Base differs from a fresh original-source block proof")
    if proof["changes"] != audit.get("changes"):
        raise ValueError("Base audit regions differ from original-source proof")
    old = graph_index(original)
    footprints = {}
    for r in proof["changes"]:
        if r["kind"] == "adds":
            names = {r["bias_add"], r["residual_add"]}
        else:
            names = {r["pre_snake"], r["depthwise_snake"]}
            dw = old["nodes"][r["depthwise_snake"]]
            previous = old["nodes"][old["producers"][dw.input[0]]]
            if previous.name != r["pre_snake"]:
                if previous.op_type != "Reshape" or previous.domain:
                    raise ValueError("Unexpected shape alias in proven chain")
                names.add(previous.name)
        footprints[r["node"]] = names
    base_index = graph_index(base)
    occupied = set()
    for node in base.graph.node:
        region = footprints.setdefault(node.name, {node.name})
        if not region <= set(old["nodes"]) or occupied & region:
            raise ValueError("Base regions are not a disjoint original partition")
        check_boundary(old, region, node, base_index["initializers"])
        occupied |= region
    if occupied != set(old["nodes"]):
        raise ValueError("Base does not cover every original node")
    return footprints


def candidate_regions(original, derivative, audit, kind):
    domain, op_type, _ = KINDS[kind]
    old, new = graph_index(original), graph_index(derivative)
    check_interface(original, derivative)
    aliases = check_initializers(old, new)
    records = audit.get("nodes") if kind == "mkl" else audit.get("changes")
    if not isinstance(records, list) or not records:
        raise ValueError("No audited candidate regions")
    if kind != "mkl" and audit.get("count") != len(records):
        raise ValueError("Candidate count mismatch")
    regions, touched = [], set()
    for record in records:
        if kind == "matrix":
            node_name = record["node"]
            names = {record[k] for k in ("matmul", "bias_add", "residual_add")}
        elif kind == "stage":
            node_name = record["node"]
            names = set(record["removed_nodes"])
            if len(names) != len(record["removed_nodes"]):
                raise ValueError("Repeated stage source node")
        else:
            node_name = record["name"]
            names = {node_name}
        if not names or not names <= set(old["nodes"]) or names & touched:
            raise ValueError("Missing or overlapping candidate original region")
        replacement = new["nodes"].get(node_name)
        if replacement is None or (replacement.domain, replacement.op_type) != (domain, op_type):
            raise ValueError("Audited custom node is missing or has wrong type")
        def ordered_bias(output, alias_name):
            final = old["nodes"][old["producers"][output]]
            if final.domain or final.op_type != "Add" or len(final.input) != 2:
                raise ValueError("Original ordered residual Add required")
            biased = old["nodes"].get(old["producers"].get(final.input[1]))
            if biased is None or biased.domain or biased.op_type != "Add" or len(biased.input) != 2:
                raise ValueError("Original ordered bias Add required")
            source_bias = old["initializers"].get(biased.input[1])
            alias = new["initializers"].get(alias_name)
            if source_bias is None or alias is None or alias_name not in aliases:
                raise ValueError("Missing proven original channel bias alias")
            expected_alias = copy.deepcopy(source_bias)
            expected_alias.name = alias_name
            del expected_alias.dims[:]
            expected_alias.dims.extend(alias.dims)
            if blob(expected_alias) != blob(alias):
                raise ValueError("Alias coefficient bytes differ from its original ordered bias")
            return final, biased
        if kind == "matrix":
            mm = old["nodes"][record["matmul"]]
            if mm.domain or mm.op_type != "MatMul" or len(mm.input) != 2 or len(replacement.input) != 4:
                raise ValueError("Original standard pointwise MatMul required")
            final, biased = ordered_bias(replacement.output[0], replacement.input[2])
            if (final.name != record["residual_add"] or biased.name != record["bias_add"]
                    or list(biased.input) != [mm.output[0], record["bias_source"]]
                    or list(replacement.input) != [mm.input[0], mm.input[1], record["bias_alias"], final.input[0]]):
                raise ValueError("Matrix replacement changes original operands or order")
        elif kind == "stage":
            constants = record.get("unit_constants", [])
            unit_outputs = record.get("unit_outputs", [])
            if not constants or len(constants) != len(unit_outputs) or any(len(row) != 8 for row in constants):
                raise ValueError("Stage audit must identify ordered unit constants and outputs")
            if list(replacement.input[1:]) != [v for row in constants for v in row]:
                raise ValueError("Stage replacement constants differ from its audit")
            for unit_output, row in zip(unit_outputs, constants):
                ordered_bias(unit_output, row[-1])
        else:
            matrix = old["nodes"][node_name]
            if matrix.domain or matrix.op_type != "MatMul" or len(matrix.input) != 2:
                raise ValueError("MKL requires an original standard MatMul")
            if list(replacement.input) != list(matrix.input) or list(replacement.output) != list(matrix.output):
                raise ValueError("MKL replacement changes original matrix operands")
        inputs, outputs = check_boundary(old, names, replacement, new["initializers"])
        regions.append({"kind": kind, "node": node_name, "original_nodes": names,
                        "replacement": replacement, "inputs": inputs, "outputs": outputs})
        touched |= names
    # The hash-bound audit may describe only the requested replacements. Any
    # other derivative mutation, including a second native-chain pass, fails.
    expected = [r["replacement"] for r in regions]
    expected += [n for n in original.graph.node if n.name not in touched]
    expected_by_name = unique(expected, lambda n: n.name, "expected derivative node")
    if set(expected_by_name) != set(new["nodes"]):
        raise ValueError("Derivative contains unaudited node additions or removals")
    for name, node in expected_by_name.items():
        if blob(node) != blob(new["nodes"][name]):
            raise ValueError("Unselected derivative node changed: " + name)
    # Validate original opsets and permit only the selected custom domain.
    original_ops = unique(original.opset_import, lambda o: o.domain or "ai.onnx", "opset")
    derivative_ops = unique(derivative.opset_import, lambda o: o.domain or "ai.onnx", "opset")
    if set(derivative_ops) != set(original_ops) | {domain}:
        raise ValueError("Unexpected derivative opset imports")
    for key, value in original_ops.items():
        if blob(value) != blob(derivative_ops[key]):
            raise ValueError("Original opset changed")
    if derivative_ops[domain].version != 1:
        raise ValueError("Candidate domain must use version 1")
    used = {v for r in regions for v in r["replacement"].input}
    if set(aliases) - used:
        raise ValueError("Derivative contains unused initializer additions")
    return regions, aliases


def compose(original, base, output, candidates, *, base_audit=None,
            package_source=None, audit_output=None):
    """Compose [(kind, derivative_path, audit_path), ...] on a proven base.

    Candidate derivatives select stages/nodes before this call. For cumulative
    MKL plus stage or matrix experiments, pass both original-derived files in
    one call. A previously composed output is deliberately not an accepted base.
    """
    if package_source:
        sys.path.insert(0, str(Path(package_source).resolve()))
    import onnx
    from fast_audiovae.graph.elementwise import copy_external_data
    original, base, output = (Path(p).resolve() for p in (original, base, output))
    base_audit = Path(base_audit or base.with_suffix(".block-fusion.json")).resolve()
    audit_output = Path(audit_output or output.with_suffix(".compose.json")).resolve()
    if not candidates or any(c[0] not in KINDS for c in candidates):
        raise ValueError("At least one known candidate kind is required")
    paths = {original, base, base_audit}
    for kind, model_path, audit_path in candidates:
        paths.add(Path(model_path).resolve())
        paths.add(Path(audit_path or Path(model_path).with_suffix(KINDS[kind][2])).resolve())
    if output in paths or audit_output in paths or output == audit_output or output.exists() or audit_output.exists():
        raise ValueError("Output and audit must be new, separate files")
    source_hash = sha(original)
    ba = audit_binding(base_audit, base, source_hash)
    models = [onnx.load(p, load_external_data=False) for p in (original, base)]
    source, base_model = models
    onnx.checker.check_model(str(original), full_check=False)
    onnx.checker.check_model(str(base), full_check=False)
    source_index, base_index = (graph_index(m) for m in models)
    check_interface(source, base_model)
    check_initializers(source_index, base_index)
    source_files = external_manifest(source, original.parent)
    if external_manifest(base_model, base.parent) != source_files:
        raise ValueError("Base external coefficient files differ from original")
    footprints = base_regions(source, base_model, ba, original.parent)
    selected, touched, derivative_reports, alias_sources = [], set(), [], {}
    final_initializers = dict(base_index["initializers"])
    final_opsets = unique(base_model.opset_import, lambda o: o.domain or "ai.onnx", "base opset")
    for kind, model_path, audit_path in candidates:
        model_path = Path(model_path).resolve()
        audit_path = Path(audit_path or model_path.with_suffix(KINDS[kind][2])).resolve()
        audit = audit_binding(audit_path, model_path, source_hash)
        derivative = onnx.load(model_path, load_external_data=False)
        onnx.checker.check_model(str(model_path), full_check=False)
        if external_manifest(derivative, model_path.parent) != source_files:
            raise ValueError("Derivative external coefficient files differ from original")
        regions, aliases = candidate_regions(source, derivative, audit, kind)
        index = graph_index(derivative)
        for alias in aliases:
            tensor = index["initializers"][alias]
            if alias in final_initializers and blob(final_initializers[alias]) != blob(tensor):
                raise ValueError("Initializer alias collision: " + alias)
            if alias in base_index["producers"] or alias in source_index["producers"]:
                raise ValueError("Initializer alias collides with an activation")
            final_initializers[alias] = tensor
            alias_sources[alias] = aliases[alias]
        for opset in derivative.opset_import:
            key = opset.domain or "ai.onnx"
            if key in final_opsets and blob(final_opsets[key]) != blob(opset):
                raise ValueError("Opset collision: " + key)
            final_opsets[key] = opset
        for region in regions:
            if region["original_nodes"] & touched:
                raise ValueError("Candidate regions overlap in the original graph")
            selected.append(region)
            touched |= region["original_nodes"]
        derivative_reports.append({"kind": kind, "model": str(model_path),
            "model_sha256": sha(model_path), "audit": str(audit_path), "audit_sha256": sha(audit_path),
            "source_sha256": audit["source_sha256"], "region_count": len(regions)})
    replace_at, removed, records = {}, set(), []
    positions = {n.name: i for i, n in enumerate(base_model.graph.node)}
    all_names = set(base_index["nodes"]) | set(final_initializers) | set(base_index["producers"])
    for region in selected:
        original_region = region["original_nodes"]
        selected_base = {name for name, footprint in footprints.items() if footprint & original_region}
        if any(not footprints[name] <= original_region for name in selected_base):
            raise ValueError("Candidate partially intersects a fused base region")
        if set().union(*(footprints[name] for name in selected_base)) != original_region:
            raise ValueError("Candidate does not cover a complete original region")
        replacement = region["replacement"]
        check_boundary(base_index, selected_base, replacement, final_initializers)
        if removed & selected_base:
            raise ValueError("Composed base regions overlap")
        if replacement.name in all_names and replacement.name not in selected_base:
            raise ValueError("Replacement node name collision: " + replacement.name)
        for value in replacement.output:
            if value not in base_index["producers"] or base_index["producers"][value] not in selected_base:
                raise ValueError("Replacement output does not belong to the removed base region")
        anchor = max(positions[name] for name in selected_base)
        replace_at[anchor] = replacement
        removed |= selected_base
        all_names.add(replacement.name)
        records.append({"kind": region["kind"], "replacement_node": replacement.name,
            "original_nodes": sorted(original_region), "removed_base_nodes": sorted(selected_base),
            "activation_inputs": sorted(region["inputs"]), "activation_outputs": sorted(region["outputs"])})
    result = copy.deepcopy(base_model)
    nodes = []
    for i, node in enumerate(base_model.graph.node):
        if i in replace_at:
            nodes.append(copy.deepcopy(replace_at[i]))
        elif node.name not in removed:
            nodes.append(copy.deepcopy(node))
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    del result.graph.initializer[:]
    result.graph.initializer.extend(copy.deepcopy(list(final_initializers.values())))
    del result.opset_import[:]
    result.opset_import.extend(copy.deepcopy(list(final_opsets.values())))
    live = {v for node in nodes for v in (*node.input, *node.output)}
    live |= {v.name for v in (*result.graph.input, *result.graph.output)}
    infos = [v for v in result.graph.value_info if v.name in live]
    del result.graph.value_info[:]
    result.graph.value_info.extend(infos)
    graph_index(result)
    check_initializers(source_index, graph_index(result))
    output.parent.mkdir(parents=True, exist_ok=True)
    external = copy_external_data(result, original.parent, output.parent)
    with tempfile.NamedTemporaryFile(prefix=".compose-", suffix=".onnx", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        onnx.save_model(result, temporary)
        onnx.checker.check_model(str(temporary), full_check=False)
        output_hash = sha(temporary)
        report = {"source_sha256": source_hash, "output_sha256": output_hash,
            "original": str(original), "base": str(base), "base_sha256": sha(base),
            "base_audit": str(base_audit), "base_audit_sha256": sha(base_audit),
            "base_variant": ba["variant"], "base_backend_override": ba.get("backend_override"),
            "derivatives": derivative_reports, "regions": records,
            "counts": dict(Counter(r["kind"] for r in records)),
            "source_nodes": len(source.graph.node), "base_nodes": len(base_model.graph.node),
            "output_nodes": len(nodes), "original_initializers_unchanged": True,
            "alias_original_sources": alias_sources, "external_files": external,
            "base_reproved_from_original": True, "complete_disjoint_original_regions": True,
            "activation_boundaries_unchanged": True, "onnx_graph_check_passed": True,
            "inference_executed": False,
            "validation": "Graph and coefficient provenance checks only; numerical parity, quality, and CPU timing remain unmeasured by this helper."}
        # Exclusive creation also protects against another composition finishing
        # at the same path after the initial preflight. Both files are local.
        os.link(temporary, output)
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        with audit_output.open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-audit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output")
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--derivative")
    parser.add_argument("--derivative-audit")
    parser.add_argument("--candidate", nargs=3, action="append", default=[], metavar=("KIND", "MODEL", "AUDIT"),
                        help="Repeat for cumulative disjoint candidates, all derived from --original")
    parser.add_argument("--package-source", default=str(Path(__file__).resolve().parents[2] / "src"))
    args = vars(parser.parse_args())
    candidates = args.pop("candidate")
    kind, derivative, derivative_audit = (args.pop(k) for k in ("kind", "derivative", "derivative_audit"))
    if kind or derivative or derivative_audit:
        if not kind or not derivative:
            parser.error("--kind and --derivative are required together")
        candidates.append((kind, derivative, derivative_audit))
    print(json.dumps(compose(candidates=candidates, **args), indent=2))


if __name__ == "__main__":
    main()
