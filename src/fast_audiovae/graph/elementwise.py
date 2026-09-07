"""Create separate, explicitly opted-in native CPU AudioVAE2 ONNX variants.

Rewrites only verified FP32 BCT Snake and causal depthwise Conv1d(k=7) patterns.
The source model/external data are never modified. External data are copied
byte-for-byte once into the output directory; tensor offsets are not repacked.
Full decoder inference/benchmarks are intentionally not part of this script.
"""
from __future__ import annotations
import argparse
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DOMAIN = "venky.audio.cpu"
VARIANTS = ("snake", "dw", "snake_dw", "fused")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_capabilities(library):
    lib = ctypes.CDLL(str(Path(library).resolve()))
    lib.ncc_abi_version.restype = ctypes.c_uint32
    lib.ncc_capabilities.restype = ctypes.c_uint64
    if lib.ncc_abi_version() != 1:
        raise RuntimeError("Native C library ABI must be 1")
    caps = int(lib.ncc_capabilities())
    return {"library": str(Path(library).resolve()), "library_sha256": sha256(library),
            "native_abi": 1, "capability_bits": caps, "vforce": bool(caps & 32)}


class Graph:
    def __init__(self, model, base_dir):
        self.model = model
        self.base_dir = Path(base_dir)
        self.nodes = list(model.graph.node)
        self.producer = {name: i for i, n in enumerate(self.nodes) for name in n.output}
        self.users = {}
        for i, n in enumerate(self.nodes):
            for name in set(n.input):
                self.users.setdefault(name, []).append(i)
        self.outputs = {v.name for v in model.graph.output}
        self.inputs = {v.name for v in model.graph.input}
        self.initializers = {t.name: t for t in model.graph.initializer}
        self.values = {v.name: v for v in [*model.graph.value_info, *model.graph.input, *model.graph.output]}
        self.const_cache = {}

    def attrs(self, node):
        return {a.name: helper.get_attribute_value(a) for a in node.attribute}

    def shape(self, name):
        v = self.values.get(name)
        if v is None or not v.type.HasField("tensor_type"):
            return None
        t = v.type.tensor_type
        if t.elem_type != TensorProto.FLOAT or not t.HasField("shape"):
            return None
        return tuple(d.dim_value if d.HasField("dim_value") else d.dim_param or None for d in t.shape.dim)

    def channels(self, name):
        shape = self.shape(name)
        return shape[1] if shape and len(shape) == 3 and isinstance(shape[1], int) and shape[1] > 0 else None

    def constant(self, name):
        if name in self.inputs:
            return None  # Overridable initializer is not an immutable coefficient.
        if name in self.const_cache:
            return self.const_cache[name]
        tensor = self.initializers.get(name)
        if tensor is None:
            index = self.producer.get(name)
            node = None if index is None else self.nodes[index]
            if node is None or node.domain or node.op_type != "Constant":
                return None
            tensor = self.attrs(node).get("value")
            if tensor is None:
                return None
        # to_array may load external bytes into the TensorProto: use a copy so
        # saving the derivative cannot accidentally inline/repack old weights.
        result = numpy_helper.to_array(copy.deepcopy(tensor), base_dir=str(self.base_dir))
        self.const_cache[name] = result
        return result

    def channel_constant(self, name, channels):
        a = self.constant(name)
        if a is None or a.dtype != np.float32 or not np.isfinite(a).all():
            return None
        # ONNX right-aligned broadcasting: [C] would broadcast over TIME and
        # must not be reinterpreted as channel-wise. [C,1] is valid for BCT.
        if a.shape in ((), (1,), (1, 1), (1, 1, 1)):
            return np.full((channels,), a.reshape(-1)[0], dtype=np.float32)
        if a.shape not in ((channels, 1), (1, channels, 1)):
            return None
        return np.array(a.reshape(channels), dtype=np.float32, copy=True)

    def only_user(self, name, op_type=None):
        users = self.users.get(name, [])
        if name in self.outputs or len(users) != 1:
            return None
        i = users[0]
        n = self.nodes[i]
        return i if not n.domain and (op_type is None or n.op_type == op_type) else None

    def snake_matches(self):
        if sys.flags.optimize:
            raise RuntimeError("Run this checked graph rewriter without python -O/PYTHONOPTIMIZE")
        matches, rejected = [], []
        for sin_i, sin in enumerate(self.nodes):
            if sin.domain or sin.op_type != "Sin" or len(sin.input) != 1:
                continue
            try:
                mul_i = self.producer.get(sin.input[0])
                assert mul_i is not None
                mul = self.nodes[mul_i]
                assert not mul.domain and mul.op_type == "Mul" and len(mul.input) == 2
                assert self.only_user(mul.output[0], "Sin") == sin_i
                candidates = []
                for const_name, x in (mul.input, reversed(mul.input)):
                    channels = self.channels(x)
                    alpha = self.channel_constant(const_name, channels) if channels else None
                    if alpha is not None:
                        candidates.append((x, const_name, channels, alpha))
                assert len(candidates) == 1
                x, alpha_name, channels, alpha = candidates[0]
                square_i = self.only_user(sin.output[0])
                assert square_i is not None
                square = self.nodes[square_i]
                if square.op_type == "Pow":
                    assert square.input[0] == sin.output[0]
                    exponent = self.constant(square.input[1])
                    assert exponent is not None and exponent.size == 1 and float(exponent.reshape(-1)[0]) == 2.0
                else:
                    assert square.op_type == "Mul" and list(square.input) == [sin.output[0], sin.output[0]]
                correction_i = self.only_user(square.output[0], "Mul")
                assert correction_i is not None
                correction = self.nodes[correction_i]
                other = [v for v in correction.input if v != square.output[0]]
                assert len(other) == 1
                reciprocal = self.channel_constant(other[0], channels)
                assert reciprocal is not None
                add_i = self.only_user(correction.output[0], "Add")
                assert add_i is not None
                add = self.nodes[add_i]
                assert sorted(add.input) == sorted([x, correction.output[0]])
                assert self.shape(add.output[0]) == self.shape(x)
                matches.append({"root": add_i, "remove": {mul_i, sin_i, square_i, correction_i, add_i},
                                "x": x, "output": add.output[0], "channels": channels,
                                "alpha_name": alpha_name, "reciprocal_name": other[0],
                                "alpha": alpha, "reciprocal": reciprocal})
            except (AssertionError, IndexError, TypeError, ValueError):
                rejected.append(sin.name or sin.output[0])
        return matches, rejected

    def identity_reshape(self, node):
        """Prove a BCT Reshape preserves dimensions, not merely annotations."""
        if node.domain or node.op_type != "Reshape" or len(node.input) != 2 or len(node.output) != 1:
            return False
        before, after = self.shape(node.input[0]), self.shape(node.output[0])
        target = self.constant(node.input[1])
        if before is None or before != after or len(before) != 3 or target is None or target.shape != (3,):
            return False
        attrs = self.attrs(node)
        allowzero = attrs.get("allowzero", 0)
        if set(attrs) - {"allowzero"} or not isinstance(allowzero, int) or allowzero not in (0, 1):
            return False
        if target.dtype != np.int64 or np.count_nonzero(target == -1) > 1:
            return False
        # With no literal zero, allowzero does not change Reshape semantics.
        # Under allowzero=1 a zero means an empty dimension, never a copy.
        # Leave all such targets intact instead of inferring identity from
        # matching annotations. This also rejects the invalid zero/-1 mix.
        if allowzero == 1 and np.any(target == 0):
            return False
        # Under allowzero=0, zero copies the corresponding input dimension.
        # -1 infers the only remaining dimension once all others are preserved.
        for old, new in zip(before, target.tolist()):
            if new in (0, -1):
                continue
            if not isinstance(old, int) or old <= 0 or new != old:
                return False
        return True

    def dw_matches(self):
        result = []
        for i, node in enumerate(self.nodes):
            if node.domain or node.op_type != "Conv" or len(node.input) != 3:
                continue
            a = self.attrs(node)
            w = self.initializers.get(node.input[1])
            b = self.initializers.get(node.input[2])
            c = self.channels(node.input[0])
            if not c or w is None or b is None or node.input[1] in self.inputs or node.input[2] in self.inputs:
                continue
            if w.data_type != TensorProto.FLOAT or b.data_type != TensorProto.FLOAT:
                continue
            if list(w.dims) != [c, 1, 7] or list(b.dims) != [c] or a.get("group", 1) != c:
                continue
            if self.shape(node.output[0]) != self.shape(node.input[0]):
                continue
            d = a.get("dilations", [1])
            if len(d) != 1 or not isinstance(d[0], int) or not 0 < d[0] <= (2**31 - 1) // 6:
                continue
            if a.get("strides", [1]) != [1] or a.get("pads", [0, 0]) != [6 * d[0], 0]:
                continue
            if a.get("auto_pad", b"NOTSET") != b"NOTSET" or a.get("kernel_shape", [7]) != [7]:
                continue
            # Reject unknown Conv attributes rather than silently changing them.
            if set(a) - {"group", "dilations", "strides", "pads", "auto_pad", "kernel_shape"}:
                continue
            result.append({"root": i, "x": node.input[0], "output": node.output[0], "channels": c,
                           "weight": node.input[1], "bias": node.input[2], "dilation": d[0]})
        return result


def prune(model):
    """Reverse liveness from graph outputs; graph inputs remain unchanged."""
    live = {v.name for v in model.graph.output}
    keep = []
    for n in reversed(model.graph.node):
        if any(name in live for name in n.output):
            keep.append(n)
            live.update(name for name in n.input if name)
    kept = list(reversed(keep))
    del model.graph.node[:]
    model.graph.node.extend(kept)
    initializers = [t for t in model.graph.initializer if t.name in live]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    defined = {v.name for v in model.graph.input} | {t.name for t in initializers}
    defined.update(name for n in kept for name in n.output)
    infos = [v for v in model.graph.value_info if v.name in defined]
    del model.graph.value_info[:]
    model.graph.value_info.extend(infos)


def rewrite(model, base_dir, variant, *, enable_snake=True, row_batches=0):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if model.graph.sparse_initializer or model.functions or any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for n in model.graph.node for a in n.attribute):
        raise ValueError("This audited rewriter handles a flat dense graph without local functions only")
    graph = Graph(model, base_dir)
    snakes, rejected = graph.snake_matches()
    depths = graph.dw_matches()
    model = copy.deepcopy(model)
    use_snake = enable_snake and variant in ("snake", "snake_dw", "fused")
    use_dw = variant in ("dw", "snake_dw", "fused")
    replacements, removed, records = {}, set(), []
    used_names = {v for n in graph.nodes for v in [*n.input, *n.output]} | set(graph.initializers)

    def coefficient(value, hint):
        name = hint
        while name in used_names:
            name += "_"
        used_names.add(name)
        model.graph.initializer.append(numpy_helper.from_array(value, name))
        return name

    common = {"native_abi": 1, "backend": 0, "row_batches": row_batches}
    fused_dw = set()
    dw_by_output = {d["output"]: d for d in depths}
    for s in snakes if use_snake else []:
        d, reshape_i = None, None
        if variant == "fused":
            d = dw_by_output.get(s["x"])
            if d is None:
                r_i = graph.producer.get(s["x"])
                r = graph.nodes[r_i] if r_i is not None else None
                if r is not None and not r.domain and r.op_type == "Reshape":
                    maybe = dw_by_output.get(r.input[0])
                    # A reshape with provably identical shape and sole DW
                    # ownership is an identity; preserve all other reshapes.
                    if maybe and graph.only_user(maybe["output"], "Reshape") == r_i and graph.identity_reshape(r):
                        d, reshape_i = maybe, r_i
            if d:
                allowed = s["remove"] | ({reshape_i} if reshape_i is not None else set())
                if s["x"] in graph.outputs or any(u not in allowed for u in graph.users.get(s["x"], [])):
                    d = None
                if d and reshape_i is None and any(u not in s["remove"] for u in graph.users.get(d["output"], [])):
                    d = None
        alpha = coefficient(s["alpha"], s["output"] + "__ncc_alpha")
        reciprocal = coefficient(s["reciprocal"], s["output"] + "__ncc_reciprocal")
        attrs = {**common, "channels": s["channels"], "require_vforce": 1}
        if d:
            attrs["dilation"] = d["dilation"]
            inputs = [d["x"], d["weight"], d["bias"], alpha, reciprocal]
            op_type = "CausalDW7SnakeF32"
            removed.add(d["root"])
            if reshape_i is not None:
                removed.add(reshape_i)
            fused_dw.add(d["root"])
        else:
            inputs, op_type = [s["x"], alpha, reciprocal], "SnakeF32"
        replacements[s["root"]] = helper.make_node(op_type, inputs, [s["output"]], name="ncc_" + graph.nodes[s["root"]].name, domain=DOMAIN, **attrs)
        removed.update(s["remove"])
        records.append({"kind": op_type, "source_node": graph.nodes[s["root"]].name, "output": s["output"],
                        "channels": s["channels"], "alpha_source": s["alpha_name"], "reciprocal_source": s["reciprocal_name"],
                        "fused_dw_source": graph.nodes[d["root"]].name if d else None})
    for d in depths if use_dw else []:
        if d["root"] in fused_dw:
            continue
        replacements[d["root"]] = helper.make_node("CausalDW7F32", [d["x"], d["weight"], d["bias"]], [d["output"]],
            name="ncc_" + graph.nodes[d["root"]].name, domain=DOMAIN, **common, channels=d["channels"], dilation=d["dilation"])
        removed.add(d["root"])
        records.append({"kind": "CausalDW7F32", "source_node": graph.nodes[d["root"]].name,
                        "output": d["output"], "channels": d["channels"], "dilation": d["dilation"]})
    nodes = []
    for i, n in enumerate(graph.nodes):
        if i in replacements:
            nodes.append(replacements[i])
        elif i not in removed:
            nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    if replacements:
        existing = next((o for o in model.opset_import if o.domain == DOMAIN), None)
        if existing is None:
            model.opset_import.append(helper.make_opsetid(DOMAIN, 1))
        elif existing.version != 1:
            raise ValueError("Custom domain opset mismatch")
    prune(model)
    return model, {"variant": variant, "matched_snakes": len(snakes), "rejected_snakes": rejected,
                   "matched_depthwise": len(depths), "snake_enabled": use_snake,
                   "source_nodes": len(graph.nodes), "output_nodes": len(model.graph.node),
                   "custom_nodes": len(records), "fused_depthwise": len(fused_dw), "replacements": records}


def copy_external_data(model, source_dir, output_dir):
    files = {}
    for t in model.graph.initializer:
        if t.data_location != TensorProto.EXTERNAL:
            continue
        fields = {v.key: v.value for v in t.external_data}
        location = fields.get("location")
        if not location or Path(location).is_absolute() or ".." in Path(location).parts:
            raise ValueError("Only safe relative external-data locations are supported")
        source = (Path(source_dir) / location).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        # Check external-data ranges without materializing the weight tensor.
        offset = int(fields.get("offset", 0))
        length = int(fields.get("length", source.stat().st_size - offset))
        if offset < 0 or length < 0 or offset + length > source.stat().st_size:
            raise ValueError(f"Invalid external-data bounds: {t.name}")
        if location in files:
            continue
        destination = Path(output_dir) / location
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256(source)
        if destination.exists():
            if sha256(destination) != digest:
                raise RuntimeError(f"Refusing to overwrite nonmatching external-data file: {destination}")
        else:
            shutil.copy2(source, destination)
        files[location] = {"source": str(source), "copy": str(destination.resolve()), "sha256": digest, "bytes": source.stat().st_size}
    return files
