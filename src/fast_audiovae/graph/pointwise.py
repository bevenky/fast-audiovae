"""Lower supported dense k1 Conv to BCT MatMul without activation transposes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def rewrite(source, output, layout="bct"):
    model = onnx.load(str(source), load_external_data=True)
    initializers = {x.name: x for x in model.graph.initializer}
    changes, nodes = [], []
    for node in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        weight = initializers.get(node.input[1]) if node.op_type == "Conv" and len(node.input)>1 else None
        eligible = (node.domain in ("", "ai.onnx") and weight is not None and weight.data_type == onnx.TensorProto.FLOAT
                    and weight.name not in {x.name for x in model.graph.input}
                    and len(weight.dims) == 3 and weight.dims[-1] == 1
                    and attrs.get("group", 1) == 1
                    and list(attrs.get("strides", [1])) == [1]
                    and list(attrs.get("dilations", [1])) == [1]
                    and not any(attrs.get("pads", [0, 0]))
                    and attrs.get("auto_pad", b"NOTSET") in (b"NOTSET", b"VALID"))
        if not eligible:
            nodes.append(node)
            continue
        prefix = node.name + "__bct_mm"
        w = numpy_helper.to_array(weight).copy().reshape(weight.dims[0], weight.dims[1])
        wname = prefix + "_weight"
        has_bias = len(node.input)>2 and bool(node.input[2])
        mm_out = prefix+"_product" if has_bias else node.output[0]
        if layout=="bct":
            model.graph.initializer.append(numpy_helper.from_array(w, wname))
            nodes.append(helper.make_node("MatMul", [wname, node.input[0]], [mm_out], name=prefix))
        else:
            model.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(w.T), wname))
            xt=prefix+"_btc_input";yt=prefix+"_btc_product"
            nodes.append(helper.make_node("Transpose",[node.input[0]],[xt],name=prefix+"_input_transpose",perm=[0,2,1]))
            nodes.append(helper.make_node("MatMul",[xt,wname],[yt],name=prefix))
            nodes.append(helper.make_node("Transpose",[yt],[mm_out],name=prefix+"_output_transpose",perm=[0,2,1]))
        if has_bias:
            bias = initializers.get(node.input[2])
            if (bias is None or bias.name in {x.name for x in model.graph.input}
                    or bias.data_type != onnx.TensorProto.FLOAT or list(bias.dims) != [weight.dims[0]]):
                raise ValueError(f"Unsupported bias on {node.name}")
            bname=prefix+"_bias"
            model.graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(bias).copy().reshape(1,-1,1), bname))
            nodes.append(helper.make_node("Add", [mm_out,bname], list(node.output), name=prefix+"_add"))
        changes.append({"node":node.name,"weight_shape":list(weight.dims),"bias":has_bias})
    if not changes:
        raise ValueError("No supported pointwise convolutions found")
    del model.graph.node[:];model.graph.node.extend(nodes)
    used={x for node in nodes for x in node.input}|{x.name for x in model.graph.output}
    kept=[x for x in model.graph.initializer if x.name in used]
    del model.graph.initializer[:];model.graph.initializer.extend(kept)
    # Embedding original external tensors gives each derived artifact its own
    # unambiguous weights; the original assets are untouched.
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            onnx.external_data_helper.convert_model_from_external_data(model)
            break
    onnx.checker.check_model(model, full_check=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,str(output))
    manifest={"source":str(source.resolve()),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
              "output":str(output.resolve()),"output_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
              "kind":"pointwise MatMul", "layout":layout, "changes":changes,
              "packing":"BCT puts weights on A, no persistent ORT B packing; BTC puts weights on B, includes two activation transposes",
              "numerics":"same FP32 weights and function; dense reduction rounding must be validated"}
    output.with_suffix(".pointwise.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest
