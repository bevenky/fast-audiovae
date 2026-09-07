"""Checked C128 stride-two projection, phase and residual-stage fusion.

Creates a separate CPU experiment graph. Coefficients and unselected nodes stay
byte-identical. Shape proofs derive from inputs and producer semantics rather
than trusting intermediate value_info. No inference or timing is performed.
"""
from __future__ import annotations
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DOMAIN = 'fast.audiovae.upsample.experimental'
STAGE = 'fast.audiovae.stage.experimental'
NATIVE = 'venky.audio.cpu.portable'
MKL = 'venky.audio.intel.decoder.experimental'
ACCEPTED_SOURCE = 'e188d0609795d256627b4e39b632d5c5ca064256d410899ecb05ac4eb6301bc2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''): h.update(part)
    return h.hexdigest()


def attrs(node):
    if len({a.name for a in node.attribute}) != len(node.attribute):
        raise ValueError('Duplicate node attributes')
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def factors(value):
    return (value, ()) if type(value) is int else value


def product(a, b):
    aa, av = factors(a); bb, bv = factors(b)
    variables = tuple(sorted((*av, *bv)))
    return aa * bb if not variables else (aa * bb, variables)


def quotient(a, b):
    aa, av = factors(a); bb, bv = factors(b)
    left, right = Counter(av), Counter(bv)
    if bb <= 0 or aa % bb or right - left:
        raise ValueError('Unproven reshape volume')
    variables = tuple(sorted((left - right).elements()))
    return aa // bb if not variables else (aa // bb, variables)


class ShapeProof:
    """Limited exact shape algebra for the already supported decoder graph."""
    def __init__(self, model, base_dir):
        self.model, self.base_dir = model, Path(base_dir)
        self.nodes = list(model.graph.node)
        self.initializers = {v.name: v for v in model.graph.initializer}
        self.inputs = {v.name: v for v in model.graph.input}
        self.values = {v.name: v for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
        self.producer = {v: i for i, n in enumerate(self.nodes) for v in n.output}
        if len(self.initializers) != len(model.graph.initializer) or len(self.producer) != sum(len(n.output) for n in self.nodes):
            raise ValueError('Duplicate initializer or produced value')
        if any(v in self.producer for v in (*self.initializers, *self.inputs)):
            raise ValueError('Produced value shadows graph input or initializer')
        if len({n.name for n in self.nodes}) != len(self.nodes) or any(not n.name for n in self.nodes):
            raise ValueError('Unique nonempty node names required')
        self.cache, self.shape_cache, self.active = {}, {}, set()

    def constant(self, name, dtype, dims=None):
        tensor = self.initializers.get(name)
        if tensor is None or name in self.inputs or tensor.data_type != dtype:
            raise ValueError('Immutable coefficient initializer with expected dtype required: ' + name)
        if dims is not None and list(tensor.dims) != list(dims):
            raise ValueError('Coefficient dimensions differ: ' + name)
        value = numpy_helper.to_array(copy.deepcopy(tensor), base_dir=str(self.base_dir))
        if dtype == TensorProto.FLOAT and not np.isfinite(value).all():
            raise ValueError('Nonfinite coefficient: ' + name)
        return value

    def shape_value(self, name):
        if name in self.shape_cache: return self.shape_cache[name]
        if name in self.initializers:
            v = self.constant(name, TensorProto.INT64)
            answer = (tuple(int(x) for x in v.reshape(-1)), tuple(v.shape))
        else:
            node = self.nodes[self.producer[name]]; a = attrs(node)
            if node.domain: raise ValueError('Unsupported shape-program domain')
            if node.op_type == 'Shape' and len(node.input) == 1 and not set(a)-{'start','end'}:
                dims = self.shape(node.input[0]); values = dims[a.get('start',0):a.get('end',len(dims))]
                answer = (values, (len(values),))
            elif node.op_type == 'Mul' and len(node.input) == 2 and not a:
                (left, ls), (right, rs) = [self.shape_value(v) for v in node.input]
                if len(left) != 1 or len(right) != 1 or ls not in ((),(1,)) or rs not in ((),(1,)):
                    raise ValueError('Only scalar shape products are supported')
                answer = ((product(left[0],right[0]),), (1,) if ls or rs else ())
            elif node.op_type == 'Concat' and a == {'axis':0}:
                parts = [self.shape_value(v) for v in node.input]
                if not parts or any(len(s) != 1 for _,s in parts): raise ValueError('Shape Concat requires vectors')
                values = tuple(v for part,_ in parts for v in part); answer = (values,(len(values),))
            elif node.op_type in ('Squeeze','Unsqueeze','Reshape'):
                values, before = self.shape_value(node.input[0])
                if node.op_type == 'Reshape':
                    if len(node.input) != 2 or set(a)-{'allowzero'}: raise ValueError('Malformed shape Reshape')
                    target, target_shape = self.shape_value(node.input[1])
                    if target_shape != (len(target),): raise ValueError('Reshape shape must be vector')
                    after = self.reshape(before,target,a.get('allowzero',0))
                else:
                    if a or len(node.input) not in (1,2): raise ValueError('Malformed shape squeeze')
                    if len(node.input) == 1:
                        if node.op_type != 'Squeeze': raise ValueError('Unsqueeze axes required')
                        axes = [i for i,v in enumerate(before) if v == 1]
                    else:
                        axes, axes_shape = self.shape_value(node.input[1])
                        if axes_shape != (len(axes),) or any(type(v) is not int for v in axes):
                            raise ValueError('Constant integer axes required')
                    rank = len(before) + (len(axes) if node.op_type == 'Unsqueeze' else 0)
                    if any(v < -rank or v >= rank for v in axes): raise ValueError('Shape axis out of range')
                    axes = [v+rank if v < 0 else v for v in axes]
                    if len(set(axes)) != len(axes): raise ValueError('Repeated shape axes')
                    if node.op_type == 'Squeeze':
                        if any(before[v] != 1 for v in axes): raise ValueError('Nonunit squeeze')
                        after = tuple(v for i,v in enumerate(before) if i not in axes)
                    else:
                        old = iter(before); after = tuple(1 if i in axes else next(old) for i in range(rank))
                answer = (values,after)
            else: raise ValueError('Unsupported shape operation: ' + node.op_type)
        self.shape_cache[name] = answer
        return answer

    @staticmethod
    def reshape(before, target, allowzero):
        if allowzero not in (0,1) or target.count(-1) > 1:
            raise ValueError('Malformed Reshape')
        if allowzero and 0 in target and -1 in target: raise ValueError('Ambiguous zero Reshape')
        target = list(target)
        for i,v in enumerate(target):
            if type(v) is int and v < -1: raise ValueError('Negative Reshape dimension')
            if v == 0 and not allowzero:
                if i >= len(before): raise ValueError('Copied Reshape dimension out of range')
                target[i] = before[i]
        total = known = 1
        for v in before: total = product(total,v)
        for v in target:
            if v != -1: known = product(known,v)
        if -1 in target: target[target.index(-1)] = quotient(total,known)
        elif total != known: raise ValueError('Reshape changes volume')
        return tuple(target)

    def shape(self, name):
        if name in self.cache: return self.cache[name]
        if name in self.active: raise ValueError('Cyclic shape dependency')
        self.active.add(name)
        if name in self.inputs:
            value = self.inputs[name]
            if value.type.tensor_type.elem_type != TensorProto.FLOAT: raise ValueError('FP32 graph input required')
            out = tuple(d.dim_value if d.HasField('dim_value') else (1,('symbol:'+d.dim_param,))
                        for d in value.type.tensor_type.shape.dim)
            if any(not d.HasField('dim_value') and not d.dim_param for d in value.type.tensor_type.shape.dim):
                raise ValueError('Declared input dimensions required')
        elif name in self.initializers:
            self.constant(name,TensorProto.FLOAT); out = tuple(self.initializers[name].dims)
        else:
            node = self.nodes[self.producer[name]]; a = attrs(node)
            if len(node.output) != 1: raise ValueError('Single-output shape producer required')
            if not node.domain and node.op_type == 'Reshape' and len(node.input) == 2 and not set(a)-{'allowzero'}:
                target, s = self.shape_value(node.input[1])
                if s != (len(target),): raise ValueError('Reshape target must be vector')
                out = self.reshape(self.shape(node.input[0]),target,a.get('allowzero',0))
            elif (not node.domain and node.op_type == 'MatMul' and not a) or (node.domain == MKL and node.op_type == 'IntelPlainMatMulF32'):
                if len(node.input) != 2: raise ValueError('Invalid matrix inputs')
                left,right = [self.shape(v) for v in node.input]
                if len(left) != 2 or len(right) != 3 or left[1] != right[1]: raise ValueError('Unsupported BCT matrix shape')
                if node.domain == MKL and (a.get('M'),a.get('K'),a.get('mode'),a.get('N')) != (*left,0,0):
                    raise ValueError('Unsupported MKL shape contract')
                out = (right[0],left[0],right[2])
            elif not node.domain and node.op_type in ('Add','Mul') and len(node.input) == 2 and not a:
                left,right = [self.shape(v) for v in node.input]; rank=max(len(left),len(right))
                left=(1,)*(rank-len(left))+left; right=(1,)*(rank-len(right))+right
                if any(x != y and x != 1 and y != 1 for x,y in zip(left,right)): raise ValueError('Unproven broadcasting')
                out=tuple(y if x == 1 else x for x,y in zip(left,right))
            elif node.domain == NATIVE and node.op_type in ('SnakeF32','CausalDW7F32','CausalDW7SnakeF32','SnakeDW7SnakeF32','BiasResidualF32','PhaseSumBiasInterleaveF32'):
                out=self.shape(node.input[0]); c=a.get('channels')
                if len(out) != 3 or a.get('native_abi') != 1: raise ValueError('Invalid native shape contract')
                if node.op_type == 'PhaseSumBiasInterleaveF32':
                    if a.get('stride') not in (2,5,6,8) or out[1] != c*a['stride']: raise ValueError('Invalid phase shape')
                    out=(out[0],c,product(out[2],a['stride']))
                elif out[1] != c: raise ValueError('Invalid native channels')
            elif node.domain == STAGE and node.op_type == 'StageStackF32':
                out=self.shape(node.input[0])
                if len(out) != 3 or out[1] != a.get('channels') or a.get('native_abi') != 1 or len(node.input) != 25:
                    raise ValueError('Invalid stage shape contract')
            else: raise ValueError('Unsupported activation shape producer: ' + node.name)
        if any(type(v) is int and v < 0 for v in out): raise ValueError('Negative dimension')
        declared=self.values.get(name)
        if declared is not None:
            t=declared.type.tensor_type
            if t.elem_type != TensorProto.FLOAT or len(t.shape.dim) != len(out): raise ValueError('Contradictory activation dtype/rank')
            if any(d.HasField('dim_value') and d.dim_value != v for d,v in zip(t.shape.dim,out)):
                raise ValueError('Declaration contradicts producer dimensions')
        self.cache[name]=out; self.active.remove(name)
        return out


def rewrite_model(model, base_dir='.', *, tile_time=128, segments=2, projection_mode=0, projection_isa=None):
    if type(tile_time) is not int or tile_time not in (64,128,256) or type(segments) is not int or not 1 <= segments <= 64:
        raise ValueError('Invalid tile or segment count')
    if type(projection_mode) is not int or projection_mode not in (0,1): raise ValueError('Projection mode must be 0 or 1')
    expected_isa=512 if projection_mode == 0 else 0
    if projection_isa is None: projection_isa=expected_isa
    if type(projection_isa) is not int or projection_isa != expected_isa: raise ValueError('Projection ISA disagrees with mode')
    if model.functions or model.graph.sparse_initializer or any(a.type in (onnx.AttributeProto.GRAPH,onnx.AttributeProto.GRAPHS) for n in model.graph.node for a in n.attribute):
        raise ValueError('Flat dense graph required')
    if DOMAIN in {v.domain for v in model.opset_import} or any(n.domain == DOMAIN for n in model.graph.node):
        raise ValueError('Already an upsample candidate')
    for domain in (NATIVE,STAGE):
        if [v.version for v in model.opset_import if v.domain == domain] != [1]: raise ValueError('Expected native/stage opset1')
    graph=ShapeProof(model,base_dir); nodes=graph.nodes
    selected=[(i,n) for i,n in enumerate(nodes) if n.domain == STAGE and n.op_type == 'StageStackF32' and attrs(n).get('channels') == 128]
    if len(selected) != 1: raise ValueError('Exactly one C128 stage required')
    stage_i,stage=selected[0]; a=attrs(stage)
    required={'native_abi','channels','tile_time','segments','backend','matrix_mode','matrix_isa'}
    if set(a) != required or any(type(v) is not int for v in a.values()) or len(stage.input) != 25 or len(stage.output) != 1:
        raise ValueError('Malformed C128 stage')
    if (a['native_abi'],a['backend'],a['matrix_mode'],a['matrix_isa']) != (1,5,0,512) or a['tile_time'] not in (64,128,256) or not 1 <= a['segments'] <= 64:
        raise ValueError('Existing C128 stage must use direct AVX512')
    internal={stage_i}
    def views(value):
        while value in graph.producer:
            i=graph.producer[value]; n=nodes[i]
            if n.domain or n.op_type != 'Reshape': break
            if len(n.input) != 2 or len(n.output) != 1 or graph.shape(value) != graph.shape(n.input[0]):
                raise ValueError('Intermediate Reshape is not an identical BCT view')
            internal.add(i); value=n.input[0]
        return value
    phase_value=views(stage.input[0]); phase_i=graph.producer.get(phase_value)
    if phase_i is None: raise ValueError('Missing phase producer')
    phase=nodes[phase_i]; pa=attrs(phase)
    if phase.domain != NATIVE or phase.op_type != 'PhaseSumBiasInterleaveF32' or len(phase.input) != 3 or len(phase.output) != 1:
        raise ValueError('Ordinary two-input phase sum required')
    if set(pa) != {'channels','native_abi','stride','previous_shift','row_batches'} or any(type(v) is not int for v in pa.values()):
        raise ValueError('Malformed phase attributes')
    if (pa['channels'],pa['native_abi'],pa['stride'],pa['previous_shift']) != (128,1,2,1) or pa['row_batches'] < 0:
        raise ValueError('Expected C128 stride2, current then unshifted previous phase contract')
    internal.add(phase_i); projections=[]
    for value in phase.input[:2]:
        value=views(value); i=graph.producer.get(value)
        if i is None: raise ValueError('Missing projection')
        n=nodes[i]
        if n.domain or n.op_type != 'MatMul' or attrs(n) or len(n.input) != 2 or len(n.output) != 1:
            raise ValueError('Projection must be an ordinary complete MatMul')
        graph.constant(n.input[0],TensorProto.FLOAT,[256,256]); internal.add(i); projections.append(n)
    if projections[0] is projections[1] or projections[0].input[1] != projections[1].input[1]:
        raise ValueError('Two distinct projections must consume the same activation')
    x=projections[0].input[1]; xs=graph.shape(x)
    if len(xs) != 3 or xs[1] != 256: raise ValueError('Expected BCT input [B,256,T]')
    expected=(xs[0],128,product(xs[2],2))
    if graph.shape(phase.output[0]) != expected or graph.shape(stage.input[0]) != expected or graph.shape(stage.output[0]) != expected:
        raise ValueError('Phase or stage dimensions differ from [B,128,2T]')
    graph.constant(phase.input[2],TensorProto.FLOAT,[128])
    shapes=([128,1,7],[128],[128],[128],[128],[128],[128,128],[128])
    for unit in range(3):
        for name,dims in zip(stage.input[1+8*unit:9+8*unit],shapes): graph.constant(name,TensorProto.FLOAT,dims)
    users={}
    for i,n in enumerate(nodes):
        for value in n.input: users.setdefault(value,[]).append(i)
    exposed={v.name for v in model.graph.output}; final=stage.output[0]
    for i in internal:
        for value in nodes[i].output:
            if value == final: continue
            if value in exposed or any(j not in internal for j in users.get(value,[])):
                raise ValueError('Region intermediate has an external consumer or graph output: ' + value)
    if x in {v for i in internal for v in nodes[i].output}: raise ValueError('Activation boundary is inside region')
    name='upsample_stage_c128'
    occupied={n.name for n in nodes}|{v for n in nodes for v in (*n.input,*n.output)}|set(graph.initializers)
    if name in occupied: raise ValueError('Generated name collision')
    inputs=[x,projections[0].input[0],projections[1].input[0],phase.input[2],*stage.input[1:]]
    replacement=helper.make_node('UpsampleStageF32',inputs,[final],name=name,domain=DOMAIN,
        native_abi=1,channels=128,stride=2,tile_time=tile_time,segments=segments,backend=5,
        matrix_mode=0,matrix_isa=512,projection_mode=projection_mode,projection_isa=projection_isa)
    result=copy.deepcopy(model)
    kept=[replacement if i == stage_i else copy.deepcopy(n) for i,n in enumerate(nodes) if i not in internal or i == stage_i]
    del result.graph.node[:]; result.graph.node.extend(kept)
    result.opset_import.append(helper.make_opsetid(DOMAIN,1))
    removed_values={v for i in internal if i != stage_i for v in nodes[i].output}
    infos=[v for v in result.graph.value_info if v.name not in removed_values]
    del result.graph.value_info[:]; result.graph.value_info.extend(infos)
    if [v.SerializeToString() for v in model.graph.initializer] != [v.SerializeToString() for v in result.graph.initializer]:
        raise RuntimeError('Original initializer changed')
    unaffected=[n.SerializeToString() for i,n in enumerate(nodes) if i not in internal]
    if unaffected != [n.SerializeToString() for n in result.graph.node if n.name != name]:
        raise RuntimeError('Unselected node changed')
    record={'node':name,'input':x,'output':final,'projection_outputs':[n.output[0] for n in projections],
        'phase_output':phase.output[0],'stage_input':stage.input[0],'stage_node':stage.name,
        'removed_nodes':[nodes[i].name for i in sorted(internal)],'input_names':inputs,
        'channels':128,'stride':2,'tile_time':tile_time,'segments':segments,'backend':5,
        'matrix_mode':0,'matrix_isa':512,'projection_mode':projection_mode,'projection_isa':projection_isa}
    return result,{'count':1,'changes':[record],'source_nodes':len(nodes),'output_nodes':len(kept),
        'original_initializers_unchanged':True,'unselected_nodes_unchanged':True,
        'proof':'Producer-derived BCT shapes, immutable FP32 coefficients, ordered phase inputs and exclusive internal consumers',
        'projection_rounding':'Complete-K FP32 projection reductions may differ from ORT MatMul; numerical validation required',
        'inference_executed':False,'default_promoted':False}


def rewrite(source, output, *, package_source=None, **options):
    if package_source: sys.path.insert(0,str(Path(package_source).resolve()))
    from fast_audiovae.graph.elementwise import copy_external_data
    source,output=Path(source),Path(output)
    audit_path=output.with_suffix('.upsample.json')
    if source.resolve() == output.resolve() or output.exists() or audit_path.exists():
        raise ValueError('Preserve source and existing output artifacts')
    source_sha=sha(source)
    onnx.checker.check_model(str(source),full_check=False)
    original=onnx.load(source,load_external_data=False)
    result,report=rewrite_model(original,source.parent,**options)
    output.parent.mkdir(parents=True,exist_ok=True)
    external=copy_external_data(result,source.parent,output.parent)
    onnx.save_model(result,output); onnx.checker.check_model(str(output),full_check=False)
    if sha(source) != source_sha: raise RuntimeError('Source changed during rewrite')
    report.update(source_sha256=source_sha,output_sha256=sha(output),external_files=external,
        accepted_source_hash_match=source_sha == ACCEPTED_SOURCE,source_ir=original.ir_version,output_ir=result.ir_version)
    audit_path.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True); p.add_argument('--output',required=True)
    p.add_argument('--tile-time',type=int,default=128); p.add_argument('--segments',type=int,default=2)
    p.add_argument('--projection-mode',type=int,choices=(0,1),default=0)
    p.add_argument('--projection-isa',type=int,choices=(0,512))
    p.add_argument('--package-source',type=Path,default=Path(__file__).resolve().parents[3]/'src')
    print(json.dumps(rewrite(**vars(p.parse_args())),indent=2))
