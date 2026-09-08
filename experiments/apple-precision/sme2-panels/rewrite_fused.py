"""Derive four Apple fused late regions and the same 22-product INT8 coverage.

Requires the exact accepted Apple FP32 source. All region connectivity, shape,
constant and exclusive-consumer proofs precede precision replacement.
"""
import argparse,copy,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper
ROOT=Path(__file__).resolve().parent
SOURCE_SHA='fa7992825e807cac7ab1be405912ae3735031fb941a18dd81006005dc597bde3'
MATRIX='fast.audiovae.precision.apple.r4.experimental'
STAGE='fast.audiovae.precision.apple.r4.fused.stage.experimental'
UP='fast.audiovae.precision.apple.r4.fused.upsample.experimental'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'fused'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def rewrite(source,output,package_source,tile_time=512,segments=4,shards=4):
    source,output=Path(source),Path(output)
    if sha(source)!=SOURCE_SHA:raise ValueError('Exact accepted Apple FP32 graph required')
    if output.exists() or output.with_suffix('.json').exists():raise ValueError('Preserve existing output')
    if tile_time not in (64,128,256,512) or not 1<=segments<=64 or not 1<=shards<=64:raise ValueError('Invalid geometry')
    sys.path.insert(0,str(Path(package_source).resolve()))
    output.parent.mkdir(parents=True,exist_ok=True)
    intermediate=output.with_suffix('.proof-stage.onnx')
    if intermediate.exists():raise ValueError('Preserve existing stage proof')
    sr=module('proof_stage').rewrite(source,intermediate,channels=(256,128,64,32),tile_time=tile_time,
        segments=segments,backend=2,matrix_mode=0,matrix_isa=128,package_source=package_source)
    model=onnx.load(intermediate);model,ur=module('proof_upsample').rewrite_model(model,intermediate.parent,
        tile_time=tile_time,segments=segments,projection_mode=0,projection_isa=512)
    constants={v.name:v for v in model.graph.initializer};saved={v.name:v.SerializeToString() for v in model.graph.initializer}
    changed=[];retained=[];nodes=[]
    for n in model.graph.node:
        a={v.name:helper.get_attribute_value(v) for v in n.attribute}
        if n.op_type=='StageStackF32' and n.domain=='fast.audiovae.stage.experimental':
            c=a['channels']
            if c not in (32,64,256):raise ValueError('Unexpected remaining stage')
            a['precision_mode']=8 if c==256 else 0
            nodes.append(helper.make_node(n.op_type,list(n.input),list(n.output),name=n.name,domain=STAGE,**a));continue
        if n.op_type=='UpsampleStageF32' and n.domain=='fast.audiovae.upsample.experimental':
            a.update(backend=2,matrix_mode=0,matrix_isa=128,projection_mode=8,projection_isa=0,precision_mode=8)
            nodes.append(helper.make_node(n.op_type,list(n.input),list(n.output),name=n.name,domain=UP,**a));continue
        if not n.domain and n.op_type=='MatMul':
            if a or len(n.input)!=2 or len(n.output)!=1:raise ValueError('Malformed matrix')
            w=numpy_helper.to_array(constants[n.input[0]])
            if w.dtype!=np.float32 or w.ndim!=2 or not np.isfinite(w).all():raise ValueError('Finite constant weight required')
            m,k=w.shape;record={'node':n.name,'M':m,'K':k,'weight':n.input[0]}
            if min(m,k)>=128:
                nodes.append(helper.make_node('PrecisionMatMulF32',list(n.input),list(n.output),name=n.name,domain=MATRIX,
                    M=m,K=k,native_abi=1,precision_mode=8,backend=3,shards=shards));changed.append(record);continue
            retained.append(record)
        nodes.append(copy.deepcopy(n))
    if len(changed)!=14 or len(retained)!=3:raise ValueError(f'Unexpected remaining matrices: {len(changed)}/{len(retained)}')
    del model.graph.node[:];model.graph.node.extend(nodes)
    used={n.domain for n in nodes};imports=[x for x in model.opset_import if x.domain in used or not x.domain]
    del model.opset_import[:];model.opset_import.extend(imports)
    for domain in (MATRIX,STAGE,UP):model.opset_import.append(helper.make_opsetid(domain,1))
    if {v.name:v.SerializeToString() for v in model.graph.initializer}!=saved:raise RuntimeError('Coefficient changed during precision rewrite')
    onnx.checker.check_model(model,check_custom_domain=False);onnx.save(model,output)
    record={'source_sha256':SOURCE_SHA,'output_sha256':sha(output),'source':str(source),'output':str(output),
        'stage_proof':sr,'upsample_proof':ur,'standalone_selected':changed,'standalone_retained_FP32':retained,
        'selected_products':22,'retained_FP32_products':9,'tile_time':tile_time,'segments':segments,'shards':shards,
        'domains':[MATRIX,STAGE,UP],'original_initializer_bytes_unchanged':True,'inference_executed':False,
        'precision_scope':'Same INT8 weights and per-time-column activation quantization; FP32 C64/C32 complete-K products require parity validation'}
    output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n');return record
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--package-source',required=True,type=Path);p.add_argument('--tile-time',type=int,default=512)
    p.add_argument('--segments',type=int,default=4);p.add_argument('--shards',type=int,default=4)
    print(json.dumps(rewrite(**vars(p.parse_args())),indent=2))
