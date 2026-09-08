"""Replace the 22 selected large matrix products in the pinned Apple graph."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper
SOURCE_SHA='fa7992825e807cac7ab1be405912ae3735031fb941a18dd81006005dc597bde3'
DOMAIN='fast.audiovae.precision.apple.experimental'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rewrite(source,output,shards=4,backend=2):
    source=Path(source);output=Path(output)
    if sha(source)!=SOURCE_SHA:raise ValueError('Expected exact accepted Apple FP32 graph')
    if output.exists():raise ValueError('Output exists')
    if backend not in (0,2) or shards not in range(1,65):raise ValueError('Invalid backend/shards')
    model=onnx.load(source);assert not model.functions
    constants={x.name:x for x in model.graph.initializer}
    saved={x.name:x.SerializeToString() for x in model.graph.initializer}
    changed=[];retained=[];nodes=[]
    for n in model.graph.node:
        if n.domain=='' and n.op_type=='MatMul':
            assert not n.attribute and len(n.input)==2 and len(n.output)==1
            assert n.input[0] in constants
            w=numpy_helper.to_array(constants[n.input[0]])
            assert w.dtype==np.float32 and w.ndim==2 and np.isfinite(w).all()
            m,k=w.shape
            row={'node':n.name,'weight':n.input[0],'shape':[m,k],
                 'weight_bytes_sha256':hashlib.sha256(w.tobytes()).hexdigest()}
            if min(m,k)>=128:
                nodes.append(helper.make_node('PrecisionMatMulF32',list(n.input),list(n.output),
                    name=n.name,domain=DOMAIN,native_abi=1,M=m,K=k,shards=shards,precision_mode=8,backend=backend))
                changed.append(row);continue
            retained.append(row)
        nodes.append(copy.deepcopy(n))
    assert len(changed)==22 and len(retained)==9,(len(changed),len(retained))
    del model.graph.node[:];model.graph.node.extend(nodes)
    model.opset_import.append(helper.make_opsetid(DOMAIN,1))
    assert {x.name:x.SerializeToString() for x in model.graph.initializer}==saved
    onnx.checker.check_model(model,check_custom_domain=False)
    output.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,output)
    record={'source_path':str(source.resolve()),'source_sha256':SOURCE_SHA,'path':str(output.resolve()),
            'sha256':sha(output),'changed':changed,'retained_FP32_products':retained,
            'learned_initializer_bytes_unchanged':True,'FP32_nonlinear_nodes_unchanged':True,
            'changed_product_count':22,'domain':DOMAIN,'backend':backend,'shards':shards,
            'GPU_used':False,'scope':'Same selective INT8 matrix recipe; Apple FP32 nonlinearities and nine small products retained'}
    output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n');return record
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--shards',type=int,default=4)
    p.add_argument('--backend',type=int,choices=(0,2),default=2);a=p.parse_args()
    print(json.dumps(rewrite(a.source,a.output,a.shards,a.backend),indent=2))
if __name__=='__main__':main()
