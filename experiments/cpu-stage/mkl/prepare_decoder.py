"""Create an explicit experimental Intel decoder copy; never edit the accepted graph."""
import argparse
import copy
import json
from pathlib import Path
import matrix_common as b

PIN = '34ccdc4b835d04c6a240c7cd2c8025995d22b4c9f3877e61e0b9ee7524d5de66'
DOMAIN = 'venky.audio.intel.decoder.experimental'
SIGNATURES = {(8192,2048):2, (1024,1024):3, (3072,1024):2, (512,512):3, (1280,512):2}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,choices=(1,2),required=True)
    a=p.parse_args();b.libraries()
    if a.source.resolve()==a.output.resolve() or a.output.exists():
        raise ValueError('Source is immutable; output must be new')
    if b.sha(a.source)!=PIN:
        raise ValueError('Expected exact accepted default Linux graph')
    model=b.onnx.load(a.source,load_external_data=True)
    if model.functions or model.graph.sparse_initializer or any(
            attr.type in (b.onnx.AttributeProto.GRAPH,b.onnx.AttributeProto.GRAPHS)
            for node in model.graph.node for attr in node.attribute):
        raise ValueError('Only a flat graph without local functions/sparse initializers is supported')
    candidates=b.matrix_nodes(model)
    selected={node.name:(node,w,v,signature) for node,w,v,signature in candidates
              if tuple(w.dims) in SIGNATURES}
    if len(selected)!=12:
        raise ValueError('Expected exactly 12 selected nodes')
    from collections import Counter
    if Counter(tuple(row[1].dims) for row in selected.values())!=Counter(SIGNATURES):
        raise ValueError('Unexpected selected matrix geometry')
    nodes=[];report=[]
    for node in model.graph.node:
        if node.name not in selected:
            nodes.append(copy.deepcopy(node));continue
        old,w,v,signature=selected[node.name]
        m,k=map(int,w.dims)
        nodes.append(b.helper.make_node('IntelPlainMatMulF32',list(node.input),list(node.output),
                    name=node.name,domain=DOMAIN,M=m,K=k,N=0,mode=0,shards=a.threads))
        report.append({'name':node.name,'M':m,'K':k,'dynamic_time':True,'weights':w.name})
    del model.graph.node[:];model.graph.node.extend(nodes)
    if any(op.domain==DOMAIN for op in model.opset_import):
        raise ValueError('Candidate domain already present')
    model.opset_import.append(b.helper.make_opsetid(DOMAIN,1))
    b.onnx.checker.check_model(model)
    a.output.parent.mkdir(parents=True,exist_ok=True);b.onnx.save(model,a.output)
    record={'source':str(a.source.resolve()),'source_sha256':PIN,'output':str(a.output.resolve()),
            'output_sha256':b.sha(a.output),'threads':a.threads,'nodes':report,
            'scope':'Experimental global matrix selection, fixed before multilingual results',
            'selection':'Five shapes with K>=512 improved at both thread counts; 12 nodes total',
            'math':'Original FP32 weights; full K; alpha1 beta0; no quantization, pruning, BF16 or GPU',
            'weights_packed':False,'blas_threading':'sequential; ORT owns 1/2 row workers'}
    a.output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'nodes':len(report),'threads':a.threads}),flush=True)

if __name__=='__main__':
    main()
