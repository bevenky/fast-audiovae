"""Prepare one explicitly selected generic-wrapped triple region; no inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper
DOMAIN='fast.audiovae.amd.rawhistory.a2.v1'
OP='RawHistorySnakeDW7SnakeF32'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def attrs(n):
    assert len({a.name for a in n.attribute})==len(n.attribute)
    return {a.name:helper.get_attribute_value(a) for a in n.attribute}
def dims(v):
    assert v.type.tensor_type.elem_type==onnx.TensorProto.FLOAT
    return [d.dim_value if d.HasField('dim_value') else None for d in v.type.tensor_type.shape.dim]
def inspect(model,name):
    nodes={n.name:n for n in model.graph.node};assert len(nodes)==len(model.graph.node)
    n=nodes[name];a=attrs(n)
    assert n.op_type=='SnakeDW7SnakeF32' and n.domain=='venky.audio.cpu.portable' and len(n.input)==7 and len(n.output)==1
    assert set(a)=={'native_abi','channels','dilation','backend','row_batches','require_vector_sine'}
    assert a['native_abi']==1 and a['backend']==5 and a['require_vector_sine']==1 and a['row_batches']>=0
    c,d=a['channels'],a['dilation'];assert c in (512,1024) and d in (1,3,9);h=6*d
    init={t.name:t for t in model.graph.initializer};inputs={v.name:v for v in model.graph.input};outputs={v.name:v for v in model.graph.output}
    coeff=[]
    for i,shape in zip(range(1,7),([c,1,7],[c],[c],[c],[c],[c])):
        t=init[n.input[i]];assert t.name not in inputs and t.data_type==onnx.TensorProto.FLOAT and list(t.dims)==shape
        assert np.isfinite(numpy_helper.to_array(t)).all();coeff.append(t)
    producer={v:n for n in model.graph.node for v in n.output};uses={}
    for row in model.graph.node:
        for v in row.input:uses.setdefault(v,[]).append(row)
    joined=producer[n.input[0]]
    assert joined.op_type=='Concat' and not joined.domain and attrs(joined)=={'axis':2} and len(joined.input)==2
    hist,x=joined.input;assert hist in inputs and dims(inputs[hist])==[1,c,h]
    consumers=uses[joined.output[0]];assert len(consumers)==2 and sum(z.name==name for z in consumers)==1
    retain=next(z for z in consumers if z.name!=name)
    outuses=uses[n.output[0]];assert len(outuses)==1;crop=outuses[0]
    for node,start in ((retain,-h),(crop,h)):
        assert node.op_type=='Slice' and not node.domain and not attrs(node) and len(node.input)==5 and len(node.output)==1
        values=[numpy_helper.to_array(init[q]).tolist() for q in node.input[1:]]
        assert values==[[start],[9223372036854775807],[2],[1]],values
    assert retain.output[0] in outputs and dims(outputs[retain.output[0]])==[1,c,h]
    assert joined.output[0] not in outputs and n.output[0] not in outputs
    return {'node':n,'attrs':a,'channels':c,'dilation':d,'halo':h,'x':x,'history':hist,
            'y':crop.output[0],'next':retain.output[0],'coefficients':coeff,
            'remove':[joined.name,retain.name,n.name,crop.name]}
def new_node(r):
    a={k:v for k,v in r['attrs'].items() if k!='native_abi'};a['candidate_abi']=1
    return helper.make_node(OP,[r['x'],r['history'],*r['node'].input[1:]],[r['y'],r['next']],
                            name=r['node'].name,domain=DOMAIN,**a)
def rewrite(model,name):
    original=copy.deepcopy(model);r=inspect(model,name);replacement=new_node(r)
    before=[v.SerializeToString() for v in (*model.graph.input,*model.graph.output)]
    weights=[v.SerializeToString() for v in model.graph.initializer]
    target=r['remove'][-1]
    nodes=[replacement if n.name==target else n for n in model.graph.node if n.name==target or n.name not in r['remove']]
    del model.graph.node[:];model.graph.node.extend(nodes)
    live={v for n in nodes for v in (*n.input,*n.output)}|{v.name for v in (*model.graph.input,*model.graph.output)}
    kept=[v for v in model.graph.value_info if v.name in live];del model.graph.value_info[:];model.graph.value_info.extend(kept)
    assert DOMAIN not in [o.domain for o in model.opset_import];model.opset_import.append(helper.make_opsetid(DOMAIN,1))
    assert before==[v.SerializeToString() for v in (*model.graph.input,*model.graph.output)]
    assert weights==[v.SerializeToString() for v in model.graph.initializer]
    unchanged={n.name:n.SerializeToString() for n in original.graph.node if n.name not in r['remove']}
    assert unchanged=={n.name:n.SerializeToString() for n in model.graph.node if n.name!=name}
    onnx.checker.check_model(model)
    return model,{'node':name,'channels':r['channels'],'dilation':r['dilation'],'halo':r['halo'],
        'removed':r['remove'],'inserted':name,'state_semantics':'unchanged raw pre-Snake history, oldest first',
        'inputs_outputs_identical':True,'all_initializers_identical':True,'all_other_nodes_identical':True}
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--source-sha',required=True)
    p.add_argument('--node',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert sha(a.source)==a.source_sha;a.output.mkdir(parents=True,exist_ok=False)
    m=onnx.load(a.source);m,report=rewrite(m,a.node);onnx.save(m,a.output/'candidate.onnx')
    report.update(source_sha256=a.source_sha,model_sha256=sha(a.output/'candidate.onnx'),
                  qualified=False,scope='Prepared single-node graph only; explicit library registration required')
    (a.output/'rewrite.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
