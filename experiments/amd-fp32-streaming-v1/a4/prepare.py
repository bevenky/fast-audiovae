"""Extract and rewrite authenticated five-node phase regions; no inference."""
from pathlib import Path
import argparse, copy, hashlib, json
import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as TP

DOMAIN='fast.audiovae.amd.phase.state.v1'
T80={1024:2,512:16,256:96,128:480,64:960,32:1920}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def attrs(n):return {a.name:H.get_attribute_value(a) for a in n.attribute}
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def rewrite(model, selected_names=()):
    require(not model.functions and not model.graph.sparse_initializer,'Flat dense graph required')
    require(not any(a.type in (onnx.AttributeProto.GRAPH,onnx.AttributeProto.GRAPHS)
                    for n in model.graph.node for a in n.attribute),'Nested graphs unsupported')
    source=copy.deepcopy(model);nodes=list(source.graph.node)
    require(len({n.name for n in nodes})==len(nodes),'Unique node names required')
    constants={w.name:w for w in source.graph.initializer}
    inputs={v.name:v for v in source.graph.input};outputs={v.name:v for v in source.graph.output}
    require(not set(constants)&set(inputs),'Overridable initializers unsupported')
    consumers={}
    for n in nodes:
        for name in n.input:consumers.setdefault(name,[]).append(n.name)
    picked=set(selected_names);found=set();regions=[];remove=set();replace={}
    for i,phase in enumerate(nodes):
        if phase.op_type!='PhaseSumBiasInterleaveF32':continue
        a=attrs(phase)
        if picked and phase.name not in picked:continue
        if not picked and a.get('channels')!=1024:continue
        require(3<=i<len(nodes)-1,'Incomplete phase region')
        prev,retain,cur,_,crop=nodes[i-3:i+2];c=a.get('channels');s=a.get('stride')
        require(c in T80 and s=={1024:8,512:6,256:5,128:2,64:2,32:2}[c],'Unexpected geometry')
        require(a==dict(channels=c,stride=s,native_abi=1,previous_shift=1,row_batches=0),'Phase attributes changed')
        require(phase.domain in ('venky.audio.cpu','venky.audio.cpu.portable'),'Unexpected baseline domain')
        require([n.op_type for n in (prev,retain,cur,phase,crop)]==
                ['Concat','Slice','Concat','PhaseSumBiasInterleaveF32','Slice'],'Unexpected phase topology')
        require(all(n.domain in ('','ai.onnx') for n in (prev,retain,cur,crop)),'Nonstandard wrapper')
        require(len(prev.input)==len(cur.input)==2 and len(phase.input)==3
                and all(len(n.output)==1 for n in (prev,retain,cur,phase,crop)),'Changed IO arity')
        require(attrs(prev)==attrs(cur)=={'axis':2},'Changed concatenation axes')
        require(prev.input[0] in inputs and retain.output[0] in outputs,'Expected external projected state')
        for value in (inputs[prev.input[0]],outputs[retain.output[0]]):
            t=value.type.tensor_type
            require(t.elem_type==TP.FLOAT and [d.dim_value for d in t.shape.dim]==[1,c*s,1],'State contract changed')
        require(list(phase.input[:2])==[cur.output[0],prev.output[0]] and
                retain.input[0]==prev.output[0] and crop.input[0]==phase.output[0],'Phase wiring changed')
        require(sorted(consumers[prev.output[0]])==sorted([retain.name,phase.name])
                and consumers[cur.output[0]]==[phase.name]
                and consumers[phase.output[0]]==[crop.name],'Additional intermediate consumer')
        require(not {prev.output[0],cur.output[0],phase.output[0]}&set(outputs),'Intermediate graph output')
        z=N.to_array(constants[cur.input[0]])
        require(z.dtype==np.float32 and z.shape==(1,c*s,1) and not z.view(np.uint32).any(),'Positive-zero pad required')
        for n,start in ((retain,-1),(crop,s)):
            require(len(n.input)==5 and not n.attribute,'Changed slice attributes')
            require([N.to_array(constants[x]).tolist() for x in n.input[1:]]==
                    [[start],[9223372036854775807],[2],[1]],'Changed slice constants')
            require(all(constants[x].data_type==TP.INT64 for x in n.input[1:]),'Slice dtype changed')
        bias=constants[phase.input[2]];b=N.to_array(bias)
        require(b.dtype==np.float32 and b.shape==(c,) and np.isfinite(b).all(),'Fixed finite bias[C] required')
        replacement=H.make_node('StatefulPhaseFinishF32',
            [cur.input[1],prev.input[1],prev.input[0],phase.input[2]],
            [crop.output[0],retain.output[0]],name=phase.name+'_a4_direct',domain=DOMAIN,
            channels=c,stride=s,phase_state_abi=1,threads=1)
        require(replacement.name not in {n.name for n in nodes},'Replacement name collision')
        group=[prev,retain,cur,phase,crop];names={n.name for n in group}
        require(not names&remove,'Overlapping replacement');remove|=names;replace[crop.name]=replacement
        mapping={cur.input[1]:'current',prev.input[1]:'previous',prev.input[0]:'history',
                 crop.output[0]:'audio',retain.output[0]:'next_history'}
        def region(ns):
            ns=copy.deepcopy(ns)
            for n in ns:
                for j,x in enumerate(n.input):n.input[j]=mapping.get(x,x)
                for j,x in enumerate(n.output):n.output[j]=mapping.get(x,x)
            used={x for n in ns for x in n.input if x in constants}
            ins=[H.make_tensor_value_info(x,TP.FLOAT,[1,c*s,'T']) for x in ('current','previous')]
            ins.append(H.make_tensor_value_info('history',TP.FLOAT,[1,c*s,1]))
            outs=[H.make_tensor_value_info('audio',TP.FLOAT,[1,c,'TS']),H.make_tensor_value_info('next_history',TP.FLOAT,[1,c*s,1])]
            ops=[copy.deepcopy(o) for o in source.opset_import if o.domain!=DOMAIN]+[H.make_opsetid(DOMAIN,1)]
            m=H.make_model(H.make_graph(ns,'a4_phase',ins,outs,[copy.deepcopy(constants[x]) for x in sorted(used)]),
                           ir_version=source.ir_version,opset_imports=ops)
            onnx.checker.check_model(m);return m
        regions.append({'node':phase.name,'channels':c,'stride':s,'T40':T80[c]//2,'T80':T80[c],
                        'removed_nodes':[n.name for n in group], 'inputs':list(replacement.input),
                        'outputs':list(replacement.output),'bias_name':bias.name,'bias_sha256':hashlib.sha256(b.tobytes()).hexdigest(),
                        'baseline':region(group),'candidate':region([replacement])})
        found.add(phase.name)
    require(found==picked if picked else len(found)==1,'Selected region inventory mismatch')
    new=copy.deepcopy(source);kept=[replace[n.name] if n.name in replace else n for n in nodes if n.name not in remove or n.name in replace]
    del new.graph.node[:];new.graph.node.extend(kept)
    require(DOMAIN not in {o.domain for o in new.opset_import},'Candidate domain already present')
    new.opset_import.append(H.make_opsetid(DOMAIN,1))
    for field in ('input','output','initializer'):
        require([x.SerializeToString() for x in getattr(source.graph,field)]==
                [x.SerializeToString() for x in getattr(new.graph,field)],'Changed '+field)
    onnx.checker.check_model(new)
    return new,regions

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--expected-source-sha256',required=True);ap.add_argument('--node',action='append',default=[])
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    require(not a.output.exists(),'Output exists');require(sha(a.source)==a.expected_source_sha256,'Source digest mismatch')
    m=onnx.load(a.source,load_external_data=False)
    require(not any(w.data_location==TP.EXTERNAL for w in m.graph.initializer),'Embedded source graph required')
    candidate,rows=rewrite(m,a.node);a.output.mkdir(parents=True)
    onnx.save(candidate,a.output/'overlay-unqualified.onnx')
    for i,row in enumerate(rows):
        for arm in ('baseline','candidate'):
            p=a.output/f'region-{i}-{arm}.onnx';onnx.save(row.pop(arm),p)
            row[arm+'_file']=p.name;row[arm+'_sha256']=sha(p)
    require(sha(a.source)==a.expected_source_sha256,'Source changed during preparation')
    dump(a.output/'plan.json',{'version':'amd_phase_state_plan_v1','status':'prepared_unqualified',
        'source':str(a.source.resolve()),'source_sha256':a.expected_source_sha256,
        'overlay':'overlay-unqualified.onnx','overlay_sha256':sha(a.output/'overlay-unqualified.onnx'),
        'initializer_count':len(m.graph.initializer),'inputs_unchanged':True,'outputs_unchanged':True,
        'initializers_byte_identical':True,'projection_nodes_unchanged':True,'regions':rows,
        'prepare_source_sha256':sha(__file__)})
    print(str((a.output/'plan.json').resolve()))
if __name__=='__main__':main()
