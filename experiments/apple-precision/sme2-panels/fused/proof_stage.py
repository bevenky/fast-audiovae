"""Checked, stage-selective derivative of the accepted native graph; no inference.

The repository block proof establishes FP32 constants, exact shape expressions,
MatMul and ordered bias/skip contracts. This adds exact three-unit connectivity,
view-only aliases, dilation 1/3/9 and zero external intermediate consumers.
Unselected graph nodes and every original initializer remain byte-identical.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

DOMAIN='fast.audiovae.stage.experimental'

def rewrite(source,output,channels=(256,),tile_time=256,segments=2,backend=2,
            matrix_mode=0,matrix_isa=128,package_source=None):
    if package_source:sys.path.insert(0,str(Path(package_source).resolve()))
    import onnx
    from onnx import helper,TensorProto
    from fast_audiovae.graph.block_fusion import rewrite_model
    from fast_audiovae.graph.elementwise import copy_external_data,sha256
    source,output=Path(source),Path(output)
    if source.resolve()==output.resolve():raise ValueError('Output must differ from source')
    if not channels or len(channels)!=len(set(channels)) or not set(channels)<={32,64,128,256}:raise ValueError('Invalid stage selection')
    if tile_time not in (64,128,256,512) or not 1<=segments<=64:raise ValueError('Invalid tile/segment count')
    if backend!=2 or matrix_mode not in (0,1,2):raise ValueError('Apple NEON backend2 and a supported mode required')
    if (matrix_mode and matrix_isa!=0) or (not matrix_mode and matrix_isa not in (1,128)):raise ValueError('Apple direct requires ISA128/1; LIBXSMM requires ISA0')
    onnx.checker.check_model(str(source),full_check=False)
    model=onnx.load(source,load_external_data=False)
    if DOMAIN in {v.domain for v in model.opset_import}:raise ValueError('Already a stage candidate')
    if {n.domain for n in model.graph.node if n.domain.startswith('venky.audio.cpu')}!={'venky.audio.cpu'}:raise ValueError('Apple native source required')
    proven,audit=rewrite_model(model,source.parent,mode='both')
    nodes=list(model.graph.node);producer={v:i for i,n in enumerate(nodes) for v in n.output}
    if len(producer)!=sum(len(n.output) for n in nodes):raise ValueError('Duplicate outputs')
    byname={n.name:i for i,n in enumerate(nodes)}
    if len(byname)!=len(nodes):raise ValueError('Unique node names required')
    users={}
    for i,n in enumerate(nodes):
        for v in n.input:users.setdefault(v,[]).append(i)
    exposed={v.name for v in model.graph.output}
    pnodes={n.name:n for n in proven.graph.node};pinit={t.name:t for t in proven.graph.initializer}
    initializers={t.name:t for t in model.graph.initializer}
    chains={r['depthwise_snake']:r for r in audit['changes'] if r['kind']=='chain'}
    units=[];skipped=[]
    def attrs(n):return {a.name:helper.get_attribute_value(a) for a in n.attribute}
    def views(v):
        consumed=[]
        while v in producer:
            i=producer[v];n=nodes[i]
            if n.domain or n.op_type!='Reshape':break
            if len(n.input)!=2 or len(n.output)!=1 or set(attrs(n))-{'allowzero'}:raise ValueError('Malformed view')
            consumed.append(i);v=n.input[0]
        return v,consumed
    for r in audit['changes']:
        if r['kind']!='adds' or r['channels'] not in channels:continue
        try:
            add=pnodes[r['node']];product,pbias,skip=add.input;c=r['channels']
            mm_i=producer[product];mm=nodes[mm_i];w=initializers.get(mm.input[0])
            if w is None or list(w.dims)!=[c,c] or w.data_type!=TensorProto.FLOAT:raise ValueError('Square FP32 pointwise required')
            dw_out,postviews=views(mm.input[1]);dw_i=producer[dw_out];dw=nodes[dw_i]
            if dw.name not in chains:raise ValueError('No proven pre-Snake/DW/post-Snake chain')
            chain=chains[dw.name];triple=pnodes[chain['node']]
            if chain['channels']!=c:raise ValueError('Channel mismatch')
            snake_i=byname[chain['pre_snake']];snake=nodes[snake_i]
            before,previews=views(snake.input[0])
            if before!=skip:raise ValueError('Residual does not skip the original unit input')
            snake_out,between=views(dw.input[0])
            if snake_out!=snake.output[0]:raise ValueError('Unexpected chain view')
            final_i=producer[add.output[0]];final=nodes[final_i];bias_i=producer[final.input[1]]
            remove={mm_i,dw_i,snake_i,bias_i,final_i,*previews,*postviews,*between}
            constants=list(triple.input[1:])+[mm.input[0],pbias]
            if len(constants)!=8:raise ValueError('Unexpected stage constants')
            units.append({'c':c,'input':skip,'output':final.output[0],'end':final_i,
                          'd':attrs(dw)['dilation'],'remove':remove,'constants':constants,
                          'nodes':[nodes[i].name for i in sorted(remove)],'bias':pbias})
        except (ValueError,KeyError,IndexError) as e:
            skipped.append({'node':r['residual_add'],'reason':str(e)})
    result=copy.deepcopy(model);replace={};remove=set();records=[]
    occupied={n.name for n in nodes}|{v for n in nodes for v in (*n.input,*n.output)}|set(initializers)
    for c in channels:
        selected=[u for u in units if u['c']==c]
        candidates=[]
        for first in selected:
            if first['d']!=1:continue
            seconds=[u for u in selected if u['d']==3 and u['input']==first['output']]
            if len(seconds)!=1:continue
            thirds=[u for u in selected if u['d']==9 and u['input']==seconds[0]['output']]
            if len(thirds)==1:candidates.append([first,seconds[0],thirds[0]])
        if len(candidates)!=1:raise ValueError(f'Expected exactly one proven C{c} three-unit stack, got {len(candidates)}; rejects={skipped}')
        stack=candidates[0];internal=set().union(*(u['remove'] for u in stack));last=stack[-1]['output']
        for i in internal:
            for v in nodes[i].output:
                if v==last:continue
                if v in exposed or any(user not in internal for user in users.get(v,[])):
                    raise ValueError(f'External intermediate consumer or graph output: {v}')
        if remove&internal:raise ValueError('Overlapping stack')
        name=f'sp_stage_c{c}'
        if name in occupied:raise ValueError('Generated name collision')
        inputs=[stack[0]['input']]+[v for u in stack for v in u['constants']]
        replace[stack[-1]['end']]=helper.make_node('StageStackF32',inputs,[last],name=name,domain=DOMAIN,
                    native_abi=1,channels=c,tile_time=tile_time,segments=segments,backend=backend,
                    matrix_mode=matrix_mode,matrix_isa=matrix_isa)
        for u in stack:result.graph.initializer.append(copy.deepcopy(pinit[u['bias']]))
        remove.update(internal-{stack[-1]['end']})
        records.append({'node':name,'channels':c,'input':inputs[0],'output':last,
                        'unit_outputs':[u['output'] for u in stack],
                        'unit_constants':[u['constants'] for u in stack],
                        'removed_nodes':[nodes[i].name for i in sorted(internal)],
                        'tile_time':tile_time,'segments':segments,'halo_frames':[6,18,54],
                        'warmup_per_noninitial_segment':78,'scratch_floats_per_worker':4*c*tile_time+78*c+566})
    kept=[replace.get(i,copy.deepcopy(n)) for i,n in enumerate(nodes) if i not in remove]
    del result.graph.node[:];result.graph.node.extend(kept)
    result.opset_import.append(helper.make_opsetid(DOMAIN,1))
    live={v for n in kept for v in (*n.input,*n.output)}|{v.name for v in model.graph.input}|exposed
    info=[v for v in result.graph.value_info if v.name in live]
    del result.graph.value_info[:];result.graph.value_info.extend(info)
    for old,new in zip(model.graph.initializer,result.graph.initializer):
        if old.SerializeToString()!=new.SerializeToString():raise RuntimeError('Original initializer changed')
    output.parent.mkdir(parents=True,exist_ok=True)
    external=copy_external_data(result,source.parent,output.parent)
    onnx.save_model(result,output);onnx.checker.check_model(str(output),full_check=False)
    report={'source_sha256':sha256(source),'output_sha256':sha256(output),'external_files':external,
            'changes':records,'skipped':skipped,'count':len(records),'source_nodes':len(nodes),'output_nodes':len(kept),
            'source_ir':model.ir_version,'output_ir':result.ir_version,
            'original_initializers_unchanged':True,'unselected_nodes_unchanged':True,
            'proof':'Repository exact shape and chain proof, then full stack connectivity and exclusive intermediate consumers',
            'matrix_rounding':'Complete-K FP32 FMA may differ from MLAS; strict output quality validation required',
            'inference_executed':False,'default_promoted':False}
    output.with_suffix('.stage.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--channels',default='256');p.add_argument('--tile-time',type=int,default=256)
    p.add_argument('--segments',type=int,default=2);p.add_argument('--backend',type=int,choices=(2,),default=2)
    p.add_argument('--matrix-mode',type=int,choices=(0,1,2),default=0)
    p.add_argument('--matrix-isa',type=int,choices=(0,1,128),default=128)
    p.add_argument('--package-source',type=Path,default=Path(__file__).resolve().parents[3]/'src')
    a=vars(p.parse_args());a['channels']=tuple(int(c) for c in a['channels'].split(','))
    print(json.dumps(rewrite(**a),indent=2))
