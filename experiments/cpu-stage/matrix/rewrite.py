"""Create a separate stage-selective pointwise fusion candidate; no inference.

Reuse the repository's checked shape proof, but apply changes to the original
graph so unselected stages and native operators remain byte-for-byte unchanged.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

DOMAIN='audio.cpu.pointwise.experimental'

def rewrite(source,output,channels=(256,128,64,32),mode=0,isa=0,tile_time=None,
            tile_channels=32,blocks_per_task=8,package_source=None,expected=None,native_chains=False):
    if package_source: sys.path.insert(0,str(Path(package_source).resolve()))
    import onnx
    from onnx import helper
    from fast_audiovae.graph.block_fusion import rewrite_model
    from fast_audiovae.graph.elementwise import copy_external_data,sha256
    source,output=Path(source),Path(output)
    if source.resolve()==output.resolve():raise ValueError('Output must differ from source')
    if not set(channels)<={32,64,128,256} or not channels:raise ValueError('Invalid stage channels')
    if mode not in (0,1,2) or isa not in (0,1,128,256,512):raise ValueError('Invalid backend')
    if mode and isa:raise ValueError('LIBXSMM modes require automatic ISA dispatch')
    if tile_time is not None and not 1<=tile_time<=256:raise ValueError('Invalid time tile')
    if not 1<=tile_channels<=64 or not 1<=blocks_per_task<=1024:raise ValueError('Invalid tile/task size')
    model=onnx.load(source,load_external_data=False)
    proven,audit=rewrite_model(model,source.parent,mode='both' if native_chains else 'adds')
    result=copy.deepcopy(model)
    producers={name:i for i,node in enumerate(model.graph.node) for name in node.output}
    if len(producers)!=sum(len(n.output) for n in model.graph.node):raise ValueError('Duplicate output names')
    users={}
    for i,node in enumerate(model.graph.node):
        for name in node.input:users.setdefault(name,[]).append(i)
    exposed={v.name for v in model.graph.output}
    proven_nodes={n.name:n for n in proven.graph.node}
    proven_init={t.name:t for t in proven.graph.initializer}
    original_init={t.name:t for t in model.graph.initializer}
    if DOMAIN in {o.domain for o in model.opset_import}:raise ValueError('Source already has experimental pointwise domain')
    replace={};remove=set();records=[];skipped=[];chain_records=[]
    occupied={n.name for n in model.graph.node}|{x for n in model.graph.node for x in (*n.input,*n.output)}
    for record in audit['changes']:
        if record['kind']!='adds' or record['channels'] not in channels:continue
        fused=proven_nodes[record['node']]
        product,bias,skip=fused.input
        mm_i=producers[product];mm=model.graph.node[mm_i]
        final_i=producers[fused.output[0]];final=model.graph.node[final_i]
        bias_i=producers[final.input[1]];bias_add=model.graph.node[bias_i]
        c=record['channels'];w=original_init.get(mm.input[0])
        reason=None
        if w is None or list(w.dims)!=[c,c]:reason='Square C=K pointwise matrix required'
        elif product in exposed or users.get(product)!=[bias_i]:reason='MatMul output has fanout or is graph output'
        elif bias_add.output[0] in exposed or users.get(bias_add.output[0])!=[final_i]:reason='Bias output has fanout'
        elif list(final.input)!=[skip,bias_add.output[0]] or list(bias_add.input)!=[product,record['bias_source']]:reason='Operand order mismatch'
        elif set((mm_i,bias_i,final_i))&remove:reason='Overlapping fusion chain'
        if reason:
            skipped.append({'node':mm.name,'reason':reason});continue
        name='fx_pointwise_'+str(final_i)
        if name in occupied:raise ValueError('Generated name collision')
        tt=tile_time or (64 if mode==0 else 8192//c)
        replace[final_i]=helper.make_node('PointwiseBiasResidualF32',
            [mm.input[0],mm.input[1],bias,skip],list(final.output),name=name,domain=DOMAIN,
            native_abi=1,channels=c,tile_time=tt,tile_channels=min(c,tile_channels),
            mode=mode,isa=isa,skip_first=1,blocks_per_task=blocks_per_task)
        result.graph.initializer.append(copy.deepcopy(proven_init[bias]))
        remove.update((mm_i,bias_i));occupied.add(name)
        records.append({'node':name,'channels':c,'matmul':mm.name,'bias_add':bias_add.name,
                        'residual_add':final.name,'weight':mm.input[0],
                        'bias_alias':bias,'bias_source':record['bias_source'],
                        'operand_order':'skip + (complete_FP32_dot + bias)',
                        'tile_time':tt,'mode':mode,'isa':isa})
    if not records:raise ValueError('No selected proven residual MatMul chains: '+str(skipped))
    if expected is not None and len(records)!=expected:raise ValueError(f'Expected {expected} fusions, got {len(records)}')
    # Compose only chain replacements proven against the same unchanged model.
    # Do not ask a generic matcher to trust the new experimental operator.
    if native_chains:
        for record in audit['changes']:
            if record['kind']!='chain':continue
            fused=proven_nodes[record['node']]
            final_i=producers[fused.output[0]];dw=model.graph.node[final_i]
            previous_i=producers.get(dw.input[0])
            if previous_i is None:raise ValueError('Proven chain lost its producer')
            previous=model.graph.node[previous_i]
            removed_chain={previous_i}
            if not previous.domain and previous.op_type=='Reshape':
                snake_i=producers.get(previous.input[0])
                if snake_i is None:raise ValueError('Proven chain lost its Snake producer')
                removed_chain.add(snake_i)
            touched=removed_chain|{final_i}
            if touched&(remove|set(replace)):raise ValueError('Matrix and native-chain replacements overlap')
            if fused.name in occupied:raise ValueError('Native-chain node name collision')
            replace[final_i]=copy.deepcopy(fused);remove.update(removed_chain);occupied.add(fused.name)
            chain_records.append(record)
    nodes=[replace.get(i,copy.deepcopy(n)) for i,n in enumerate(model.graph.node) if i not in remove]
    del result.graph.node[:];result.graph.node.extend(nodes)
    result.opset_import.append(helper.make_opsetid(DOMAIN,1))
    live={x for n in nodes for x in (*n.input,*n.output)}|{x.name for x in result.graph.input}|exposed
    infos=[v for v in result.graph.value_info if v.name in live]
    del result.graph.value_info[:];result.graph.value_info.extend(infos)
    for old,new in zip(model.graph.initializer,result.graph.initializer):
        if old.SerializeToString()!=new.SerializeToString():raise RuntimeError('Original initializer changed')
    output.parent.mkdir(parents=True,exist_ok=True)
    external=copy_external_data(result,source.parent,output.parent)
    onnx.save_model(result,output)
    onnx.checker.check_model(str(output),full_check=False)
    report={'source_sha256':sha256(source),'output_sha256':sha256(output),'external_files':external,
            'domain':DOMAIN,'changes':records,'skipped':skipped,'count':len(records),
            'native_chain_count':len(chain_records),'native_chain_changes':chain_records,
            'shape_proof':'fast_audiovae.graph.block_fusion.rewrite_model adds proof',
            'source_nodes':len(model.graph.node),'output_nodes':len(nodes),
            'original_initializers_unchanged':True,'inference_executed':False}
    output.with_suffix('.fusion.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--channels',default='256,128,64,32')
    p.add_argument('--mode',type=int,choices=[0,1,2],default=0)
    p.add_argument('--isa',type=int,choices=[0,1,128,256,512],default=0)
    p.add_argument('--tile-time',type=int);p.add_argument('--tile-channels',type=int,default=32)
    p.add_argument('--blocks-per-task',type=int,default=8)
    p.add_argument('--expected',type=int)
    p.add_argument('--native-chains',action='store_true',help='Compose proven Snake/DW/Snake chains from the same original source')
    p.add_argument('--package-source',type=Path,default=Path(__file__).resolve().parents[3]/'src')
    args=vars(p.parse_args());args['channels']=tuple(int(v) for v in args['channels'].split(','))
    print(json.dumps(rewrite(**args),indent=2))
