"""One-pass integration of explicitly named, numerically passed A2 regions.

No runtime execution or automatic speed selection. The original single-node
rewrite.py remains unchanged because it is a frozen running-screen dependency.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper
import rewrite as rw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,s):
    if not v:raise ValueError(s)
def qualification(screen_path,screen_sha,source_sha,names):
    require(sha(screen_path)==screen_sha,'Screen receipt SHA mismatch')
    s=json.loads(Path(screen_path).read_text())
    require(s.get('complete') is True and s.get('math_passed') is True and not s.get('probe_empty_control_only'),
            'Completed six-region numerical screen required')
    require(s.get('atol')==1e-5 and s.get('rtol')==1e-4,'Original numerical gates required')
    require(s.get('providers')==['CPUExecutionProvider'],'CPU screen required')
    before=s.get('artifact_hashes_before',{});require(before==s.get('artifact_hashes_after'),'Artifacts changed during screen')
    require(source_sha in before.values(),'Source graph was not the numerically screened graph')
    rows=s.get('math',[]);states=s.get('state_checks',[])
    require(len(rows)==72 and len(states)==12,'Wrong completed numerical/state inventory')
    patterns={'empty','single','tail3','short40','normal80','below_halo','at_halo','above_halo','tile_tail','zero','quiet','alternating_large'}
    all_names={r['node'] for r in rows};require(len(all_names)==6 and set(names)<=all_names,'Unscreened selected node')
    inventory={}
    for name in all_names:
        rs=[r for r in rows if r['node']==name];ss=[r for r in states if r['node']==name]
        require(len(rs)==12 and {r['pattern'] for r in rs}==patterns,'Missing/duplicate numerical pattern')
        require(len(ss)==2 and {r['arm'] for r in ss}=={'control','candidate'},'Missing state comparisons')
        require(all(r.get('raw_next_history_exact') and r.get('inputs_unchanged') for r in rs),'Raw-history/input gate failed')
        require(all(r.get('raw_history_exact') and r.get('old_states_unchanged') and r.get('independent_interleaving') for r in ss),
                'State behavior gate failed')
        require(len({(r['channels'],r['dilation']) for r in rs})==1,'Inconsistent screened geometry')
        c,d=rs[0]['channels'],rs[0]['dilation'];require(c in (512,1024) and d in (1,3,9),'Wrong screened geometry')
        expected_t={'empty':0,'single':1,'tail3':3,'short40':8 if c==1024 else 48,
                    'normal80':16 if c==1024 else 96,'below_halo':6*d-1,'at_halo':6*d,
                    'above_halo':6*d+1,'tile_tail':257,'zero':16 if c==1024 else 96,
                    'quiet':16 if c==1024 else 96,'alternating_large':16 if c==1024 else 96}
        require(all(r['T']==expected_t[r['pattern']] for r in rs),'Wrong tested shape inventory')
        errors=[r['waveform']['max_abs'] for r in rs]+[r[k]['max_abs'] for r in ss for k in ('partition','future_prefix')]
        require(all(np.isfinite(e) and e>=0 for e in errors),'Invalid numerical summary')
        inventory[name]={'channels':c,'dilation':d,'math_cases':12,'state_cases':2,
                         'maximum_reported_absolute_error':max(errors)}
    require({(r['channels'],r['dilation']) for r in inventory.values()}=={(c,d) for c in (512,1024) for d in (1,3,9)},
            'Incomplete six-region geometry inventory')
    measured={}
    for t in s.get('timings',[]):
        require(t['node'] in all_names,'Timing node outside math inventory')
        pairs=t['pairs'];require(len(pairs)==3,'Expected original three matched trials')
        for p in pairs:
            require(np.isfinite(p['control_seconds']) and np.isfinite(p['candidate_seconds']) and
                    p['control_seconds']>0 and p['candidate_seconds']>0,'Invalid timing')
        gains=[100*(1-p['candidate_seconds']/p['control_seconds']) for p in pairs]
        measured.setdefault(t['node'],[]).append({'T':t['T'],'median_paired_reduction_percent':float(np.median(gains)),
             'wins':sum(g>0 for g in gains),'pairs':3})
    return s,inventory,measured

def rewrite_many(model,names):
    require(names and len(names)==len(set(names)),'Explicit unique --node selections required')
    require(rw.DOMAIN not in [o.domain for o in model.opset_import],'Apply to the original tested control, not an existing A2 graph')
    regions=[rw.inspect(model,n) for n in names]
    counts=Counter(n for r in regions for n in r['remove']);require(all(c==1 for c in counts.values()),'Selected regions overlap')
    removed=set(counts);insert={r['remove'][-1]:rw.new_node(r) for r in regions}
    result=copy.deepcopy(model)
    nodes=[insert[n.name] if n.name in insert else n for n in result.graph.node if n.name in insert or n.name not in removed]
    del result.graph.node[:];result.graph.node.extend(nodes)
    live={v for n in nodes for v in (*n.input,*n.output)}|{v.name for v in (*result.graph.input,*result.graph.output)}
    metadata=[v for v in result.graph.value_info if v.name in live]
    del result.graph.value_info[:];result.graph.value_info.extend(metadata)
    result.opset_import.append(helper.make_opsetid(rw.DOMAIN,1))
    for field in ('input','output','initializer'):
        require([v.SerializeToString() for v in getattr(model.graph,field)]==
                [v.SerializeToString() for v in getattr(result.graph,field)],field+' protobuf changed')
    unchanged={n.name:n.SerializeToString() for n in model.graph.node if n.name not in removed}
    require(unchanged=={n.name:n.SerializeToString() for n in result.graph.node if n.name not in names},'Unselected node changed')
    expected_metadata=[v.SerializeToString() for v in model.graph.value_info if v.name in live]
    require(expected_metadata==[v.SerializeToString() for v in result.graph.value_info],'Retained source shape metadata changed')
    onnx.checker.check_model(result)
    changes=[{'node':r['node'].name,'channels':r['channels'],'dilation':r['dilation'],'halo':r['halo'],
              'raw_input':r['x'],'history_input':r['history'],'output':r['y'],'history_output':r['next'],
              'removed_nodes':r['remove'],'inserted_operator':rw.OP} for r in regions]
    return result,changes

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','screen','build','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('source-sha','screen-sha','build-sha'):p.add_argument('--'+k,required=True)
    p.add_argument('--node',action='append',required=True)
    a=p.parse_args();require(sha(a.source)==a.source_sha and sha(a.build)==a.build_sha,'Source/build hash mismatch')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report={'complete':False,'qualified_for_promotion':False,'scope':'Prepared selected A2 regions; no integrated numerical or timing run'}
    try:
        s,inventory,measured=qualification(a.screen,a.screen_sha,a.source_sha,a.node)
        require(a.build_sha in s['artifact_hashes_before'].values(),'Unscreened candidate build')
        build=json.loads(a.build.read_text());require(build.get('complete') is True,'Candidate build incomplete')
        library=Path(build['library']);require(sha(library)==build['library_sha256'],'Candidate library changed')
        require(build['library_sha256'] in s['artifact_hashes_before'].values(),'Candidate library was not screened')
        source=onnx.load(a.source,load_external_data=False)
        require(all(t.data_location!=onnx.TensorProto.EXTERNAL for t in source.graph.initializer),'Embedded source required')
        result,changes=rewrite_many(source,a.node)
        for row in changes:require((row['channels'],row['dilation'])==(inventory[row['node']]['channels'],inventory[row['node']]['dilation']),
                                  'Graph and numerical receipt geometry differ')
        onnx.save(result,out/'candidate.onnx')
        report.update(complete=True,source_sha256=a.source_sha,model_sha256=sha(out/'candidate.onnx'),
            screen_sha256=a.screen_sha,build_sha256=a.build_sha,library=str(library),library_sha256=sha(library),
            additional_domain=rw.DOMAIN,changes=changes,nodes_removed_net=3*len(changes),
            selected_numerical_evidence={n:inventory[n] for n in a.node},
            selected_regional_timing_evidence={n:measured.get(n,[]) for n in a.node},
            selected_nodes_without_regional_timing=[n for n in a.node if n not in measured],
            all_weights_and_external_schemas_identical=True,all_unselected_nodes_identical=True,
            live_source_value_info_preserved=True,state_semantics='unchanged raw pre-Snake history, oldest first',
            inference_executed=False,script_sha256=sha(__file__),single_node_helper_sha256=sha(Path(rw.__file__)))
        require(sha(a.source)==a.source_sha and sha(a.screen)==a.screen_sha and sha(a.build)==a.build_sha and
                sha(library)==build['library_sha256'],'Input artifact changed while integrating')
    except BaseException as e:report['error']=repr(e);raise
    finally:(out/'integration.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
