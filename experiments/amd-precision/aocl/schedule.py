"""Use the existing four-CPU AMD budget in the unchanged selective decoder math."""
import argparse,copy,hashlib,json,os
from pathlib import Path
import onnx
PIN='26b545641a43380b2f012c44f600f9e9f4413f424402bb35d9aa758b45849304'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser()
    for name in ('source','output','source-config','output-config'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-config-sha256',required=True);a=p.parse_args()
    if a.output.exists() or a.output_config.exists():raise ValueError('Fresh output files required')
    assert sha(a.source)==PIN and sha(a.source_config)==a.source_config_sha256
    original=onnx.load(a.source);model=copy.deepcopy(original);changed=[]
    for n in model.graph.node:
        attrs={x.name:x for x in n.attribute}
        if n.domain=='fast.audiovae.precision.matrix.experimental' and n.op_type=='PrecisionMatMulF32':
            assert attrs['shards'].i==2;attrs['shards'].i=4;changed.append({'name':n.name,'attribute':'shards','old':2,'new':4})
        elif n.op_type in ('StageStackF32','UpsampleStageF32'):
            assert n.domain in ('fast.audiovae.precision.stage.experimental','fast.audiovae.precision.upsample.experimental','fast.audiovae.stage.experimental')
            assert attrs['segments'].i==2;attrs['segments'].i=4;changed.append({'name':n.name,'attribute':'segments','old':2,'new':4})
    assert len(changed)==18 and sum(x['attribute']=='shards' for x in changed)==14
    assert all(a.SerializeToString()==b.SerializeToString() for a,b in zip(original.graph.initializer,model.graph.initializer))
    changed_names={x['name'] for x in changed}
    for old,new in zip(original.graph.node,model.graph.node):
        if old.name not in changed_names:assert old.SerializeToString()==new.SerializeToString()
        else:
            restored=copy.deepcopy(new)
            for at in restored.attribute:
                if at.name in ('shards','segments'):at.i=2
            assert restored.SerializeToString()==old.SerializeToString()
    onnx.checker.check_model(model,check_custom_domain=False);a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,a.output)
    cfg=json.loads(a.source_config.read_text());srcbase=a.source_config.resolve().parent;outbase=a.output_config.resolve().parent
    assert srcbase==outbase,'Keep sibling configs for transparent relative paths'
    candidate=next(x for x in cfg['models'] if x['name']=='int8_large');assert sha(srcbase/candidate['path'])==PIN
    oldgraph=candidate['path'];candidate['path']=os.path.relpath(a.output.resolve(),outbase);candidate['scheduling']='AMD AOCL INT8 + 4 native workers: 14 matrix row shards and all four stage segment counts set to four. ORT budget four, integer-library inner budget one.'
    del cfg['artifact_sha256'][oldgraph];cfg['artifact_sha256'][candidate['path']]=sha(a.output)
    cfg['source_config_sha256']=a.source_config_sha256;cfg['scope']='AMD AOCL selective INT8 with four native workers. One bounded architecture-specific port configuration; library and outer scheduling changed together versus the rejected MKL two-worker portability screen.'
    cfg['aocl_derivation']['outer_shards_segments']=4;cfg['aocl_derivation']['quality_status']='pending'
    cfg['scheduling_derivation']={'source_graph_sha256':PIN,'output_graph_sha256':sha(a.output),'source_config_sha256':a.source_config_sha256,'script_sha256':sha(Path(__file__)),'changes':changed,'initializer_bytes_unchanged':True,'unselected_nodes_unchanged':True,'full_K_unchanged':True,'native_execution':False,'numerical_parity':'pending'}
    a.output_config.write_text(json.dumps(cfg,indent=2)+'\n');audit=a.output.with_suffix('.schedule.json');audit.write_text(json.dumps(cfg['scheduling_derivation'],indent=2)+'\n')
    print(json.dumps({'graph_sha256':sha(a.output),'config_sha256':sha(a.output_config),'changed_nodes':len(changed)},indent=2))
if __name__=='__main__':main()
