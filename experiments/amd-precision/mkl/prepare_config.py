"""Prepare a hash-bound AMD four-model screen using the unchanged selective INT8 graph."""
import argparse,copy,hashlib,json,os
from pathlib import Path
GRAPH_SHA='26b545641a43380b2f012c44f600f9e9f4413f424402bb35d9aa758b45849304'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-config','graph','precision-build','fused-build','mkl-pins','mkl-library-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-config-sha256',required=True)
    a=p.parse_args();source=a.source_config.resolve();out=a.output.resolve();base=out.parent
    if out.exists():raise ValueError('Fresh config path required')
    if sha(source)!=a.source_config_sha256 or sha(a.graph)!=GRAPH_SHA:raise ValueError('Frozen source/graph hash differs')
    cfg=json.loads(source.read_text());models={m['name']:m for m in cfg['models']}
    if cfg['expected_case_count']!=60 or len(cfg['expected_uids'])!=60 or len(cfg['timing_uids'])!=10:raise ValueError('Original 60/10 cohort required')
    def resolve(value):return (source.parent/value).resolve()
    def rel(path):return os.path.relpath(Path(path).resolve(),base)
    oldpins={str(resolve(k)):v for k,v in cfg['artifact_sha256'].items()}
    artifact={}
    def add(path,expected=None):
        path=Path(path).resolve();actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('Artifact hash differs: '+str(path))
        artifact[rel(path)]=actual;return rel(path)
    def old(value):
        path=resolve(value)
        if str(path) not in oldpins:raise ValueError('Source artifact lacks hash: '+str(path))
        return add(path,oldpins[str(path)])
    def converted(model,name=None):
        m=copy.deepcopy(model)
        if name:m['name']=name
        m['path']=old(m['path'])
        if m.get('custom_library'):m['custom_library']=old(m['custom_library'])
        if m.get('custom_libraries'):m['custom_libraries']=[old(x) for x in m['custom_libraries']]
        m['external_data']=[old(x) for x in m.get('external_data',[])]
        return m
    stock=converted(models['audio_stock']);fast=converted(models['stage_late3'],'fast_fp32');mimi=converted(models['mimi'])
    precision=json.loads(a.precision_build.read_text());fused=json.loads(a.fused_build.read_text())
    core=Path(precision['core_path']);ops=Path(precision['ops_path'])
    add(core,precision['libraries'][core.name]);add(ops,precision['libraries'][ops.name])
    library_paths=[resolve(x) for x in models['stage_late3']['custom_libraries']]
    library_paths+=[ops,*map(Path,fused['libraries'])]
    for path in map(Path,fused['libraries']):add(path,fused['sha256'][str(path)])
    mkl=json.loads(a.mkl_pins.read_text())
    for name,digest in mkl['cpu_library_sha256'].items():add(a.mkl_library_dir/name,digest)
    candidate={'name':'int8_large','kind':'audio','causal':True,'approximate':True,'path':add(a.graph,GRAPH_SHA),
               'external_data':[],'custom_libraries':[rel(x) for x in library_paths],
               'precision_description':'Same 22 selective matrix products and quantization as accepted Intel: per-output-row weight scales, per-time-column activation scales, full-K INT32 dot and ordered FP32 output. All nonlinearities and waveform interfaces remain FP32.',
               'scheduling':'Unchanged Intel graph: 2 shards/segments, run inside the same 4-thread ORT budget as all compared models.'}
    cfg['models']=[stock,fast,candidate,mimi]
    for kind in ('audio','mimi'):
        value=cfg[kind+'_cases'];cfg[kind+'_cases']=old(value);old(str(Path(value).with_suffix('.json')))
    cfg['artifact_sha256']=artifact
    cfg['scope']='AMD CPU-only selective INT8 portability screen. No training, weight change, GPU or FP16 trial. Perceptual quality is separate from FP32 runtime gates.'
    cfg['source_config_sha256']=a.source_config_sha256
    cfg['preparation']={'script_sha256':sha(Path(__file__)),'precision_build_sha256':sha(a.precision_build),'fused_build_sha256':sha(a.fused_build),'graph_sha256':GRAPH_SHA,'math_changes_vs_accepted_intel':False,'quality_validated_on_amd':False}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps({'config':str(out),'sha256':sha(out),'models':[m['name'] for m in cfg['models']],'artifacts':len(artifact)},indent=2))
if __name__=='__main__':main()
